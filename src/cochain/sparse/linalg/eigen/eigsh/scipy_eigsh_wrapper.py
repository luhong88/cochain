from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass, replace
from typing import Literal

import numpy as np
import scipy.sparse
import scipy.sparse.linalg
import torch
from jaxtyping import Float, Integer
from torch import Tensor

from .....utils.parsing import to_np
from ....decoupled_tensor import SparseDecoupledTensor, SparsityPattern
from ....decoupled_tensor._conversion import sdt_to_scipy_csc, sdt_to_scipy_csr
from ..base._backward import dLdA_backward, dLdA_dLdM_backward


@dataclass
class SciPyEigshConfig:
    """
    A dataclass encapsulating SciPy `eigsh()` optional parameters.

    Parameters
    ----------
    sigma
        Whether to find eigenvalues near sigma using shift-invert mode.
    which
        Which $k$ eigenvalues and eigenvectors to find.
    v0
        Starting vectors for iteration. If the input matrix is block-diagonal,
        then `v0` should be a sequence of arrays, one for each batch element.
    nvc
        The number of Lanczos vectors generated.
    maxiter
        Maximum number of Arnoldi update iterations allowed.
    tol
        Relative accuracy for eigenvalues (stopping criterion).
    mode
        Specify strategy to use for shift-invert mode.

    Notes
    -----
    The `Minv` and `OPinv` arguments to SciPy `eigsh()` are not supported.
    """

    sigma: float | None = None
    which: Literal["LM", "SM", "LA", "SA", "BE"] = "LM"
    v0: (
        Float[Tensor | np.typing.NDArray, " c"]
        | Sequence[Float[Tensor | np.typing.NDArray, " c"] | None]
        | None
    ) = None
    ncv: int | None = None
    maxiter: int | None = None
    tol: float | int = 0
    mode: Literal["normal", "buckling", "cayley"] = "normal"

    def __post_init__(self):
        match self.v0:
            case Tensor():
                self.v0 = to_np(self.v0, contiguous=True)
            case Sequence():
                v0_list = []
                for v0 in self.v0:
                    match v0:
                        case Tensor():
                            v0_list.append(to_np(v0, contiguous=True))
                        case _:
                            v0_list.append(v0)
                self.v0 = v0_list

    def expand(self, n: int) -> list[SciPyEigshConfig]:
        """
        Duplicate self for batched processing.

        Parameters
        ----------
        n
            How many times to duplicate self. If `v0` is a list of arrays, then
            `n` must be equal to the length of `v0`.

        Returns
        -------
        config_list
            A list of `SciPyEigshConfig` duplicated from self. If `v0` is a list
            of arrays, then each duplicate is assigned one element from `v0`.
        """
        config_list = []
        if isinstance(self.v0, list):
            config_list.append(replace(self, v0=v0) for v0 in self.v0)
            if len(config_list) != n:
                raise ValueError("Inconsistent v0 specification.")
        else:
            config_list = [self] * n

        return config_list


class SciPyEigshStandardAutogradFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        a_val: Float[Tensor, " nz"],
        a_pattern: Integer[SparsityPattern, "r c"],
        k: int,
        eps: float | int,
        compute_eig_vecs: bool,
        config: SciPyEigshConfig,
    ) -> tuple[Float[Tensor, " k"], Float[Tensor, "c k"] | None]:
        # When solving the standard A@x=λx, the CSR format is preferred for
        # matrix-vector multiplication.
        if config.sigma is None:
            a_scipy = sdt_to_scipy_csr(a_val, a_pattern)

        # In the shift-invert mode, an LU factorization of A + σI is required,
        # and therefore the CSC format is preferred.
        else:
            a_scipy = sdt_to_scipy_csc(a_val, a_pattern)

        # In both cases, scipy supports CSC/CSR arrays with int64 index tensors.
        results = scipy.sparse.linalg.eigsh(
            A=a_scipy,
            k=k,
            return_eigenvectors=compute_eig_vecs,
            **asdict(config),
        )

        if compute_eig_vecs:
            eig_vals_np, eig_vecs_np = results

            eig_vecs = torch.from_numpy(eig_vecs_np).to(
                dtype=a_val.dtype, device=a_val.device
            )

        else:
            eig_vals_np = results
            eig_vecs = None

        eig_vals = torch.from_numpy(eig_vals_np).to(
            dtype=a_val.dtype, device=a_val.device
        )

        return eig_vals, eig_vecs

    @staticmethod
    def setup_context(ctx, inputs, output):
        a_val, a_pattern, k, eps, compute_eig_vecs, config = inputs
        eig_vals, eig_vecs = output

        ctx.save_for_backward(eig_vals, eig_vecs)
        ctx.a_pattern = a_pattern
        ctx.eps = eps

    @staticmethod
    def backward(
        ctx, dLdl: Float[Tensor, " k"], dLdv: Float[Tensor, "c k"] | None
    ) -> tuple[
        Float[Tensor, " nz"] | None,
        None,
        None,
        None,
        None,
        None,
    ]:
        needs_grad_A_val = ctx.needs_input_grad[0]

        dLdA_val = dLdA_backward(ctx, dLdl, dLdv) if needs_grad_A_val else None

        return dLdA_val, None, None, None, None, None


class SciPyEigshGEPAutogradFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        a_val: Float[Tensor, " A_nz"],
        a_pattern: Integer[SparsityPattern, "r c"],
        m_val: Float[Tensor, " M_nz"],
        m_pattern: Integer[SparsityPattern, "r c"],
        k: int,
        eps: float | int,
        compute_eig_vecs: bool,
        config: SciPyEigshConfig,
    ) -> tuple[Float[Tensor, " k"], Float[Tensor, "c k"] | None]:
        # When solving the standard A@x=λx, the CSR format is preferred for
        # matrix-vector multiplication.
        if config.sigma is None:
            a_scipy = sdt_to_scipy_csr(a_val, a_pattern)

        # In the shift-invert mode, an LU factorization of A + σI is required,
        # and therefore the CSC format is preferred.
        else:
            a_scipy = sdt_to_scipy_csc(a_val, a_pattern)

        # M should always be in CSC format for LU factorization.
        m_scipy = sdt_to_scipy_csc(m_val, m_pattern)

        # In both cases, scipy supports CSC/CSR arrays with int64 index tensors.
        results = scipy.sparse.linalg.eigsh(
            A=a_scipy,
            k=k,
            M=m_scipy,
            return_eigenvectors=compute_eig_vecs,
            **asdict(config),
        )

        if compute_eig_vecs:
            eig_vals_np, eig_vecs_np = results

            eig_vecs = torch.from_numpy(eig_vecs_np).to(
                dtype=a_val.dtype, device=a_val.device
            )

        else:
            eig_vals_np = results
            eig_vecs = None

        eig_vals = torch.from_numpy(eig_vals_np).to(
            dtype=a_val.dtype, device=a_val.device
        )

        return eig_vals, eig_vecs

    @staticmethod
    def setup_context(ctx, inputs, output):
        a_val, a_pattern, m_val, m_pattern, k, eps, compute_eig_vecs, config = inputs
        eig_vals, eig_vecs = output

        ctx.save_for_backward(eig_vals, eig_vecs)
        ctx.a_pattern = a_pattern
        ctx.m_pattern = m_pattern
        ctx.eps = eps

    @staticmethod
    def backward(
        ctx, dLdl: Float[Tensor, " k"], dLdv: Float[Tensor, "c k"] | None
    ) -> tuple[
        Float[Tensor, " A_nz"] | None,
        None,
        Float[Tensor, " M_nz"] | None,
        None,
        None,
        None,
        None,
        None,
    ]:
        needs_grad_A_val = ctx.needs_input_grad[0]
        needs_grad_M_val = ctx.needs_input_grad[2]

        dLdA_val, dLdM_val = dLdA_dLdM_backward(
            ctx, dLdl, dLdv, needs_grad_A_val, needs_grad_M_val
        )

        return dLdA_val, None, dLdM_val, None, None, None, None, None


def _scipy_eigsh_no_batch(
    a: Float[SparseDecoupledTensor, "r c"],
    m: Float[SparseDecoupledTensor, "r c"] | None,
    k: int,
    eps: float | int,
    compute_eig_vecs: bool,
    config: SciPyEigshConfig,
) -> tuple[Float[Tensor, " k"], Float[Tensor, "c k"]]:
    """Dispatch the autograd functions whn there is no block-diagonal batching."""
    if m is None:
        eig_vals, eig_vecs = SciPyEigshStandardAutogradFunction.apply(
            a.values, a.pattern, k, eps, compute_eig_vecs, config
        )
    else:
        eig_vals, eig_vecs = SciPyEigshGEPAutogradFunction.apply(
            a.values, a.pattern, m.values, m.pattern, k, eps, compute_eig_vecs, config
        )

    return eig_vals, eig_vecs


def _scipy_eigsh_batch(
    a_batched: Float[SparseDecoupledTensor, "r c"],
    m_batched: Float[SparseDecoupledTensor, "r c"] | None,
    k: int,
    eps: float | int,
    compute_eig_vecs: bool,
    config_batched: SciPyEigshConfig,
) -> tuple[Float[Tensor, "b k"], Float[Tensor, "c k"]]:
    """Dispatch the autograd functions when there is block-diagonal batching."""
    # Unpack the A, M (if not None), and v0 (if not None) for looping
    a_list = a_batched.unpack_block_diag()
    if m_batched is None:
        m_list = [None] * len(a_list)
    else:
        m_list = m_batched.unpack_block_diag()

    config_list = config_batched.expand(n=len(a_list))

    eig_val_list = []
    eig_vec_list = []
    for a_sdt, m_sdt, config in zip(a_list, m_list, config_list, strict=True):
        eig_val, eig_vec = _scipy_eigsh_no_batch(
            a_sdt, m_sdt, k, eps, compute_eig_vecs, config
        )
        eig_val_list.append(eig_val)
        eig_vec_list.append(eig_vec)

    eig_vals = torch.vstack(eig_val_list)

    if eig_vec_list[0] is None:
        eig_vecs = None
    else:
        eig_vecs = torch.vstack(eig_vec_list)

    return eig_vals, eig_vecs


def scipy_eigsh(
    a: Float[SparseDecoupledTensor, "r c"],
    m: Float[SparseDecoupledTensor, "r c"] | None = None,
    *,
    block_diag_batch: bool = False,
    k: int = 6,
    eps: float | int = 1e-6,
    return_eigenvectors: bool = True,
    config: SciPyEigshConfig | None = None,
) -> Float[Tensor, "*b k"] | tuple[Float[Tensor, "*b k"], Float[Tensor, "coord k"]]:
    """
    Sparse eigensolver for symmetric square matrices using SciPy.

    This function provides a differentiable wrapper for the CPU-based
    `scipy.sparse.linalg.eigsh()` method.

    Parameters
    ----------
    a : [r, c]
        A real symmetric square sparse matrix.
    m : [r, c]
        A real symmetric square sparse matrix that induces an inner product
        on the column space of `a`. If `m` is provided, solve a generalized
        eigenvalue problem.
    block_diag_batch
        Whether the input `a` matrix (and `m` if not `None`) is block-diagonal.
        If `a` and `m` are block-diagonal, then they must both have valid and
        matching `block_diag_config`, in which case each block/batch element
        will be solved sequentially. Note that the `eigsh()` function does not
        natively support batching.
    k
        The number of eigenvalues/eigenvectors to find.
    eps
        The strength of Lorentzian broadening/regularization, which removes
        singularities in backward gradient calculation when some of the
        eigenvalues are (near) degenerate. Set to integer 0 to disable regularization.
    return_eigenvectors
        Whether to compute and return the eigenvectors in addition to the eigenvalues.
        Note that, if `a` and/or `m` requires gradient tracking, the eigenvectors
        will be computed regardless of this argument.
    config
        Additional optional arguments for `eigsh()`.

    Returns
    -------
    eig_vals : [*b, k]
        A tensor of `k` eigenvalues. If `block_diag_batch` is `True`, then the
        tensor also has a leading batch dimension corresponding to the blocks
        in the input `a` matrix.
    eig_vecs : [coord, k]
        A tensor of `k` M-orthonormal eigenvectors. If `block_diag_batch` is `False`,
        then each column represents an eigenvector; if `block_diag_batch` is `True`,
        then the eigenvectors for each block are stacked along the first dimension.
        If `return_eigenvectors` is False, then this tensor is not returned.

    Notes
    -----
    The autograd through eigenvectors do not account for contributions from the
    unresolved eigenvectors.

    The `a` and `m` matrices will be converted to SciPy CSR/CSC arrays and copied
    to CPU. The sparse CSR/CSC index tensors of `a` and `m` can be either in `int32`
    or `int64` dtype, but will be automatically downcast to `int32` if possible.
    The use of SciPy `LinearOperator` objects for `a` and `m` is not supported.

    For finding the lowest non-zero eigenvalues of a positive semi-definite matrix,
    one can either use `which='SM'` or `which='LM'` with a `sigma` close to zero.
    While the latter approach typically lead to faster convergence, it has higher
    memory requirement due to the need to factorize the matrix.
    """
    # Eigenvectors are required for backward().
    compute_eig_vecs = return_eigenvectors
    if a.requires_grad:
        compute_eig_vecs = True
    if (m is not None) and m.requires_grad:
        compute_eig_vecs = True

    if config is None:
        config = SciPyEigshConfig()

    if block_diag_batch:
        eig_vals, eig_vecs = _scipy_eigsh_batch(a, m, k, eps, compute_eig_vecs, config)
    else:
        eig_vals, eig_vecs = _scipy_eigsh_no_batch(
            a, m, k, eps, compute_eig_vecs, config
        )

    if return_eigenvectors:
        return eig_vals, eig_vecs
    else:
        return eig_vals

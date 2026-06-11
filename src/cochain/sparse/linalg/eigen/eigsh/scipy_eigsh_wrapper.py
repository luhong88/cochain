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
        config_list = []
        if isinstance(self.v0, list):
            config_list.append(replace(self, v0=v0) for v0 in self.v0)
            if len(config_list) != n:
                raise ValueError("Inconsistent v0 specification.")
        else:
            config_list = [self] * n

        return config_list


class _SciPyEigshStandardAutogradFunction(torch.autograd.Function):
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


class _SciPyEigshGEPAutogradFunction(torch.autograd.Function):
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
    if m is None:
        eig_vals, eig_vecs = _SciPyEigshStandardAutogradFunction.apply(
            a.values, a.pattern, k, eps, compute_eig_vecs, config
        )
    else:
        eig_vals, eig_vecs = _SciPyEigshGEPAutogradFunction.apply(
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
) -> tuple[Float[Tensor, "*b k"], Float[Tensor, "c k"] | None]:
    """
    Sparse eigensolver for symmetric square matrices using SciPy.

    This function provides a differentiable wrapper for the CPU-based
    `scipy.sparse.linalg.eigsh()` method.

    The API for `eigsh()` is almost reproduced one-to-one, with the following
    exceptions:

    * The arguments `sigma`, `which`, `v0`, `ncv`, `maxiter`, `tol`, and `mode`
      are collected in a `ScipyEigshConfig` dataclass object, whilc the rest of
      the arguments are exposed as direct arguments to this function.
    * The `A` and `M` matrices must be `SparseDecoupledTensor` objects and will be
      converted to SciPy CSR/CSC arrays and copied to CPU. The `v0` argument
      can be a torch tensor, but will be converted to a numpy array and copied
      to CPU. The use of SciPy `LinearOperator` objects for `A` and `M` is not
      supported.
    * The `Minv` and `OPinv` arguments are not supported.
    * The `eps` argument is used for Lorentzian broadening/regularization in the
      gradient calculation to prevent gradient explosion when the eigenvalues are
      (near) degenerate.

    Note on performance:

    * If either `A` or `M` requires gradient tracking, eigenvectors will be
      computed; in this case, if `return_eigenvectors=False`, the computed
      eigenvectors are not returned.
    * The autograd through eigenvectors do not account for contributions from the
      unresolved eigenvectors.
    * The sparse CSR/CSC index tensors of `A` and `M` can be either in int32 or
      int64 dtype, but will be automatically downcast to int32 if possible.
    * The `eigsh()` function does not natively support batching. if
      `block_diag_batch` is True, the `A` `SparseDecoupledTensor` (and `M` if not `None`)
      will be split into individual sparse matrices and solved sequentially. The
      resulting eigenvalue tensor will contain a leading batch dimension, but the
      resulting eigenvector tensor will respect the original concatenated/packed
      format. This requires that `A` (and `M` if not `None`) has a valid
      `BlockDiagConfig` for unpacking the block diagonal batch structure.

    Note on shift-invert mode: for finding the lowest non-zero eigenvalues of a
    positive semi-definite matrix, one can either use `which='SM'` or `which='LM'`
    with a `sigma` close to zero. While the latter approach typically lead to
    faster convergence, it has higher memory requirement due to the need to factorize
    the matrix.
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

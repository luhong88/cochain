from __future__ import annotations

__all__ = ["CuPyEigshConfig", "cupy_eigsh"]

from collections.abc import Sequence
from dataclasses import asdict, dataclass, replace
from typing import Literal

import torch
from jaxtyping import Float, Integer
from torch import Tensor

from .....utils.stream import cupy_in_torch_stream
from ....decoupled_tensor import SparseDecoupledTensor, SparsityPattern
from ....decoupled_tensor._conversion import sdt_to_cupy_csr
from ...solvers import DirectSolverConfig
from ..base._backward import dLdA_backward
from ..base.utils import compute_lorentzian_eps
from ._cupy_eigsh_operators import CuPyShiftInvSymOp

try:
    import cupy as cp
    import cupyx.scipy.sparse.linalg as cp_sp_linalg

    _HAS_CUPY = True

except ImportError:
    _HAS_CUPY = False

try:
    import nvmath.sparse.advanced

    _HAS_NVMATH = True

except ImportError:
    _HAS_NVMATH = False


@dataclass
class CuPyEigshConfig:
    """
    A dataclass encapsulating CuPy `eigsh()` optional parameters.

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
    """

    sigma: float | None = None
    which: Literal["LM", "LA", "SA"] = "LM"
    v0: (
        Float[Tensor | cp.ndarray, " c"]
        | Sequence[Float[Tensor | cp.ndarray, " c"] | None]
        | None
    ) = None
    ncv: int | None = None
    maxiter: int | None = None
    tol: float | int = 0

    def __post_init__(self):
        if not _HAS_CUPY:
            raise ImportError("CuPy backend required.")

        match self.v0:
            case Tensor():
                self.v0 = cp.from_dlpack(self.v0.detach().contiguous())
            case Sequence():
                v0_list = []
                for v0 in self.v0:
                    match v0:
                        case Tensor():
                            v0_list.append(cp.from_dlpack(v0.detach().contiguous()))
                        case _:
                            v0_list.append(v0)
                self.v0 = v0_list

    def expand(self, n: int) -> list[CuPyEigshConfig]:
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
            A list of `CuPyEigshConfig` duplicated from self. If `v0` is a list
            of arrays, then each duplicate is assigned one element from `v0`.
        """
        if isinstance(self.v0, list):
            config_list = [replace(self, v0=v0) for v0 in self.v0]
            if len(config_list) != n:
                raise ValueError("Inconsistent v0 specification.")
        else:
            config_list = [self] * n

        return config_list


class CuPyEigshAutogradFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        a_val: Float[Tensor, " nz"],
        a_pattern: Integer[SparsityPattern, "r c"],
        k: int,
        eps: float | int,
        compute_eig_vecs: bool,
        cp_config: CuPyEigshConfig,
        nvmath_config: DirectSolverConfig | None,
    ) -> tuple[Float[Tensor, " k"], Float[Tensor, "c k"] | None]:
        # Force CuPy to use the current Pytorch stream.
        with cupy_in_torch_stream():
            if cp_config.sigma is None:
                # cupy supports CSR matrices with int64 indices.
                a_cp = sdt_to_cupy_csr(a_val, a_pattern)
            else:
                # CuPyShiftInvSymOp only supports int32 index tensors due to
                # sparse solver limitations.
                a_cp = CuPyShiftInvSymOp(
                    a_val, a_pattern, cp_config.sigma, nvmath_config
                )

            cp_config_dict = asdict(cp_config)
            del cp_config_dict["sigma"]

            results = cp_sp_linalg.eigsh(
                a=a_cp,
                k=k,
                return_eigenvectors=compute_eig_vecs,
                **cp_config_dict,
            )

            if compute_eig_vecs:
                eig_vals_cp, eig_vecs_cp = results
                eig_vecs = torch.from_dlpack(eig_vecs_cp)
            else:
                eig_vals_cp = results
                eig_vecs = None

            eig_vals = torch.from_dlpack(eig_vals_cp)

        if cp_config.sigma is None:
            eig_vals_true = eig_vals
        else:
            # In shift-invert mode, λ_returned = 1 / (λ_true - σ)
            eig_vals_true = cp_config.sigma + (1.0 / eig_vals)

        return eig_vals_true, eig_vecs

    @staticmethod
    def setup_context(ctx, inputs, output):
        a_val, a_pattern, k, eps, compute_eig_vecs, cp_config, nvmath_config = inputs
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
        None,
    ]:
        needs_grad_A_val = ctx.needs_input_grad[0]

        dLdA_val = dLdA_backward(ctx, dLdl, dLdv) if needs_grad_A_val else None

        return dLdA_val, None, None, None, None, None, None


def _cupy_eigsh_no_batch(
    a: Float[SparseDecoupledTensor, "r c"],
    k: int,
    eps: float | int,
    compute_eig_vecs: bool,
    cp_config: CuPyEigshConfig,
    nvmath_config: DirectSolverConfig | None,
) -> tuple[Float[Tensor, " k"], Float[Tensor, "c k"]]:
    """Dispatch the autograd functions whn there is no block-diagonal batching."""
    eig_vals, eig_vecs = CuPyEigshAutogradFunction.apply(
        a.values, a.pattern, k, eps, compute_eig_vecs, cp_config, nvmath_config
    )

    return eig_vals, eig_vecs


def _cupy_eigsh_batch(
    a_batched: Float[SparseDecoupledTensor, "r c"],
    k: int,
    eps: float | int,
    compute_eig_vecs: bool,
    cp_config_batched: CuPyEigshConfig,
    nvmath_config: DirectSolverConfig | None,
) -> tuple[Float[Tensor, "b k"], Float[Tensor, "c k"]]:
    """Dispatch the autograd functions when there is block-diagonal batching."""
    a_list = a_batched.unpack_block_diag()

    cp_config_list = cp_config_batched.expand(n=len(a_list))

    eig_val_list = []
    eig_vec_list = []
    for a_sdt, cp_config in zip(a_list, cp_config_list):
        eig_val, eig_vec = _cupy_eigsh_no_batch(
            a_sdt, k, eps, compute_eig_vecs, cp_config, nvmath_config
        )
        eig_val_list.append(eig_val)
        eig_vec_list.append(eig_vec)

    eig_vals = torch.vstack(eig_val_list)

    if eig_vec_list[0] is None:
        eig_vecs = None
    else:
        eig_vecs = torch.vstack(eig_vec_list)

    return eig_vals, eig_vecs


def cupy_eigsh(
    a: Float[SparseDecoupledTensor, "r c"],
    *,
    block_diag_batch: bool = False,
    k: int = 6,
    eps: float | int | Literal["auto"] = "auto",
    return_eigenvectors: bool = True,
    cp_config: CuPyEigshConfig | None = None,
    nvmath_config: DirectSolverConfig | None = None,
) -> Float[Tensor, "*b k"] | tuple[Float[Tensor, "*b k"], Float[Tensor, "r k"]]:
    r"""
    Sparse eigensolver for symmetric square matrices using CuPy.

    This function provides a differentiable wrapper for the GPU-based
    `cupyx.scipy.sparse.linalg.eigsh()` method.

    Parameters
    ----------
    a : [r, c]
        A real symmetric square sparse matrix.
    block_diag_batch
        Whether the input `a` matrix is block-diagonal. If `a` is
        block-diagonal, then it must have a valid `block_diag_config`,
        in which case each block/batch element will be solved sequentially.
        Note that the `eigsh()` function does not natively support batching.
    k
        The number of eigenvalues/eigenvectors to find.
    eps
        The strength of Lorentzian broadening/regularization, which removes
        singularities in backward gradient calculation when some of the
        eigenvalues are (near) degenerate. As a heuristic, the regularization starts
        to dominate the gradient calculation as the spectral gap approaches the
        square root of `eps`. Set to integer 0 to disable regularization; set to
        "auto" to select `eps` based on the input dtype and matrix inf-norm.
    return_eigenvectors
        Whether to compute and return the eigenvectors in addition to the
        eigenvalues. Note that, if `a` requires gradient tracking, the
        eigenvectors will be computed regardless of this argument.
    cp_config
        Additional optional arguments for CuPy `eigsh()`.
    nvmath_config
        Additional optional arguments for nvmath `DirectSolver()`; only
        relevant for the shift-invert mode.

    Returns
    -------
    eig_vals : [*b, k]
        A tensor of `k` eigenvalues. If `block_diag_batch` is `True`,
        then the tensor also has a leading batch dimension corresponding
        to the blocks in the input `a` matrix.
    eig_vecs : [r, k]
        A tensor of `k` orthonormal eigenvectors. If `block_diag_batch` is
        `False`, then each column represents an eigenvector; if `block_diag_batch`
        is `True`, then the eigenvectors for each block are stacked along
        the first dimension. If `return_eigenvectors` is False, then this
        tensor is not returned.

    Notes
    -----
    The autograd through eigenvectors do not account for contributions from the
    unresolved eigenvectors.

    The algorithm used by the CuPy `eigsh()` often underestimates the multiplicity
    of degenerate eigenvalues; consider using the `lobpcg()` solver instead.

    For a standard eigenvalue problem, the `a` matrix will be converted to
    a CuPy CSR matrix; both `int32` and `int64` index dtypes are supported, but
    the matrix will automatically be downcast to `int32` if possible.

    The CuPy `eigsh()` does not natively support the shift-invert mode. Here,
    the shift-invert mode is implemented by constructing a CuPy `LinearOperator`
    object for the $L = (A - \sigma I)^{-1}$ matrix; computing the matrix-vector
    multiplication $L x = b$ is then equivalent to solving a sparse linear system
    with the LHS matrix $A - \sigma I$ and RHS vector $x$. This approach corresonds
    to the `mode='normal'` option in SciPy `eigsh()`. Currently, the sparse solver
    is implemented using the nvmath-python `DirectSolver()` class; the behavior
    of this solver can be controlled via the `nvmath_config` argument. Only
    `int32` index dtype for `a` is supported in this case due to the sparse
    solver requirement.

    The CuPy `eigsh()` does not support generalized eigenvalue problems.
    """
    if not _HAS_CUPY:
        raise ImportError("CuPy backend required.")

    shift_invert = (cp_config is not None) and (cp_config.sigma is not None)

    if shift_invert and (not _HAS_NVMATH):
        raise ImportError("nvmath-python backend required for shift-invert mode.")

    # Eigenvectors are required for backward().
    compute_eig_vecs = return_eigenvectors
    if a.requires_grad:
        compute_eig_vecs = True

    if cp_config is None:
        cp_config = CuPyEigshConfig()

    if shift_invert and (nvmath_config is None):
        nvmath_config = DirectSolverConfig()

    if eps == "auto":
        eps = compute_lorentzian_eps(a, None)

    if block_diag_batch:
        eig_vals, eig_vecs = _cupy_eigsh_batch(
            a, k, eps, compute_eig_vecs, cp_config, nvmath_config
        )
    else:
        eig_vals, eig_vecs = _cupy_eigsh_no_batch(
            a, k, eps, compute_eig_vecs, cp_config, nvmath_config
        )

    if return_eigenvectors:
        return eig_vals, eig_vecs
    else:
        return eig_vals

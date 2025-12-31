from dataclasses import asdict, dataclass
from typing import Any, Literal

import numpy as np
import scipy.sparse
import scipy.sparse.linalg
import torch as t
from jaxtyping import Float, Integer

from ..operators import SparseOperator, SparseTopology
from ._eigsh_utils import (
    compute_dLdA_val,
    compute_dLdM_val,
    compute_eig_vec_grad_proj,
    compute_lorentz_matrix,
)


@dataclass
class SciPyEigshConfig:
    sigma: float | None = None
    which: Literal["LM", "SM", "LA", "SA", "BE"] = "LM"
    v0: Float[t.Tensor | np.typing.NDArray, " c"] | None = None
    ncv: int | None = None
    maxiter: int | None = None
    tol: float | int = 0
    mode: Literal["normal", "buckling", "cayley"] = "normal"

    def __post_init__(self):
        if isinstance(self.v0, t.Tensor):
            self.v0 = self.v0.detach().contiguous().cpu().numpy()


class _SciPyEigshWrapperStandard(t.autograd.Function):
    @staticmethod
    def forward(
        A_val: Float[t.Tensor, " nnz"],
        A_sp_topo: Integer[SparseTopology, "r c"],
        k: int,
        eps: float | int,
        compute_eig_vecs: bool,
        config: SciPyEigshConfig,
    ) -> tuple[Float[t.Tensor, " k"], Float[t.Tensor, "c k"] | None]:
        # When solving the standard A@x=λx, the CSR format is preferred for
        # matrix-vector multiplication.
        if config.sigma is None:
            A_scipy = scipy.sparse.csr_array(
                (
                    A_val.detach().contiguous().cpu().numpy(),
                    A_sp_topo.idx_col_int32.detach().contiguous().cpu().numpy(),
                    A_sp_topo.idx_crow_int32.detach().contiguous().cpu().numpy(),
                ),
                shape=A_sp_topo.shape,
            )

        # In the shift-invert mode, an LU factorization of A + σI is required,
        # and therefore the CSC format is preferred.
        else:
            A_scipy = scipy.sparse.csc_array(
                (
                    A_val[A_sp_topo.coo_to_csc_perm]
                    .detach()
                    .contiguous()
                    .cpu()
                    .numpy(),
                    A_sp_topo.idx_row_csc_int32.detach().contiguous().cpu().numpy(),
                    A_sp_topo.idx_ccol_int32.detach().contiguous().cpu().numpy(),
                ),
                shape=A_sp_topo.shape,
            )

        results = scipy.sparse.linalg.eigsh(
            A=A_scipy,
            k=k,
            return_eigenvectors=compute_eig_vecs,
            **asdict(config),
        )

        if compute_eig_vecs:
            eig_vals_np, eig_vecs_np = results

            eig_vecs = t.from_numpy(eig_vecs_np).to(
                dtype=A_val.dtype, device=A_val.device
            )

        else:
            eig_vals_np = results
            eig_vecs = None

        eig_vals = t.from_numpy(eig_vals_np).to(dtype=A_val.dtype, device=A_val.device)

        return eig_vals, eig_vecs

    @staticmethod
    def setup_context(ctx, inputs, output):
        A_val, A_sp_topo, k, eps, compute_eig_vecs, config = inputs
        eig_vals, eig_vecs = output

        ctx.save_for_backward(eig_vals, eig_vecs)
        ctx.A_sp_topo = A_sp_topo
        ctx.k = k
        ctx.eps = eps

    @staticmethod
    def backward(
        ctx, dLdl: Float[t.Tensor, " k"], dLdv: Float[t.Tensor, "c k"] | None
    ) -> tuple[
        Float[t.Tensor, " nnz"] | None,
        None,
        None,
        None,
        None,
        None,
    ]:
        needs_grad_A_val = ctx.needs_input_grad[0]

        dLdA_val = None

        if needs_grad_A_val:
            # The eigenvectors need to be length-normalized for the following
            # calculation; scipy eigsh() by default returns orthonormal eigenvectors.
            eig_vals, eig_vecs = ctx.saved_tensors
            A_sp_topo: SparseTopology = ctx.A_sp_topo

            # This error should never be triggered if the user-facing wrapper does
            # its job.
            if eig_vecs is None:
                raise ValueError("Eigenvectors are required for backward().")

            if dLdv is None:
                eig_vec_grad_proj = None
                lorentz = None
            else:
                eig_vec_grad_proj = compute_eig_vec_grad_proj(eig_vecs, dLdv)
                lorentz = compute_lorentz_matrix(eig_vals, ctx.k, ctx.eps)

            dLdA_val = compute_dLdA_val(
                A_sp_topo, eig_vecs, dLdl, dLdv, eig_vec_grad_proj, lorentz
            )

        return dLdA_val, None, None, None, None, None


class _SciPyEigshWrapperGeneralized(t.autograd.Function):
    @staticmethod
    def forward(
        A_val: Float[t.Tensor, " A_nnz"],
        A_sp_topo: Integer[SparseTopology, "r c"],
        M_val: Float[t.Tensor, " M_nnz"],
        M_sp_topo: Integer[SparseTopology, "r c"],
        k: int,
        eps: float | int,
        compute_eig_vecs: bool,
        config: SciPyEigshConfig,
    ) -> tuple[Float[t.Tensor, " k"], Float[t.Tensor, "c k"] | None]:
        # When solving the standard A@x=λx, the CSR format is preferred for
        # matrix-vector multiplication.
        if config.sigma is None:
            A_scipy = scipy.sparse.csr_array(
                (
                    A_val.detach().contiguous().cpu().numpy(),
                    A_sp_topo.idx_col_int32.detach().contiguous().cpu().numpy(),
                    A_sp_topo.idx_crow_int32.detach().contiguous().cpu().numpy(),
                ),
                shape=A_sp_topo.shape,
            )

        # In the shift-invert mode, an LU factorization of A + σI is required,
        # and therefore the CSC format is preferred.
        else:
            A_scipy = scipy.sparse.csc_array(
                (
                    A_val[A_sp_topo.coo_to_csc_perm]
                    .detach()
                    .contiguous()
                    .cpu()
                    .numpy(),
                    A_sp_topo.idx_row_csc_int32.detach().contiguous().cpu().numpy(),
                    A_sp_topo.idx_ccol_int32.detach().contiguous().cpu().numpy(),
                ),
                shape=A_sp_topo.shape,
            )

        # M should always be in CSC format for LU factorization.
        M_scipy = scipy.sparse.csc_array(
            (
                M_val[M_sp_topo.coo_to_csc_perm].detach().contiguous().cpu().numpy(),
                M_sp_topo.idx_row_csc_int32.detach().contiguous().cpu().numpy(),
                M_sp_topo.idx_ccol_int32.detach().contiguous().cpu().numpy(),
            ),
            shape=M_sp_topo.shape,
        )

        results = scipy.sparse.linalg.eigsh(
            A=A_scipy,
            k=k,
            M=M_scipy,
            return_eigenvectors=compute_eig_vecs,
            **asdict(config),
        )

        if compute_eig_vecs:
            eig_vals_np, eig_vecs_np = results

            eig_vecs = t.from_numpy(eig_vecs_np).to(
                dtype=A_val.dtype, device=A_val.device
            )

        else:
            eig_vals_np = results
            eig_vecs = None

        eig_vals = t.from_numpy(eig_vals_np).to(dtype=A_val.dtype, device=A_val.device)

        return eig_vals, eig_vecs

    @staticmethod
    def setup_context(ctx, inputs, output):
        A_val, A_sp_topo, M_val, M_sp_topo, k, eps, compute_eig_vecs, config = inputs
        eig_vals, eig_vecs = output

        ctx.save_for_backward(eig_vals, eig_vecs)
        ctx.A_sp_topo = A_sp_topo
        ctx.M_sp_topo = M_sp_topo
        ctx.k = k
        ctx.eps = eps

    @staticmethod
    def backward(
        ctx, dLdl: Float[t.Tensor, " k"], dLdv: Float[t.Tensor, "c k"] | None
    ) -> tuple[
        Float[t.Tensor, " A_nnz"] | None,
        None,
        Float[t.Tensor, " M_nnz"] | None,
        None,
        None,
        None,
        None,
        None,
    ]:
        needs_grad_A_val = ctx.needs_input_grad[0]
        needs_grad_M_val = ctx.needs_input_grad[2]

        dLdA_val = None
        dLdM_val = None

        # The eigenvectors need to be orthonormal wrt M for the following
        # calculation; should be true by scipy eigsh() default.
        eig_vals, eig_vecs = ctx.saved_tensors
        A_sp_topo: SparseTopology = ctx.A_sp_topo
        M_sp_topo: SparseTopology = ctx.M_sp_topo

        if needs_grad_A_val or needs_grad_M_val:
            # This error should never be triggered if the user-facing wrapper does
            # its job.
            if eig_vecs is None:
                raise ValueError("Eigenvectors are required for backward().")

            if dLdv is None:
                eig_vec_grad_proj = None
                lorentz = None
            else:
                eig_vec_grad_proj = compute_eig_vec_grad_proj(eig_vecs, dLdv)
                lorentz = compute_lorentz_matrix(eig_vals, ctx.k, ctx.eps)

        if needs_grad_A_val:
            dLdA_val = compute_dLdA_val(
                A_sp_topo, eig_vecs, dLdl, dLdv, eig_vec_grad_proj, lorentz
            )

        if needs_grad_M_val:
            dLdM_val = compute_dLdM_val(
                M_sp_topo, eig_vecs, dLdl, dLdv, eig_vec_grad_proj, lorentz
            )

        return dLdA_val, None, dLdM_val, None, None, None, None, None


def _scipy_eigsh_no_batch(
    A: Float[SparseOperator, "r c"],
    M: Float[SparseOperator, "r c"] | None,
    kwargs: dict[str, Any],
) -> tuple[Float[t.Tensor, " k"], Float[t.Tensor, "c k"]]:
    if M is None:
        eig_vals, eig_vecs = _SciPyEigshWrapperStandard.apply(
            A.val, A.sp_topo, **kwargs
        )
    else:
        eig_vals, eig_vecs = _SciPyEigshWrapperGeneralized.apply(
            A.val, A.sp_topo, M.val, M.sp_topo, **kwargs
        )

    return eig_vals, eig_vecs


def _scipy_eigsh_batch(
    A_batched: Float[SparseOperator, "r c"],
    M_batched: Float[SparseOperator, "r c"] | None,
    kwargs: dict[str, Any],
) -> tuple[Float[t.Tensor, "b k"], Float[t.Tensor, "c k"]]:
    A_list = A_batched.unpack_block_diag()
    if M_batched is None:
        M_list = [None] * len(A_list)
    else:
        M_list = M_batched.unpack_block_diag()

    for A, M in zip(A_list, M_list, strict=True):
        eig_val_list, eig_vec_list = _scipy_eigsh_no_batch(A, M, kwargs)

    eig_vals = t.vstack(eig_val_list)
    eig_vecs = t.vstack(eig_vec_list)

    return eig_vals, eig_vecs


def scipy_eigsh(
    A: Float[SparseOperator, "r c"],
    M: Float[SparseOperator, "r c"] | None = None,
    block_diag_batch: bool = False,
    k: int = 6,
    eps: float | int = 1e-6,
    return_eigenvectors: bool = True,
    config: SciPyEigshConfig | None = None,
) -> tuple[Float[t.Tensor, "*b k"], Float[t.Tensor, "c k"] | None]:
    """
    This function provides a differentiable wrapper for the CPU-based
    `scipy.sparse.linalg.eigsh()` method.

    The API for `eigsh()` is almost reproduced one-to-one, with the following
    exceptions:

    * The arguments `sigma`, `which`, `v0`, `ncv`, `maxiter`, `tol`, and `mode`
      are collected in a `ScipyEigshConfig` dataclass object, whilc the rest of
      the arguments are exposed as direct arguments to this function.
    * The `A` and `M` matrices must be `SparseOperator` objects and will be
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
      computed and the `return_eigenvectors` argument will be ignored.
    * The `eigsh()` function does not natively support batching. if
      `block_diag_batch` is True, the `A` `SparseOperator` (and `M` if not `None`)
      will be split into individual sparse matrices and solved sequentially. The
      resulting eigenvalue tensor will contain a leading batch dimension, but the
      resulting eigenvector tensor will respect the original concatenated/packed
      format. This requires that `A` (and `M` if not `None`) has a valid
      `BlockDiagConfig` for unpacking the block diagonal batch structure.
    """
    # Eigenvectors are required for backward().
    compute_eig_vecs = return_eigenvectors
    if A.requires_grad:
        compute_eig_vecs = True
    if (M is not None) and M.requires_grad:
        compute_eig_vecs = True

    if config is None:
        config = SciPyEigshConfig()

    kwargs = {
        "k": k,
        "eps": eps,
        "compute_eig_vecs": compute_eig_vecs,
        "config": config,
    }

    if block_diag_batch:
        eig_vals, eig_vecs = _scipy_eigsh_batch(A, M, kwargs)
    else:
        eig_vals, eig_vecs = _scipy_eigsh_no_batch(A, M, kwargs)

    if return_eigenvectors:
        return eig_vals, eig_vecs
    else:
        return eig_vals

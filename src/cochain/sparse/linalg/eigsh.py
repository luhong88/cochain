from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any, Literal

import scipy.sparse
import scipy.sparse.linalg
import torch as t
from jaxtyping import Float, Integer

from ..operators import SparseOperator, SparseTopology

try:
    import cupy as cp
    import cupyx.scipy.sparse as cp_sp
    import cupyx.scipy.sparse.linalg as cp_sp_linalg

    _HAS_CUPY = True
except ImportError:
    _HAS_CUPY = False

if TYPE_CHECKING:
    import cupyx.scipy.sparse as cp_sp
    import cupyx.scipy.sparse.linalg as cp_sp_linalg


@dataclass
class SciPyEigshConfig:
    sigma: float | None = None
    which: Literal["LM", "SM", "LA", "SA", "BE"] = "LM"
    ncv: int | None = None
    maxiter: int | None = None
    tol: float | int = 0
    mode: Literal["normal", "buckling", "cayley"] = "normal"


def _compute_eig_vec_grad_proj(
    eig_vecs: Float[t.Tensor, "c k"],
    dLdv: Float[t.Tensor, "c k"],
) -> Float[t.Tensor, "k k"]:
    # Compute the projection of eigenvector gradients onto the eigensapce.
    return eig_vecs.T @ dLdv


def _compute_lorentz_matrix(
    eig_vals: Float[t.Tensor, " k"], k: int, eps: float | int
) -> Float[t.Tensor, "k k"]:
    # Compute the matrix F, where F_ij = 1/(λ_j - λ_i) and F_ii = 0.
    eig_val_diffs = eig_vals.view(1, -1) - eig_vals.view(-1, 1)

    if eps > 0:
        # If eps > 0, compute a regularized version where
        # F_ij = Δ_ji / (Δ_ji^2 + ϵ), where Δ_ji = λ_j - λ_i.
        # When Δ >> 0, this recovers the true definition; when Δ is close
        # to 0, this prevents the gradient from exploding by decaying to 0.
        lorentz = eig_val_diffs / (eig_val_diffs.pow(2) + eps)

    else:
        lorentz_diag = 1.0 / (
            t.eye(k, dtype=eig_vals.dtype, device=eig_vals.device) + eig_val_diffs
        )
        lorentz = lorentz_diag - t.eye(k, dtype=eig_vals.dtype, device=eig_vals.device)

    return lorentz


def _compute_dLdA_val(
    A_sp_topo: Integer[SparseTopology, "r c"],
    eig_vecs: Float[t.Tensor, "c k"],
    dLdl: Float[t.Tensor, " k"],
    dLdv: Float[t.Tensor, "c k"] | None,
    eig_vec_grad_proj: Float[t.Tensor, "k k"],
    lorentz: Float[t.Tensor, "k k"],
) -> Float[t.Tensor, " nnz"]:
    """
    The formula is the same for standard and generalized eigenvalue problems.
    """
    eig_vecs_row = eig_vecs[A_sp_topo.idx_coo[0]]
    eig_vecs_col = eig_vecs[A_sp_topo.idx_coo[1]]

    # If the loss does not depend on the eigenvectors, then the "eigenvalue"
    # component of the gradient is given by sum_i[dLdλ_i * v_i @ v_i.T]
    dLdA_eig_vals = t.sum(
        dLdl.view(1, -1) * eig_vecs_row * eig_vecs_col,
        dim=1,
    )

    if dLdv is None:
        dLdA_val = dLdA_eig_vals

    else:
        anti_symmetric_proj = 0.5 * (eig_vec_grad_proj - eig_vec_grad_proj.T)

        # Compute the Hadamard product K = F * P
        kernel: Float[t.Tensor, "k k"] = lorentz * anti_symmetric_proj

        # The "eigenvector" component is given by V @ K @ V.T
        dLdA_eig_vecs = t.sum(
            (eig_vecs_row @ kernel) * eig_vecs_col,
            dim=1,
        )

        # Sum together the "eigenvalue" and "eigenvector" components of the gradient.
        dLdA_val = dLdA_eig_vals + dLdA_eig_vecs

    return dLdA_val


def _compute_dLdM_val(
    M_sp_topo: Integer[SparseTopology, "r c"],
    eig_vals: Float[t.Tensor, " k"],
    eig_vecs: Float[t.Tensor, "c k"],
    dLdl: Float[t.Tensor, " k"],
    dLdv: Float[t.Tensor, "c k"] | None,
    eig_vec_grad_proj: Float[t.Tensor, "k k"],
    lorentz: Float[t.Tensor, "k k"],
) -> Float[t.Tensor, " nnz"]:
    eig_vecs_row = eig_vecs[M_sp_topo.idx_coo[0]]
    eig_vecs_col = eig_vecs[M_sp_topo.idx_coo[1]]

    # If the loss does not depend on the eigenvectors, then the "eigenvalue"
    # component of the gradient is given by -sum_i[λ_i * dLdλ_i * v_i @ v_i.T]
    dLdM_eig_vals = -t.sum(
        eig_vals.view(1, -1) * dLdl.view(1, -1) * eig_vecs_row * eig_vecs_col,
        dim=1,
    )

    if dLdv is None:
        dLdM_val = dLdM_eig_vals

    else:
        # The elements of the kernel is given by
        # off-diagonal: K_ij = F_ij * (λ_i*P_ij - λ_j*P_ji)
        # diagonal: K_ii = - P_ii/2
        lP = eig_vals.view(-1, 1) * eig_vec_grad_proj
        kernel_off_diag = lorentz * (lP - lP.T)
        kernel_diag = -0.5 * t.diag(eig_vec_grad_proj)
        kernel: Float[t.Tensor, "k k"] = t.diagflat(kernel_diag) + kernel_off_diag

        # The "eigenvector" component is given by V @ K @ V.T
        dLdM_eig_vecs = t.sum(
            (eig_vecs_row @ kernel) * eig_vecs_col,
            dim=1,
        )

        # Sum together the "eigenvalue" and "eigenvector" components of the
        # gradient.
        dLdM_val = dLdM_eig_vals + dLdM_eig_vecs

    return dLdM_val


class _SciPyEigshWrapperStandard(t.autograd.Function):
    @staticmethod
    def forward(
        A_val: Float[t.Tensor, " nnz"],
        A_sp_topo: Integer[SparseTopology, "r c"],
        k: int,
        v0: Float[t.Tensor, " c"] | None,
        esp: float | int,
        return_eig_vecs: bool,
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

        if v0 is None:
            v0_np = None
        else:
            v0_np = v0.detach().contiguous().cpu().numpy()

        results = scipy.sparse.linalg.eigsh(
            A=A_scipy,
            k=k,
            v0=v0_np,
            return_eigenvectors=return_eig_vecs,
            **asdict(config),
        )

        if return_eig_vecs:
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
        A_val, A_sp_topo, k, v0, esp, return_eig_vecs, config = inputs
        eig_vals, eig_vecs = output

        ctx.save_for_backward(eig_vals, eig_vecs)
        ctx.A_sp_topo = A_sp_topo
        ctx.k = k
        ctx.esp = esp

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

            eig_vec_grad_proj = _compute_eig_vec_grad_proj(eig_vecs, dLdv)
            lorentz = _compute_lorentz_matrix(eig_vals, ctx.k, ctx.eps)

            dLdA_val = _compute_dLdA_val(
                A_sp_topo, eig_vecs, dLdl, dLdv, eig_vec_grad_proj, lorentz
            )

        return dLdA_val, None, None, None, None, None, None


class _SciPyEigshWrapperGeneralized(t.autograd.Function):
    @staticmethod
    def forward(
        A_val: Float[t.Tensor, " A_nnz"],
        A_sp_topo: Integer[SparseTopology, "r c"],
        M_val: Float[t.Tensor, " M_nnz"],
        M_sp_topo: Integer[SparseTopology, "r c"],
        k: int,
        v0: Float[t.Tensor, " c"] | None,
        esp: float | int,
        return_eig_vecs: bool,
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

        if v0 is None:
            v0_np = None
        else:
            v0_np = v0.detach().contiguous().cpu().numpy()

        results = scipy.sparse.linalg.eigsh(
            A=A_scipy,
            k=k,
            M=M_scipy,
            v0=v0_np,
            return_eigenvectors=return_eig_vecs,
            **asdict(config),
        )

        if return_eig_vecs:
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
        A_val, A_sp_topo, M_val, M_sp_topo, k, v0, esp, return_eig_vecs, config = inputs
        eig_vals, eig_vecs = output

        ctx.save_for_backward(eig_vals, eig_vecs)
        ctx.A_sp_topo = A_sp_topo
        ctx.M_sp_topo = M_sp_topo
        ctx.k = k
        ctx.esp = esp

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

            eig_vec_grad_proj = _compute_eig_vec_grad_proj(eig_vecs, dLdv)
            lorentz = _compute_lorentz_matrix(eig_vals, ctx.k, ctx.eps)

        if needs_grad_A_val:
            dLdA_val = _compute_dLdA_val(
                A_sp_topo, eig_vecs, dLdl, dLdv, eig_vec_grad_proj, lorentz
            )

        if needs_grad_M_val:
            dLdM_val = _compute_dLdM_val(
                M_sp_topo, eig_vecs, dLdl, dLdv, eig_vec_grad_proj, lorentz
            )

        return dLdA_val, None, dLdM_val, None, None, None, None, None, None

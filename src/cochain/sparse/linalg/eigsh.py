from __future__ import annotations

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


class _SciPyEigshWrapperStandard(t.autograd.Function):
    @staticmethod
    def forward(
        A_val: Float[t.Tensor, " nnz"],
        A_sp_topo: Integer[SparseTopology, "r c"],
        k: int,
        which: Literal["LM", "SM", "LA", "SA", "BE"],
        v0: Float[t.Tensor, " c"] | None,
        ncv: int | None,
        maxiter: int | None,
        tol: float,
        esp: float,
        return_eigenvectors: bool,
    ) -> tuple[Float[t.Tensor, " k"], Float[t.Tensor, "c k"] | None]:
        val = A_val[A_sp_topo.coo_to_csc_perm].detach().contiguous().cpu().numpy()
        idx_ccol = A_sp_topo.idx_ccol_int32.detach().contiguous().cpu().numpy()
        idx_row = A_sp_topo.idx_row_csc_int32.detach().contiguous().cpu().numpy()

        A_scipy: Float[scipy.sparse.csc_array, "r c"] = scipy.sparse.csc_array(
            (val, idx_row, idx_ccol),
            shape=A_sp_topo.shape,
        )

        if v0 is None:
            v0_np = None
        else:
            v0_np = v0.detach().contiguous().cpu().numpy()

        results = scipy.sparse.linalg.eigsh(
            A=A_scipy,
            k=k,
            which=which,
            v0=v0_np,
            ncv=ncv,
            maxiter=maxiter,
            tol=tol,
            return_eigenvectors=return_eigenvectors,
        )

        if return_eigenvectors:
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
        A_val, A_sp_topo, k, which, v0, ncv, maxiter, tol, esp, return_eigenvectors = (
            inputs
        )
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
        None,
        None,
        None,
    ]:
        needs_grad_A_val = ctx.needs_input_grad[0]

        if not needs_grad_A_val:
            return (None,) * 10

        # The eigenvectors need to be length-normalized for the following
        # calculation; scipy eigsh() by default returns orthonormal eigenvectors.
        eig_vals, eig_vecs = ctx.saved_tensors
        A_sp_topo: SparseTopology = ctx.A_sp_topo

        # This error should never be triggered if the user-facing wrapper does
        # its job.
        if eig_vecs is None:
            raise ValueError("Eigenvectors are required for backward().")

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
            # Compute the projection P between the eigenvectors and the
            # gradients of the eigenvectors.
            grad_proj = 0.5 * (eig_vecs.T @ dLdv - dLdv.T @ eig_vecs)

            # Compute the matrix F, where F_ij = 1/(λ_j - λ_i) and F_ii = 0.
            eig_val_diffs = eig_vals.view(1, -1) - eig_vals.view(-1, 1)

            if ctx.eps > 0:
                # If eps > 0, compute a regularized version where
                # F_ij = Δ_ji / (Δ_ji^2 + ϵ), where Δ_ji = λ_j - λ_i.
                # When Δ >> 0, this recovers the true definition; when Δ is close
                # to 0, this prevents the gradient from exploding by decaying to 0.
                lorentz = eig_val_diffs / (eig_val_diffs.pow(2) + ctx.eps)

            else:
                lorentz_diag = 1.0 / (
                    t.eye(ctx.k, dtype=eig_vals.dtype, device=eig_vals.device)
                    + eig_val_diffs
                )
                lorentz = lorentz_diag - t.eye(
                    ctx.k, dtype=eig_vals.dtype, device=eig_vals.device
                )

            # Compute the Hadamard product K = F * P
            kernel: Float[t.Tensor, "k k"] = lorentz * grad_proj

            # The "eigenvector" component is given by V @ K @ V.T
            dLdA_eig_vecs = t.sum(
                (eig_vecs_row @ kernel) * eig_vecs_col,
                dim=1,
            )

            # Sum together the "eigenvalue" and "eigenvector" components of the
            # gradient.
            dLdA_val = dLdA_eig_vals + dLdA_eig_vecs

        return (dLdA_val,) + (None,) * 9

import torch as t
from jaxtyping import Float, Integer

from ..operators import SparseTopology


def compute_eig_vec_grad_proj(
    eig_vecs: Float[t.Tensor, "c k"],
    dLdv: Float[t.Tensor, "c k"],
) -> Float[t.Tensor, "k k"]:
    # Compute the projection of eigenvector gradients onto the eigensapce.
    return eig_vecs.T @ dLdv


def compute_lorentz_matrix(
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


def compute_dLdA_val(
    A_sp_topo: Integer[SparseTopology, "r c"],
    eig_vecs: Float[t.Tensor, "c k"],
    dLdl: Float[t.Tensor, " k"],
    dLdv: Float[t.Tensor, "c k"] | None,
    eig_vec_grad_proj: Float[t.Tensor, "k k"] | None,
    lorentz: Float[t.Tensor, "k k"] | None,
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


def compute_dLdM_val(
    M_sp_topo: Integer[SparseTopology, "r c"],
    eig_vals: Float[t.Tensor, " k"],
    eig_vecs: Float[t.Tensor, "c k"],
    dLdl: Float[t.Tensor, " k"],
    dLdv: Float[t.Tensor, "c k"] | None,
    eig_vec_grad_proj: Float[t.Tensor, "k k"] | None,
    lorentz: Float[t.Tensor, "k k"] | None,
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

import torch as t
from jaxtyping import Float, Integer

from ...operators import SparseTopology


def compute_eig_vec_grad_proj(
    eig_vecs: Float[t.Tensor, "c k"],
    dLdv: Float[t.Tensor, "c k"],
) -> Float[t.Tensor, "k k"]:
    # Compute the projection of eigenvector gradients onto the eigenspace.
    return eig_vecs.T @ dLdv


def compute_cauchy_matrix(
    eig_vals: Float[t.Tensor, " k"], k: int, eps: float | int
) -> Float[t.Tensor, "k k"]:
    """Compute the matrix F, where F_ij = 1/(λ_j - λ_i) and F_ii = 0."""
    eig_val_diffs = eig_vals.view(1, -1) - eig_vals.view(-1, 1)

    if eps > 0:
        # If eps > 0, compute a regularized version where
        # F_ij = Δ_ji / (Δ_ji^2 + ϵ), where Δ_ji = λ_j - λ_i.
        # When Δ >> 0, this recovers the true definition; when Δ is close
        # to 0, this prevents the gradient from exploding by decaying to 0.
        cauchy = eig_val_diffs / (eig_val_diffs.pow(2) + eps)

    else:
        eig_val_diffs.fill_diagonal_(float("inf"))
        cauchy = 1.0 / eig_val_diffs

    return cauchy


def compute_dLdA_val(
    A_sp_topo: Integer[SparseTopology, "r c"],
    eig_vecs: Float[t.Tensor, "c k"],
    dLdl: Float[t.Tensor, " k"],
    dLdv: Float[t.Tensor, "c k"] | None,
    eig_vec_grad_proj: Float[t.Tensor, "k k"] | None,
    cauchy: Float[t.Tensor, "k k"] | None,
) -> Float[t.Tensor, " nnz"]:
    """
    The formula is the same for standard and generalized eigenvalue problems.
    """
    eig_vecs_row = eig_vecs[A_sp_topo.idx_coo[0]]
    eig_vecs_col = eig_vecs[A_sp_topo.idx_coo[1]]

    # If the loss does not depend on the eigenvectors, then the "eigenvalue"
    # component of the gradient is given by dLdA_ij = sum_k[dLdλ_k * V_ik * V_jk]
    dLdA_eig_vals = t.sum(
        dLdl.view(1, -1) * eig_vecs_row * eig_vecs_col,
        dim=1,
    )

    if dLdv is None:
        dLdA_val = dLdA_eig_vals

    else:
        anti_symmetric_proj = 0.5 * (eig_vec_grad_proj - eig_vec_grad_proj.T)

        # Compute the Hadamard product K = F * P
        kernel: Float[t.Tensor, "k k"] = cauchy * anti_symmetric_proj

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
    cauchy: Float[t.Tensor, "k k"] | None,
) -> Float[t.Tensor, " nnz"]:
    eig_vecs_row = eig_vecs[M_sp_topo.idx_coo[0]]
    eig_vecs_col = eig_vecs[M_sp_topo.idx_coo[1]]

    # If the loss does not depend on the eigenvectors, then the "eigenvalue"
    # component of the gradient is given by
    # dLdM_ij = -sum_k[λ_k * dLdλ_k * V_ik * V_jk]
    dLdM_eig_vals = -t.sum(
        eig_vals.view(1, -1) * dLdl.view(1, -1) * eig_vecs_row * eig_vecs_col,
        dim=1,
    )

    if dLdv is None:
        dLdM_val = dLdM_eig_vals

    else:
        # The elements of the kernel is given by
        # off-diagonal: K_ij = F_ij * (λ_i*P_ji - λ_j*P_ij)/2
        # diagonal: K_ii = - P_ii/2
        lP_T = eig_vals.view(-1, 1) * eig_vec_grad_proj.T
        kernel_off_diag = 0.5 * cauchy * (lP_T - lP_T.T)
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


def dLdA_backward(
    ctx, dLdl: Float[t.Tensor, " k"], dLdv: Float[t.Tensor, "c k"] | None
):
    """
    A function that encapsulates the shared backward() gradient logic for eigenvalue
    problems.
    """
    # The eigenvectors need to be length-normalized for the following calculation.
    eig_vals, eig_vecs = ctx.saved_tensors
    A_sp_topo: SparseTopology = ctx.A_sp_topo

    # This error should never be triggered if the user-facing wrapper does
    # its job.
    if eig_vecs is None:
        raise ValueError("Eigenvectors are required for backward().")

    if dLdv is None:
        eig_vec_grad_proj = None
        cauchy = None
    else:
        eig_vec_grad_proj = compute_eig_vec_grad_proj(eig_vecs, dLdv)
        cauchy = compute_cauchy_matrix(eig_vals, ctx.k, ctx.eps)

    dLdA_val = compute_dLdA_val(
        A_sp_topo, eig_vecs, dLdl, dLdv, eig_vec_grad_proj, cauchy
    )

    return dLdA_val


def dLdA_dLdM_backward(
    ctx,
    dLdl: Float[t.Tensor, " k"],
    dLdv: Float[t.Tensor, "c k"] | None,
    needs_grad_A_val: bool,
    needs_grad_M_val: bool,
):
    """
    A function encapsulates the shared backward() gradient logic for generalized
    eigenvalue problems.
    """
    dLdA_val = None
    dLdM_val = None

    # The eigenvectors need to be orthonormal wrt M for the following calculation.
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
            cauchy = None
        else:
            eig_vec_grad_proj = compute_eig_vec_grad_proj(eig_vecs, dLdv)
            cauchy = compute_cauchy_matrix(eig_vals, ctx.k, ctx.eps)

    if needs_grad_A_val:
        dLdA_val = compute_dLdA_val(
            A_sp_topo, eig_vecs, dLdl, dLdv, eig_vec_grad_proj, cauchy
        )

    if needs_grad_M_val:
        dLdM_val = compute_dLdM_val(
            M_sp_topo, eig_vecs, dLdl, dLdv, eig_vec_grad_proj, cauchy
        )

    return dLdA_val, dLdM_val

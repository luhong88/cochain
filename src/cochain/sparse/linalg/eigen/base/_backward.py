import torch
from einops import einsum, rearrange
from jaxtyping import Float, Integer
from torch import Tensor

from ....decoupled_tensor import SparsityPattern


def compute_eig_vec_grad_proj(
    eig_vecs: Float[Tensor, "c k"],
    dLdv: Float[Tensor, "c k"],
) -> Float[Tensor, "k k"]:
    """Compute the projection of eigenvector gradients onto the eigenspace."""
    return eig_vecs.T @ dLdv


def compute_cauchy_matrix(
    eig_vals: Float[Tensor, " k"], eps: float | int
) -> Float[Tensor, "k k"]:
    """Compute the Cauchy/Lorentz matrix with optional regularization."""
    delta = rearrange(eig_vals, "j -> 1 j") - rearrange(eig_vals, "i -> i 1")

    if eps == 0:
        delta.fill_diagonal_(float("inf"))
        cauchy = 1.0 / delta

    else:
        # If eps > 0, compute a regularized version, where F_ij = Δ_ji / (Δ_ji^2 + ϵ).
        cauchy = delta / (delta.pow(2) + eps)

    return cauchy


def compute_dLdA_val(
    a_pattern: Integer[SparsityPattern, "r c"],
    eig_vecs: Float[Tensor, "c k"],
    dLdl: Float[Tensor, " k"],
    dLdv: Float[Tensor, "c k"] | None,
    eig_vec_grad_proj: Float[Tensor, "k k"] | None,
    cauchy: Float[Tensor, "k k"] | None,
) -> Float[Tensor, " nz"]:
    """
    Compute the gradient with respect to the nonzero values of A.

    Note that the formula implemented in this function is applicable to both
    standard and generalized eigenvalue problems.
    """
    eig_vecs_row = eig_vecs[a_pattern.idx_coo[0]]
    eig_vecs_col = eig_vecs[a_pattern.idx_coo[1]]

    # If the loss does not depend on the eigenvectors, then the eigenvalue
    # component of the gradient is given by dLdA_ij = sum_k[dLdλ_k * V_ik * V_jk]
    dLdA_eig_vals = einsum(
        eig_vecs_row,
        eig_vecs_col,
        dLdl,
        "nz eig, nz eig, eig -> nz",
    )

    if dLdv is None:
        dLdA_val = dLdA_eig_vals

    else:
        anti_sym_proj = 0.5 * (eig_vec_grad_proj - eig_vec_grad_proj.T)

        # Compute the Hadamard product K = F * P
        kernel: Float[Tensor, "k k"] = cauchy * anti_sym_proj

        # The eigenvector component is given by V @ K @ V.T, or
        # dLdA_ij = V_ik * K_kl * V_jl
        dLdA_eig_vecs = einsum(
            eig_vecs_row,
            eig_vecs_col,
            kernel,
            "nz eig_k, nz eig_l, eig_k eig_l -> nz",
        )

        # Sum together the eigenvalue and eigenvector components of the gradient.
        dLdA_val = dLdA_eig_vals + dLdA_eig_vecs

    return dLdA_val


def compute_dLdM_val(
    m_pattern: Integer[SparsityPattern, "r c"],
    eig_vals: Float[Tensor, " k"],
    eig_vecs: Float[Tensor, "c k"],
    dLdl: Float[Tensor, " k"],
    dLdv: Float[Tensor, "c k"] | None,
    eig_vec_grad_proj: Float[Tensor, "k k"] | None,
    cauchy: Float[Tensor, "k k"] | None,
) -> Float[Tensor, " nz"]:
    eig_vecs_row = eig_vecs[m_pattern.idx_coo[0]]
    eig_vecs_col = eig_vecs[m_pattern.idx_coo[1]]

    # If the loss does not depend on the eigenvectors, then the eigenvalue
    # component of the gradient is given by
    # dLdM_ij = -sum_k[λ_k * dLdλ_k * V_ik * V_jk]
    dLdM_eig_vals = -einsum(
        eig_vals,
        dLdl,
        eig_vecs_row,
        eig_vecs_col,
        "eig, eig, nz eig, nz eig -> nz",
    )

    if dLdv is None:
        dLdM_val = dLdM_eig_vals

    else:
        # The elements of the kernel is given by
        # off-diagonal: K_ij = F_ij * (λ_i*P_ji - λ_j*P_ij)/2
        # diagonal: K_ii = - P_ii/2
        lP = eig_vals.view(1, -1) * eig_vec_grad_proj
        kernel_off_diag = 0.5 * cauchy * (lP.T - lP)
        kernel_diag = -0.5 * torch.diag(eig_vec_grad_proj)
        kernel: Float[Tensor, "k k"] = torch.diagflat(kernel_diag) + kernel_off_diag

        # The "eigenvector" component is given by V @ K @ V.T, or
        # dLdM_ij = V_ik * K_kl * V_jl
        dLdM_eig_vecs = einsum(
            eig_vecs_row,
            eig_vecs_col,
            kernel,
            "nz eig_k, nz eig_l, eig_k eig_l -> nz",
        )

        # Sum together the eigenvalue and eigenvector components of the gradient.
        dLdM_val = dLdM_eig_vals + dLdM_eig_vecs

    return dLdM_val


def dLdA_backward(
    ctx, dLdl: Float[Tensor, " k"], dLdv: Float[Tensor, "c k"] | None
) -> Float[Tensor, " nz"]:
    """Backward gradient logic for eigenvalue problems."""
    # The eigenvectors need to be length-normalized for the following calculation.
    eig_vals, eig_vecs = ctx.saved_tensors
    A_pattern: SparsityPattern = ctx.A_pattern

    # This error should never be triggered if the user-facing wrapper does its job.
    if eig_vecs is None:
        raise ValueError("Eigenvectors are required for backward().")

    if dLdv is None:
        eig_vec_grad_proj = None
        cauchy = None
    else:
        eig_vec_grad_proj = compute_eig_vec_grad_proj(eig_vecs, dLdv)
        cauchy = compute_cauchy_matrix(eig_vals, ctx.eps)

    dLdA_val = compute_dLdA_val(
        A_pattern, eig_vecs, dLdl, dLdv, eig_vec_grad_proj, cauchy
    )

    return dLdA_val


def dLdA_dLdM_backward(
    ctx,
    dLdl: Float[Tensor, " k"],
    dLdv: Float[Tensor, "c k"] | None,
    needs_grad_A_val: bool,
    needs_grad_M_val: bool,
) -> tuple[Float[Tensor, " A_nz"], Float[Tensor, " M_nz"]]:
    """Backward gradient logic for generalized eigenvalue problems."""
    dLdA_val = None
    dLdM_val = None

    # The eigenvectors are assumed to be orthonormal wrt M, which is required for
    # the following calculation.
    eig_vals, eig_vecs = ctx.saved_tensors
    A_pattern: SparsityPattern = ctx.A_pattern
    M_pattern: SparsityPattern = ctx.M_pattern

    if needs_grad_A_val or needs_grad_M_val:
        # This error should never be triggered if the user-facing wrapper does its job.
        if eig_vecs is None:
            raise ValueError("Eigenvectors are required for backward().")

        if dLdv is None:
            eig_vec_grad_proj = None
            cauchy = None
        else:
            eig_vec_grad_proj = compute_eig_vec_grad_proj(eig_vecs, dLdv)
            cauchy = compute_cauchy_matrix(eig_vals, ctx.eps)

    if needs_grad_A_val:
        dLdA_val = compute_dLdA_val(
            A_pattern, eig_vecs, dLdl, dLdv, eig_vec_grad_proj, cauchy
        )

    if needs_grad_M_val:
        dLdM_val = compute_dLdM_val(
            M_pattern, eig_vals, eig_vecs, dLdl, dLdv, eig_vec_grad_proj, cauchy
        )

    return dLdA_val, dLdM_val

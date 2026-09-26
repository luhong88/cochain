from jaxtyping import Float, Integer
from torch import Tensor

from ...sparse.decoupled_tensor import SparseDecoupledTensor, SparsityPattern
from ...sparse.linalg.eigen.base._backward import compute_dLdA_val, compute_dLdM_val


def compute_dLdM_km1_val(
    mass_km1_pattern: Integer[SparsityPattern, "km1_splx km1_splx"],
    eig_vecs_codiff: Float[Tensor, "km1_splx eig"],
    dLdl: Float[Tensor, " eig"],
    dLdv: Float[Tensor, "k_splx eig"] | None,
    eig_vec_grad_proj: Float[Tensor, "k_splx k_splx"] | None,
    cauchy: Float[Tensor, "k_splx k_splx"] | None,
) -> Float[Tensor, " nz"]:
    """
    Compute the gradient with respect to the nonzero values of $M_{k-1}$.

    Note that $dL/dM_{k-1}$ essentially reuses the same logic as implemented in
    compute_dLdA_val(), but with the codifferential of the eigenvectors in place
    of the eigenvectors.
    """
    return -compute_dLdA_val(
        a_pattern=mass_km1_pattern,
        eig_vecs=eig_vecs_codiff,
        dLdl=dLdl,
        dLdv=dLdv,
        eig_vec_grad_proj=eig_vec_grad_proj,
        cauchy=cauchy,
    )


def compute_dLdM_k_val(
    cbd_km1: Float[SparseDecoupledTensor, "k_splx km1_splx"],
    mass_k_pattern: Integer[SparsityPattern, "k_splx k_splx"],
    eig_vals: Float[Tensor, " eig"],
    eig_vecs: Float[Tensor, "k_splx eig"],
    eig_vecs_codiff: Float[Tensor, "km1_splx eig"],
    dLdl: Float[Tensor, " eig"],
    dLdv: Float[Tensor, "k_splx eig"] | None,
    eig_vec_grad_proj: Float[Tensor, "k_splx k_splx"] | None,
    cauchy: Float[Tensor, "k_splx k_splx"] | None,
):
    """Compute the gradient with respect to the nonzero values of $M_k$."""
    # For the eigenvalue problem S_k@v_i = λ_i@M_k@v_i, we compute separately
    # the gradient w.r.t. M_k through the left-hand side path (via its contribution
    # to the definition of S_k) and through the right-hand side path (via its role
    # as the metric).

    # The LHS path logic is similar to that in compute_dLdA_val().
    d_eig_vec_codiffs = cbd_km1 @ eig_vecs_codiff

    eig_vecs_row = eig_vecs[mass_k_pattern.idx_coo[0]]
    d_eig_vec_codiffs_col = d_eig_vec_codiffs[mass_k_pattern.idx_coo[1]]

    eig_vecs_col = eig_vecs[mass_k_pattern.idx_coo[1]]
    d_eig_vec_codiffs_row = d_eig_vec_codiffs[mass_k_pattern.idx_coo[0]]

    # If the loss does not depend on the eigenvectors, then the eigenvalue
    # component of the gradient is given by
    #
    # dLdM = V@dLdλ@(d_{k-1}@W).T + (d_{k-1}@W)@dLdλ@V.T
    #
    # or, componentwise,
    #
    # dLdM_ij = sum_k[dLdλ_k * V_ik * dW_jk] + sum_k[dLdλ_k * V_jk * dW_ik]
    dLdM_eig_vals = einsum(
        eig_vecs_row,
        d_eig_vec_codiffs_col,
        dLdl,
        "nz eig, nz eig, eig -> nz",
    )
    dLdM_eig_vals_T = einsum(
        d_eig_vec_codiffs_row,
        eig_vecs_col,
        dLdl,
        "nz eig, nz eig, eig -> nz",
    )

    if dLdv is None:
        dLdM_lhs = dLdM_eig_vals + dLdM_eig_vals_T

    else:
        anti_sym_proj = 0.5 * (eig_vec_grad_proj - eig_vec_grad_proj.T)

        # Compute the Hadamard product K = F * P.
        kernel: Float[Tensor, "k k"] = cauchy * anti_sym_proj

        # The eigenvector component is given by V @ K @ (d_{k-1} @ W).T.
        dLdM_eig_vecs = einsum(
            eig_vecs_row,
            d_eig_vec_codiffs_col,
            kernel,
            "nz eig_k, nz eig_l, eig_k eig_l -> nz",
        )
        dLdM_eig_vecs_T = einsum(
            d_eig_vec_codiffs_row,
            eig_vecs_col,
            kernel,
            "nz eig_k, nz eig_l, eig_k eig_l -> nz",
        )

        # Sum together the eigenvalue and eigenvector components of the gradient.
        dLdM_lhs = dLdM_eig_vals + dLdM_eig_vals_T + dLdM_eig_vecs + dLdM_eig_vecs_T

    # The RHS path resues the compute_dLdM_val() logic exactly.
    dLdM_rhs = compute_dLdM_val(
        m_pattern=mass_k_pattern,
        eig_vals=eig_vals,
        eig_vecs=eig_vecs,
        dLdl=dLdl,
        dLdv=dLdv,
        eig_vec_grad_proj=eig_vec_grad_proj,
        cauchy=cauchy,
    )

    return dLdM_lhs + dLdM_rhs


def compute_dLdM_kp1_val(
    cbd_k: Float[SparseDecoupledTensor, "kp1_splx k_splx"],
    mass_kp1_pattern: Integer[SparsityPattern, "km1_splx km1_splx"],
    eig_vecs: Float[Tensor, "k_splx eig"],
    dLdl: Float[Tensor, " eig"],
    dLdv: Float[Tensor, "k_splx eig"] | None,
    eig_vec_grad_proj: Float[Tensor, "k_splx k_splx"] | None,
    cauchy: Float[Tensor, "k_splx k_splx"] | None,
) -> Float[Tensor, " nz"]:
    """
    Compute the gradient with respect to the nonzero values of $M_{k+1}$.

    Note that $dL/dM_{k+1}$ essentially reuses the same logic as implemented in
    compute_dLdA_val(), but with $d_k V$ in place of $V$.
    """
    return compute_dLdA_val(
        a_pattern=mass_kp1_pattern,
        eig_vecs=cbd_k.T @ eig_vecs,
        dLdl=dLdl,
        dLdv=dLdv,
        eig_vec_grad_proj=eig_vec_grad_proj,
        cauchy=cauchy,
    )

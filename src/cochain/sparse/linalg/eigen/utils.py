from typing import Literal

import torch
from jaxtyping import Float
from torch import Tensor

from ...decoupled_tensor import SparseDecoupledTensor


def M_orthonormalize(
    V: Float[Tensor, "m n"],
    M_op: Float[SparseDecoupledTensor, "m m"],
    *,
    rtol: float | None = None,
    n_min: int | None = None,
    generator: torch.Generator | None = None,
    max_iter: int = 3,
) -> Float[Tensor, "m l"]:
    """
    Convert the column vectors of V into M-orthonormal vectors using iterative
    canonical/PCA orthonormalization.

    The method implemented in this function follows the SVQB method (Singular Value
    QR Blocking) from Stathopoulos & Wu., SIAM J. Sci. Comput. (2002). This function
    differs from SVQB in that it is rank-adaptive; i.e., instead of clamping,
    linearly dependent columns of V are dropped and the function returns fewer
    orthonormalized vectors (l <= n). If n_min is provided and the returned V matrix
    has fewer columns than n_min, then a "soft restart" will be attempted where
    new, random M-orthonormal vectors are appended to V to ensure that it has n_min
    columns.

    An issue with the SVQB approach is that it requires a Gram matrix whose condition
    number is the square of the condition number of V. It is therefore recommended
    to perform iterative refinment to suppress the condition number issue. To
    further help with the issue, this function performs the orthonormalization
    in float64.
    """
    # Force double precision to further suppress the condition number issue.
    V_dtype = V.dtype
    V_double = V.to(torch.float64)
    M_op_double = M_op.to(torch.float64)

    V_current = V_double
    for _ in range(max_iter):
        V_ortho_double, cond = _M_orthonormalize_one_iter(V_current, M_op_double, rtol)

        pad_cond = 0.0
        # If the number of columns in V_ortho drops below the minimum, perform
        # a "soft restart" by padding random vectors to V_ortho.
        if (n_min is not None) and (V_ortho_double.size(-1) < n_min):
            pad = torch.randn(
                (V_ortho_double.size(0), n_min - V_ortho_double.size(-1)),
                generator=generator,
                dtype=V_ortho_double.dtype,
                device=V_ortho_double.device,
            )

            # The padded vectors need to form a subspace that is orthogonal to
            # the current V_ortho column space
            pad_overlap = V_ortho_double.T @ (M_op_double @ pad)
            pad_proj = V_ortho_double @ pad_overlap
            pad_res = pad - pad_proj

            # The padded vectors need to be M-orthonormal
            pad_res_ortho, pad_cond = _M_orthonormalize_one_iter(
                pad_res, M_op_double, rtol
            )

            # Concat to form the new basis
            V_ortho_double = torch.hstack((V_ortho_double, pad_res_ortho))

        V_current = V_ortho_double

        # If V is exactly M-orthonormal, then V.T@M@V = I and the condition number
        # is 1; here, we allow for small deviation up to 1e-3.
        if (cond <= 1.0 + 1e-3) and (pad_cond <= 1.0 + 1e-3):
            break

    V_ortho = V_current.to(V_dtype)

    return V_ortho


def _M_orthonormalize_one_iter(
    V: Float[Tensor, "m n"],
    M_op: Float[SparseDecoupledTensor, "m m"],
    rtol: float | None = None,
) -> tuple[Float[Tensor, "m l"], Float[Tensor, ""]]:
    if rtol is None:
        rtol = V.size(0) * torch.finfo(V.dtype).eps

    # Compute the M-orthogonal gram matrix.
    G: Float[Tensor, "n n"] = V.T @ (M_op @ V)

    # Implicit normalization of G. Let D be a diagonal matrix whose diagonal
    # elements are the inverse square roots of the diagonal elements of G; then
    # the normalized G is G' = D@G@D. This is equivalent to normalizing the columns
    # of V, but computationally cheaper. This scaling improves the condition number
    # of G for eigh(). Note that columns of V that are zero or very close to
    # zero are not length-normalized.
    V_col_norm2 = torch.diag(G).clamp(min=0.0)
    D = 1.0 / torch.sqrt(V_col_norm2)

    zero_col_mask = (~torch.isfinite(D)) | (V_col_norm2 < torch.finfo(V.dtype).eps * 10)
    D[zero_col_mask] = 1.0

    G_scaled = D.view(-1, 1) * G * D.view(1, -1)

    # Perform an eigendecomposition of G = Q@Λ@Q.T.
    eig_vals, eig_vecs = torch.linalg.eigh(G_scaled)

    # Drop very small eigenvalues corresponding to linearly dependent columns.
    eps = rtol * eig_vals.max()
    mask = eig_vals > eps

    # If V is basically zero, return a single zero vector and a condition number of 0.
    if not mask.any():
        return torch.zeros_like(V[:, :1]), torch.tensor(
            0.0, dtype=V.dtype, device=V.device
        )

    eig_vals_masked = eig_vals[mask]
    inv_eig_vals_masked = 1.0 / torch.sqrt(eig_vals_masked)
    eig_vecs_masked = eig_vecs[:, mask]

    # Check the condition number using the masked eigenvalues, for assessing
    # progress of iterative refinement.
    cond = torch.sqrt(eig_vals_masked.max() / eig_vals_masked.min())

    # Compute the whitening matrix W = Q@Λ^(-1/2) as the inverse square root
    # of G. Need to apply the D vector here to undo the implicit normalization.
    W = D.view(-1, 1) * eig_vecs_masked * inv_eig_vals_masked.view(1, -1)

    # Find V_ortho = V@W, the M-orthonormal version of V. With some algebra,
    # one can check that V_ortho.T@M@V_ortho = I.
    V_ortho = V @ W

    return V_ortho, cond


def canonicalize_eig_vec_signs(
    eig_vecs: Float[Tensor, "m k"],
) -> Float[Tensor, "m k"]:
    """
    "Canonicalize" the orientation of eigenvectors via the convention that the
    element with the largest absolute value has a positive sign.
    """
    max_idx = eig_vecs.abs().max(dim=0, keepdim=True).indices
    max_sign = torch.gather(input=eig_vecs, dim=0, index=max_idx).sign()
    canon_eig_vecs = eig_vecs * max_sign
    return canon_eig_vecs


def grassmann_proj_dists(
    eig_vecs_pred: Float[Tensor, "m k"],
    eig_vecs_true: Float[Tensor, "m k"],
    M: Float[Tensor, "m m"] | Float[SparseDecoupledTensor, "m m"] | None = None,
    mode: Literal["pairwise", "subspace"] = "subspace",
) -> Float[Tensor, "*k"]:
    """
    Compute the Grassmann projection distance between two sets of eigenvectors.

    If `mode='pairwise'`, the function compares the i-th eigenspace of `eig_vecs_pred`
    with the i-th eigenspace of `eig_vecs_true`; mathematically, it computes
    1 - (v_i_pred.T @ M @ v_i_true)^2, or ||P_i_pred - P_i_true||_F^2/2. For this
    mode to produce meaningful results, the eigenvector pairs in `eig_vecs_pred`
    and `eig_vecs_true` need to correspond to the same true eigenvalue, and there
    need to be no degenerate eigenvalues (eigenvectors of degenerate eigenvalues
    can differ by a rotation and it is only meaningful to compare the projection
    matrices of the degenerate eigenspaces, not the individual eigenvectors).

    If `mode=subspace`', the function compares the eigenspace spanned by the entire
    k eigenvectors in `eig_vecs_pred` and `eig_vecs_true`; mathematically, it
    computes k - ||V_pred.T @ M @ V_true||_F^2, or ||P_pred - P_true||_F^2/2. This
    mode is robust to eigenvalue/eigenvector permutations and degenerate eigenvalues.
    For this mode to produce meaningful results, there needs to be a gap between
    λ_k and λ_(k+1) to prevent the possibility of a degeneracy at the exact
    spetral cutoff point (if such degeneracy exists, then the comparison of the kth
    eigenspace is not meaningful). If k = m (i.e., a full eigendecomposition),
    then this mode returns 0, assuming that `eig_vecs_pred` and `eig_vecs_true`
    both satisfies the M-orthonormality condition.

    If the M matrix is provided, it is assumed that the eigenvectors are derived
    from a generalized eigenvalue problem, and both the `eig_vecs_pred` and
    `eig_vecs_true` are M-orthonormal. If M is None, then the eigenvectors are
    assumed to be orthonormal w.r.torch. the standard Euclidean metric.
    """
    if M is None:
        W = eig_vecs_true
    else:
        W = M @ eig_vecs_true

    match mode:
        case "pairwise":
            dist = 1 - torch.sum(eig_vecs_pred * W, dim=0).pow(2)
        case "subspace":
            k = eig_vecs_true.size(-1)
            dist = k - torch.sum((eig_vecs_pred.T @ W).pow(2))
        case _:
            raise ValueError()

    return dist

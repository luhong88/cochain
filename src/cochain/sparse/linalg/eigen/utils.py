from typing import Literal

import torch as t
from jaxtyping import Float

from ...operators import SparseOperator


def canonicalize_eig_vec_signs(
    eig_vecs: Float[t.Tensor, "m k"],
) -> Float[t.Tensor, "m k"]:
    """
    "Canonicalize" the orientation of eigenvectors via the convention that the
    element with the largest absolute value has a positive sign.
    """
    max_idx = eig_vecs.abs().max(dim=0, keepdim=True).indices
    max_sign = t.gather(input=eig_vecs, dim=0, index=max_idx).sign()
    canon_eig_vecs = eig_vecs * max_sign
    return canon_eig_vecs


def grassmann_proj_dists(
    eig_vecs_pred: Float[t.Tensor, "m k"],
    eig_vecs_true: Float[t.Tensor, "m k"],
    M: Float[t.Tensor, "m m"] | Float[SparseOperator, "m m"] | None = None,
    mode: Literal["pairwise", "subspace"] = "subspace",
) -> Float[t.Tensor, "*k"]:
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
    assumed to be orthonormal w.r.t. the standard Euclidean metric.
    """
    if M is None:
        W = eig_vecs_true
    else:
        W = M @ eig_vecs_true

    match mode:
        case "pairwise":
            dist = 1 - t.sum(eig_vecs_pred * W, dim=0).pow(2)
        case "subspace":
            k = eig_vecs_true.size(-1)
            dist = k - t.sum((eig_vecs_pred.T @ W).pow(2))
        case _:
            raise ValueError()

    return dist

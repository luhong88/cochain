__all__ = ["m_orthonormalize", "canonicalize_eig_vec_signs", "grassmann_proj_dists"]

from typing import Literal

import torch
from jaxtyping import Float
from torch import Tensor

from ....decoupled_tensor import BaseDecoupledTensor, SparseDecoupledTensor


def _m_normalize(
    v: Float[Tensor, "m n"],
    m: Float[BaseDecoupledTensor, "m m"],
) -> tuple[
    Float[Tensor, "m l"],
    Float[Tensor, "m l"],
    Float[Tensor, " l"],
    Float[Tensor, " l"],
]:
    r"""
    M-normalize the nonzero column vectors of a matrix.

    Parameters
    ----------
    v : [m, n]
        A dense 2D matrix whose columns are to be M-normalized.
    m : [m, m]
        A sparse 2D symmetric positive definite matrix that induces an inner product
        on the column space of $V$.

    Returns
    -------
    v_normed : [m, l]
        A dense 2D matrix whose columns consist of the M-normalized, nonzero
        column vectors of `v`.
    mv_normed: [m, l]
        A dense 2D matrix computed as `m @ v_normed`.
    col_norm: [l,]
        The M-norms of the nonzero column vectors of `v`.
    col_mask: [l,]
        A boolean mask marking the nonzero columns of `v`.

    Notes
    -----
    This function considers a column vector to be zero if every elements in that
    column is identically zero. The calculation of column vector norms after
    scaling by `1/col_max` avoids potential numerical issues for very large or
    small column vectors.
    """
    empty = torch.empty((v.size(0), 0), dtype=v.dtype, device=v.device)

    if v.numel() == 0:
        return empty, empty, v.new_empty(0), v.new_empty(0)

    # Normalize the length of each column of V by dividing each column by its
    # absolute max entry, which gives V_scaled.
    col_max = v.abs().amax(dim=0)
    col_mask = col_max > 0

    if not col_mask.any():
        return empty, empty, v.new_empty(0), v.new_empty(0)

    v_scaled = v[:, col_mask] / col_max[col_mask]
    mv_scaled = m @ v_scaled

    # Compute the M-norms of the columns of V_scaled.
    col_norm2 = (v_scaled * mv_scaled).sum(dim=0)
    col_norm = torch.sqrt(col_norm2)

    return (
        v_scaled / col_norm,
        mv_scaled / col_norm,
        col_norm * col_max[col_mask],
        col_mask,
    )


def _m_orthogonalize(
    v: Float[Tensor, "m n"],
    u: Float[Tensor, "m l"],
    m: Float[BaseDecoupledTensor, "m m"],
    mv: Float[Tensor, "m n"] | None = None,
) -> Float[Tensor, "m l"]:
    """
    Find the vector components perpendicular from an M-orthonormal vector set.

    Assuming that V contains column vectors that are M-orthonormal, then this
    function computes the perpendicular projection U_perp = U - V@(V.T@M@U). This
    projection is performed twice to guard against the possibility that U lies
    almost entirely in span(V). It is possible to perform this iterative
    refinement selectively by checking the change in vector norm after the
    first projection, but that would require computing U.T@M@U, which is usually
    more expensive than just projecting again.

    Note that this function does not guard against input V that deviates from
    the M-orthonormality condition or an input M that is ill-conditioned.
    """
    if mv is None:
        mv = m @ v

    u_perp = u - v @ (mv.T @ u)
    u_perp_again = u_perp - v @ (mv.T @ u_perp)
    return u_perp_again


def m_orthonormalize(
    v: Float[Tensor, "m n"],
    m: Float[BaseDecoupledTensor, "m m"],
    *,
    rtol: float | None = None,
    n_min: int | None = None,
    generator: torch.Generator | None = None,
    max_iter: int = 3,
) -> Float[Tensor, "m l"]:
    r"""
    $M$-orthonormalize the $V$ column vectors with iterative canonical orthonormalization.

    Parameters
    ----------
    v : [m, n]
        A dense 2D matrix whose columns are to be $M$-orthonormalized.
    m : [m, m]
        A sparse 2D symmetric positive definite matrix that induces an inner product
        on the column space of $V$.
    rtol
        A relative tolerance threshold on the normalized Gram eigenvalues for
        checking linearly dependent vectors. Defaults to `m` times the double
        precision machine epsilon.
    n_min
        The minimum number of column vectors to be returned.
    max_iter
        The maximum number of iterations to perform.

    Returns
    -------
    v_ortho : [m, l]
        A dense 2D matrix whose columns form an $M$-orthonormal basis for the column
        space of V, up to the relative rank tolerance. Without a soft restart,
        the number of columns satisfies $0 \le l \le n$; an all-zero or empty
        input returns an empty basis. A soft restart can supplement this basis
        with vectors outside the input column space.

    Notes
    -----
    The method implemented in this function follows the SVQB method (Singular Value
    QR Blocking) from Stathopoulos & Wu., SIAM J. Sci. Comput. (2002). This function
    differs from SVQB in that it is rank-adaptive; i.e., instead of clamping,
    linearly dependent columns of $V$ are dropped and the function returns fewer
    orthonormalized vectors ($l \le n$). If `n_min` is provided and the returned $V$
    matrix has fewer columns than `n_min`, then a "soft restart" will be attempted
    where new, random $M$-orthonormal vectors are appended to $V$ to ensure that it
    has `n_min` columns. The rest of this section is a full accounting of the
    algorithm implemented in this function.

    **Canonical orthogonalization**
    Consider a matrix $V$ whose column vectors are to be M-orthonormalized. Let us
    denote this unknown $M$-orthogonal matrix as $U$, which satisfies the condition
    $U^T M U = I$. Since the column vectors of $V$ and $U$ span the same space,
    there is a linear transformation, represented by the whitening matrix $W$,
    such that $U = V W$.

    Let $G = V^T M V$ be the symmetric positive definite Gram matrix with the
    eigendecomposition $G = Q \Lambda Q^T$, where $Q$ is orthogonal. Then, we claim
    that a valid choice of $W$ is to set $W = Q \Lambda^{-1/2}$. To see why, note that

    $$
    U^T M U = W^T (V^T M V) W = (Q \Lambda^{-1/2})^T Q \Lambda Q^T (Q \Lambda^{-1/2}) = I
    $$

    that is, $U$ is an $M$-orthogonal matrix, as desired. This technique is also known
    as PCA whitening.

    **Rank-adaptive orthonormalization**
    A useful spectral property of the Gram matrix $G$ is that the eigenvectors
    corresponding to the zero eigenvalue represent linear dependence relations
    among the column vectors of $V$. To see why, let $x$ be such an eigenvector,
    then

    $$x^T G x = x^T V^T M V x = \|Vx\|_M^2 = 0$$

    that is, $V x = 0$ and $x$ is the coefficient vector encoding a linear dependence
    relation. Therefore, a numerically robust way to handle $V$ with (nearly) linearly
    dependent column vectors is to define $W$ using reduced $\Lambda$ and $Q$
    matrices, where the (near) zero eigenvalues and their corresponding eigenvectors
    have been deleted. This approach returns a matrix $U$ with potentially fewer
    columns than $V$, but it guarantees that $U$ has linearly independent columns
    that span a subspace of the $V$ column space.

    **Soft restart**
    In some applications, there is a minimum number of required orthonormal vectors.
    To prevent the matrix $U$ from becoming too small, we introduce a set of random
    vectors $R$ to "supplement" $U$. To ensure that the combined column vectors of
    $U$ and $R$ still form an $M$-orthonormal basis set, we first project out the
    component of $R$ that's not perpendicular to the column space of $U$,

    $$R^\perp = R - P R$$

    where $P$ is the projection matrix for the column space of $U$. Then, $R^\perp$
    itself undergoes the same rank-adaptive canonical orthogonalization step as
    described above before being appended to $U$.

    Here, the $M$-orthogonal projection matrix $P$ is defined as

    $$P = U U^T M$$

    Briefly, we show that $P$ is indeed the projection matrix to the column space
    of $U$. First, note that

    $$P^2 = U (U^T M U) U^T M = U U^T M = P$$

    which shows that $P$ is idempotent and thus a projection matrix. Next, note that

    $$
    U^T M R^\perp
    = U^T M (R - P R)
    = U^T M R - (U^T M U) U^T M R = 0
    $$

    which shows that the column vectors of $U$ and $R^\perp$ are $M$-orthogonal, and
    thus the column space of $R^\perp$ is $M$-orthogonal to the column space of $U$.

    **Jacobi preconditioning**
    One potential problem with the canonical orthogonalization method described
    above is that the Gram matrix can have a very large condition number if
    the norms of the column vectors of $V$ span a wide range of magnitude. Therefore,
    it is numerically preferable to perform the orthogonalization after the column
    vectors of $V$ have been normalized.

    To avoid underflow/overflow from computing the norms of very small or large
    column entries, first discard exactly zero columns and divide each remaining
    column by its largest absolute entry,

    $$s_j = \max_i |V_{ij}|, \qquad B_j = V_j / s_j$$

    Then compute the squared metric norms of the scaled columns and normalize
    explicitly,

    $$q_j = B_j^T M B_j, \qquad C_j = B_j / \sqrt{q_j}$$

    The normalized Gram matrix is $\bar G = C^T M C$. Given its eigendecomposition
    $\bar G = \bar Q \bar\Lambda \bar Q^T$, retain eigenvalues greater than
    `rtol` times the largest eigenvalue and construct the output directly from
    the normalized columns

    $$U = C \bar Q_\text{kept} \bar\Lambda^{-1/2}_\text{kept}$$

    **Iterative refinement**
    Since $G = V^T M V$, the condition number of $G$ is roughly the square of the
    condition number of $V$. Therefore, if $V$ is highly ill-conditioned, the
    Jacobi preconditioning procedure may be insufficient to fully $M$-orthonormalize
    the column vectors. Therefore, it is recommended to run the full algorithm
    iteratively a few times in double precision to minimize any residual
    non-orthogonality.
    """
    # Force double precision to further suppress the condition number issue.
    v_dtype = v.dtype
    v_double = v.to(torch.float64)

    # In LOBPCG this function may be called with m as an IdOp object, which does
    # not have the dtype attribute.
    if getattr(m, "dtype", torch.float64) == torch.float64:
        m_double = m
    else:
        m_double = m.to(torch.float64)

    v_current = v_double
    for _ in range(max_iter):
        v_ortho_double, cond = _m_orthonormalize_one_iter(v_current, m_double, rtol)

        pad_cond = 0.0
        # If the number of columns in V_ortho drops below the minimum, perform
        # a "soft restart" by padding random vectors to V_ortho.
        if (n_min is not None) and (v_ortho_double.size(-1) < n_min):
            pad = torch.randn(
                (v_ortho_double.size(0), n_min - v_ortho_double.size(-1)),
                generator=generator,
                dtype=v_ortho_double.dtype,
                device=v_ortho_double.device,
            )

            # The padded vectors need to form a subspace that is orthogonal to
            # the current V_ortho column space.
            pad_perp = _m_orthogonalize(v_ortho_double, pad, m_double)

            # The padded vectors need to be M-orthonormal.
            pad_res_ortho, pad_cond = _m_orthonormalize_one_iter(
                pad_perp, m_double, rtol
            )

            # Concat to form the new basis.
            v_ortho_double = torch.hstack((v_ortho_double, pad_res_ortho))

        v_current = v_ortho_double

        # If V is exactly M-orthonormal, then V.T @ M @ V = I and the condition
        # number is 1; here, we allow for small deviation up to 1e-3.
        if (cond <= 1.0 + 1e-3) and (pad_cond <= 1.0 + 1e-3):
            break

    v_ortho = v_current.to(v_dtype)

    return v_ortho


def _m_orthonormalize_one_iter(
    v: Float[Tensor, "m n"],
    m: Float[BaseDecoupledTensor, "m m"],
    rtol: float | None = None,
) -> tuple[Float[Tensor, "m l"], Float[Tensor, ""]]:
    """Perform one iteration of M-orthonormalization."""
    eps = torch.finfo(v.dtype).eps

    if rtol is None:
        rtol = v.size(0) * eps

    # Compute V_normed, which contains the M-normalized nonzero column vectors of V.
    v_normed, mv_normed, _, _ = _m_normalize(v, m)
    if v_normed.size(1) == 0:
        return torch.zeros_like(v[:, :0]), v.new_tensor(0.0)

    # Form the M-orthogonal gram matrix G using V_normed, the M-normalized V.
    g_scaled = v_normed.T @ mv_normed
    if not torch.isfinite(g_scaled).all():
        raise ValueError("The normalized Gram matrix must be finite.")

    # Enforce symmetry on G.
    g_scaled = (g_scaled + g_scaled.T) / 2.0

    # Perform an eigendecomposition of G = Q @ Λ @ Q.T.
    eig_vals, eig_vecs = torch.linalg.eigh(g_scaled)

    # Drop very small eigenvalues corresponding to linearly dependent columns.
    eps = rtol * eig_vals.max()
    mask = eig_vals > eps

    # If V is basically zero, return an empty basis zero and a condition number of 0.
    if not mask.any():
        return torch.empty(
            (v.size(0), 0), dtype=v.dtype, device=v.device
        ), v.new_tensor(0.0)

    eig_vals_masked = eig_vals[mask]
    eig_vecs_masked = eig_vecs[:, mask]

    # Check the condition number using the masked eigenvalues, for assessing
    # progress of iterative refinement.
    cond = torch.sqrt(eig_vals_masked.max()) / torch.sqrt(eig_vals_masked.min())

    # Find V_ortho = V_normed @ Q @ Λ^(-1/2) using the retained eigenpairs.
    v_ortho = (v_normed @ eig_vecs_masked) / torch.sqrt(eig_vals_masked)

    return v_ortho, cond


def canonicalize_eig_vec_signs(
    eig_vecs: Float[Tensor, "m k"],
) -> Float[Tensor, "m k"]:
    """
    Canonicalize the orientation of eigenvectors.

    This function adjusts the signs of the column vectors such that the
    vector with the largest absolute value element has a positive sign.

    Parameters
    ----------
    eig_vecs : [m, k]
        The input eigenvector matrix.

    Returns
    -------
    canon_eig_vecs : [m, k]
        The canonicalized eigenvector matrix.
    """
    max_idx = eig_vecs.abs().max(dim=0, keepdim=True).indices
    max_sign = torch.gather(input=eig_vecs, dim=0, index=max_idx).sign()
    canon_eig_vecs = eig_vecs * max_sign
    return canon_eig_vecs


def grassmann_proj_dists(
    eig_vecs_pred: Float[Tensor, "m k"],
    eig_vecs_true: Float[Tensor, "m k"],
    m: Float[Tensor, "m m"] | Float[SparseDecoupledTensor, "m m"] | None = None,
    mode: Literal["pairwise", "subspace"] = "subspace",
) -> Float[Tensor, "*k"]:
    r"""   
    Compute the Grassmann projection distance between two sets of eigenvectors.

    Parameters
    ----------
    eig_vecs_pred : [m, k]
        A matrix whose columns are the predicted eigenvectors.
    eig_vecs_true : [m, k]
        A matrix whose columns are the true eigenvectors.
    m : [m, m]
        A symmetric positive definite matrix that induces an inner product on
        the column space.
    mode
        If `mode` is `"pairwise"`, this function compares the $i$-th eigenspace
        of `eig_vecs_pred` with the $i$-th eigenspace of `eig_vecs_true`.
        If `mode` is `"subspace"`, this function compares the eigenspace spanned
        by the entire $k$ eigenvectors in `eig_vecs_pred` and `eig_vecs_true`.

    Returns
    -------
    dist : [*k]
        The Grassmann projection distance. If `mode` is `"pairwise"`, then
        `k` distances are calculated, one for each pair of `pred` and `true`
        eigenvectors. If `mode` is `"subspace"`, then a single distance is
        returned.

    Notes
    -----
    If the $M$ matrix is provided, it is assumed that the eigenvectors are 
    derived from a generalized eigenvalue problem, and both the `eig_vecs_pred` 
    and `eig_vecs_true` are $M$-orthonormal. If $M$ is `None`, then the 
    eigenvectors are assumed to be orthonormal w.r.t. the standard Euclidean 
    metric.

    In general, consider two $M$-orthogonal matrices $U$ and $V$. To compare 
    the distance between the column spaces of $U$ and $V$, we define the 
    chordal distance

    $$d^2(U, V) = \frac 1 2 \text{tr}[(P_U - P_V)^2]$$

    where $P_U = U U^T M$ is the $M$-orthogonal projection matrix onto the 
    column space of $U$ and $P_V$ is the $M$-orthogonal projection matrix onto the
    column space of $V$.

    This definition can be further simplified to avoid the need to explicitly compute 
    the projection matrices,

    $$
    \begin{aligned}
    d^2(U, V) &= \frac 1 2 \text{tr}(P_U^2 - P_U P_V - P_V P_U + P_V^2) \\
    & \overset{(1)}{=} \frac 1 2 \text{tr}(P_U - P_U P_V - P_V P_U + P_V) \\
    & \overset{(2)}{=} k - \text{tr}(P_U P_V) \\
    & = k - \text{tr}[(U^T M V) (V^T M U)] \\
    & \overset{(3)}{=} k - \|U^T M V\|_F^2
    \end{aligned}
    $$

    where $\|\cdot\|_F$ is the Frobenius matrix norm. Here, equality (1) follows
    from the fact that the projection matrix is idempotent (e.g., $P_U^2 = P_U$), 
    equality (2) follows from the fact that the trace of a projection operator is 
    equal to the dimensionality of the subspace it projects onto (i.e., $k$) and the 
    trace operator is invariant to cyclic permutation of matrix multiplication, and 
    equality (3) follows from the fact that $\|A\|_F^2 = \text{tr}(A^TA)$.

    In the `"pairwise"` mode, we compare the eigenspace spanned by the $i$-th
    columns of $U$ and $V$ independently,

    $$d^2(u_i, v_i) = 1 - (u_i^T M v_i)^2$$

    For this mode to produce meaningful results, each eigenvector pairs in 
    `eig_vecs_pred` and `eig_vecs_true` need to correspond to the same true 
    eigenvalue, and there need to be no degenerate eigenvalues (eigenvectors 
    of degenerate eigenvalues can differ by a rotation and it is only meaningful 
    to compare the projection matrices of the degenerate eigenspaces, not 
    the individual eigenvectors).

    In the `"subspace"` mode, we compare the eigenspace spanned by the entire
    $k$ eigenvectors in `eig_vecs_pred` and `eig_vecs_true`. This mode is 
    robust to eigenvalue/eigenvector permutations and degenerate eigenvalues.
    However, for this mode to produce meaningful results, there needs to be 
    a gap between $\lambda_k$ and $\lambda_{k+1}$ to prevent the possibility 
    of a degeneracy at the exact spetral cutoff point (if such degeneracy 
    exists, then the comparison of the $k$-th eigenspace is not meaningful). 
    If $k = m$ (i.e., a full eigendecomposition), then this mode returns 0, 
    assuming that `eig_vecs_pred` and `eig_vecs_true` both satisfies the 
    $M$-orthonormality condition.
    """
    if m is None:
        w = eig_vecs_true
    else:
        w = m @ eig_vecs_true

    match mode:
        case "pairwise":
            dist = 1 - torch.sum(eig_vecs_pred * w, dim=0).pow(2)
        case "subspace":
            k = eig_vecs_true.size(-1)
            dist = k - torch.sum((eig_vecs_pred.T @ w).pow(2))
        case _:
            raise ValueError(f"Unknown mode argument '{mode}'.")

    return dist


def matrix_inf_norm(sdt: Float[SparseDecoupledTensor, "m m"] | None) -> float:
    """Compute the matrix infinity norm."""
    if sdt is None:
        return 1.0
    else:
        ones = torch.ones(sdt.size(0), dtype=sdt.dtype, device=sdt.device)
        row_sum = sdt.abs() @ ones
        return row_sum.max().item()


def compute_lorentzian_eps_via_norm(
    a: Float[SparseDecoupledTensor, "m m"],
    m: Float[SparseDecoupledTensor, "m m"] | None,
) -> float:
    """
    Select the strength of Lorentzian broadening from matrix norms.

    The parameter `eps` should be small enough to allow accurate gradients through
    the eigenvectors of near-degenerate eigenvalues, but large enough to stabilize
    the backward pass for true degeneracies. In addition, `eps` should scale with
    the square of the spectral radius so that the regularization floor adapts to
    the physical scale of the operators.

    To avoid computing the exact spectral radius ahead of time, we scale `eps`
    using the matrix infinity norm (recall that |λ_max| <= ||A||_∞). For GEP, the
    spectral radius roughly scales with the ratio ||A||_∞ / ||M||_∞.

    For shift-invert mode, the backward pass still differentiates through the
    original operators using the original unshifted eigenvalues, so this scaling
    heuristic remains mathematically consistent without further adjustment.
    """
    machine_eps = torch.finfo(a.dtype).eps

    a_norm = matrix_inf_norm(a)
    m_norm = matrix_inf_norm(m)

    # Prevent division by zero if M is severely ill-scaled.
    safe_m_norm = max(m_norm, machine_eps)

    lorentz_eps = 10.0 * machine_eps * max(1.0, (a_norm / safe_m_norm) ** 2.0)

    return lorentz_eps


def compute_lorentzian_eps_via_eigs(eig_vals: Float[Tensor, " eig"]) -> float:
    """
    Select the strength of Lorentzian broadening from resolved eigenvalues.

    This function is similar to `compute_lorentzian_eps_via_norm()`, but it
    estimate the spectral scale from the computed eigenvalues directly, which is
    useful for matrix-free linear operators.
    """
    scale = eig_vals.abs().max().item()
    lorentz_eps = 10.0 * torch.finfo(eig_vals.dtype).eps * max(1.0, scale**2)
    return lorentz_eps

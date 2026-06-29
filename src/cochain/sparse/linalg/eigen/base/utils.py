__all__ = ["m_orthonormalize", "canonicalize_eig_vec_signs", "grassmann_proj_dists"]

from typing import Literal

import torch
from jaxtyping import Float
from torch import Tensor

from ....decoupled_tensor import SparseDecoupledTensor


def m_orthonormalize(
    v: Float[Tensor, "m n"],
    m: Float[SparseDecoupledTensor, "m m"],
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
        A relative tolerance threshold for checking linearly dependent vectors.
    n_min
        The minimum number of column vectors to be returned.
    max_iter
        The maximum number of iterations to perform.

    Returns
    -------
    v_ortho : [m, l]
        A dense 2D matrix whose columns form an $M$-orthonormal basis for the column
        space of V. The number of column vectors `l` should be between `n_min` and
        `n`, depending on the number of linearly independent column vectors in $V$.

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
    $U^T M U = I$. Since the column vectors of $V$ and $U$ spans the same space,
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

    which shows that the column vectors of U and $R^\perp$ are $M$-orthogonal, and
    thus the column space of $R^\perp$ is $M$-orthogonal to the column space of $U$.

    **Jacobi preconditioning**
    One potential problem with the canonical orthogonalization method described
    above is that the Gram matrix can have a very large condition number if
    the norms of the column vectors of $V$ span a wide range of magnitude. Therefore,
    it is numerically preferable to perform the orthogonalization after the column
    vectors of $V$ have been normalized.

    Instead of directly modifying $V$, we can achieve the same normalization by
    preconditioning the Gram matrix $G$, which tends to be computationally cheaper.
    To do so, let us define a diagonal matrix $D$ whose diagonal elements
    correspond to the norms of the column vectors of $V$. Since the Gram matrix
    $G$ contains all pairwise inner products of the column vectors of $V$,
    we can define $D$ in terms of $G$ as $D_{ii} = 1/\sqrt{G_{ii}}$.

    With the matrix $D$, we define the normalized matrix $\bar V$ as $\bar V = VD$.
    The Gram matrix $\bar G$ of $\bar V$ is then given by

    $$\bar G = \bar V^T M \bar V = D^T V^T M V D = D^T G D = D G D$$

    Given the eigendecomposition $\bar G = \bar Q \bar\Lambda \bar Q^T$, we define
    the whitening matrix $\bar W = D \bar Q \bar \Lambda^{-1/2}$, such that

    $$
    U^T M U
    = (D \bar Q \bar \Lambda^{-1/2})^T (V^T M V) (D \bar Q \bar \Lambda^{-1/2})
    = (\bar Q \bar \Lambda^{-1/2})^T \bar G (\bar Q \bar \Lambda^{-1/2}) = I
    $$

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
            pad_overlap = v_ortho_double.T @ (m_double @ pad)
            pad_proj = v_ortho_double @ pad_overlap
            pad_perp = pad - pad_proj

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
    m: Float[SparseDecoupledTensor, "m m"],
    rtol: float | None = None,
) -> tuple[Float[Tensor, "m l"], Float[Tensor, ""]]:
    """Perform one iteration of M-orthonormalization."""
    eps = torch.finfo(v.dtype).eps

    if rtol is None:
        rtol = v.size(0) * eps

    # Compute the M-orthogonal gram matrix.
    gram: Float[Tensor, "n n"] = v.T @ (m @ v)

    # Implicit normalization of G. Let D be a diagonal matrix whose diagonal
    # elements are the inverse square roots of the diagonal elements of G; then
    # the normalized G is G' = D@G@D. This is equivalent to normalizing the columns
    # of V, but computationally cheaper. This scaling improves the condition number
    # of G for eigh(). Note that columns of V that are zero or very close to
    # zero are not length-normalized.
    v_col_norm2 = torch.diag(gram).clamp(min=0.0)
    diag = 1.0 / torch.sqrt(v_col_norm2)

    zero_col_mask = (~torch.isfinite(diag)) | (v_col_norm2 < eps * 10)
    diag[zero_col_mask] = 1.0

    g_scaled = torch.einsum("i,ij,j->ij", diag, gram, diag)

    # Perform an eigendecomposition of G = Q @ Λ @ Q.T.
    eig_vals, eig_vecs = torch.linalg.eigh(g_scaled)

    # Drop very small eigenvalues corresponding to linearly dependent columns.
    eps = rtol * eig_vals.max()
    mask = eig_vals > eps

    # If V is basically zero, return a single zero vector and a condition number of 0.
    if not mask.any():
        return torch.zeros_like(v[:, :1]), torch.tensor(
            0.0, dtype=v.dtype, device=v.device
        )

    eig_vals_masked = eig_vals[mask]
    inv_eig_vals_masked = 1.0 / torch.sqrt(eig_vals_masked)
    eig_vecs_masked = eig_vecs[:, mask]

    # Check the condition number using the masked eigenvalues, for assessing
    # progress of iterative refinement.
    cond = torch.sqrt(eig_vals_masked.max() / eig_vals_masked.min())

    # Compute the whitening matrix W = D @ Q @ Λ^(-1/2) as the inverse square root
    # of G. Need to apply the D vector here to undo the implicit normalization.
    whiten = torch.einsum("i,ij,j->ij", diag, eig_vecs_masked, inv_eig_vals_masked)

    # Find V_ortho = V @ W, the M-orthonormal version of V. With some algebra,
    # one can check that V_ortho.T @ M @ V_ortho = I.
    v_ortho = v @ whiten

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


def compute_lorentzian_eps(
    a: Float[SparseDecoupledTensor, "m m"],
    m: Float[SparseDecoupledTensor, "m m"] | None,
) -> float:
    """
    Automatically select the strength of Lorentzian broadening/regularization.

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

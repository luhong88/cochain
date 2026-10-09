import warnings
from typing import Literal

try:
    from typing import TypeAlias
except ImportError:
    from typing_extensions import TypeAlias


import torch
from jaxtyping import Float
from torch import Tensor

from ....decoupled_tensor import SparseDecoupledTensor
from ...solvers import DirectSolverConfig
from ..base.utils import (
    _m_normalize,
    _m_orthonormalize_one_iter,
    m_orthonormalize,
)
from ._lobpcg_operators import (
    IdOp,
    LinearOp,
    ShiftInvSymGEPSpOp,
    ShiftInvSymSpOp,
)
from ._lobpcg_preconditioners import LOBPCGPreconditioner

SparseDecoupledTensorLike: TypeAlias = (
    IdOp
    | Float[LinearOp, "m m"]
    | Float[SparseDecoupledTensor, "m m"]
    | Float[ShiftInvSymSpOp, "m m"]
    | Float[ShiftInvSymGEPSpOp, "m m"]
)


def _m_orthogonalize_safe(
    v: Float[Tensor, "m n"],
    u: Float[Tensor, "m l"],
    mv: Float[Tensor, "m n"],
    l: Float[Tensor, "n n"],
) -> Float[Tensor, "m l"]:
    """
    Find the vector components perpendicular from a linearly independent vector set.

    Assuming that V contains linearly independent column vectors. Let G = V^T@M@V
    be a Gram matrix. This function computes the perpendicular projection
    U_perp = U - V@inv(G)@(V.T@M@U).
    """
    u_perp = u - v @ torch.cholesky_solve(mv.T @ u, l)
    u_perp_again = u_perp - v @ torch.cholesky_solve(mv.T @ u_perp, l)

    return u_perp_again


def _orthonormalize_search_directions(
    new_dir: Float[Tensor, "m k"],
    x_current: Float[Tensor, "m n"],
    m_op: Float[SparseDecoupledTensor, "m m"] | IdOp,
    rtol: float | None = None,
) -> Float[Tensor, "m l"]:
    """M-orthonormalize new search directions against a fixed Ritz block X."""
    x_dtype = x_current.dtype

    x_double = x_current.to(torch.float64)
    new_dir_double = new_dir.to(torch.float64)

    if getattr(m_op, "dtype", torch.float64) == torch.float64:
        m_double = m_op
    else:
        m_double = m_op.to(torch.float64)

    empty = torch.empty(
        (x_current.size(0), 0),
        dtype=x_current.dtype,
        device=x_current.device,
    )

    # Compute the rank of the complement of span(X). We can assume that the column
    # vectors of X are linearly independent, and the number of column vectors/trial
    # eigenvectors (i.e., n) is smaller than the dimension of the ambient space
    # for the eigenvectors (i.e., m), so rank(X) = n and the complement of span(X)
    # has dimension m - n.
    complement_rank = x_double.size(0) - x_double.size(1)

    # If m = n already, then there are no possible new search directions.
    if complement_rank == 0:
        return empty

    # Set a heuristic for dropping small columns, which is the machine eps scaled
    # by the vector length.
    if rtol is None:
        rtol = x_double.size(0) * torch.finfo(torch.float64).eps

    # M-normalize the search direction column vectors to produce the matrix D;
    # exactly-zero (soft-locked) columns are removed.
    d, _, _, _ = _m_normalize(new_dir_double, m_double)
    if d.size(1) == 0:
        return empty

    # Find the Cholesky decomposition of G = X.T@M@X for use with the
    # _m_orthogonalize_safe() function.
    mx = m_double @ x_double
    g = x_double.T @ mx
    g = (g + g.T) / 2.0
    l = torch.linalg.cholesky(g)

    # Find the components of D that are perpendicular to span(X).
    d = _m_orthogonalize_safe(x_double, d, mx, l)

    # Drop columns of D whose M-norm is < rtol, and M-normalize the rest of
    # the column vectors. Note that, since the columns of D were first normalized
    # before the perpendicular projection, dropping columns with norm < rtol effectively
    # drops columns that are substantially contained in span(X).
    d, _, col_norms, _ = _m_normalize(d, m_double)
    d = d[:, col_norms > rtol]

    # Ensure that the columns of D form a set of M-orthonormal vectors. This
    # process could amplify any residual component of D still in span(X); so
    # we insert a perpendicular projection step in between two SVQB steps.
    # Note that, in _lobpcg_one_iter(), B' = V.T@B@V is not explicitly assumed to
    # be equal to I; therefore, it is more important for this function to ensure
    # the M-orthogonality between span(D) and span(X) (i.e., linear independent
    # of the search directions) than the M-orthonormality of the columns of D itself.
    d, _ = _m_orthonormalize_one_iter(d, m_double)
    # If D has more columns than needed (unlikely, since m >> n typically), then
    # only pick enough columns to fill the complement_rank. We would prefer to
    # pick orthonormalized column vectors corresponding to the largest eigenvalues
    # of the gram matrix inside _m_orthonormalize_one_iter() for numerical stability
    # reason, and, conveniently, the column vectors outputted by this function is
    # ordered by eigenvalues in ascending order.
    if d.size(1) > complement_rank:
        d = d[:, -complement_rank:]
    d = _m_orthogonalize_safe(x_double, d, mx, l)
    d, _ = _m_orthonormalize_one_iter(d, m_double)

    return d.to(x_dtype)


def _lobpcg_one_iter(
    t_op: SparseDecoupledTensorLike,
    m_op: Float[SparseDecoupledTensor, "m m"] | IdOp,
    m_double: Float[SparseDecoupledTensor, "m m"] | IdOp,
    s_op: Float[SparseDecoupledTensor, "m m"] | IdOp,
    res: Float[Tensor, "m n"],
    x_current: Float[Tensor, "m n"],
    tx_current: Float[Tensor, "m n"],
    p_current: Float[Tensor, "m n"],
    precond: LOBPCGPreconditioner,
    largest: bool,
    tol_current: Float[Tensor, " n"],
) -> tuple[
    Float[Tensor, " n"],
    Float[Tensor, "m n"],
    Float[Tensor, "m n"],
    Float[Tensor, "m n"],
]:
    n = x_current.size(-1)

    # Perform soft locking/deflation to lock in converged eigenvectors by zeroing
    # out the corresponding residual vectors.
    res_norm = torch.linalg.norm(res, dim=0, keepdim=True)
    mask = (res_norm > tol_current).to(res_norm.dtype)
    res_masked = res * mask

    # For a given preconditioner Pr, the new search directions W is given by
    # W = Pr@R for the residual vectors R.
    search_dir = precond @ res_masked

    # Perform the same soft locking on the momentum/conjugate directions.
    conj_dir = p_current * mask

    # Ensure that the new search directions consist of (approximately) M-orthonormal
    # vectors that are outside of span(X).
    new_dir = torch.hstack((search_dir, conj_dir))
    new_dir_ortho = _orthonormalize_search_directions(new_dir, x_current, m_double)

    # The new Ritz block/trial eigenvectors consist of the existing eigenvectors
    # plus the processed new search directions. The union of these two sets should
    # still form an M-orthonormal set of vectors.
    v_ortho = torch.hstack((x_current, new_dir_ortho))

    # Apply T to the new set of trial eigenvectors in V_ortho. Since X_current
    # has not been changed, the existing T@X_current can be reused instead of
    # applying T to the entire V_ortho.
    tc_ortho = t_op @ new_dir_ortho
    tv_ortho = torch.hstack((tx_current, tc_ortho))

    # Perform the Rayleigh-Ritz projection.
    #
    # Let us approximate the eigenvectors using the trial subspace basis vectors;
    # i.e., X_next = V@C, where C is a coefficient matrix. Then, the eigenvalue
    # equation T@X = B@X@Λ can be approximated as T@V@C = B@V@C@Λ. The best
    # approximation ensures that the error is orthogonal to the trial subspace
    # with respect to the inner product induced by S, i.e., <V, T@V@C - B@V@C@Λ>_S = 0.
    # In most cases, S = I and B = M; therefore, this is equivalent to solving a
    # "reduced" generalized eigenvalue problem T'@C = B'@C@Λ, where T' = V.T@T@V
    # and B' = V.T@B@V.
    #
    # Note that, for a GEP in the shift-invert mode where B = I, the definition
    # B' = V.T@B@V = V.T@V does not actually reduce to I, since V is M-orthonormal.
    # In this case, the projection needs to use the inner product induced by S = M,
    # i.e.,  <V, T@V@C - B@V@C@Λ>_M = 0. This results in the "reduced" GEP where
    # T' = V.T@M@T@V and B' = V.T@M@B@V = I. Note that, for regular GEP, since
    # B = M, using the inner product induced by M would have resulted in "double
    # counting" of M in B' (which is why S = I above).
    #
    # If V is perfectly M-orthonormal, B' is identical to I and this reduces to
    # a standard eigenvalue problem for T'. This can be achieved (up to the limit
    # of floating point precision) by recycling the m_orthonormalize() function
    # a few times. A complementary approach is to just accept that B' is not
    # an identity matrix and solve the reduced GEP. Since eigh() does not support
    # GEP, we achieve the same thing by "whitening" the GEP. Let B' = L@L.T be the
    # Cholesky decomposition of B', and write the reduced GEP as
    #
    # inv(L)@T'@inv(L).T@(L.T@C) = (L.T@C)@Λ.
    #
    # Then, the operator T'' = inv(L)@T'@inv(L).T satisfies a standard eigenvalue
    # problem T''@Y = Y@Λ and Y = L.T@C.
    b_reduced = v_ortho.T @ (m_op @ v_ortho)
    b_lower = torch.linalg.cholesky(b_reduced, upper=False)
    # Applying solve_triangular() to L with I as the RHS is equivalent to finding
    # the inverse of L.
    b_lower_inv = torch.linalg.solve_triangular(
        b_lower,
        torch.eye(v_ortho.size(-1), dtype=b_lower.dtype, device=b_lower.device),
        upper=False,
    )

    t_reduced = v_ortho.T @ (s_op @ tv_ortho)
    t_sym = b_lower_inv @ t_reduced @ b_lower_inv.T

    lambda_next_all, y_next_reduced_all = torch.linalg.eigh(t_sym)
    x_next_reduced_all = b_lower_inv.T @ y_next_reduced_all

    # Extract the n largest (or smallest) eigenvalue-eigenvector pairs.
    # Note that torch.linalg.eigh() returns eigenvalues in ascending order.
    if largest:
        # if largest=True, sort eigenvalues in descending order.
        x_next_reduced = torch.flip(x_next_reduced_all[:, -n:], dims=(-1,))
        lambda_next = torch.flip(lambda_next_all[-n:], dims=(0,))
    else:
        # if largest=False, keep eigenvalues in ascending order.
        x_next_reduced = x_next_reduced_all[:, :n]
        lambda_next = lambda_next_all[:n]

    # Lift the reduced eigenvectors back to the full space.
    x_next = v_ortho @ x_next_reduced
    tx_next = tv_ortho @ x_next_reduced

    # Determine the next momentum/conjugate direction block P.
    #
    # Since X_next = V_ortho @ X_next_reduced and V_ortho = [X_current, D_ortho],
    # We can similarly split X_next_reduced into two blocks of the appropriate
    # shapes (stacked vertically) X_next_reduced= [C_current C_correct].T, such
    # that X_next can be decomposed into X_next = X_current@C_current + D_ortho@C_correct,
    # where the first matmul describes linear combinations within span(X_current),
    # while the second matmul describes orthogonal corrections coming from the
    # new search directions. Therefore, instead of computing P_next = X_next - X_current,
    # which can suffer from catastrophic cancellations, we directly set
    # P_next = D_ortho@C_correct. This is similar to just taking the step
    # X_next - X_current, but we ignore contributions to the step coming from
    # linear recombination within span(X_current) itself, which does not provide
    # new search directions anyways.
    #
    # Tensor shapes:
    # X_current: [m, n], D_ortho: [m, l] (l in [0, 2*n]) => V_ortho: [m, n+l]
    # X_next_reduced: [n+l, n] => C_current: [n, n], C_correct: [l, n]
    p_next = v_ortho[:, n:] @ x_next_reduced[n:, :]

    return lambda_next, x_next, tx_next, p_next


def _lobpcg_loop(
    t_op: SparseDecoupledTensorLike,
    b_op: Float[SparseDecoupledTensor, "m m"] | IdOp,
    m_op: Float[SparseDecoupledTensor, "m m"] | IdOp,
    s_op: Float[SparseDecoupledTensor, "m m"] | IdOp,
    x_0: Float[Tensor, "m n"],
    precond: LOBPCGPreconditioner,
    largest: bool,
    tol: float,
    op_scale: float | Literal["auto"],
    a_norm: float | None,
    m_norm: float | None,
    sigma: float | int | None,
    niter: int,
    generator: torch.Generator | None,
) -> tuple[Float[Tensor, " n"], Float[Tensor, "m n"]]:
    # If m_op is not in double precision, create a double copy for numerical
    # routines that demand double precision to avoid repeated dtype casting.
    if getattr(m_op, "dtype", torch.float64) == torch.float64:
        m_double = m_op
    else:
        m_double = m_op.to(torch.float64)

    x_current = m_orthonormalize(
        x_0, m_double, n_min=x_0.size(-1), generator=generator, max_iter=3
    )

    tx_current = t_op @ x_current

    # Initialize the momentum/conjugate search direction P as zero.
    p_current = torch.zeros_like(x_current)

    if a_norm is None:
        if op_scale == "auto":
            # Compute a lower bound estimate of the matrix norm ||A||_2, used
            # later as part of the matrix-free convergence criteria.
            a_norm_lower_bound = (
                torch.linalg.norm(tx_current, dim=0)
                / torch.linalg.norm(x_current, dim=0)
            ).max()
        else:
            a_norm_lower_bound = op_scale

    # Compute the eigenvalues using the Rayleigh quotient X.T@S@T@X/X.T@M@X.
    # Since X is M-orthonormal, X.T@M@X = 1. In most cases, S = I and B = M so
    # the quotient further reduces to X.T@T@X = X.T@M@X@Λ = Λ. For the shift-invert
    # GEP mode, T@X = X@Λ' and S = M so that X.T@M@T@X correctly reduces to the
    # shift-inverted eigenvalues Λ'.
    lambda_current = torch.diag(x_current.T @ (s_op @ tx_current))

    converged = False
    for i in range(niter + 1):
        # Compute the residual vectors R = T@X - B@X@Λ.
        bx_current = b_op @ x_current
        res = tx_current - bx_current * lambda_current.view(1, -1)
        res_norm = torch.linalg.norm(res, dim=0)
        x_norm = torch.linalg.norm(x_current, dim=0)

        # The PyTorch implementation of LOBPCG uses the tolerance threshold
        #
        # ||R_i||_2 < tol*(||X_i||_2||A||_2 + ||X_i||_2||M||_2|λ_i|)
        #
        # where tol is the square root of the machine epsilon. Here, we replace
        # the matrix 2-norms with the inf-norm to reduce the computational cost,
        #
        # ||R_i||_2 < tol*(||A||_∞ + ||M||_∞|λ_i|)*||X_i||_2
        #
        # For the shift-invert mode, we need to use a different tolerance threshold.
        # Let μ_i = 1/(λ_i - σ) be the shift-inverted eigenvalue. Note that, in the
        # SI mode,
        #
        # R_i = inv(A - σM)@M@x_i - μ_i*x_i
        #
        # Multiplying both sides on the left with -(A - σM)/μ_i gives
        #
        # -(A - σM)@R_i/μ_i = A@x_i - λM@x_i
        #
        # Note that the RHS is the true residue vector in the absence of SI. That
        # is
        #
        # ||R_true_i||_2 <= ||A - σM||_2||R_i||_2/|μ_i|
        #
        # By the triangle inequality, ||A - σM||_2 <= ||A||_2 + |σ|*||M||_2. If
        # we require that ||R_i||_2 < tol*|μ_i|*||X_i||_2 and assume that σ is
        # approximately λ_i, then
        #
        # ||R_true_i||_2 < tol*(||A||_2 + ||M||_2*|λ_i|)*||X_i||_2
        #
        # which is effectively the same error bound as before.
        #
        # When A is provided as a matrix-free linear operator, it may not be possible
        # to explicitly compute the inf-norm ||A||_∞. Instead, we simply ask whether
        #
        # ||R_i||_2 < tol*(||A@X_i||_2 + |λ_i|*||M@X_i||_2)
        #
        # The ratio ||R_i||_2/(||A@X_i||_2 + |λ_i|*||M@X_i||_2) provides a
        # scale-invariant test of the error A@X_i - λ_iM@X_i. To see why, let
        # u = A@X_i and v = λ_iM@X_i. Then, by the law of cosines,
        #
        # ||u - v||^2 = ||u||^2 + ||v||^2 - 2*||u||*||v||*cos(θ)
        #
        # As the algorithm converges, u and v have roughly the same length L,
        #
        # ||u - v||^2 = 4*L^2*sin(θ/2)^2
        #
        # and thus
        #
        # ||u - v||/(||u|| + ||v||) = sin(θ/2)
        #
        # That is, the ratio measures the difference in direction that is invariant
        # to L.
        #
        # A problem with this approach is that, for λ_i equal to or close to zero,
        # both u and v should be close to zero as well, which can result in
        # artificially stringent convergence criteria. Therefore, we add in an
        # absolute floor to the tolerance that is scaled by the matrix norm of
        # A and the length of X_i; we approximate ||A||_2 as s = max(||A@X_0||/||X_0||)
        # over the initial trial eigenvectors. Taken together,
        #
        # ||R_i||_2 < tol*(s*||X_i||_2 + ||A@X_i||_2 + |λ_i|*||M@X_i||_2)
        if a_norm is None:
            # The absolute floor makes convergence attainable for harmonic modes,
            # whose individual ||Ax|| and |lambda| ||Mx|| both approach zero.
            abs_floor = a_norm_lower_bound * x_norm
            rel_criterion = torch.linalg.norm(
                tx_current, dim=0
            ) + lambda_current.abs() * torch.linalg.norm(bx_current, dim=0)
            tol_current = tol * (abs_floor + rel_criterion)

        elif sigma is None:
            tol_current = tol * (a_norm + m_norm * lambda_current.abs()) * x_norm

        else:
            tol_current = tol * lambda_current.abs() * x_norm

        if (res_norm <= tol_current).all():
            converged = True
            break

        if i == niter:
            break

        lambda_next, x_next, tx_next, p_next = _lobpcg_one_iter(
            t_op,
            m_op,
            m_double,
            s_op,
            res,
            x_current,
            tx_current,
            p_current,
            precond,
            largest,
            tol_current,
        )

        x_current = x_next
        tx_current = tx_next
        p_current = p_next
        lambda_current = lambda_next

    if not converged:
        max_res_norm_idx = res_norm.argmax()
        warnings.warn(
            f"LOBPCG did not converge after {niter} iterations. "
            f"Max residual norm: {res_norm[max_res_norm_idx].item():.2e} "
            f"(tol: {tol_current[max_res_norm_idx].item():.2e}).",
            UserWarning,
        )

    return lambda_current, x_current


def _dispatch_ops(
    n: int,
    a_op: SparseDecoupledTensorLike,
    m_op: Float[SparseDecoupledTensor, "m m"] | None,
    sigma: float | int | None,
    nvmath_config: DirectSolverConfig,
) -> tuple[
    SparseDecoupledTensorLike,
    SparseDecoupledTensorLike,
    SparseDecoupledTensorLike,
    SparseDecoupledTensorLike,
]:
    match (m_op, sigma):
        case (None, None):
            t_op = a_op
            b_op = IdOp()
            m_op = IdOp()
            s_op = IdOp()

        case (m_op, None):
            t_op = a_op
            b_op = m_op
            m_op = m_op
            s_op = IdOp()

        case (None, sigma):
            # a_op needs to be int32-safe.
            t_op = ShiftInvSymSpOp(a_sdt=a_op, sigma=sigma, n=n, config=nvmath_config)
            b_op = IdOp()
            m_op = IdOp()
            s_op = IdOp()

        case (m_op, sigma):
            # a_op and m_op need to be int32-safe.
            t_op = ShiftInvSymGEPSpOp(
                a_sdt=a_op, m_sdt=m_op, sigma=sigma, n=n, config=nvmath_config
            )
            b_op = IdOp()
            m_op = m_op
            s_op = m_op

        case _:
            raise ValueError("Invalid eigenvalue problem definition.")

    return t_op, b_op, m_op, s_op


def lobpcg_forward(
    a_op: SparseDecoupledTensorLike,
    m_op: Float[SparseDecoupledTensor, "m m"] | None,
    a_norm: float | None,
    m_norm: float | None,
    sigma: float | int | None,
    v0: Float[Tensor, "m n"],
    largest: bool,
    tol: float,
    op_scale: float | Literal["auto"],
    maxiter: int,
    precond: LOBPCGPreconditioner,
    nvmath_config: DirectSolverConfig,
    generator: torch.Generator | None,
) -> tuple[Float[Tensor, " n"], Float[Tensor, "m n"]]:
    """
    Solve a (generalized) eigenvalue problem of the form A@x = λ*M@x.

    In order to account for generalized eigenvalue problems (GEP) and shift-invert
    mode (SI), we reformulate the eigenvalue problem in terms of four operators
    T, B, M, and S. With these operators, we rewrite the problem as T@x = λ*B@x,
    subject to the orthonormality condition x.T@M@x = I with the metric M. The S
    matrix acts as a symmetrizer for computing Rayleigh quotients and the
    Rayleigh-Ritz projection.

    | Setup    | Equation                         | T               | B | M | S |
    |----------|----------------------------------|-----------------|---|---|---|
    | Standard | A@x = λx                         | A               | I | I | I |
    | GEP      | A@x = λM@x                       | A               | M | M | I |
    | SI       | inv(A - σI)@x = (λ - σ)^-1 * x   | inv(A - σI)     | I | I | I |
    | GEP + SI | inv(A - σM)@M@x = (λ - σ)^-1 * x | inv(A - σM) @ M | I | M | M |

    This function can accept matrix-free `a_op` as a `LinearOp` object, although
    this is currently not compatible with the shift-invert mode.
    """
    if (a_norm is None) != (m_norm is None):
        raise ValueError(
            "'a_norm' and 'm_norm' must either both be None or both be provided."
        )
    if a_norm is None and sigma is not None:
        raise NotImplementedError(
            "The matrix-free residual stopping criterion via 'op_scale' does not "
            "support the shift-invert mode."
        )

    if isinstance(a_op, LinearOp):
        if sigma is not None:
            raise NotImplementedError(
                "The shift-invert mode is not implemented when 'a_op' is "
                "represented as a matrix-free linear operator."
            )

    n = v0.size(-1)

    t_op, b_op, m_op, s_op = _dispatch_ops(n, a_op, m_op, sigma, nvmath_config)

    return _lobpcg_loop(
        t_op=t_op,
        b_op=b_op,
        m_op=m_op,
        s_op=s_op,
        x_0=v0,
        largest=largest,
        tol=tol,
        op_scale=op_scale,
        a_norm=a_norm,
        m_norm=m_norm,
        sigma=sigma,
        niter=maxiter,
        precond=precond,
        generator=generator,
    )

import functools

import torch as t
from jaxtyping import Float

from ...operators import SparseOperator
from ..solvers.nvmath_wrapper import DirectSolverConfig
from ._lobpcg_operators import (
    IdentityPrecond,
    ShiftInvSymGEPSpOp,
    ShiftInvSymSpOp,
    SparseOperatorLike,
    SpPrecond,
)


def _enforce_M_orthonormality(
    V: Float[t.Tensor, "m 3*n"],
    M_op: Float[SparseOperator, "m m"] | None,
    rtol: float | None,
) -> Float[t.Tensor, "m 3*n"]:
    """
    Convert the column vectors of V into M-orthonormal vectors using symmetric
    orthogonalization.

    Currently batched sparse-dense matrix operations are not well supported in
    torch; therefore, this function cannot support batch dimensions.
    """
    if M_op is None:
        return V

    if rtol is None:
        rtol = M_op.size(-1) * t.finfo(M_op.dtype).eps

    # Compute the M-orthogonal gram matrix.
    G: Float[t.Tensor, "3*n 3*n"] = V.T @ (M_op @ V)

    # Perform an eigendecomposition of G = Q@Λ@Q.T.
    eig_vals, eig_vecs = t.linalg.eigh(G)

    # Clamp very small eigenvalues.
    eps = rtol * eig_vals.max()
    eig_vals_clamped = t.clip(eig_vals, min=eps)
    inv_eig_vals = 1.0 / t.sqrt(eig_vals_clamped)

    # Compute the whitening matrix W = Q@Λ^(-1/2)@Q.T as the inverse square root
    # of G.
    W = (eig_vecs * inv_eig_vals.view(1, -1)) @ eig_vecs.T

    # Find V_ortho = V@W, the M-orthonormal version of V. With some algebra,
    # one can check that V_ortho@M@V_ortho = I.
    V_ortho = eig_vecs @ W

    return V_ortho


def _lobpcg_one_iter(
    iter_: int,
    A_op: Float[SparseOperatorLike, "m m"],
    M_op: Float[SparseOperator, "m m"] | None,
    R: Float[t.Tensor, "m n"],
    X_current: Float[t.Tensor, "m n"],
    X_prev: Float[t.Tensor, "m n"],
    precond: SpPrecond | IdentityPrecond,
    largest: bool,
    tol: float,
) -> tuple[Float[t.Tensor, " n"], Float[t.Tensor, "m n"]]:
    """
    Perform one iteration of LOBPCG.
    """
    # Perform soft locking/deflation to lock in converged eigenvectors by zeroing
    # out the corresponding residual vectors.
    R_norm = t.linalg.norm(R, dim=0, keepdim=True)
    mask = (R_norm > tol).to(R_norm.dtype)
    R_masked = R * mask

    # Use the residual vectors R to compute the precondition directions W = inv(M)@R.
    W = precond @ R_masked

    # Compute the momentum/conjugate directions P. During the first iteration,
    # X_current = X_prev so P = 0. Perform the same soft locking on the momentum.
    P = (X_current - X_prev) * mask

    # Assemble the new trial subspace and enforce M-orthonormality condition on
    # the subspace basis vectors. We perform the orthonomalization twice to
    # minimize numerical error. Since P = 0 during the first iteration, skip it.
    if iter_ == 0:
        V = t.hstack((X_current, W))
    else:
        V = t.hstack((X_current, W, P))

    V_ortho = _enforce_M_orthonormality(_enforce_M_orthonormality(V, M_op), M_op)

    # Rayleigh-Ritz projection
    # Let us approximate the eigenvectors using the subspace basis vectors; i.e.,
    # X_next = V@C, where C is a coefficient matrix. Then, the eigenvalue
    # equation A@X = M@X@Λ can be approximated as A@V@C = M@V@C@Λ. The best
    # approximation ensures that the error is orthogonal to the trial subspace,
    # i.e., V.T@(A@V@C - M@V@C@Λ) = 0. This is equivalent to solving a "reduced"
    # generalized eigenvalue problem A'@C = M'@C@Λ, where A' = V.T@A@V and
    # M' = V.T@M@V. Since V is M-orthonormal, M' should be close to the identity
    # matrix.
    A_reduced = V_ortho.T @ (A_op @ V_ortho)
    Lambda_next_all, X_next_reduced_all = t.linalg.eigh(A_reduced)

    # Lift the reduced eigenvectors back to the full space
    X_next_all = V_ortho @ X_next_reduced_all

    # Extract the n largest (or smallest) eigenvalue-eigenvector pairs.
    # Note that t.linalg.eigh() returns eigenvalues in ascending order
    n = X_current.size(-1)
    if largest:
        # if largest=True, sort eigenvalues in descending order
        X_next = t.flip(X_next_all[-n:], dim=-1)
        Lambda_next = t.flip(Lambda_next_all[-n:])
    else:
        # if largest=False, keep eigenvalues in ascending order
        X_next = X_next_all[:n]
        Lambda_next = Lambda_next[:n]

    return Lambda_next, X_next


def _lobpcg_loop(
    A_op: Float[SparseOperatorLike, "m m"],
    M_op: Float[SparseOperator, "m m"] | None,
    X_0: Float[t.Tensor, "m n"],
    precond: SpPrecond | IdentityPrecond,
    largest: bool,
    tol: float,
    niter: int,
) -> tuple[Float[t.Tensor, " n"], Float[t.Tensor, "m n"]]:
    X_current = _enforce_M_orthonormality(X_0, M_op)
    X_prev = X_current

    # Compute the eigenvalues using the Rayleigh quotient
    # X.T@A@X/X.T@M@X = X.T@A@X = X.T@M@X@Λ = Λ
    Lambda_current = t.diag(X_current.T @ (M_op @ X_current))

    for iter_ in range(niter):
        # Compute the residual vectors R = A@X - M@X@Λ
        R = A_op @ X_current - (M_op @ X_current) * Lambda_current.view(1, -1)
        R_norm = t.linalg.norm(R, dim=0)

        if (R_norm < tol).all():
            break
        else:
            Lambda_next, X_next = _lobpcg_one_iter(
                iter_, A_op, M_op, R, X_current, X_prev, precond, largest, tol
            )

            X_prev = X_current
            X_current = X_next
            Lambda_current = Lambda_next

    return Lambda_current, X_current


def lobpcg_forward(
    A_op: Float[SparseOperatorLike, "m m"],
    M_op: Float[SparseOperator, "m m"] | None,
    sigma: float | int | None,
    v0: Float[t.Tensor, "m n"],
    diag_damp: float | int | None,
    largest: bool,
    tol: float,
    maxiter: int,
    nvmath_config: DirectSolverConfig,
) -> tuple[Float[t.Tensor, " n"], Float[t.Tensor, "m n"]]:
    lobpcg_loop_partial = functools.partial(
        _lobpcg_loop, X_0=v0, largest=largest, tol=tol, niter=maxiter
    )

    n = v0.size(-1)

    match (M_op, sigma):
        case (None, None):
            return lobpcg_loop_partial(
                A_op=A_op,
                M_op=None,
                precond=IdentityPrecond(),
            )

        case (M_op, None):
            return lobpcg_loop_partial(
                A_op=A_op,
                M_op=M_op,
                precond=SpPrecond(
                    A_op=A_op, n=n, diag_damp=diag_damp, convig=nvmath_config
                ),
            )

        case (None, sigma):
            return lobpcg_loop_partial(
                A_op=ShiftInvSymSpOp(A_op=A_op, sigma=sigma, n=n, config=nvmath_config),
                M_op=None,
                precond=IdentityPrecond(),
            )

        case (M_op, sigma):
            return lobpcg_loop_partial(
                A_op=ShiftInvSymGEPSpOp(
                    A_op=A_op, M_op=M_op, sigma=sigma, n=n, config=nvmath_config
                ),
                M_op=M_op,
                precond=IdentityPrecond(),
            )

        case _:
            raise ValueError()

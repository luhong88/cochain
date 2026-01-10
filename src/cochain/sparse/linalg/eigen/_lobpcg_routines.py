import functools

import torch as t
from jaxtyping import Float

from ...operators import SparseOperator
from ..solvers.nvmath_wrapper import DirectSolverConfig
from ._lobpcg_operators import (
    IdentityOperator,
    ShiftInvSymGEPSpOp,
    ShiftInvSymSpOp,
    SpPrecond,
)
from .utils import M_orthonormalize

type SparseOperatorLike = (
    Float[SparseOperator, "m m"]
    | Float[ShiftInvSymSpOp, "m m"]
    | Float[ShiftInvSymGEPSpOp, "m m"]
)


def _lobpcg_one_iter(
    A_op: SparseOperatorLike,
    M_op: Float[SparseOperator, "m m"] | IdentityOperator,
    R: Float[t.Tensor, "m n"],
    X_current: Float[t.Tensor, "m n"],
    X_prev: Float[t.Tensor, "m n"],
    precond: SpPrecond | IdentityOperator,
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
    # minimize numerical error.
    V = t.hstack((X_current, W, P))

    V_ortho = M_orthonormalize(M_orthonormalize(V, M_op), M_op)

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
        X_next = t.flip(X_next_all[:, -n:], dims=(-1,))
        Lambda_next = t.flip(Lambda_next_all[-n:], dims=(0,))
    else:
        # if largest=False, keep eigenvalues in ascending order
        X_next = X_next_all[:, :n]
        Lambda_next = Lambda_next_all[:n]

    return Lambda_next, X_next


def _lobpcg_loop(
    A_op: SparseOperatorLike,
    M_op: Float[SparseOperator, "m m"] | IdentityOperator,
    X_0: Float[t.Tensor, "m n"],
    precond: SpPrecond | IdentityOperator,
    largest: bool,
    tol: float,
    niter: int,
) -> tuple[Float[t.Tensor, " n"], Float[t.Tensor, "m n"]]:
    X_current = M_orthonormalize(X_0, M_op)
    X_prev = X_current

    # Compute the eigenvalues using the Rayleigh quotient
    # X.T@A@X/X.T@M@X = X.T@A@X = X.T@M@X@Λ = Λ
    Lambda_current = t.diag(X_current.T @ (A_op @ X_current))

    for iter_ in range(niter):
        # Compute the residual vectors R = A@X - M@X@Λ
        R = A_op @ X_current - (M_op @ X_current) * Lambda_current.view(1, -1)
        R_norm = t.linalg.norm(R, dim=0)

        if (R_norm < tol).all():
            break
        else:
            Lambda_next, X_next = _lobpcg_one_iter(
                A_op, M_op, R, X_current, X_prev, precond, largest, tol
            )

            X_prev = X_current
            X_current = X_next
            Lambda_current = Lambda_next

    return Lambda_current, X_current


def lobpcg_forward(
    A_op: SparseOperatorLike,
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
                M_op=IdentityOperator(),
                precond=SpPrecond(
                    A_op=A_op, n=n, diag_damp=diag_damp, config=nvmath_config
                ),
            )

        case (M_op, None):
            return lobpcg_loop_partial(
                A_op=A_op,
                M_op=M_op,
                precond=SpPrecond(
                    A_op=A_op, n=n, diag_damp=diag_damp, config=nvmath_config
                ),
            )

        case (None, sigma):
            return lobpcg_loop_partial(
                A_op=ShiftInvSymSpOp(A_op=A_op, sigma=sigma, n=n, config=nvmath_config),
                M_op=IdentityOperator(),
                precond=IdentityOperator(),
            )

        case (M_op, sigma):
            return lobpcg_loop_partial(
                A_op=ShiftInvSymGEPSpOp(
                    A_op=A_op, M_op=M_op, sigma=sigma, n=n, config=nvmath_config
                ),
                M_op=M_op,
                precond=IdentityOperator(),
            )

        case _:
            raise ValueError()

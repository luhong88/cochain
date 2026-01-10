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
    B_op: Float[SparseOperator, "m m"] | IdentityOperator,
    M_op: Float[SparseOperator, "m m"] | IdentityOperator,
    R: Float[t.Tensor, "m n"],
    X_current: Float[t.Tensor, "m n"],
    X_prev: Float[t.Tensor, "m n"],
    precond: SpPrecond | IdentityOperator,
    largest: bool,
    tol: float,
    shift_invert: bool,
) -> tuple[Float[t.Tensor, " n"], Float[t.Tensor, "m n"], Float[t.Tensor, "m n"]]:
    """
    Perform one iteration of LOBPCG.
    """
    # Perform soft locking/deflation to lock in converged eigenvectors by zeroing
    # out the corresponding residual vectors.
    R_norm = t.linalg.norm(R, dim=0, keepdim=True)
    mask = (R_norm > tol).to(R_norm.dtype)
    R_masked = R * mask

    # Use the residual vectors R to compute the precondition directions W.
    W = precond @ R_masked

    # Compute the momentum/conjugate directions P. During the first iteration,
    # X_current = X_prev so P = 0. Perform the same soft locking on the momentum.
    P = (X_current - X_prev) * mask

    # Assemble the new trial subspace and enforce M-orthonormality condition on
    # the subspace basis vectors. We perform the orthonomalization twice to
    # minimize numerical error.
    V = t.hstack((X_current, W, P))

    # TODO: add a check to avoid orthonormalize twice by default
    V_ortho = M_orthonormalize(M_orthonormalize(V, M_op), M_op)
    AV_ortho = A_op @ V_ortho

    # Rayleigh-Ritz projection
    # Let us approximate the eigenvectors using the subspace basis vectors; i.e.,
    # X_next = V@C, where C is a coefficient matrix. Then, the eigenvalue
    # equation A@X = B@X@Λ can be approximated as A@V@C = B@V@C@Λ. The best
    # approximation ensures that the error is orthogonal to the trial subspace
    # with respect to the standard inner product, i.e., <V, A@V@C - B@V@C@Λ> = 0.
    # This is equivalent to solving a "reduced" generalized eigenvalue problem
    # A'@C = B'@C@Λ, where A' = V.T@A@V and B' = V.T@B@V. Since V is M-orthonormal,
    # B' should be close to the identity matrix and this reduces to a standard
    # eigenvalue problem for A'. Note that, in the shift-invert mode, A is not
    # in general symmetric and we need to use the inner product induced by M
    # for the projection, i.e., <V, A@V@C - B@V@C@Λ>_M = 0. This results in a
    # "redued" GEP where A' = V.T@M@A@V and B' = V.T@M@B@V.
    if shift_invert:
        A_reduced = V_ortho.T @ (M_op @ AV_ortho)
    else:
        A_reduced = V_ortho.T @ AV_ortho

    Lambda_next_all, X_next_reduced_all = t.linalg.eigh(A_reduced)

    # Extract the n largest (or smallest) eigenvalue-eigenvector pairs.
    # Note that t.linalg.eigh() returns eigenvalues in ascending order
    n = X_current.size(-1)
    if largest:
        # if largest=True, sort eigenvalues in descending order
        X_next_reduced = t.flip(X_next_reduced_all[:, -n:], dims=(-1,))
        Lambda_next = t.flip(Lambda_next_all[-n:], dims=(0,))
    else:
        # if largest=False, keep eigenvalues in ascending order
        X_next_reduced = X_next_reduced_all[:, :n]
        Lambda_next = Lambda_next_all[:n]

    # Lift the reduced eigenvectors back to the full space
    X_next = V_ortho @ X_next_reduced
    AX_next = AV_ortho @ X_next_reduced

    return Lambda_next, X_next, AX_next


def _lobpcg_loop(
    A_op: SparseOperatorLike,
    B_op: Float[SparseOperator, "m m"] | IdentityOperator,
    M_op: Float[SparseOperator, "m m"] | IdentityOperator,
    X_0: Float[t.Tensor, "m n"],
    precond: SpPrecond | IdentityOperator,
    largest: bool,
    tol: float,
    niter: int,
    shift_invert: bool,
) -> tuple[Float[t.Tensor, " n"], Float[t.Tensor, "m n"]]:
    """
    Solve an eigenvalue problem of the type A@x = λ*B@x, subject to the condition
    x.T@M@x = I. Typically, M = B, but they can differ in shift-invert mode.
    """
    X_current = M_orthonormalize(X_0, M_op)
    X_prev = X_current

    AX_current = A_op @ X_current

    # Compute the eigenvalues using the Rayleigh quotient:
    # X.T@A@X/X.T@M@X = X.T@A@X = X.T@M@X@Λ = Λ
    # For the shift-invert mode, use X.T@M@A@X instead since A is self-adjoint
    # with respect to the inner product induced by M.
    if shift_invert:
        Lambda_current = t.diag(X_current.T @ (M_op @ AX_current))
    else:
        Lambda_current = t.diag(X_current.T @ AX_current)

    for _ in range(niter):
        # Compute the residual vectors R = A@X - B@X@Λ
        R = AX_current - (B_op @ X_current) * Lambda_current.view(1, -1)
        R_norm = t.linalg.norm(R, dim=0)

        if (R_norm < tol).all():
            break
        else:
            Lambda_next, X_next, AX_next = _lobpcg_one_iter(
                A_op,
                B_op,
                M_op,
                R,
                X_current,
                X_prev,
                precond,
                largest,
                tol,
                shift_invert,
            )

            X_prev = X_current
            X_current = X_next
            AX_current = AX_next
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
        _lobpcg_loop,
        X_0=v0,
        largest=largest,
        tol=tol,
        niter=maxiter,
        shift_invert=sigma is not None,
    )

    n = v0.size(-1)

    match (M_op, sigma):
        case (None, None):
            # Equation: A@x = λx
            # A_op: A
            # B_op: I
            # M_op: I
            return lobpcg_loop_partial(
                A_op=A_op,
                B_op=IdentityOperator(),
                M_op=IdentityOperator(),
                precond=SpPrecond(
                    A_op=A_op, n=n, diag_damp=diag_damp, config=nvmath_config
                ),
            )

        case (M_op, None):
            # Equation: A@x = λM@x
            # A_op: A
            # B_op: M
            # M_op: M
            return lobpcg_loop_partial(
                A_op=A_op,
                B_op=M_op,
                M_op=M_op,
                precond=SpPrecond(
                    A_op=A_op, n=n, diag_damp=diag_damp, config=nvmath_config
                ),
            )

        case (None, sigma):
            # Equation: inv(A - σI)@x = (λ - σ)^-1 * x
            # A_op: inv(A - σI)
            # B_op: I
            # M_op: I
            return lobpcg_loop_partial(
                A_op=ShiftInvSymSpOp(A_op=A_op, sigma=sigma, n=n, config=nvmath_config),
                B_op=IdentityOperator(),
                M_op=IdentityOperator(),
                precond=IdentityOperator(),
            )

        case (M_op, sigma):
            # Equation: inv(A - σM)@M@x = (λ - σ)^-1 * x
            # A_op: inv(A - σM)@M
            # B_op: I
            # M_op: M
            return lobpcg_loop_partial(
                A_op=ShiftInvSymGEPSpOp(
                    A_op=A_op, M_op=M_op, sigma=sigma, n=n, config=nvmath_config
                ),
                B_op=IdentityOperator(),
                M_op=M_op,
                precond=IdentityOperator(),
            )

        case _:
            raise ValueError()

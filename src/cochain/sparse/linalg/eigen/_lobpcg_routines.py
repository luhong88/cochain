import functools
import warnings

import torch as t
from jaxtyping import Float

from ...operators import SparseOperator
from ..solvers.nvmath_wrapper import DirectSolverConfig
from ._lobpcg_operators import (
    IdentityOperator,
    ShiftInvSymGEPSpOp,
    ShiftInvSymSpOp,
)
from ._lobpcg_preconditioners import (
    ChoPrecond,
    IdentityPrecond,
    ILUPrecond,
    JacobiPrecond,
    LOBPCGPrecondConfig,
)
from .utils import M_orthonormalize

type SparseOperatorLike = (
    Float[SparseOperator, "m m"]
    | Float[ShiftInvSymSpOp, "m m"]
    | Float[ShiftInvSymGEPSpOp, "m m"]
)

type LOBPCGPreconditioner = IdentityPrecond | JacobiPrecond | ILUPrecond | ChoPrecond


def _lobpcg_one_iter(
    T_op: SparseOperatorLike,
    M_op: Float[SparseOperator, "m m"] | IdentityOperator,
    S_op: Float[SparseOperator, "m m"] | IdentityOperator,
    R: Float[t.Tensor, "m n"],
    X_current: Float[t.Tensor, "m n"],
    X_prev: Float[t.Tensor, "m n"],
    precond: LOBPCGPreconditioner,
    largest: bool,
    tol: float,
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

    # TODO: if V_ortho has fewer than k columns, pad with random vectors to restart.
    V_ortho = M_orthonormalize(V, M_op)
    TV_ortho = T_op @ V_ortho

    # Rayleigh-Ritz projection
    #
    # Let us approximate the eigenvectors using the subspace basis vectors; i.e.,
    # X_next = V@C, where C is a coefficient matrix. Then, the eigenvalue
    # equation T@X = B@X@Λ can be approximated as T@V@C = B@V@C@Λ. The best
    # approximation ensures that the error is orthogonal to the trial subspace
    # with respect to the standard inner product, i.e., <V, T@V@C - B@V@C@Λ> = 0.
    # This is equivalent to solving a "reduced" generalized eigenvalue problem
    # T'@C = B'@C@Λ, where T' = V.T@T@V and B' = V.T@B@V. Since V is M-orthonormal,
    # B' should be close to the identity matrix and this reduces to a standard
    # eigenvalue problem for T'.
    #
    # Note that, for a GEP in the shift-invert mode where B = I, the definition
    # B' = V.T@B@V = V.T@V does not actually reduce to I, since V is M-orthonormal.
    # In this case, the projection needs to use the inner product induced by M,
    # i.e.,  <V, T@V@C - B@V@C@Λ>_M = 0. This results in the "reduced" GEP where
    # T' = V.T@M@T@V and B' = V.T@M@B@V = I. We achieve this by setting S = M in
    # this case. On the other hand, for regular GEP, since B = M, using the inner
    # product induced by M would have resulted in "double counting" of M in B'.
    T_reduced = V_ortho.T @ (S_op @ TV_ortho)

    Lambda_next_all, X_next_reduced_all = t.linalg.eigh(T_reduced)

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
    TX_next = TV_ortho @ X_next_reduced

    return Lambda_next, X_next, TX_next


def _lobpcg_loop(
    T_op: SparseOperatorLike,
    B_op: Float[SparseOperator, "m m"] | IdentityOperator,
    M_op: Float[SparseOperator, "m m"] | IdentityOperator,
    S_op: Float[SparseOperator, "m m"] | IdentityOperator,
    X_0: Float[t.Tensor, "m n"],
    precond: LOBPCGPreconditioner,
    largest: bool,
    tol: float,
    niter: int,
) -> tuple[Float[t.Tensor, " n"], Float[t.Tensor, "m n"]]:
    X_current = M_orthonormalize(X_0, M_op)
    X_prev = X_current

    TX_current = T_op @ X_current

    # Compute the eigenvalues using the Rayleigh quotient X.T@S@T@X/X.T@M@X.
    # In most cases, S = I so the quotient reduces to X.T@T@X = X.T@M@X@Λ = Λ.
    # For GEP in the shift-invert mode, T@X = X@Λ' and S = M so that X.T@M@T@X
    # correctly reduces to the shift-inverted eigenvalues Λ'.
    Lambda_current = t.diag(X_current.T @ (S_op @ TX_current))

    converged = False
    for _ in range(niter):
        # Compute the residual vectors R = T@X - B@X@Λ
        R = TX_current - (B_op @ X_current) * Lambda_current.view(1, -1)
        R_norm = t.linalg.norm(R, dim=0)

        if (R_norm < tol).all():
            converged = True
            break

        else:
            Lambda_next, X_next, TX_next = _lobpcg_one_iter(
                T_op,
                M_op,
                S_op,
                R,
                X_current,
                X_prev,
                precond,
                largest,
                tol,
            )

            X_prev = X_current
            X_current = X_next
            TX_current = TX_next
            Lambda_current = Lambda_next

    if not converged:
        warnings.warn(
            f"LOBPCG did not converge after {niter} iterations. "
            f"Max residual norm: {R_norm.max().item():.2e} (tol: {tol:.2e}).",
            UserWarning,
        )

    return Lambda_current, X_current


def lobpcg_forward(
    A_op: SparseOperatorLike,
    M_op: Float[SparseOperator, "m m"] | None,
    sigma: float | int | None,
    v0: Float[t.Tensor, "m n"],
    largest: bool,
    tol: float,
    maxiter: int,
    nvmath_config: DirectSolverConfig,
    precond_config: LOBPCGPrecondConfig,
) -> tuple[Float[t.Tensor, " n"], Float[t.Tensor, "m n"]]:
    """
    Solve a (generalized) eigenvalue problem of the form A@x = λ*M@x.

    In order to account for generalized eigenvalue problems (GEP) and shift-invert
    mode (SI), we reformulate the eigenvalue problem in terms of four operators
    T, B, M, and S. With these operators, we rewrite the problem as T@x = λ*B@x,
    subject to the orthonormality condition x.T@M@x = I with the metric M. The S
    matrix acts as a symmetrizer for computing Rayleigh quotients and the
    Rayleigh-Ritz projection.

    ------------------------------------------------------------------
    Setup     Equation                          T              B  M  S
    ------------------------------------------------------------------
    Standard  A@x = λx                          A              I  I  I
    GEP       A@x = λM@x                        A              M  M  I
    SI        inv(A - σI)@x = (λ - σ)^-1 * x    inv(A - σI)    I  I  I
    GEP + SI  inv(A - σM)@M@x = (λ - σ)^-1 * x  inv(A - σM)@M  I  M  M
    ------------------------------------------------------------------
    """
    n = v0.size(-1)

    if sigma is not None:
        # If doing shift-invert mode, always use the identity preconditioner and
        # ignore the user inputs.
        precond = IdentityPrecond()
    else:
        match precond_config.method:
            case "identity":
                precond = IdentityPrecond()
            case "jacobi":
                precond = JacobiPrecond(A_op=A_op)
            case "ilu":
                precond = ILUPrecond(
                    A_op=A_op,
                    diag_damp=precond_config.diag_damp,
                    spilu_kwargs=precond_config.spilu_kwargs,
                )
            case "cholesky":
                precond = ChoPrecond(
                    A_op=A_op,
                    n=n,
                    diag_damp=precond_config.diag_damp,
                    nvmath_config=precond_config.nvmath_config,
                )
            case _:
                raise ValueError()

    lobpcg_loop_partial = functools.partial(
        _lobpcg_loop, X_0=v0, largest=largest, tol=tol, niter=maxiter, precond=precond
    )

    match (M_op, sigma):
        case (None, None):
            return lobpcg_loop_partial(
                T_op=A_op,
                B_op=IdentityOperator(),
                M_op=IdentityOperator(),
                S_op=IdentityOperator(),
            )

        case (M_op, None):
            return lobpcg_loop_partial(
                T_op=A_op,
                B_op=M_op,
                M_op=M_op,
                S_op=IdentityOperator(),
            )

        case (None, sigma):
            return lobpcg_loop_partial(
                T_op=ShiftInvSymSpOp(A_op=A_op, sigma=sigma, n=n, config=nvmath_config),
                B_op=IdentityOperator(),
                M_op=IdentityOperator(),
                S_op=IdentityOperator(),
            )

        case (M_op, sigma):
            return lobpcg_loop_partial(
                T_op=ShiftInvSymGEPSpOp(
                    A_op=A_op, M_op=M_op, sigma=sigma, n=n, config=nvmath_config
                ),
                B_op=IdentityOperator(),
                M_op=M_op,
                S_op=M_op,
            )

        case _:
            raise ValueError()

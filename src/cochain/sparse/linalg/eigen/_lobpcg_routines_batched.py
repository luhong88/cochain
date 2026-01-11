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
from ._lobpcg_routines import _dispatch_operators
from .utils import M_orthonormalize_batched

type SparseOperatorLikeBatched = (
    IdentityOperator
    | Float[SparseOperator, "b*m b*m"]
    | Float[ShiftInvSymSpOp, "b*m b*m"]
    | Float[ShiftInvSymGEPSpOp, "b*m b*m"]
)

type LOBPCGPreconditioner = IdentityPrecond | JacobiPrecond | ILUPrecond | ChoPrecond


def _lobpcg_one_iter_batched(
    b: int,
    T_op: SparseOperatorLikeBatched,
    M_op: Float[SparseOperator, "b*m b*m"] | IdentityOperator,
    S_op: Float[SparseOperator, "b*m b*m"] | IdentityOperator,
    R: Float[t.Tensor, "b*m n"],
    X_current: Float[t.Tensor, "b*m n"],
    X_prev: Float[t.Tensor, "b*m n"],
    precond: LOBPCGPreconditioner,
    largest: bool,
    tol: float,
    generator: t.Generator | None,
) -> tuple[Float[t.Tensor, "b n"], Float[t.Tensor, "b*m n"], Float[t.Tensor, "b*m n"]]:
    """
    Perform one iteration of LOBPCG.

    Variables with an explicit batch dimension are named with the _batched suffix.
    """
    bm, n = X_current.shape
    m = bm // b

    R_batched = R.view(b, m, n)
    R_norm_batched = t.linalg.norm(R_batched, dim=1, keepdim=True)
    mask = (
        (R_norm_batched > tol).to(R_norm_batched.dtype).expand_as(R_batched).view(bm, n)
    )
    R_masked = R * mask

    W = precond @ R_masked
    P = (X_current - X_prev) * mask
    V = t.hstack((X_current, W, P))

    V_ortho = M_orthonormalize_batched(
        V, M_op, n_min=n, generator=generator, max_iter=3
    )
    TV_ortho = T_op @ V_ortho

    # Due to rank-adaptive M-orthonormalization, the shape of V_ortho is (b*m, l),
    # where l ∈ [n, 3n]; therefore, need to view it as (b, m, -1) to keep l variable.
    T_reduced_batched = t.bmm(
        V_ortho.view(b, m, -1).transpose(-1, -2), (S_op @ TV_ortho).view(b, m, -1)
    )

    # Note that the eigenvalues retain the explicit batch dimension.
    Lambda_next_all, X_next_reduced_all_batched = t.linalg.eigh(T_reduced_batched)
    X_next_reduced_all = X_next_reduced_all_batched.reshape(bm, -1)

    if largest:
        X_next_reduced = t.flip(X_next_reduced_all[:, -n:], dims=(-1,))
        Lambda_next = t.flip(Lambda_next_all[:, -n:], dims=(-1,))
    else:
        X_next_reduced = X_next_reduced_all[:, :n]
        Lambda_next = Lambda_next_all[:, :n]

    # Need to lift the reduced eigenvectors separately over the batch items.
    X_next = t.bmm(V_ortho.view(b, m, n), X_next_reduced.view(b, m, n)).view(bm, n)
    TX_next = t.bmm(TV_ortho.view(b, m, n), X_next_reduced.view(b, m, n)).view(bm, n)

    return Lambda_next, X_next, TX_next


def _lobpcg_loop_batched(
    b: int,
    T_op: SparseOperatorLikeBatched,
    B_op: Float[SparseOperator, "b*m b*m"] | IdentityOperator,
    M_op: Float[SparseOperator, "b*m b*m"] | IdentityOperator,
    S_op: Float[SparseOperator, "b*m b*m"] | IdentityOperator,
    X_0: Float[t.Tensor, "b*m n"],
    precond: LOBPCGPreconditioner,
    largest: bool,
    tol: float,
    niter: int,
    generator: t.Generator | None,
) -> tuple[Float[t.Tensor, "b n"], Float[t.Tensor, "b*m n"]]:
    """
    Variables with an explicit batch dimension are named with the _batched suffix.
    """
    bm, n = X_0.shape

    X_current = M_orthonormalize_batched(
        X_0, M_op, n_min=X_0.size(-1), generator=generator, max_iter=3
    )
    X_prev = X_current

    TX_current = T_op @ X_current

    Lambda_current = t.einsum(
        "bii->bi",
        t.bmm(X_current.view(b, -1, n) @ (S_op @ TX_current).view(b, -1, n)),
    )

    converged = False
    for _ in range(niter):
        R_batched = TX_current.view(b, -1, n) - (
            (B_op @ X_current).view(b, -1, n) * Lambda_current.view(b, 1, -1)
        )
        R_norm_batched = t.linalg.norm(R_batched, dim=1)

        if (R_norm_batched < tol).all():
            converged = True
            break

        else:
            Lambda_next, X_next, TX_next = _lobpcg_one_iter_batched(
                b,
                T_op,
                M_op,
                S_op,
                R_batched.view(bm, n),
                X_current,
                X_prev,
                precond,
                largest,
                tol,
                generator,
            )

            X_prev = X_current
            X_current = X_next
            TX_current = TX_next
            Lambda_current = Lambda_next

    if not converged:
        warnings.warn(
            f"LOBPCG did not converge after {niter} iterations. "
            f"Max residual norm: {R_norm_batched.max().item():.2e} (tol: {tol:.2e}).",
            UserWarning,
        )

    return Lambda_current, X_current


def lobpcg_forward_batched(
    b: int,
    A_op: SparseOperatorLikeBatched,
    M_op: Float[SparseOperator, "b*m b*m"] | None,
    sigma: float | int | None,
    v0: Float[t.Tensor, "b*m n"],
    largest: bool,
    tol: float,
    maxiter: int,
    nvmath_config: DirectSolverConfig,
    precond_config: LOBPCGPrecondConfig,
    generator: t.Generator | None,
) -> tuple[Float[t.Tensor, "b n"], Float[t.Tensor, "b*m n"]]:
    bm, n = v0.shape
    if bm % b != 0:
        raise ValueError(
            f"The v0 tensor cannot be evenly divided into {b} batch items along its first dimension."
        )

    T_op, B_op, M_op, S_op, precond = _dispatch_operators(
        n, A_op, M_op, sigma, nvmath_config, precond_config
    )

    return _lobpcg_loop_batched(
        b=b,
        T_op=T_op,
        B_op=B_op,
        M_op=M_op,
        S_op=S_op,
        X_0=v0,
        largest=largest,
        tol=tol,
        niter=maxiter,
        precond=precond,
        generator=generator,
    )

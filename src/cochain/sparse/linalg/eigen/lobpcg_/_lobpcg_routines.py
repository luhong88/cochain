import warnings

try:
    from typing import TypeAlias
except ImportError:
    from typing_extensions import TypeAlias

import torch
from jaxtyping import Float
from torch import Tensor

from ....decoupled_tensor import SparseDecoupledTensor
from ...solvers import DirectSolverConfig
from ..base.utils import m_orthonormalize
from ._lobpcg_operators import (
    IdOp,
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

SparseDecoupledTensorLike: TypeAlias = (
    IdOp
    | Float[SparseDecoupledTensor, "m m"]
    | Float[ShiftInvSymSpOp, "m m"]
    | Float[ShiftInvSymGEPSpOp, "m m"]
)

LOBPCGPreconditioner: TypeAlias = (
    IdentityPrecond | JacobiPrecond | ILUPrecond | ChoPrecond
)


def _lobpcg_one_iter(
    t_op: SparseDecoupledTensorLike,
    m_op: Float[SparseDecoupledTensor, "m m"] | IdOp,
    s_op: Float[SparseDecoupledTensor, "m m"] | IdOp,
    res: Float[Tensor, "m n"],
    x_current: Float[Tensor, "m n"],
    x_prev: Float[Tensor, "m n"],
    precond: LOBPCGPreconditioner,
    largest: bool,
    rtol: float,
    generator: torch.Generator | None,
) -> tuple[Float[Tensor, " n"], Float[Tensor, "m n"], Float[Tensor, "m n"]]:
    """Perform one iteration of LOBPCG."""
    n = x_current.size(-1)

    # Perform soft locking/deflation to lock in converged eigenvectors by zeroing
    # out the corresponding residual vectors.
    res_norm = torch.linalg.norm(res, dim=0, keepdim=True)
    mask = (res_norm > rtol).to(res_norm.dtype)
    res_masked = res * mask

    # Use the residual vectors R to compute the precondition directions W.
    search_dir = precond @ res_masked

    # Compute the momentum/conjugate directions P. During the first iteration,
    # X_current = X_prev so P = 0. Perform the same soft locking on the momentum.
    conj_dir = (x_current - x_prev) * mask

    # Assemble the new trial subspace and enforce M-orthonormality condition on
    # the subspace basis vectors.
    v = torch.hstack((x_current, search_dir, conj_dir))

    v_ortho = m_orthonormalize(v, m_op, n_min=n, generator=generator, max_iter=3)
    tv_ortho = t_op @ v_ortho

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
    t_reduced = v_ortho.T @ (s_op @ tv_ortho)

    lambda_next_all, x_next_reduced_all = torch.linalg.eigh(t_reduced)

    # Extract the n largest (or smallest) eigenvalue-eigenvector pairs.
    # Note that torch.linalg.eigh() returns eigenvalues in ascending order
    if largest:
        # if largest=True, sort eigenvalues in descending order
        x_next_reduced = torch.flip(x_next_reduced_all[:, -n:], dims=(-1,))
        lambda_next = torch.flip(lambda_next_all[-n:], dims=(0,))
    else:
        # if largest=False, keep eigenvalues in ascending order
        x_next_reduced = x_next_reduced_all[:, :n]
        lambda_next = lambda_next_all[:n]

    # Lift the reduced eigenvectors back to the full space
    x_next = v_ortho @ x_next_reduced
    tx_next = tv_ortho @ x_next_reduced

    return lambda_next, x_next, tx_next


def _lobpcg_loop(
    t_op: SparseDecoupledTensorLike,
    b_op: Float[SparseDecoupledTensor, "m m"] | IdOp,
    m_op: Float[SparseDecoupledTensor, "m m"] | IdOp,
    s_op: Float[SparseDecoupledTensor, "m m"] | IdOp,
    x_0: Float[Tensor, "m n"],
    precond: LOBPCGPreconditioner,
    largest: bool,
    atol: float,
    rtol: float,
    niter: int,
    generator: torch.Generator | None,
) -> tuple[Float[Tensor, " n"], Float[Tensor, "m n"]]:
    x_current = m_orthonormalize(
        x_0, m_op, n_min=x_0.size(-1), generator=generator, max_iter=3
    )
    x_prev = x_current

    tx_current = t_op @ x_current

    # Compute the eigenvalues using the Rayleigh quotient X.T@S@T@X/X.T@M@X.
    # In most cases, S = I so the quotient reduces to X.T@T@X = X.T@M@X@Λ = Λ.
    # For GEP in the shift-invert mode, T@X = X@Λ' and S = M so that X.T@M@T@X
    # correctly reduces to the shift-inverted eigenvalues Λ'.
    lambda_current = torch.diag(x_current.T @ (s_op @ tx_current))

    converged = False
    for _ in range(niter):
        # Compute the residual vectors R = T@X - B@X@Λ.
        res = tx_current - (b_op @ x_current) * lambda_current.view(1, -1)
        res_norm = torch.linalg.norm(res, dim=0)

        # Adjust the tolerance based on the eigenvalue magnitudes.
        tol_current = atol + rtol * lambda_current.abs()

        if (res_norm <= tol_current).all():
            converged = True
            break

        else:
            lambda_next, x_next, tx_next = _lobpcg_one_iter(
                t_op,
                m_op,
                s_op,
                res,
                x_current,
                x_prev,
                precond,
                largest,
                rtol,
                generator,
            )

            x_prev = x_current
            x_current = x_next
            tx_current = tx_next
            lambda_current = lambda_next

    if not converged:
        warnings.warn(
            f"LOBPCG did not converge after {niter} iterations. "
            f"Max residual norm: {res_norm.max().item():.2e} (rtol: {rtol:.2e}).",
            UserWarning,
        )

    return lambda_current, x_current


def _dispatch_operators(
    n: int,
    a_op: SparseDecoupledTensorLike,
    m_op: Float[SparseDecoupledTensor, "m m"] | None,
    sigma: float | int | None,
    nvmath_config: DirectSolverConfig,
    precond_config: LOBPCGPrecondConfig,
) -> tuple[
    SparseDecoupledTensorLike,
    SparseDecoupledTensorLike,
    SparseDecoupledTensorLike,
    SparseDecoupledTensorLike,
    LOBPCGPreconditioner,
]:
    if sigma is not None:
        # If doing shift-invert mode, always use the identity preconditioner and
        # ignore the user inputs.
        precond = IdentityPrecond()
    else:
        match precond_config.method:
            case "identity":
                precond = IdentityPrecond()
            case "jacobi":
                # a_op is not required to be int32-safe.
                precond = JacobiPrecond(a_sdt=a_op)
            case "ilu":
                # a_op is required to be int32-safe.
                precond = ILUPrecond(
                    a_sdt=a_op,
                    diag_damp=precond_config.diag_damp,
                    spilu_kwargs=precond_config.spilu_kwargs,
                )
            case "cholesky":
                # a_op is required to be int32-safe.
                precond = ChoPrecond(
                    a_sdt=a_op,
                    n=n,
                    diag_damp=precond_config.diag_damp,
                    nvmath_config=precond_config.nvmath_config,
                )
            case _:
                raise ValueError()

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
            raise ValueError()

    return t_op, b_op, m_op, s_op, precond


def lobpcg_forward(
    a_op: SparseDecoupledTensorLike,
    m_op: Float[SparseDecoupledTensor, "m m"] | None,
    sigma: float | int | None,
    v0: Float[Tensor, "m n"],
    largest: bool,
    atol: float,
    rtol: float,
    maxiter: int,
    nvmath_config: DirectSolverConfig,
    precond_config: LOBPCGPrecondConfig,
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
    """
    n = v0.size(-1)

    t_op, b_op, m_op, s_op, precond = _dispatch_operators(
        n, a_op, m_op, sigma, nvmath_config, precond_config
    )

    return _lobpcg_loop(
        t_op=t_op,
        b_op=b_op,
        m_op=m_op,
        s_op=s_op,
        x_0=v0,
        largest=largest,
        atol=atol,
        rtol=rtol,
        niter=maxiter,
        precond=precond,
        generator=generator,
    )

from __future__ import annotations

__all__ = [
    "mixed_weak_laplacian_lobpcg",
    "LOBPCGConfig",
    "LaplacianLOBPCGPrecondConfig",
]

from dataclasses import asdict, replace
from typing import Any, Literal

import torch
from jaxtyping import Float, Integer
from torch import Tensor
from torch.autograd.function import once_differentiable

from ...sparse.decoupled_tensor import (
    BaseDecoupledTensor,
    DiagDecoupledTensor,
    SparseDecoupledTensor,
    SparsityPattern,
)
from ...sparse.linalg.eigen.base._backward import (
    compute_cauchy_matrix,
    compute_eig_vec_grad_proj,
)
from ...sparse.linalg.eigen.base.utils import compute_lorentzian_eps_via_eigs
from ...sparse.linalg.eigen.lobpcg_._lobpcg_preconditioners import (
    IdentityPrecond,
    LOBPCGPrecondConfig,
    LOBPCGPreconditioner,
)
from ...sparse.linalg.eigen.lobpcg_._lobpcg_routines import lobpcg_forward
from ...sparse.linalg.eigen.lobpcg_.lobpcg_ import LOBPCGConfig
from ...sparse.linalg.solvers import DirectSolverConfig
from ..laplacians import MixedWeakLaplacianBlocks
from ._backward import compute_dLdM_k_val, compute_dLdM_km1_val, compute_dLdM_kp1_val
from ._operator import MassKm1Solver, MixedWeakLaplacianOp
from ._preconditioners import (
    LaplacianLOBPCGPrecondConfig,
    ShiftedLumpedPrecond,
    ShiftedUpPrecond,
)


def _reconstruct_op(
    val: Tensor, pattern: SparsityPattern | None
) -> BaseDecoupledTensor:
    if pattern is None:
        return DiagDecoupledTensor(val)
    else:
        return SparseDecoupledTensor(pattern, val)


class MixedWeakLaplacianLOBPCGAutogradFunction(torch.autograd.Function):
    @staticmethod
    def _dispatch_precond(
        mixed_weak_laplacian: Float[MixedWeakLaplacianBlocks, "k_splx k_splx"],
        lobpcg_config: LOBPCGConfig,
        precond_config: LaplacianLOBPCGPrecondConfig,
    ) -> LOBPCGPreconditioner:
        match precond_config.method:
            case "identity":
                precond = IdentityPrecond()
            case "shifted_up":
                if mixed_weak_laplacian._block_11 is None:
                    raise ValueError(
                        "The ShiftedUpPrecond is not applicable for a down-only "
                        "MixedWeakLaplacianBlocks object."
                    )

                precond = ShiftedUpPrecond(
                    weak_up_laplacian=mixed_weak_laplacian._block_11,
                    mass_k=mixed_weak_laplacian.mass_k,
                    star_k=precond_config.star_k,
                    tau=precond_config.tau,
                    n=lobpcg_config.v0.size(-1),
                    nvmath_config=precond_config.nvmath_config,
                )

            case "shifted_lumped":
                precond = ShiftedLumpedPrecond(
                    cbd_km1=mixed_weak_laplacian.cbd_km1,
                    cbd_k=mixed_weak_laplacian.cbd_k,
                    star_km1=precond_config.star_km1,
                    mass_k=mixed_weak_laplacian.mass_k,
                    star_k=precond_config.star_k,
                    mass_kp1=mixed_weak_laplacian.mass_kp1,
                    tau=precond_config.tau,
                    n=lobpcg_config.v0.size(-1),
                    nvmath_config=precond_config.nvmath_config,
                )

            case _:
                raise ValueError(f"Unknown preconditioner '{precond_config.method}'.")

        return precond

    @staticmethod
    def forward(
        cbd_km1: Float[SparseDecoupledTensor, "k_splx km1_splx"],
        cbd_k: Float[SparseDecoupledTensor, "kp1_splx k_splx"] | None,
        mass_km1_val: Float[Tensor, " km1_nz"],
        mass_km1_pattern: Integer[SparsityPattern, "km1_splx km1_splx"],
        mass_k_val: Float[Tensor, " k_nz"],
        mass_k_pattern: Integer[SparsityPattern, "k_splx k_splx"] | None,
        mass_kp1_val: Float[Tensor, " kp1_nz"] | None,
        mass_kp1_pattern: Integer[SparsityPattern, "kp1_splx kp1_splx"] | None,
        n: int,
        l: int,
        eps: float | int | Literal["auto"],
        op_scale: float | Literal["auto"],
        solver_type: Literal["scipy_splu", "cupy_splu", "nvmath_direct_solver"],
        solver_config: DirectSolverConfig | dict[str, Any],
        lobpcg_config: LOBPCGConfig,
        precond_config: LaplacianLOBPCGPrecondConfig,
    ) -> tuple[Float[Tensor, " l"], Float[Tensor, "k_splx l"], MassKm1Solver]:
        mass_km1 = SparseDecoupledTensor(mass_km1_pattern, mass_km1_val)
        mass_k = _reconstruct_op(mass_k_val, mass_k_pattern)

        if mass_kp1_val is None:
            mass_kp1 = None
        else:
            mass_kp1 = _reconstruct_op(mass_kp1_val, mass_kp1_pattern)

        laplacian = MixedWeakLaplacianBlocks(cbd_km1, cbd_k, mass_km1, mass_k, mass_kp1)
        precond = MixedWeakLaplacianLOBPCGAutogradFunction._dispatch_precond(
            laplacian, lobpcg_config, precond_config
        )

        mass_km1_solver = MassKm1Solver(mass_km1, solver_type, solver_config, n)
        laplacian_op = MixedWeakLaplacianOp(laplacian, mass_km1_solver)

        # The nvmath_config argument is only relevant for the shift-invert mode,
        # which is currently not supported for matrix-free linear operators.
        eig_vals, eig_vecs = lobpcg_forward(
            a_op=laplacian_op,
            m_op=mass_k,
            a_norm=None,
            m_norm=None,
            op_scale=op_scale,
            precond=precond,
            nvmath_config=DirectSolverConfig(),
            **asdict(lobpcg_config),
        )

        eig_vals_true = eig_vals[:l]

        return eig_vals_true, eig_vecs[:, :l], mass_km1_solver

    @staticmethod
    def setup_context(ctx, inputs, output):
        (
            cbd_km1,
            cbd_k,
            mass_km1_val,
            mass_km1_pattern,
            mass_k_val,
            mass_k_pattern,
            mass_kp1_val,
            mass_kp1_pattern,
            n,
            l,
            eps,
            op_scale,
            solver_type,
            solver_config,
            lobpcg_config,
            precond_config,
        ) = inputs
        eig_vals, eig_vecs, mass_km1_solver = output

        needs_grad_mass_km1_val = ctx.needs_input_grad[2]
        needs_grad_mass_k_val = ctx.needs_input_grad[4]
        needs_grad_mass_kp1_val = ctx.needs_input_grad[6]

        needs_codiff = needs_grad_mass_km1_val or needs_grad_mass_k_val

        ctx.save_for_backward(eig_vals, eig_vecs, mass_k_val if needs_codiff else None)

        ctx.cbd_km1 = cbd_km1
        ctx.mass_km1_pattern = mass_km1_pattern
        ctx.mass_k_pattern = mass_k_pattern
        ctx.mass_kp1_pattern = mass_kp1_pattern
        ctx.eps = compute_lorentzian_eps_via_eigs(eig_vals) if eps == "auto" else eps

        if needs_codiff:
            ctx.mass_km1_solver = mass_km1_solver

        if needs_grad_mass_kp1_val:
            ctx.cbd_k = cbd_k

    @staticmethod
    @once_differentiable
    def backward(
        ctx, dLdl: Float[Tensor, " k"], dLdv: Float[Tensor, "m k"] | None, _
    ) -> tuple[
        None,
        None,
        Float[Tensor, " km1_nz"] | None,
        None,
        Float[Tensor, " k_nz"] | None,
        None,
        Float[Tensor, " kp1_nz"] | None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
    ]:
        needs_grad_mass_km1_val = ctx.needs_input_grad[2]
        needs_grad_mass_k_val = ctx.needs_input_grad[4]
        needs_grad_mass_kp1_val = ctx.needs_input_grad[6]

        needs_codiff = needs_grad_mass_km1_val or needs_grad_mass_k_val

        # The eigenvectors need to be M_k-normalized for the following calculation.
        eig_vals, eig_vecs, mass_k_val = ctx.saved_tensors

        cbd_km1: SparseDecoupledTensor = ctx.cbd_km1
        mass_km1_pattern: SparsityPattern = ctx.mass_km1_pattern
        mass_k_pattern: SparsityPattern | None = ctx.mass_k_pattern
        mass_kp1_pattern: SparsityPattern | None = ctx.mass_kp1_pattern

        # This error should never be triggered if the user-facing wrapper does its job.
        if eig_vecs is None:
            raise ValueError("Eigenvectors are required for backward().")

        if (dLdv is None) or (not torch.any(dLdv)):
            eig_vec_grad_proj = None
            cauchy = None
        else:
            eig_vec_grad_proj = compute_eig_vec_grad_proj(eig_vecs, dLdv)
            cauchy = compute_cauchy_matrix(eig_vals, ctx.eps)

        if needs_codiff:
            mass_km1_solver: MassKm1Solver = ctx.mass_km1_solver

            mass_k = _reconstruct_op(mass_k_val, mass_k_pattern)
            rhs = cbd_km1.T @ mass_k @ eig_vecs

            eig_vec_codiffs = mass_km1_solver.solve(rhs)

        if needs_grad_mass_km1_val:
            dLdM_km1 = compute_dLdM_km1_val(
                mass_km1_pattern, eig_vec_codiffs, dLdl, dLdv, eig_vec_grad_proj, cauchy
            )
        else:
            dLdM_km1 = None

        if needs_grad_mass_k_val:
            dLdM_k = compute_dLdM_k_val(
                cbd_km1,
                mass_k_pattern,
                eig_vals,
                eig_vecs,
                eig_vec_codiffs,
                dLdl,
                dLdv,
                eig_vec_grad_proj,
                cauchy,
            )
        else:
            dLdM_k = None

        if needs_grad_mass_kp1_val:
            cbd_k: SparseDecoupledTensor = ctx.cbd_k

            dLdM_kp1 = compute_dLdM_kp1_val(
                cbd_k, mass_kp1_pattern, eig_vecs, dLdl, dLdv, eig_vec_grad_proj, cauchy
            )
        else:
            dLdM_kp1 = None

        return (
            None,
            None,
            dLdM_km1,
            None,
            dLdM_k,
            None,
            dLdM_kp1,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


def mixed_weak_laplacian_lobpcg(
    mixed_weak_laplacian: Float[MixedWeakLaplacianBlocks, "k_splx k_splx"],
    n: int | None = None,
    l: int = 6,
    eps: float | int | Literal["auto"] = "auto",
    op_scale: float | Literal["auto"] = "auto",
    solver_type: Literal[
        "scipy_splu", "cupy_splu", "nvmath_direct_solver"
    ] = "scipy_splu",
    solver_config: DirectSolverConfig | dict[str, Any] | None = None,
    lobpcg_config: LOBPCGConfig | None = None,
    precond_config: LaplacianLOBPCGPrecondConfig | None = None,
) -> tuple[Float[Tensor, " l"], Float[Tensor, "k_splx l"]]:
    r"""
    Sparse differentiable eigensolver for mixed weak Hodge Laplacians using LOBPCG.

    This function implements a specialized version of `lobpcg()` that avoids
    materializing the matrix inverse of non-diagonal $M_{k-1}$ in the down-component
    of the weak Hodge $k$-Laplacian by representing the Laplacian as a matrix-free
    operator.

    Parameters
    ----------
    mixed_weak_laplacian : [k_splx, k_splx]
        A weak Hodge Laplacian represented as a `MixedWeakLaplacianBlocks` object.
    n
        The number of approximated eigenvalues/eigenvectors, which should be in
        the range [`l`, `k_splx`] (default value: `l`). In general, it is recommended
        to set the `n` argument somewhat higher than `l`, to make the convergence
        of the `l` desired eigenvalues faster and to account for possible degenerate
        eigenvalues.
    l
        The number of eigenvalues/eigenvectors to find. Note that, by default,
        this function finds the `l` smallest eigenvalues of `mixed_weak_laplacian`;
        this behavior can be changed in `lobpcg_config`.
    eps
        The strength of Lorentzian broadening/regularization, which removes
        singularities in backward gradient calculation when some of the
        eigenvalues are (near) degenerate. As a heuristic, the regularization starts
        to dominate the gradient calculation as the spectral gap approaches the
        square root of `eps`. Set to integer 0 to disable regularization; set to
        "auto" to select `eps` based on the computed eigenvalues.
    op_scale
        Operator scale for the matrix-free residual floor: the floor for each
        eigenvector x is tol * op_scale * ||x||. By default, estimate this scale from
        the first block of operator applications.
    solver_type
        Which sparse linear solver backend to use for representing the action
        of the inverse of $M_{k-1}$.
    solver_config
        Additional optional arguments for the sparse linear solver constructors.
        If solver_type is "nvmath_direct_solver", then this should be an
        instance of `DirectSolverConfig`; if the solver_type is "*_splu", then this
        should be a dict.
    lobpcg_config
        Additional optional LOBPCG configurations.
    precond_config
        Additional optional arguments for LOBPCG preconditioners. Note that the
        preconditioner config is ignored in the shift-invert mode.

    Returns
    -------
    eig_vals : [l,]
        A tensor of `l` eigenvalues.
    eig_vecs : [k_splx, l]
        A tensor of `l` $M_k$-orthonormal eigenvectors; each column represents
        an eigenvector.

    Notes
    -----
    Block-diagonal batching and the shift-invert mode is not supported.

    Automatic operator-scale estimation uses only the initial trial eigenvectors
    and can fail for operators with very large nullspaces, where the initial vectors
    are likely to lie in or near the nullspace. The estimated scale can then
    be too small, making the convergence criteria unattainably. In such cases,
    supply a positive `op_scale` representative of the weak Laplacian matrix norm
    instead of using "auto".

    The autograd through eigenvectors do not account for contributions from the
    unresolved eigenvectors. Currently, only first-order derivatives are supported.
    The coboundary operators are treated as nondifferentiable.

    This implementation accepts specific preconditioners, including: identity,
    "shifted up" ($(S_k^up + \tau M_k)^{-1}$), and "shifted lumped"
    (approximately, $(S_k + \tau M_k)^{-1}$) preconditioners; in particular, the
    "shifted up" preconditioner is not applicable for down-component only
    `MixedWeakLaplacianBlocks`. The preconditioner can be configured using a
    `LaplacianLOBPCGPrecondConfig` and will be generated internally. Both the
    "shifted up" and "shifted lumped" preconditioners require `nvmath-python`.
    """
    # Note that we delegate the CuPy and nvmath-python dependency checks to
    # the operator and preconditioner constructors, rather than performing a
    # top-level check.

    if lobpcg_config is None:
        lobpcg_config = LOBPCGConfig()
    if precond_config is None:
        precond_config = LOBPCGPrecondConfig()

    if solver_config is None:
        match solver_type:
            case "nvmath_direct_solver":
                solver_config = DirectSolverConfig()
            case "scipy_splu" | "cupy_splu":
                solver_config = {}
            case _:
                raise ValueError(f"Unrecognized 'solver_type' argument '{solver_type}'")

    # Process raw LOBPCG config.
    n_k_splx = mixed_weak_laplacian.size(0)

    if lobpcg_config.v0 is None:
        if n is None:
            n = l
    else:
        v0 = lobpcg_config.v0

        if not isinstance(v0, Tensor):
            raise ValueError("v0 must be a tensor.")
        if n is None:
            n = v0.size(-1)
        if v0.ndim != 2 or v0.shape != (n_k_splx, n):
            raise ValueError("v0 must have shape (k, n).")

    if n < l or n > n_k_splx:
        raise ValueError("n must be in the range [l, k].")

    if lobpcg_config.v0 is None:
        v0 = torch.randn(
            (n_k_splx, n),
            generator=lobpcg_config.generator,
            dtype=mixed_weak_laplacian.dtype,
            device=mixed_weak_laplacian.device,
        )

    tol = (
        torch.finfo(mixed_weak_laplacian.dtype).eps ** 0.5
        if lobpcg_config.tol == "auto"
        else lobpcg_config.tol
    )

    processed_lobpcg_config = replace(lobpcg_config, v0=v0, tol=tol)

    eig_vals, eig_vecs, _ = MixedWeakLaplacianLOBPCGAutogradFunction.apply(
        mixed_weak_laplacian.cbd_km1,
        mixed_weak_laplacian.cbd_k,
        mixed_weak_laplacian.mass_km1.values,
        mixed_weak_laplacian.mass_km1.pattern,
        mixed_weak_laplacian.mass_k.values,
        getattr(mixed_weak_laplacian.mass_k, "pattern", None),
        getattr(mixed_weak_laplacian.mass_kp1, "values", None),
        getattr(mixed_weak_laplacian.mass_kp1, "pattern", None),
        n,
        l,
        eps,
        op_scale,
        solver_type,
        solver_config,
        processed_lobpcg_config,
        precond_config,
    )

    return eig_vals, eig_vecs

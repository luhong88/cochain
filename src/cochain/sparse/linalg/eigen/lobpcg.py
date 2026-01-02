from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import TYPE_CHECKING, Any

import torch as t
from jaxtyping import Float, Integer

from ...operators import SparseOperator, SparseTopology
from ._backward import dLdA_backward, dLdA_dLdM_backward
from ._lobpcg_routines import lobpcg_forward

try:
    import nvmath.sparse.advanced as nvmath_sp

    _HAS_NVMATH = True

except ImportError:
    _HAS_NVMATH = False

if TYPE_CHECKING:
    from ..solvers.nvmath_wrapper import DirectSolverConfig


@dataclass
class LOBPCGConfig:
    sigma: float | int | None = None
    v0: Float[t.Tensor, "m n"] | None = None
    diag_damp: float | int | None
    largest: bool = False
    tol: float | None = None
    maxiter: int | None = 1000


class _LOBPCGAutogradFunction(t.autograd.Function):
    @staticmethod
    def forward(
        A_val: Float[t.Tensor, " A_nnz"],
        A_sp_topo: Integer[SparseTopology, "m m"],
        M_val: Float[t.Tensor, " M_nnz"] | None,
        M_sp_topo: Integer[SparseTopology, "m m"] | None,
        k: int,
        eps: float | int,
        lobpcg_config: LOBPCGConfig,
        nvmath_config: DirectSolverConfig,
    ) -> tuple[Float[t.Tensor, " k"], Float[t.Tensor, "m k"]]:
        A_op = SparseOperator(A_sp_topo, A_val)

        if (M_val is None) and (M_sp_topo is None):
            M_op = None
        elif (M_val is not None) and (M_sp_topo is not None):
            M_op = SparseOperator(M_sp_topo, M_val)
        else:
            raise ValueError()

        eig_vals, eig_vecs = lobpcg_forward(
            A_op=A_op,
            M_op=M_op,
            nvmath_config=nvmath_config,
            **asdict(lobpcg_config),
        )

        return eig_vals[:k], eig_vecs[:k]

    @staticmethod
    def setup_context(ctx, inputs, output):
        A_val, A_sp_topo, M_val, M_sp_topo, k, eps, lobpcg_config, nvmath_config = (
            inputs
        )
        eig_vals, eig_vecs = output

        ctx.save_for_backward(eig_vals, eig_vecs)
        ctx.A_sp_topo = A_sp_topo
        ctx.M_sp_topo = M_sp_topo
        ctx.k = k
        ctx.eps = eps

    @staticmethod
    def backward(
        ctx, dLdl: Float[t.Tensor, " k"], dLdv: Float[t.Tensor, "m k"] | None
    ) -> tuple[
        Float[t.Tensor, " A_nnz"] | None,
        None,
        Float[t.Tensor, " A_nnz"] | None,
        None,
        None,
        None,
        None,
        None,
    ]:
        needs_grad_A_val = ctx.needs_input_grad[0]
        needs_grad_M_val = ctx.needs_input_grad[2]

        if ctx.M_sp_topo is None:
            dLdA_val = dLdA_backward(ctx, dLdl, dLdv) if needs_grad_A_val else None
            dLdM_val = None
        else:
            dLdA_val, dLdM_val = dLdA_dLdM_backward(
                ctx, dLdl, dLdv, needs_grad_A_val, needs_grad_M_val
            )

        return dLdA_val, None, dLdM_val, None, None, None, None, None


def _lobpcg_no_batch(
    A: Float[SparseOperator, "m m"],
    M: Float[SparseOperator, "m m"] | None,
    kwargs: dict[str, Any],
) -> tuple[Float[t.Tensor, " k"], Float[t.Tensor, "c k"]]:
    if M is None:
        eig_vals, eig_vecs = _LOBPCGAutogradFunction.apply(
            A.val, A.sp_topo, None, None, **kwargs
        )
    else:
        eig_vals, eig_vecs = _LOBPCGAutogradFunction.apply(
            A.val, A.sp_topo, M.val, M.sp_topo, **kwargs
        )

    return eig_vals, eig_vecs


def _lobpcg_batch(
    A_batched: Float[SparseOperator, "m m"],
    M_batched: Float[SparseOperator, "m m"] | None,
    kwargs: dict[str, Any],
) -> tuple[Float[t.Tensor, " k"], Float[t.Tensor, "m k"]]:
    A_list = A_batched.unpack_block_diag()
    if M_batched is None:
        M_list = [None] * len(A_list)
    else:
        M_list = M_batched.unpack_block_diag()

    for A, M in zip(A_list, M_list, strict=True):
        eig_val_list, eig_vec_list = _lobpcg_no_batch(A, M, kwargs)

    eig_vals = t.vstack(eig_val_list)
    eig_vecs = t.vstack(eig_vec_list)

    return eig_vals, eig_vecs


def lobpcg(
    A: Float[SparseOperator, "m m"],
    M: Float[SparseOperator, "m m"] | None = None,
    block_diag_batch: bool = False,
    n: int | None = 10,
    k: int = 6,
    eps: float | int = 1e-6,
    lobpcg_config: LOBPCGConfig | None = None,
    nvmath_config: DirectSolverConfig | None = None,
    generator: t.Generator | None = None,
) -> tuple[Float[t.Tensor, "*b k"], Float[t.Tensor, "m k"]]:
    """
    A custom implementation of LOBPCG.

    Note that this function requires `nvmath-python` for its sparse linear solver
    utilities.

    This function implements a version of LOBPCG that's roughly equivalent to
    `torch.lobpcg(method='ortho')`, but with the following key differences:

    * This implementation is differentiable with respect to the `SparseOperator`s
      `A` and `M`. The `eps` argument is used for Lorentzian broadening/regularization
      in the gradient calculation to prevent gradient explosion when the eigenvalues
      are (near) degenerate.
    * This implementation supports shift-invert mode for both standard and
      generalized eigenvalue problems.
    * This implementation employs a damped exact solver for preconditioning. The
      diagonal damping is controlled by the `diag_damp` argument in `LOBPCGConfig`.
    * This implementation does not support explicit batch dimensions in `A` or `M`.
      If `block_diag_batch=True`, `A` and `M` will be split into individual sparse
      matrices and solved sequentially. This requires that `A` (and `M` if not `None`)
      has a valid `BlockDiagConfig` for unpacking the block diagonal batch structure.
    """
    if not _HAS_NVMATH:
        raise ImportError("nvmath-python backends required.")

    from ..solvers.nvmath_wrapper import DirectSolverConfig

    if lobpcg_config is None:
        lobpcg_config = LOBPCGConfig()
    if nvmath_config is None:
        nvmath_config = DirectSolverConfig()

    # Process raw LOBPCG config.

    if lobpcg_config.v0 is None:
        if n is None:
            n = k
        else:
            if n < k or n > A.size(-1):
                raise ValueError("n must be in the range [k, m].")

        v0 = t.randn(
            (A.size(0), n), generator=generator, dtype=A.dtype, device=A.device
        )

    else:
        v0 = lobpcg_config.v0

    tol = (
        t.finfo(A.dtype).eps ** 0.5 if lobpcg_config.tol is None else lobpcg_config.tol
    )

    processed_lobpcg_config = replace(lobpcg_config, v0=v0, tol=tol)

    kwargs = {
        "k": k,
        "eps": eps,
        "lobpcg_config": processed_lobpcg_config,
        "nvmath_config": nvmath_config,
    }

    if block_diag_batch:
        eig_vals, eig_vecs = _lobpcg_batch(A, M, kwargs)
    else:
        eig_vals, eig_vecs = _lobpcg_no_batch(A, M, kwargs)

    return eig_vals, eig_vecs

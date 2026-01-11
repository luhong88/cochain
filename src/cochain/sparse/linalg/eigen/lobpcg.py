from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import TYPE_CHECKING, Sequence

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
    from ._lobpcg_preconditioners import LOBPCGPrecondConfig


@dataclass
class LOBPCGConfig:
    sigma: float | int | None = None
    v0: Float[t.Tensor, "m n"] | Sequence[Float[t.Tensor, "m c"] | None] | None = None
    largest: bool = True
    tol: float | None = None
    maxiter: int | None = 1000
    generator: t.Generator | None = None

    def expand(self, n: int) -> list[LOBPCGConfig]:
        if isinstance(self.v0, Sequence):
            config_list = [replace(self, v0=v0) for v0 in self.v0]
            if len(config_list) != n:
                raise ValueError("Inconsistent v0 specification.")
        else:
            config_list = [self] * n

        return config_list


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
        precond_config: LOBPCGPrecondConfig,
        nvmath_config: DirectSolverConfig,
    ) -> tuple[Float[t.Tensor, " k"], Float[t.Tensor, "m k"]]:
        A_op = SparseOperator(A_sp_topo, A_val)

        if (M_val is None) and (M_sp_topo is None):
            M_op = None
        elif (M_val is not None) and (M_sp_topo is not None):
            M_op = SparseOperator(M_sp_topo, M_val)
        else:
            raise ValueError()

        lobpcg_config_dict = asdict(lobpcg_config)
        del lobpcg_config_dict["generator"]

        eig_vals, eig_vecs = lobpcg_forward(
            A_op=A_op,
            M_op=M_op,
            nvmath_config=nvmath_config,
            precond_config=precond_config,
            **lobpcg_config_dict,
        )

        if lobpcg_config.sigma is None:
            eig_vals_true = eig_vals[:k]
        else:
            # In shift-invert mode, λ_returned = 1 / (λ_true - σ)
            eig_vals_true = lobpcg_config.sigma + (1.0 / eig_vals[:k])

        return eig_vals_true, eig_vecs[:, :k]

    @staticmethod
    def setup_context(ctx, inputs, output):
        (
            A_val,
            A_sp_topo,
            M_val,
            M_sp_topo,
            k,
            eps,
            lobpcg_config,
            precond_config,
            nvmath_config,
        ) = inputs
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

        return dLdA_val, None, dLdM_val, None, None, None, None, None, None


def _lobpcg_no_batch(
    A: Float[SparseOperator, "m m"],
    M: Float[SparseOperator, "m m"] | None,
    k: int,
    eps: float | int,
    lobpcg_config: LOBPCGConfig,
    precond_config: LOBPCGPrecondConfig,
    nvmath_config: DirectSolverConfig,
) -> tuple[Float[t.Tensor, " k"], Float[t.Tensor, "c k"]]:
    if M is None:
        eig_vals, eig_vecs = _LOBPCGAutogradFunction.apply(
            A.val,
            A.sp_topo,
            None,
            None,
            k,
            eps,
            lobpcg_config,
            precond_config,
            nvmath_config,
        )
    else:
        eig_vals, eig_vecs = _LOBPCGAutogradFunction.apply(
            A.val,
            A.sp_topo,
            M.val,
            M.sp_topo,
            k,
            eps,
            lobpcg_config,
            precond_config,
            nvmath_config,
        )

    return eig_vals, eig_vecs


def _lobpcg_batch(
    A_list: list[Float[SparseOperator, "m m"]],
    M_batched: Float[SparseOperator, "m m"] | None,
    k: int,
    eps: float | int,
    lobpcg_config_batched: LOBPCGConfig,
    precond_config: LOBPCGPrecondConfig,
    nvmath_config: DirectSolverConfig,
) -> tuple[Float[t.Tensor, " k"], Float[t.Tensor, "m k"]]:
    if M_batched is None:
        M_list = [None] * len(A_list)
    else:
        M_list = M_batched.unpack_block_diag()

    lobpcg_config_list = lobpcg_config_batched.expand(n=len(A_list))

    eig_val_list = []
    eig_vec_list = []
    for A, M, lobpcg_config in zip(A_list, M_list, lobpcg_config_list, strict=True):
        eig_val, eig_vec = _lobpcg_no_batch(
            A, M, k, eps, lobpcg_config, precond_config, nvmath_config
        )
        eig_val_list.append(eig_val)
        eig_vec_list.append(eig_vec)

    eig_vals = t.vstack(eig_val_list)

    if eig_vec_list[0] is None:
        eig_vecs = None
    else:
        eig_vecs = t.vstack(eig_vec_list)

    return eig_vals, eig_vecs


# TODO: relax nvmath import if not doing shift invert mode
def lobpcg(
    A: Float[SparseOperator, "m m"],
    M: Float[SparseOperator, "m m"] | None = None,
    block_diag_batch: bool = False,
    n: int | None = None,
    k: int = 6,
    eps: float | int = 1e-6,
    lobpcg_config: LOBPCGConfig | None = None,
    nvmath_config: DirectSolverConfig | None = None,
    precond_config: LOBPCGPrecondConfig | None = None,
) -> tuple[Float[t.Tensor, "*b k"], Float[t.Tensor, "m k"]]:
    """
    A custom implementation of LOBPCG.

    Note that this function requires `nvmath-python` for the shift-invert mode.
    In addition, some preconditioners have `nvmath-python` or `cupy` dependencies.

    This function implements a version of LOBPCG that's roughly equivalent to
    `torch.lobpcg(method='ortho')`, but with the following key differences:

    * This implementation is differentiable with respect to the `SparseOperator`s
      `A` and `M`. The `eps` argument is used for Lorentzian broadening/regularization
      in the gradient calculation to prevent gradient explosion when the eigenvalues
      are (near) degenerate.
    * The autograd through eigenvectors do not account for contributions from the
      unresolved eigenvectors.
    * This implementation supports shift-invert mode for both standard and
      generalized eigenvalue problems.
    * This implementation accepts specific preconditioners, including: identity,
      Jacobi, incomplete LU, and Cholesky; the latter two support diagonal dampling.
    * This implementation does not support explicit batch dimensions in `A` or `M`.
      If `block_diag_batch=True`, `A` and `M` will be split into individual sparse
      matrices and solved sequentially. This requires that `A` (and `M` if not `None`)
      has a valid `BlockDiagConfig` for unpacking the block diagonal batch structure.

    Notes on the `k` and `n` arguments:
    * in general, it is recommended to set the `n` argument somewhat higher than
      `k`, to make the convergence of the `k` desired eigenvalues faster and to
      account for possible degenerate eigenvalues.
    * This implementation employs a rank-adaptive orthonormalization strategy for
      the trial subspace. Unlike the standard PyTorch implementation, this allows
      the solver to handle cases where the size of A (`m`) is smaller than 3x the
      block size `n`. However, if the dimension of the trial subspace (which is
      <= 3n) is equal to or larger than `m`, the Rayleigh-Ritz projection becomes
      a full similarity transformation. In this limit, the algorithm effectively
      performs an exact diagonalization similar to `torch.linalg.eigh()`, but less
      efficiently due to the subspace construction and projection steps.
    """
    if not _HAS_NVMATH:
        raise ImportError("nvmath-python backends required.")

    from ..solvers.nvmath_wrapper import DirectSolverConfig
    from ._lobpcg_preconditioners import LOBPCGPrecondConfig

    if lobpcg_config is None:
        lobpcg_config = LOBPCGConfig()
    if precond_config is None:
        precond_config = LOBPCGPrecondConfig()
    if nvmath_config is None:
        nvmath_config = DirectSolverConfig()

    if block_diag_batch:
        A_list = A.unpack_block_diag()

    # Process raw LOBPCG config.
    if lobpcg_config.v0 is None:
        if n is None:
            n = k
        else:
            if n < k or n > A.size(-1):
                raise ValueError("n must be in the range [k, m].")

        if block_diag_batch:
            v0 = [
                t.randn(
                    (a.size(0), n),
                    generator=lobpcg_config.generator,
                    dtype=A.dtype,
                    device=A.device,
                )
                for a in A_list
            ]
        else:
            v0 = t.randn(
                (A.size(0), n),
                generator=lobpcg_config.generator,
                dtype=A.dtype,
                device=A.device,
            )

    else:
        v0 = lobpcg_config.v0

    tol = (
        t.finfo(A.dtype).eps ** 0.5 if lobpcg_config.tol is None else lobpcg_config.tol
    )

    processed_lobpcg_config = replace(lobpcg_config, v0=v0, tol=tol)

    if block_diag_batch:
        eig_vals, eig_vecs = _lobpcg_batch(
            A_list, M, k, eps, processed_lobpcg_config, precond_config, nvmath_config
        )
    else:
        eig_vals, eig_vecs = _lobpcg_no_batch(
            A, M, k, eps, processed_lobpcg_config, precond_config, nvmath_config
        )

    return eig_vals, eig_vecs

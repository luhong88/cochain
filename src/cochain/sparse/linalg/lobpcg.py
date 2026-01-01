from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any, Callable, Literal

import torch as t
from jaxtyping import Float, Integer

from ..operators import SparseOperator, SparseTopology
from ._eigsh_utils import dLdA_backward, dLdA_dLdM_backward

try:
    import nvmath.sparse.advanced as nvmath_sp
    from cuda.core.experimental import Device

    _HAS_NVMATH = True

except ImportError:
    _HAS_NVMATH = False

if TYPE_CHECKING:
    import nvmath.sparse.advanced as nvmath_sp
    from cuda.core.experimental import Device

    from .nvmath_wrapper import DirectSolverConfig


def _batched_csr_eye(
    n: int, b: int, val_dtype: t.dtype, idx_dtype: t.dtype, device: t.device
) -> Float[t.Tensor, "*b n n"]:
    if b == 0:
        identity = t.sparse_csr_tensor(
            crow_indices=t.arange(n + 1, dtype=idx_dtype, device=device),
            col_indices=t.arange(n, dtype=idx_dtype, device=device),
            values=t.ones(n, dtype=val_dtype, device=device),
        )
    else:
        identity = t.sparse_csr_tensor(
            crow_indices=t.tile(
                t.arange(n + 1, dtype=idx_dtype, device=device), (b, 1)
            ),
            col_indices=t.tile(t.arange(n, dtype=idx_dtype, device=device), (b, 1)),
            values=t.tile(t.ones(n, dtype=val_dtype, device=device), (b, 1)),
        )

    return identity


@dataclass
class LOBPCGConfig:
    X: Float[t.Tensor, "*b c k"] | None = None
    n: int | None = None
    niter: int | None = None
    tol: float | None = None
    largest: bool = True
    method: Literal["basic", "ortho"] = "ortho"
    tracker: Callable | None = None
    ortho_iparams: dict[str, Any] | None = None
    ortho_fparams: dict[str, Any] | None = None
    ortho_bparams: dict[str, Any] | None = None


class LOBPCGPreconditioner:
    def __init__(
        self,
        A_val: Float[t.Tensor, " nnz"],
        A_sp_topo: Integer[SparseTopology, "*b r c"],
        n: int,
        regularization: float | int,
        config: DirectSolverConfig,
    ):
        A_csr = SparseOperator(A_val, A_sp_topo).to_sparse_csr(int32=True)

        if A_sp_topo.n_batch_dim > 0:
            b_dummy = t.zeros(
                (A_csr.size(0), A_csr.size(-1), n),
                dtype=A_csr.dtype,
                device=A_csr.device,
            )

        else:
            b_dummy = t.zeros(
                (A_csr.size(-1), n), dtype=A_csr.dtype, device=A_csr.device
            )

        if regularization < 0:
            raise ValueError("The regularization constant must be nonnegative.")

        # a good heuristic is reg = 1e-4 * mean(diag(A))
        if regularization > 0:
            if A_sp_topo.n_batch_dim > 0:
                eye = _batched_csr_eye(
                    n=A_csr.size(-1),
                    b=A_csr.size(0),
                    val_dtype=A_val.dtype,
                    idx_dtype=t.int32,
                    device=A_csr.device,
                )
            else:
                eye = _batched_csr_eye(
                    n=A_csr.size(-1),
                    b=0,
                    val_dtype=A_val.dtype,
                    idx_dtype=t.int32,
                    device=A_csr.device,
                )

            # If A_csr uses int32 indices, the op should also be int32.
            op = A_csr + regularization * eye

        else:
            op = A_csr

        from .nvmath_wrapper import sp_literal_to_matrix_type

        # Prepare nvmath DirectSolver.
        config.options.sparse_system_type = sp_literal_to_matrix_type["symmetric"]

        # Do not give DirectSolver constructor the current stream to prevent
        # possible stream mismatch in subsequent solver calls; instead, pass the
        # torch/cupy stream to individual methods to ensure sync.
        self.solver = nvmath_sp.DirectSolver(
            op, b_dummy, options=config.options, execution=config.execution
        )

        # force blocking operation to make it memory-safe to potentially call
        # free() immediately after solve().
        self.solver.options.blocking = True

        # Amortize planning and factorization costs upfront in __init__()
        t_stream = t.cuda.current_stream()

        for k, v in config.plan_kwargs.items():
            setattr(self.solver.plan_config, k, v)
        self.solver.plan(stream=t_stream)

        for k, v in config.factorization_kwargs.items():
            setattr(self.solver.factorization_config, k, v)
        self.solver.factorize(stream=t_stream)

        for k, v in config.solution_kwargs.items():
            setattr(self.solver.solution_config, k, v)

    def __matmul__(self, res: Float[t.Tensor, "*b c n"]) -> Float[t.Tensor, "*b r n"]:
        t.cuda.set_device(res.device)
        Device(res.device.index).set_current()

        stream = t.cuda.current_stream()

        res_col_major = res.transpose(-1, -2).contiguous().transpose(-1, -2)

        self.solver.reset_operands(b=res_col_major, stream=stream)

        return self.solver.solve(stream=stream)

    def __del__(self):
        # DirectSolver needs an explicit free() step to free up memory/resources.
        if hasattr(self, "solver"):
            if hasattr(self.solver, "free"):
                self.solver.free()
                self.solver = None

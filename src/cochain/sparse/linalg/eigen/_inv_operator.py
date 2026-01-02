import nvmath.sparse.advanced as nvmath_sp
import torch as t
from jaxtyping import Float

from ..solvers.nvmath_wrapper import DirectSolverConfig, sp_literal_to_matrix_type


class BaseInvSymSpOp:
    def __init__(
        self,
        a: Float[t.Tensor, " m m"],
        b: Float[t.Tensor, "m n"],
        config: DirectSolverConfig,
    ):
        self.dtype = a.dtype
        self.shape = a.shape

        stream = t.cuda.current_stream()

        # Prepare nvmath DirectSolver.
        config.options.sparse_system_type = sp_literal_to_matrix_type["symmetric"]

        # Do not give DirectSolver constructor the current stream to prevent
        # possible stream mismatch in subsequent solver calls; instead, pass the
        # torch/cupy stream to individual methods to ensure sync.
        self.solver = nvmath_sp.DirectSolver(
            a, b, options=config.options, execution=config.execution
        )

        # force blocking operation to make it memory-safe to potentially call
        # free() immediately after solve().
        self.solver.options.blocking = True

        # Amortize planning and factorization costs upfront in __init__()
        for k, v in config.plan_kwargs.items():
            setattr(self.solver.plan_config, k, v)
        self.solver.plan(stream=stream)

        for k, v in config.factorization_kwargs.items():
            setattr(self.solver.factorization_config, k, v)
        self.solver.factorize(stream=stream)

        for k, v in config.solution_kwargs.items():
            setattr(self.solver.solution_config, k, v)

    def __del__(self):
        # DirectSolver needs an explicit free() step to free up memory/resources.
        if hasattr(self, "solver"):
            if hasattr(self.solver, "free"):
                self.solver.free()
                self.solver = None

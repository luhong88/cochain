from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Literal

import torch
from jaxtyping import Float, Integer
from torch import Tensor

from ....utils.parsing import to_col_major
from ...decoupled_tensor import SparseDecoupledTensor, SparsityPattern
from ._sparse_solver import SparseSolver

try:
    import nvmath.sparse.advanced as nvmath_sp

    _HAS_NVMATH = True

except ImportError:
    _HAS_NVMATH = False

if _HAS_NVMATH:
    try:
        from cuda.core import Device
    except ImportError:
        from cuda.core.experimental import Device


@dataclass
class DirectSolverConfig:
    """
    A dataclass for all nvmath `DirectSolver` configurations.

    Parameters
    ----------
    options
        Options for the direct sparse solver as a `DirectSolverOptions` object.
        This parameter will be pased as an argument of the same name to the
        `DirectSolver` constructor.
    execution
        Execution space options for the direct solver as a `ExecutionCUDA`
        or `ExecutionHybrid` object. This parameter will be passed as an argument
        of the same name to the `DirectSolver` constructor.
    plan_kwargs
        A dict for modifying attributes of `DirectSolver.plan_config`; the keys
        of this dict must match the attributes of `plan_config`.
    factorization_kwargs
        A dict for modifying attributes of `DirectSolver.factorization_config`;
        the keys of this dict must match the attributes of `factorization_config`.
    solution_kwargs
        A dict for modifying attributes of `DirectSolver.solution_config`; the keys
        of this dict must match the attributes of `solution_config`.

    Notes
    -----
    THe `DirectSolver` constructor allows for direct control of the CUDA execution
    stream, which is not allowed here to prevent potential stream mismatch during
    backward passes.
    """

    options: nvmath_sp.DirectSolverOptions | None = None
    execution: nvmath_sp.ExecutionCUDA | nvmath_sp.ExecutionHybrid | None = None

    plan_kwargs: dict[str, Any] = field(default_factory=dict)
    factorization_kwargs: dict[str, Any] = field(default_factory=dict)
    solution_kwargs: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not _HAS_NVMATH:
            warnings.warn(
                "A DirectSolverConfig is defined but the nvmath-python backend is not detected.",
                UserWarning,
            )


if _HAS_NVMATH:
    sp_literal_to_matrix_type = {
        "general": nvmath_sp.DirectSolverMatrixType.GENERAL,
        "symmetric": nvmath_sp.DirectSolverMatrixType.SYMMETRIC,
        "spd": nvmath_sp.DirectSolverMatrixType.SPD,
    }

    class NVMathSparseSolver(SparseSolver):
        def __init__(
            self,
            a: Float[SparseDecoupledTensor, "*b r c"],
            b: Float[Tensor, " r *ch"] | Float[Tensor, "b r *ch"],
            *,
            matrix_type: Literal["general", "symmetric", "spd"] = "general",
            config: DirectSolverConfig | None = None,
        ):
            if a.n_dense_dim > 0:
                raise ValueError("Dense dimension in 'a' is not supported.")

            if not a.pattern._is_int32_safe:
                warnings.warn(
                    "The sparse indices of the input tensor 'A' cannot be "
                    "safely cast to int32 dtype.",
                    UserWarning,
                )

            # Register the type, shape, dtype, and device of the input matrix.
            self.matrx_type = matrix_type
            self.dtype = a.dtype
            self.device = a.device
            self.shape = a.shape

            # Adjust solver config.
            if config is None:
                config = DirectSolverConfig()

            if config.options is None:
                config.options = nvmath_sp.DirectSolverOptions(
                    sparse_system_type=sp_literal_to_matrix_type[self.matrix_type]
                )
            else:
                config.options.sparse_system_type = sp_literal_to_matrix_type[
                    self.matrix_type
                ]

            self.config = config

            # Configure the matrix and vector inputs to the solver.
            self.a_val = a.values
            self.a_pattern = a.pattern

            a_csr = SparseDecoupledTensor(self.a_pattern, self.a_val).to_sparse_csr()

            # Put b in the correct shape and then the col-major layout.
            b_ready = self._ready_rhs(a.n_batch_dim, b)

            # Do not give DirectSolver constructor the current stream to prevent
            # possible stream mismatch in subsequent calls to the solver by other
            # methods; instead, pass the stream to individual solver methods to ensure
            # sync between pytorch and nvmath.
            solver = nvmath_sp.DirectSolver(
                a=a_csr, b=b_ready, options=config.options, execution=config.execution
            )

            # Execute the plan() and factorize() phase upfront and cache the solver.
            stream = torch.cuda.current_stream()

            for k, v in config.plan_kwargs.items():
                setattr(solver.plan_config, k, v)
            solver.plan(stream=stream)

            for k, v in config.factorization_kwargs.items():
                setattr(solver.factorization_config, k, v)
            solver.factorize(stream=stream)

            for k, v in config.solution_kwargs.items():
                setattr(solver.solution_config, k, v)
            self.solver = solver

        @staticmethod
        def _ready_rhs(
            n_batch_dim: int,
            b: Float[Tensor, " r *ch"] | Float[Tensor, "b r *ch"],
        ) -> Float[Tensor, " r *ch"] | Float[Tensor, "b r *ch"]:
            # Get the number of batch and non-channel dimensions in b and x.
            n_non_ch_dims = n_batch_dim + 1

            # Put b in the correct shape and then the col-major layout.
            if b.ndim > n_non_ch_dims:
                # If b has channel dimension(s), flatten them.
                b_flat = b.flatten(start_dim=n_non_ch_dims)
            elif n_batch_dim > 0:
                # If b has no channel dimension but a batch dimension, add in a trivial
                # channel dimension.
                b_flat = b.unsqueeze(-1)
            else:
                # Otherwise b is a 1D vector and no reshaping is required.
                b_flat = b

            b_ready = to_col_major(
                b_flat,
                batch_first=n_batch_dim > 0,
            )

            return b_ready

        def solve(
            self,
            b: Float[Tensor, " r *ch"] | Float[Tensor, "b r *ch"],
            trans: Literal["N", "T"] = "N",
        ) -> Float[Tensor, " c *ch"] | Float[Tensor, "b c *ch"]:
            stream = torch.cuda.current_stream()

            # Flatten the channel dims of b and then enforce col-major layout.
            b_ready = self._ready_rhs(self.a_pattern.n_batch_dim, b)

            if (self.matrix_type == "general" and trans == "N") or (
                self.matrix_type in ["symmetric", "spd"]
            ):
                self.solver.reset_operands(b=b, stream=stream)

            else:
                # If A is a generic matrix, explicitly compute its transpose and
                # update both the LHS and RHS of the solver. Since the LHS is
                # updated, we need to redo the plan() and factorize() steps. Note
                # that, currently, the DirectSolver class does not expose
                # a transpose mode option, which would have been preferred over
                # re-initializing the solver.
                a_T = SparseDecoupledTensor(
                    self.a_pattern, self.solver.a.tensor.values().flatten()
                ).to_sparse_csr_transposed()
                self.solver.reset_operands(a=a_T, b=b_ready, stream=stream)
                self.solver.plan(stream=stream)
                self.solver.factorize(stream=stream)

            x_flat = self.solver.solve(stream=stream)
            x_shaped = x_flat.view(-1, *b.shape[1:])

            return x_shaped

        def free(self):
            if hasattr(self, "solver") and self.solver is not None:
                if getattr(self.solver, "valid_state", False) and hasattr(
                    self.solver, "free"
                ):
                    # Force device sync before gc.
                    try:
                        import torch

                        if torch.cuda.is_initialized():
                            torch.cuda.synchronize(self.device)

                    except Exception:
                        pass

                    self.solver.free()
                    self.solver = None

        def __del__(self):
            self.free()

else:
    sp_literal_to_matrix_type = {}

    class NVMathSparseSolver:
        def __init__(self, *args, **kwargs):
            raise ImportError("nvmath-python backend required.")

from __future__ import annotations

__all__ = ["NVMathDirectSolver"]

from typing import TYPE_CHECKING, Literal

import torch
from jaxtyping import Float, Integer
from torch import Tensor
from torch.autograd.function import once_differentiable

from .....utils.parsing import to_col_major
from ....decoupled_tensor import SparseDecoupledTensor, SparsityPattern
from .._inv_sparse_operator import InvSparseOperator
from .nvmath_utils import (
    DirectSolverConfig,
    nvmath_adjoint_method,
    sp_literal_to_matrix_type,
)

try:
    import nvmath.sparse.advanced as nvmath_sp
    from cuda.core.experimental import Device

    _HAS_NVMATH = True

except ImportError:
    _HAS_NVMATH = False

if TYPE_CHECKING:
    import nvmath.sparse.advanced as nvmath_sp
    from cuda.core.experimental import Device


if _HAS_NVMATH:

    class _PersistentNvmathDirectSolverAutogradFunction(torch.autograd.Function):
        @staticmethod
        def forward(
            a_val: Float[Tensor, " nz"],
            a_pattern: Integer[SparsityPattern, "r c"],
            b: Float[Tensor, " r"] | Float[Tensor, " r *ch"],
            solver: nvmath_sp.DirectSolver,
        ) -> Float[Tensor, " c *ch"]:
            # The a_val and a_pattern are still required for gradient tracking
            # purposes, even though they are not used in the forward pass.
            stream = torch.cuda.current_stream()

            # For a linear system A@x=b, if only b gets resetted, there is no need
            # to call plan() and factorize() again. In addition, unlike the functional
            # wrapper, the memory layout of b is assumed to be already handled.
            solver.reset_operands(b=b, stream=stream)
            x = solver.solve(stream=stream)

            return x

        @staticmethod
        def setup_context(ctx, inputs, output):
            a_val, a_pattern, b, solver = inputs

            ctx.save_for_backward(output)  # the output is simply the x tensor.
            ctx.solver = solver
            ctx.a_pattern = a_pattern

        @staticmethod
        @once_differentiable
        def backward(
            ctx,
            dLdx: Float[Tensor, " c *ch"],
        ) -> tuple[
            Float[Tensor, " nz"] | None,
            None,
            Float[Tensor, " r *ch"] | None,
            None,
        ]:
            needs_grad_A_val = ctx.needs_input_grad[0]
            needs_grad_b = ctx.needs_input_grad[2]

            if not (needs_grad_A_val or needs_grad_b):
                return (None,) * 4

            solver: nvmath_sp.DirectSolver = ctx.solver

            (x,) = ctx.saved_tensors
            a_pattern: SparsityPattern = ctx.a_pattern

            if x.is_cuda:
                # torch calls backward() on a separate thread where nvmath's internal
                # cuda state is uninitialized. We must explicitly call Device(id).set_current()
                # to bind the active CUDA context to this thread's local state, ensuring
                # nvmath operations recognize the device.
                torch.cuda.set_device(x.device)
                Device(x.device.index).set_current()

            stream = torch.cuda.current_stream()

            dLdx_col_major: Float[Tensor, " c *ch"] = to_col_major(
                dLdx, batch_first=False
            )

            solver.reset_operands(b=dLdx_col_major, stream=stream)

            # Note that, unlike the functional wrapper, lambda_ already has the correct
            # shape that matches the shape of b.
            lambda_: Float[Tensor, " r *ch"] = solver.solve(stream=stream)

            if needs_grad_b and not needs_grad_A_val:
                return (None, None, lambda_, None)

            # dLdA will have the same sparsity pattern as A.
            dLdA_val = nvmath_adjoint_method(a_pattern, x, lambda_)

            if needs_grad_A_val and not needs_grad_b:
                return (dLdA_val, None, None, None)

            if needs_grad_A_val and needs_grad_b:
                return (dLdA_val, None, lambda_, None)

    class NVMathDirectSolver(InvSparseOperator):
        """
        "Stateful" differentiable wrapper for `nvmath.sparse.advanced.DirectSolver`.

        Given a sparse 2D matrix `a` and a vector `b`, this class computes and caches
        the LU factorization of `a` for the purpose of solving the linear system
        `a @ x = b` for `x`.

        Parameters
        ----------
        a : [r, c]
            A sparse 2D matrix represented as a `SparseDecoupledTensor`; no batch
            dimension is allowed.
        b : [r, *ch]
            A dummy RHS vector; no batch dimension is allowed but `b` can have arbitrary
            channel dimensions. This argument is required for the purpose of setting
            the size of the RHS vector; all subsequent calls to the solver must use
            RHS with the same shape as this tensor.
        sparse_system_type
            Whether the matrix `a` is symmetric or symmetric positive definite; note that
            this class does not support general 2D matrices. This argument will override
            the `options.sparse_system_type` attribute in the `config`.
        config
            Additional config arguments to the `DirectSolver`; see the `DirectSolverConfig`
            class for more information.

        Notes
        -----
        This class implements a similar nvmath `DirectSolver` wrapper as the functional
        `nvmath_direct_solver()`. Both wrappers cache the solver object and the factorized
        matrix for backward passes; however, in the functional wrapper, the cached solver
        is tied to the computation graph and gc'ed after the backward pass, whereas,
        in this class, the cached solver is tied to the lifecycle of the class instances,
        which allows the same factorized matrix to be re-used to solve different linear
        systems. See the `nvmath_direct_solver()` function for more details on the
        requirements and limitations of this wrapper.
        """

        def __init__(
            self,
            a: Float[SparseDecoupledTensor, "r c"],
            b: Float[Tensor, " r *ch"],
            *,
            sparse_system_type: Literal["symmetric", "spd"] = "symmetric",
            config: DirectSolverConfig | None = None,
        ):
            if not _HAS_NVMATH:
                raise ImportError("nvmath-python backend required.")

            if a.n_batch_dim > 0:
                raise ValueError("Batch dimension in 'a' is not supported.")
            if a.n_dense_dim > 0:
                raise ValueError("Dense dimension in 'a' is not supported.")
            if not a.pattern._is_int32_safe:
                raise ValueError(
                    "The sparse indices of the input tensor 'A' cannot be "
                    "safely cast to int32 dtype."
                )

            # Adjust solver config.
            if config is None:
                config = DirectSolverConfig()

            if config.options is None:
                config.options = nvmath_sp.DirectSolverOptions(
                    sparse_system_type=sp_literal_to_matrix_type[sparse_system_type]
                )
            else:
                config.options.sparse_system_type = sp_literal_to_matrix_type[
                    sparse_system_type
                ]

            if (
                config.options.sparse_system_type
                == nvmath_sp.DirectSolverMatrixType.GENERAL
            ):
                raise NotImplementedError(
                    "Non-symmetric matrices are not supported in the stateful nvmath DirectSolver wrapper."
                )

            self.config = config

            # Register the shape, dtype, and device of the input matrix.
            self.dtype = a.dtype
            self.device = a.device
            self.shape = a.shape

            # Configure the matrix and vector inputs to the solver.
            self.a_val = a.values
            self.a_pattern = a.pattern

            a_csr = SparseDecoupledTensor(self.a_pattern, self.a_val).to_sparse_csr()

            # Flatten the channel dims of b and then enforce col-major layout.
            if b.ndim > 1:
                b_flat = b.flatten(start_dim=1)
            else:
                b_flat = b

            b_ready = to_col_major(b_flat, batch_first=False)

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

        def __del__(self):
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

        def __call__(self, b: Float[Tensor, " r *ch"]) -> Float[Tensor, " c *ch"]:
            """
            Solve a sparse linear system with the given RHS vector using `DirectSolver`.

            Parameters
            ----------
            b : [r, *ch]
                The RHS vector as a dense tensor; `b` can have at most one batch
                dimension but arbitrary channel dimensions. Internally, the `DirectSolver`
                expects `b` to be in column-major memory layout (i.e., the `r` dimension
                has stride 1) and have at most one channel dimension. If the input
                tensor `b` does not conform to this requirement, a reshaped copy will
                be created; see the `nvmath_direct_solver()` function for more details.

            Returns
            -------
            [c, *ch]
                The unknown vector `x` with channel dimensions matching those of `b`.
            """
            # Flatten the channel dims of b and then enforce col-major layout.
            if b.ndim > 1:
                b_flat = b.flatten(start_dim=1)
            else:
                b_flat = b

            b_ready = to_col_major(b_flat, batch_first=False)

            x_flat = _PersistentNvmathDirectSolverAutogradFunction.apply(
                self.a_val, self.a_pattern, b_ready, self.solver
            )
            x_shaped = x_flat.view(-1, *b.shape[1:])

            return x_shaped

else:

    class NVMathDirectSolver:
        def __init__(self, *args, **kwargs):
            raise ImportError("nvmath-python backend required.")

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import torch
from jaxtyping import Float, Integer
from torch import Tensor
from torch.autograd.function import once_differentiable

from ....decoupled_tensor import SparseDecoupledTensor, SparsityPattern

try:
    import nvmath.sparse.advanced as nvmath_sp
    from cuda.core.experimental import Device

    from .nvmath_utils import (
        DirectSolverConfig,
        nvmath_adjoint_method,
        sp_literal_to_matrix_type,
    )

    _HAS_NVMATH = True

except ImportError:
    _HAS_NVMATH = False

if TYPE_CHECKING:
    import nvmath.sparse.advanced as nvmath_sp
    from cuda.core.experimental import Device

__all__ = ["NVMathDirectSolver"]


class _StatefulNvmathDirectSolverAutogradFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        A_val: Float[Tensor, " nnz"],
        A_pattern: Integer[SparsityPattern, "*b r c"],
        b: Float[Tensor, " r"] | Float[Tensor, "*b *ch r"],
        solver: nvmath_sp.DirectSolver,
    ) -> Float[Tensor, "*b c *ch"]:
        # The A_val and A_pattern are still required for gradient tracking
        # purposes, even though they are not used in the forward pass.
        if b.ndim > 1:
            b_col_major = b.contiguous().transpose(-1, -2)
        else:
            b_col_major = b.contiguous()

        stream = torch.cuda.current_stream()

        # For a linear system A@x=b, if only b gets resetted, there is no need
        # to call plan() and factorize() again.
        solver.reset_operands(b=b_col_major, stream=stream)
        x = solver.solve(stream=stream)

        return x

    @staticmethod
    def setup_context(ctx, inputs, output):
        A_val, A_pattern, b, solver = inputs

        ctx.save_for_backward(output)  # the output is simply the x tensor.
        ctx.solver = solver
        ctx.A_pattern = A_pattern

    @staticmethod
    @once_differentiable
    def backward(
        ctx,
        dLdx: Float[Tensor, "*b c *ch"],
    ) -> tuple[
        Float[Tensor, " nnz"] | None,
        None,
        Float[Tensor, "*b *ch r"] | None,
        None,
    ]:
        needs_grad_A_val = ctx.needs_input_grad[0]
        needs_grad_b = ctx.needs_input_grad[2]

        if not (needs_grad_A_val or needs_grad_b):
            return (None,) * 4

        solver: nvmath_sp.DirectSolver = ctx.solver

        (x,) = ctx.saved_tensors
        A_pattern: SparsityPattern = ctx.A_pattern

        if x.is_cuda:
            # torch calls backward() on a separate thread where nvmath's internal
            # cuda state is uninitialized. We must explicitly call Device(id).set_current()
            # to bind the active CUDA context to this thread's local state, ensuring
            # nvmath operations recognize the device.
            torch.cuda.set_device(x.device)
            Device(x.device.index).set_current()

        stream = torch.cuda.current_stream()

        if dLdx.ndim > 1:
            dLdx_col_major: Float[Tensor, "*b c *ch"] = (
                dLdx.transpose(-1, -2).contiguous().transpose(-1, -2)
            )
        else:
            dLdx_col_major = dLdx

        solver.reset_operands(b=dLdx_col_major, stream=stream)
        lambda_: Float[Tensor, "*b r *ch"] = solver.solve(stream=stream)

        if needs_grad_b and not needs_grad_A_val:
            return (None, None, lambda_, None)

        # dLdA will have the same sparsity pattern as A.
        dLdA_val = nvmath_adjoint_method(A_pattern, x, lambda_)

        if needs_grad_A_val and not needs_grad_b:
            return (dLdA_val, None, None, None)

        if needs_grad_A_val and needs_grad_b:
            return (dLdA_val, None, lambda_, None)


class NVMathDirectSolver:
    """
    A "statefull" version of the nvmath `DirectSolver` wrapper.

    This class implements a similar wrapper to the nvmath `DirectSolver` class as
    the functional `nvmath_direct_solver()`. Both wrappers cache the solver object
    and the factorized matrix for backward passes; however, in the functional
    wrapper, the cached solver is tied to the computation graph and gc'ed after
    the backward pass, whereas, in this class, the cached solver is tied to the
    lifecycle of the class instances, which allows the same factorized matrix to
    be re-used to solve different linear systems.

    See the `nvmath_direct_solver()` function for more details on the requirements
    and limitations of this wrapper. Note that, unlike the functional wrapper,
    this class only supports symmetric matrices.

    The `b` vector passed to the constructor is used for specifying the size
    of the RHS of the quation. The expected shape of b depends on whether the input
    matrix A has a batch dimension. If A has the shape (b, r, c), then the shape
    of b must be either (b, 1, r) or (b, ch, r). If A has the shape (r, c), then the
    shape of b must be either (r,) or (ch, r). Note that, once the solver has been
    initialized, all subsequent calls to the solver must use RHS with the same
    shape as b.
    """

    def __init__(
        self,
        A: Float[SparseDecoupledTensor, "r c"] | Float[SparseDecoupledTensor, "b r c"],
        b: Float[Tensor, "*ch r"] | Float[Tensor, "b ch r"],
        *,
        sparse_system_type: Literal["symmetric", "SPD"] = "symmetric",
        config: DirectSolverConfig | None = None,
    ):
        if not _HAS_NVMATH:
            raise ImportError("nvmath-python backend required.")

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

        # Configure the matrix and vector inputs to the solver.
        self.device = A.device
        self.A_val = A.val
        self.A_pattern = A.pattern

        A_csr = SparseDecoupledTensor(self.A_pattern, self.A_val).to_sparse_csr(
            int32=True
        )

        if b.ndim > 1:
            b_col_major = b.contiguous().transpose(-1, -2)
        else:
            b_col_major = b.contiguous()

        # Do not give DirectSolver constructor the current stream to prevent
        # possible stream mismatch in subsequent calls to the solver by other
        # methods; instead, pass the stream to individual solver methods to ensure
        # sync between pytorch and nvmath.
        solver = nvmath_sp.DirectSolver(
            a=A_csr, b=b_col_major, options=config.options, execution=config.execution
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
            if hasattr(self.solver, "free"):
                # Force device sync before gc.
                try:
                    import torch

                    if torch.cuda.is_initialized():
                        torch.cuda.synchronize(self.device)

                except Exception:
                    pass

                self.solver.free()

    def __call__(self, b: Float[Tensor, "*b *ch r"]) -> Float[Tensor, "*b c *ch"]:
        return _StatefulNvmathDirectSolverAutogradFunction.apply(
            self.A_val, self.A_pattern, b, self.solver
        )

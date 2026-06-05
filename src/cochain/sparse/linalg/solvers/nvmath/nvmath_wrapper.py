from __future__ import annotations

__all__ = ["nvmath_direct_solver"]

from typing import TYPE_CHECKING, Literal

import torch
from jaxtyping import Float, Integer
from torch import Tensor
from torch.autograd.function import once_differentiable

from .....utils.parsing import to_col_major
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


if _HAS_NVMATH:

    class AutogradDirectSolver(nvmath_sp.DirectSolver):
        """
        Autograd-compatible `DirectSolver` class.

        This class adapts `DirectSolver` for autograd by tying resource release
        to gc (`__del__`) rather than scope (`__exit__`).
        """

        def __init__(self, device, *args, **kwargs):
            self.device = device
            super().__init__(*args, **kwargs)

        def __del__(self):
            if getattr(self, "valid_state", False) and hasattr(self, "free"):
                # Force device sync before gc.
                try:
                    import torch

                    if torch.cuda.is_initialized():
                        torch.cuda.synchronize(self.device)

                except Exception:
                    pass

                self.free()


class _NvmathDirectSolverAutogradFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        a_val: Float[Tensor, " nz"],
        a_pattern: Integer[SparsityPattern, "*b r c"],
        b: Float[Tensor, " r *ch_flat"] | Float[Tensor, "b r *ch_flat"],
        config: DirectSolverConfig,
    ) -> tuple[
        Float[Tensor, " c *ch_flat"] | Float[Tensor, "b c *ch_flat"],
        AutogradDirectSolver,
    ]:
        A_csr = SparseDecoupledTensor(a_pattern, a_val).to_sparse_csr()

        stream = torch.cuda.current_stream()

        # Do not give DirectSolver constructor the current stream to prevent
        # possible stream mismatch in backward(); instead, pass the stream to
        # individual methods to ensure sync between pytorch and nvmath.
        solver = AutogradDirectSolver(
            a_val.device,
            A_csr,
            b,
            options=config.options,
            execution=config.execution,
        )

        for k, v in config.plan_kwargs.items():
            setattr(solver.plan_config, k, v)
        solver.plan(stream=stream)

        for k, v in config.factorization_kwargs.items():
            setattr(solver.factorization_config, k, v)
        solver.factorize(stream=stream)

        for k, v in config.solution_kwargs.items():
            setattr(solver.solution_config, k, v)
        x = solver.solve(stream=stream)

        return x, solver

    @staticmethod
    def setup_context(ctx, inputs, output):
        a_val, a_pattern, b, config = inputs

        x, solver = output

        ctx.save_for_backward(x)
        ctx.solver = solver
        ctx.a_pattern = a_pattern

    @staticmethod
    @once_differentiable
    def backward(
        ctx,
        dLdx: Float[Tensor, " c *ch_flat"] | Float[Tensor, "b c *ch_flat"],
        _,
    ) -> tuple[
        Float[Tensor, " nz"] | None,
        None,
        Float[Tensor, " r *ch_flat"] | Float[Tensor, "b r *ch_flat"] | None,
        None,
    ]:
        needs_grad_a_val = ctx.needs_input_grad[0]
        needs_grad_b = ctx.needs_input_grad[2]

        if not (needs_grad_a_val or needs_grad_b):
            return (None,) * 4

        if ctx.solver is None:
            raise RuntimeError(
                "Solver was released. Calling backward() twice with retain_graph=True "
                + "is currently not supported."
            )
        else:
            solver: AutogradDirectSolver = ctx.solver

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

        dLdx_col_major = to_col_major(dLdx, batch_first=a_pattern.n_batch_dim > 0)

        if (
            ctx.solver.options.sparse_system_type
            == nvmath_sp.DirectSolverMatrixType.GENERAL
        ):
            # If A is a generic matrix, explicitly compute its transpose and
            # update both the LHS and RHS of the solver for the adjoint method.
            # Since the LHS is updated, we need to redo the plan() and factorize()
            # steps. Note that, currently, the DirectSolver class does not expose
            # a transpose mode option, which would have been preferred over
            # re-initializing the solver.
            A_T = SparseDecoupledTensor(
                a_pattern, solver.a.tensor.values().flatten()
            ).to_sparse_csr_transposed()
            solver.reset_operands(a=A_T, b=dLdx_col_major, stream=stream)
            solver.plan(stream=stream)
            solver.factorize(stream=stream)

        else:
            # If A is symmetric, only update the RHS of the solver.
            solver.reset_operands(b=dLdx_col_major, stream=stream)

        # lambda_ has the same shape as b.
        lambda_ = solver.solve(stream=stream)

        if needs_grad_b and not needs_grad_a_val:
            return (None, None, lambda_, None)

        # dLdA will have the same sparsity pattern as A.
        dLda_val = nvmath_adjoint_method(a_pattern, x, lambda_)

        if needs_grad_a_val and not needs_grad_b:
            return (dLda_val, None, None, None)

        if needs_grad_a_val and needs_grad_b:
            return (dLda_val, None, lambda_, None)


def nvmath_direct_solver(
    a: Float[SparseDecoupledTensor, "*b r c"],
    b: Float[Tensor, " r *ch"] | Float[Tensor, "b r *ch"],
    *,
    sparse_system_type: Literal["general", "symmetric", "SPD"] = "general",
    config: DirectSolverConfig | None = None,
) -> Float[Tensor, " c *ch"] | Float[Tensor, "b c *ch"]:
    """
    "Stateless" differentiable wrapper for `nvmath.sparse.advanced.DirectSolver`.

    Given a sparse 2D matrix `a` and a vector `b`, solve the linear system
    `a @ x = b` for `x`.

    Parameters
    ----------
    a : [*b, r, c]
        A sparse 2D matrix represented as a `SparseDecoupledTensor`; at most
        one batch dimenion is supported.
    b : [*b, r, *ch]
        The RHS vector as a dense tensor; `b` can have at most one batch dimension
        but arbitrary channel dimensions.
    sparse_system_type
        Whether the matrix `a` is symmetric, symmetric positive definite, or
        a general 2D matrix. It is recommended to specify the `sparse_system_type`
        argument to exploit the symmetry and spectral properties of `a`. Note that,
        `sparse_system_type` will override the `options.sparse_system_type`
        attribute in the `config`.
    config
        Additional config arguments to the `DirectSolver`; see the `DirectSolverConfig`
        class for more information.

    Returns
    -------
    [*b, c, *ch]
        The unknwon vector `x`; the batch and channel dimensions, if there is any,
        match those of the input `b`.

    Notes
    -----
    The `DirectSolver` class supports batching/channel dimensions in both `a` and
    `b`. More specifically, this class supports the following four batching
    configurations:

    | Batch | Channel | `a.shape`   | `b.shape`     | `x.shape`     |
    |-------|---------|-------------|---------------|---------------|
    | False | False   | `[r, c]`    | `[r,]`        | `[c,]`        |
    | False | True    | `[r, c]`    | `[r, *ch]`    | `[c, *ch]`    |
    | True  | False   | `[b, r, c]` | `[b, r, (1)]` | `[b, c, (1)]` |
    | True  | True    | `[b, r, c]` | `[b, r, *ch]` | `[b, c, *ch]` |

    Note that, if batch dimensions are present, the `DirectSolver` class requires
    that `b` have a channel dimension, even if it is trivial (indicated by the
    `(1)` notation). In this function, the trivial channel dimensions are handled
    internally.

    When both `b` and `ch` dimensions are "active", the solver effectively
    solves `b*ch` linear systems of the form `a_i@x_ij = b_ij` for matrices `a_i`
    and vectors `x_ij` and `b_ij`, where `i` iterates over `b` and `j` iterates
    over the flattened channel dimensions.

    The `DirectSolver` class expects the `b` tensor to be in the column-major
    memory layout (i.e., the `r` dimension has stride 1) and will return an `x`
    tensor in the same layout (i.e., the `c` dimension has stride 1). By default,
    Pytorch construct tensors in row-major ordering; the conversion of `b` to
    column-major ordering is handled internally in this function.

    If either `A` or `b` requires gradient, then a `DirectSolver` object will be
    cached in memory to accelerate the backward pass; this memory will not be
    cleaned up until one of the following conditions is met:

    1) a backward() call with `retain_graph=False` has been made through the
    computation graph containing `A` or `b`, or
    2) all references to the output tensor (and its `grad_fn` and `ctx` attributes)
    from this function (and any derived tensors thereof) has gone out of scope/been
    detached from the computation graph.

    This function currently has the following limitations:

    * Double backward through this function is currently not supported.
    * `DirectSolver` also supports explicit batching, where a tensor with a batch
    dimension is represented as a list of tensors; this method is not supported
    in this function.
    * The `DirectSolver` class supports `a` as a batched sparse CSR tensor with an
    arbitrary number of batch dimensions, but this function only supports `a` with
    at most one batch dimension.
    * The sparse indices of `a` will be downcasted to `int32` for compatibility with
    the cuDSS backend.
    * If `a` is a general, non-symmetric matrix, the solver will need to redo
    the factorization step in `backward()` for the transposed matrix, because the
    `DirectSolver` class currently does not expose an option to solve the adjoint
    system directly.
    """
    if not _HAS_NVMATH:
        raise ImportError("nvmath-python backend required.")

    if a.n_dense_dim > 0:
        raise ValueError("Dense dimension in 'a' is not supported.")
    if not a.pattern._is_int32_safe:
        raise ValueError(
            "The sparse indices of the input tensor 'A' cannot be "
            "safely cast to int32 dtype."
        )

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

    # Get the number of batch and non-channel dimensions in b and x.
    n_non_ch_dims = a.n_batch_dim + 1

    # Put b in the correct shape and then the col-major layout.
    if b.ndim > n_non_ch_dims:
        # If b has channel dimension(s), flatten them.
        b_flat = b.flatten(start_dim=n_non_ch_dims)
    elif a.n_batch_dim > 0:
        # If b has no channel dimension but a batch dimension, add in a trivial
        # channel dimension.
        b_flat = b.unsqueeze(-1)
    else:
        # Otherwise b is a 1D vector and no reshaping is required.
        b_flat = b

    b_ready = to_col_major(
        b_flat,
        batch_first=a.n_batch_dim > 0,
    )

    x_flat, solver = _NvmathDirectSolverAutogradFunction.apply(
        a.values,
        a.pattern,
        b_ready,
        config,
    )

    # Restore the channel dims in the output.
    x = x_flat.reshape(*x_flat.shape[:n_non_ch_dims], *b.shape[n_non_ch_dims:])

    return x

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

import torch as t
from jaxtyping import Float, Integer
from torch.autograd.function import once_differentiable

from ...operators import SparseOperator, SparseTopology

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

    class AutogradDirectSolver(nvmath_sp.DirectSolver):
        """
        Adapts DirectSolver for Autograd by tying resource release to gc (__del__)
        rather than scope (__exit__).
        """

        def __del__(self):
            if hasattr(self, "free"):
                self.free()

    sp_literal_to_matrix_type = {
        "general": nvmath_sp.DirectSolverMatrixType.GENERAL,
        "symmetric": nvmath_sp.DirectSolverMatrixType.SYMMETRIC,
        "SPD": nvmath_sp.DirectSolverMatrixType.SPD,
    }

    @dataclass
    class DirectSolverConfig:
        """
        Encapsulates all nvmath DirectSolver configuration.

        The `options` and `execution` arguments are directly passed to the arguments
        of the same names to the `DirectSolver` constructor. Note that direct control
        of stream is not allowed to prevent potential stream mismatch during backward().
        Finer grained control over the `plan_config`, `factorization_config`, and
        `solution_config` attributes of the `DirectSolver` is also possible through
        `plan_kwargs`, `factorization_kwargs`, and `solution_kwargs`; the dicts
        passed to these arguments should contain specific attributes of `plan_config`,
        `factorization_config`, and `solution_config` as keys, respectively.
        """

        options: nvmath_sp.DirectSolverOptions | None = None
        execution: nvmath_sp.ExecutionCUDA | nvmath_sp.ExecutionHybrid | None = None

        plan_kwargs: dict[str, Any] = field(default_factory=dict)
        factorization_kwargs: dict[str, Any] = field(default_factory=dict)
        solution_kwargs: dict[str, Any] = field(default_factory=dict)


class _NvmathDirectSolverAutogradFunction(t.autograd.Function):
    @staticmethod
    def forward(
        A_val: Float[t.Tensor, " nnz"],
        A_sp_topo: Integer[SparseTopology, "*b r c"],
        b: Float[t.Tensor, " r"] | Float[t.Tensor, "*b *ch r"],
        config: DirectSolverConfig,
    ) -> tuple[Float[t.Tensor, "*b c *ch"], AutogradDirectSolver]:
        A_csr = SparseOperator(A_sp_topo, A_val).to_sparse_csr(int32=True)

        if b.ndim > 1:
            b_col_major = b.contiguous().transpose(-1, -2)
        else:
            b_col_major = b.contiguous()

        stream = t.cuda.current_stream()

        # Do not give DirectSolver constructor the current stream to prevent
        # possible stream mismatch in backward(); instead, pass the stream to
        # individual methods to ensure sync between pytorch and nvmath.
        solver = AutogradDirectSolver(
            A_csr,
            b_col_major,
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
        A_val, A_sp_topo, b, config = inputs

        x, solver = output

        ctx.save_for_backward(x)
        ctx.solver = solver
        ctx.A_sp_topo = A_sp_topo

    @staticmethod
    @once_differentiable
    def backward(
        ctx, dLdx: Float[t.Tensor, "*b c *ch"], _
    ) -> tuple[
        Float[t.Tensor, " nnz"] | None,
        None,
        Float[t.Tensor, "*b *ch r"] | None,
        None,
    ]:
        needs_grad_A_val = ctx.needs_input_grad[0]
        needs_grad_b = ctx.needs_input_grad[2]

        if not (needs_grad_A_val or needs_grad_b):
            return (None,) * 4

        if ctx.solver is None:
            raise RuntimeError(
                "Solver was released. Calling backward() twice with retain_graph=True "
                + "is currently not supported."
            )
        else:
            solver: AutogradDirectSolver = ctx.solver

        (x,) = ctx.saved_tensors
        A_sp_topo: SparseTopology = ctx.A_sp_topo

        if x.is_cuda:
            # torch calls backward() on a separate thread where nvmath's internal
            # cuda state is uninitialized. We must explicitly call Device(id).set_current()
            # to bind the active CUDA context to this thread's local state, ensuring
            # nvmath operations recognize the device.
            t.cuda.set_device(x.device)
            Device(x.device.index).set_current()

        stream = t.cuda.current_stream()

        if dLdx.ndim > 1:
            dLdx_col_major: Float[t.Tensor, "*b c *ch"] = (
                dLdx.transpose(-1, -2).contiguous().transpose(-1, -2)
            )
        else:
            dLdx_col_major = dLdx

        # force blocking operation to make it memory-safe to call free() immediately
        # after solve().
        solver.options.blocking = True

        if (
            ctx.solver.options.sparse_system_type
            == nvmath_sp.DirectSolverMatrixType.GENERAL
        ):
            # If A is a generic matrix, explicitly compute its transpose and
            # update both the LHS and RHS of the solver for the adjoint method.
            # Since the LHS is updated, we need to redo the plan() and factorize()
            # steps. Note that, currently, the DirectSolver class does not expose
            # a transpose mode option.
            A_T = SparseOperator(
                A_sp_topo, solver.a.tensor.values().flatten()
            ).to_sparse_csr_transposed(int32=True)
            solver.reset_operands(a=A_T, b=dLdx_col_major, stream=stream)
            solver.plan(stream=stream)
            solver.factorize(stream=stream)

        else:
            # If A is symmetric, only update the RHS of the solver.
            solver.reset_operands(b=dLdx_col_major, stream=stream)

        lambda_: Float[t.Tensor, "*b r *ch"] = solver.solve(stream=stream)

        # Free up solver memory usage.
        solver.free()
        ctx.solver = None

        if needs_grad_b and not needs_grad_A_val:
            return (None, None, lambda_, None)

        # dLdA will have the same sparsity pattern as A.
        if lambda_.ndim > 1:
            # Sum over the channel dimension while accounting for zero or more
            # batch dimensions.

            # Recall that A.sp_topo.idx_coo has shape (sp, nnz), where sp has size
            # equal to the number of batch dims plus 2 (i.e., sp = len(*b) + 2).
            n_batch = A_sp_topo.n_batch_dim

            # Extract the nonzero dLdA element row and col indices, accounting for
            # batch dimensions, and use them to extract the corresponding elements
            # from lambda_ and x to construct the nonzero outer product elements.
            r_idx: Integer[t.LongTensor, "n_batch+1 nnz"] = A_sp_topo.idx_coo[
                list(range(n_batch)) + [n_batch]
            ]
            c_idx: Integer[t.LongTensor, "n_batch+1 nnz"] = A_sp_topo.idx_coo[
                list(range(n_batch)) + [n_batch + 1]
            ]
            # Note that r_idx.unbind(0) is equivalent to *r_idx for indexing.
            dLdA_val = t.sum(-lambda_[r_idx.unbind(0)] * x[c_idx.unbind(0)], dim=-1)

        else:
            # if there are no batch dimensions, then the A_sp_topo.idx_coo is of
            # shape (sp=2, nnz).
            dLdA_val = -lambda_[A_sp_topo.idx_coo[0]] * x[A_sp_topo.idx_coo[1]]

        if needs_grad_A_val and not needs_grad_b:
            return (dLdA_val, None, None, None)

        if needs_grad_A_val and needs_grad_b:
            return (dLdA_val, None, lambda_, None)


def nvmath_direct_solver(
    A: Float[SparseOperator, "*b r c"],
    b: Float[t.Tensor, "*b *ch r"],
    *,
    sparse_system_type: Literal["general", "symmetric", "SPD"] = "general",
    config: DirectSolverConfig | None = None,
) -> Float[t.Tensor, "*b c *ch"]:
    """
    This function provides a differentiable wrapper for the `nvmath.sparse.advanced.DirectSolver`
    class in `nvmath-python` for solving sparse linear systems of the form `A@x=b`.

    Here, `A` is a SparseOperator and `b` is a dense tensor. The `DirectSolver`
    class supports batching in `A` and/or `b` tensors. In particular, there are
    four supported batching configurations:

    * No batching in either `A` or `b`; in this case `A` has shape `(r, c)`,
    `b` has shape `(r,)` and the output `x` has shape `(c,)`.

    * No batching in `A`, but `b` has a channel dimension `ch`; in this case, `A`
    has shape `(r, c)`, `b` has shape `(ch, r)`, and `x` has shape `(c, ch)`.

    * `A` and `b` has one matching batch dimension `b`; in this case, `A` has shape
    `(b, r, c)`, `b` has shape `(b, 1, r)`, and `x` has shape `(b, c, 1)`.

    * `A` and `b` has one matching batch dimension `b`, and the `b` tensor has a
    channel dimension `ch`; in this case, `A` has shape `(b, r, c)`, `b` has shape
    `(b, ch, r)`, and `x` has shape `(b, c, ch)`. When both `b` and `ch` dimensions
    are "active", the solver effectively solves `b*ch` linear systems of the form
    `A_i@x_ij = b_ij` for matrices `A_i` and vectors `x_ij` and `b_ij`, where `i`
    iterates over `b` and `j` iterates over `ch`.

    Note that, if `b` has more than one dimension, the `DirectSolver` object expects
    the `b` tensor to have the shape `(*b, r, ch)` where the stride of the `r`
    dimension is 1 (i.e., `b` is "column-major"). Because PyTorch, by default,
    construct tensors in row-major ordering, we achieve this by ensuring that `b`
    is contiguous in memory in the `(*b, ch, r)` shape, so that it can pass a view
    `(*b, r, ch)` to the solver with no copying. Similarly, the solver always returns
    a "column-major" tensor of shape `(*b, c, ch)`, where the stride of the `c`
    dimension is 1.

    While it is possible to call the `DirectSolver` using the default config,
    it is recommended to at least specify the `sparse_system_type` argument to
    exploit the symmetry and spectral properties of `A`. Note that, `sparse_system_type`
    will override the `options.sparse_system_type` attribute in the `config`.

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
    * The `DirectSolver` class supports `A` as a batched sparse CSR tensor with an
    arbitrary number of batch dimensions, but this function only supports A with
    at most one batch dimension.
    * The sparse indices of `A` will be downcasted to `int32` for compatibility with
    the cuDSS backend.
    * If `A` is a general, non-symmetric property, the solver will need to redo
    the factorization step in backward() for the transposed tensor, because the
    `DirectSolver` class currently does not expose an option to solve the adjoint
    system directly.
    """
    if not _HAS_NVMATH:
        raise ImportError("nvmath-python backend required.")

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

    x, solver = _NvmathDirectSolverAutogradFunction.apply(
        A.val,
        A.sp_topo,
        b,
        config,
    )

    return x

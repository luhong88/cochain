from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import torch as t
from jaxtyping import Float, Integer

try:
    import nvmath.sparse.advanced as nvmath_sp

    _HAS_NVMATH = True
except ImportError:
    _HAS_NVMATH = False

if TYPE_CHECKING:
    import nvmath.sparse.advanced as nvmath_sp


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


def _coalesced_coo_to_int32_csr(
    idx: Integer[t.LongTensor, "2 nnz"],
    val: Float[t.Tensor, " nnz"],
    shape: t.Size,
):
    """
    The input tensor must be coalesced.
    """
    rows, cols = shape

    row_idx = idx[0].to(t.int32)
    col_idx = idx[1].to(t.int32)

    # Compress row idx using bincount; minlength=rows accounts for empty rows
    counts = t.bincount(row_idx, minlength=rows)

    crow_indices = t.zeros(rows + 1, dtype=t.int32, device=idx.device)
    t.cumsum(counts, dim=0, out=crow_indices[1:])

    return t.sparse_csr_tensor(
        crow_indices,
        col_idx,
        val,
        size=shape,
    )


def _transpose_sp_csr(sp_csr: Float[t.Tensor, "r c"]) -> Float[t.Tensor, "c r"]:
    """
    Compute the transpose of a sparse csr matrix.
    """
    sp_csc = sp_csr.to_sparse_csc()

    return t.sparse_csr_tensor(
        crow_indices=sp_csc.ccol_indices(),
        col_indices=sp_csc.row_indices(),
        values=sp_csc.values(),
        size=(sp_csc.size(1), sp_csc.size(0)),
        device=sp_csc.device,
        dtype=sp_csc.dtype,
    )


class _NvmathDirectSolverWrapper(t.autograd.Function):
    @staticmethod
    def forward(
        A_val: Float[t.Tensor, " nnz"],
        A_coo_idx: Integer[t.LongTensor, "2 nnz"],
        A_shape: t.Size,
        b: Float[t.Tensor, " r"],
        solver_options: nvmath_sp.DirectSolverOptions | None,
        exec_space_options: nvmath_sp.ExecutionCUDA | nvmath_sp.ExecutionHybrid | None,
        plan_config: dict[str, Any],
        fac_config: dict[str, Any],
        sol_config: dict[str, Any],
    ) -> tuple[Float[t.Tensor, " c"], AutogradDirectSolver]:
        A_csr = _coalesced_coo_to_int32_csr(A_val, A_coo_idx, A_shape)

        stream = t.cuda.current_stream()

        # Do not give DirectSolver constructor the current stream to prevent
        # possible stream mismatch in backward(); instead, pass the stream to
        # individual methods to ensure sync between pytorch and nvmath.
        solver = AutogradDirectSolver(
            A_csr,
            b,
            options=solver_options,
            execution=exec_space_options,
        )

        for k, v in plan_config.items():
            setattr(solver.plan_config, k, v)
        solver.plan(stream=stream)

        for k, v in fac_config.items():
            setattr(solver.factorization_config, k, v)
        solver.factorize(stream=stream)

        for k, v in sol_config.items():
            setattr(solver.solution_config, k, v)
        x = solver.solve(stream=stream)

        return x, solver

    @staticmethod
    def setup_context(ctx, inputs, output):
        (
            A_val,
            A_coo_idx,
            A_shape,
            b,
            solver_options,
            exec_space_options,
            plan_config,
            fac_config,
            sol_config,
        ) = inputs

        x, solver = output
        ctx.mark_non_differentiable(solver)

        ctx.save_for_backward(A_coo_idx, x)
        ctx.solver = solver
        ctx.A_shape = A_shape

    @staticmethod
    def backward(
        ctx, dLdx: Float[t.Tensor, " c"], _
    ) -> tuple[
        Float[t.Tensor, " nnz"] | None,
        None,
        None,
        Float[t.Tensor, " r"] | None,
        None,
        None,
        None,
        None,
        None,
    ]:
        needs_grad_A_val = ctx.needs_input_grad[0]
        needs_grad_b = ctx.needs_input_grad[3]

        if not (needs_grad_A_val or needs_grad_b):
            return (None,) * 9

        A_coo_idx, x = ctx.saved_tensors

        if ctx.solver is None:
            raise RuntimeError(
                "Solver was released. Calling backward() twice with retain_graph=True "
                + "is currently not supported."
            )
        else:
            solver: AutogradDirectSolver = ctx.solver

        stream = t.cuda.current_stream()

        if (
            ctx.solver.options.sparse_system_type
            == nvmath_sp.DirectSolverMatrixType.GENERAL
        ):
            # If A is a generic matrix, explicitly compute its transpose and
            # update both the LHS and RHS of the solver for the adjoint method.
            # Since the LHS is updated, we need to redo the plan() and factorize()
            # steps.
            A_t = _transpose_sp_csr(solver.a)
            solver.reset_operands(a=A_t, b=dLdx, stream=stream)
            solver.plan(stream=stream)
            solver.factorize(stream=stream)

        else:
            # If A is symmetric, only update the RHS of the solver.
            solver.reset_operands(b=dLdx, stream=stream)

        lambda_: Float[t.Tensor, " r"] = solver.solve(stream=stream)

        # Free up solver memory usage.
        solver.free()
        ctx.solver = None

        if needs_grad_b and not needs_grad_A_val:
            return (None, None, None, lambda_, None, None, None, None, None)

        # dLdA will have the same sparsity pattern as A
        dLdA_val = -lambda_[A_coo_idx[0]] * x[A_coo_idx[1]]

        if needs_grad_A_val and not needs_grad_b:
            return (dLdA_val, None, None, None, None, None, None, None, None)

        if needs_grad_A_val and needs_grad_b:
            return (dLdA_val, None, None, lambda_, None, None, None, None, None)


# TODO: support RHS implicit batching
def nvmath(
    A: Float[t.Tensor, "r c"],
    b: Float[t.Tensor, " r"],
    *,
    sparse_system_type: Literal["general", "symmetric", "SPD"] = "general",
    options: nvmath_sp.DirectSolverOptions | None = None,
    execution: nvmath_sp.ExecutionCUDA | nvmath_sp.ExecutionHybrid | None = None,
    plan_config: dict[str, Any] | None = None,
    factorization_config: dict[str, Any] | None = None,
    solution_config: dict[str, Any] | None = None,
) -> Float[t.Tensor, " c"]:
    """
    This function provides a differentiable wrapper for the `nvmath.sparse.advanced.DirectSolver`
    class in nvmath-python.

    Here, A is a coalesced sparse coo tensor and b is a dense, 1D tensor.

    The `options` and `execution` arguments are directly passed to the arguments
    of the same names to the `DirectSolver` constructor. Note that direct control
    of stream is not allowed to prevent potential stream mismatch during backward().
    Finer grained control over the `plan_config`, `factorization_config`, and
    `solution_config` attributes of the `DirectSolver` is also possible through
    arguments of the same name to this function; the dicts passed to these arguments
    should contain specific attributes of `plan_config`, `factorization_config`,
    and `solution_config` as keys, respectively.

    While it is possible to call the `DirectSolver` using the default config,
    it is recommended to at least specify the `sparse_system_type` argument to
    exploit the symmetry and spectral properties of `A`. Note that, if both
    `sparse_system_type` and `options` arguments are specified, the `sparse_system_type`
    argument overrides the `options.sparse_system_type` attribute.

    If either `A` or `b` requires gradient, then a `DirectSolver` object will be
    cached in memory; this memory will not be cleaned up until one of the following
    conditions is met:

    1) a backward() call with `retain_graph=False` has been made through the
    computation graph containing `A` or `b`, or
    2) all references to the output tensor (and its `grad_fn` and `ctx` attributes)
    from this function (and any derived tensors thereof) has gone out of scope/been
    detached from the computation graph.

    This function currently has the following limitations:

    * Double backward through this function is not supported.
    * While batching of `A` and/or `b` is supported by `DirectSolver`, it is not
    supported in this function.
    * The sparse indices of `A` will be downcasted to `int32` for compatibility with
    the cuDSS backend.
    * If `A` is a general, non-symmetric property, the solver will need to redo
    the factorization step in backward() for the transposed tensor.
    * Unlike the SuperLU wrapper, the input `A` tensor must be coalesced. This is
    because a manual conversion is performed from the coo to the csr format with
    `int32` index tensors; this step assumes that `A` is coalesced to avoid expensive
    index sorting steps.
    """
    if not _HAS_NVMATH:
        raise ImportError("nvmath-python backend required.")

    if options is None:
        options = nvmath_sp.DirectSolverOptions(
            sparse_system_type=sp_literal_to_matrix_type[sparse_system_type]
        )
    else:
        options.sparse_system_type = sp_literal_to_matrix_type[sparse_system_type]

    if plan_config is None:
        plan_config = {}
    if factorization_config is None:
        factorization_config = {}
    if solution_config is None:
        solution_config = {}

    x, solver = _NvmathDirectSolverWrapper.apply(
        A.values(),
        A.indices(),
        A.shape,
        b,
        options,
        execution,
        plan_config,
        factorization_config,
        solution_config,
    )

    return x

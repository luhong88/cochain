from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

import torch as t
from jaxtyping import Float, Integer
from torch.autograd.function import once_differentiable

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


# TODO: consider caching the csr index tensors for performance
def _coalesced_coo_to_int32_csr(
    idx: Integer[t.LongTensor, "sp nnz"],
    val: Float[t.Tensor, " nnz"],
    shape: t.Size,
    strict_batch_nnz_check: bool = False,
):
    """
    Convert a coalesced, sparse coo tensor to a csr tensor with `int32` indices.

    Caveats:
    * Batching is supported, but only if there is a single batch dimension; i.e.,
    the input sparse coo tensor must either be of shape (r, c) or (b, r, c).
    * Although torch supports batched CSR format, it requires that all tensors
    in the batch have the same number of nonzero elements. This function will
    throw a ValueError() if this condition is not met.
    """
    match len(shape):
        case 1:
            raise ValueError(
                "The input sparse coo tensor must have at least two sparse dimensions."
            )

        case 2:
            rows, cols = shape

            row_idx = idx[0].to(t.int32)
            col_idx = idx[1].to(t.int32)

            # Compress row idx using bincount; minlength=rows accounts for empty rows
            counts = t.bincount(row_idx, minlength=rows)

            crow_idx = t.zeros(rows + 1, dtype=t.int32, device=idx.device)
            t.cumsum(counts, dim=0, out=crow_idx[1:])

            sp_csr_int32 = t.sparse_csr_tensor(
                crow_idx,
                col_idx,
                val,
                size=shape,
            )

            return sp_csr_int32

        case 3:
            batch, row, col = shape
            nnz = val.shape[0]

            # If the input tensor has equal nnz along the batch dimension, then
            # the nnz per tensor in the batch is given by nnz // batch.
            if nnz % batch != 0:
                raise ValueError(
                    f"Total nnz ({nnz}) is not divisible by batch size ({batch})."
                )

            nnz_per_batch = nnz // batch

            # It is possible for a tensor to have non-equal nnz along the batch
            # dimension but still satisfies nnz % batch = 0 (e.g., if the first
            # tensor has 6 nnz, while the second has 2). This optional (but somewhat
            # expensive) check rules out this possibility.
            if strict_batch_nnz_check:
                batch_idx = idx[0]
                batch_counts = t.bincount(batch_idx, minlength=batch)
                if not (batch_counts == nnz_per_batch).all():
                    raise ValueError("Batched CSR requires equal nnz per batch item.")

            row_idx_batched = idx[1].view(batch, nnz_per_batch).to(dtype=t.int32)

            # Compute histogram of row counts per batch. The scatter_add_() function
            # effectively counts how many times the k-th row idx shows up per batch,
            # and then put this count in the k-th position. The +1 is needed to
            # account for the fact that compressed row idx always starts at 0. The
            # cumsum() converts the row counts to compressed row idx.
            counts = t.zeros(batch, row + 1, dtype=t.int32, device=val.device)
            ones = t.tensor(1, dtype=t.int32, device=val.device).expand_as(
                row_idx_batched
            )
            counts.scatter_add_(1, row_idx_batched + 1, ones)
            crow_idx = counts.cumsum(dim=1, dtype=t.int32).contiguous()

            col_idx_batched = (
                idx[2].view(batch, nnz_per_batch).to(dtype=t.int32).contiguous()
            )

            csr_val = val.view(batch, nnz_per_batch).contiguous()

            sp_csr_int32 = t.sparse_csr_tensor(
                crow_idx,
                col_idx_batched,
                csr_val,
                size=shape,
            )

            return sp_csr_int32

        case _:
            raise NotImplementedError(
                "More than one batch dimensions is not supported."
            )


def _transpose_sp_csr(sp_csr: Float[t.Tensor, "*b r c"]) -> Float[t.Tensor, "*b c r"]:
    """
    Compute the transpose of a sparse csr matrix.
    """
    sp_csc = sp_csr.to_sparse_csc()

    if sp_csr.ndim == 2:
        transposed_size = (sp_csc.size(1), sp_csc.size(0))
    else:
        transposed_size = sp_csc.shape[:-2] + (sp_csc.shape[-1], sp_csc.shape[-2])

    return t.sparse_csr_tensor(
        crow_indices=sp_csc.ccol_indices(),
        col_indices=sp_csc.row_indices(),
        values=sp_csc.values(),
        size=transposed_size,
        device=sp_csc.device,
        dtype=sp_csc.dtype,
    )


class _NvmathDirectSolverWrapper(t.autograd.Function):
    @staticmethod
    def forward(
        A_val: Float[t.Tensor, " nnz"],
        A_coo_idx: Integer[t.LongTensor, "sp nnz"],
        A_shape: t.Size,
        b: Float[t.Tensor, " r"] | Float[t.Tensor, "*b *ch r"],
        config: DirectSolverConfig,
    ) -> tuple[Float[t.Tensor, "*b c *ch"], AutogradDirectSolver]:
        A_csr = _coalesced_coo_to_int32_csr(A_coo_idx, A_val, A_shape)

        if b.ndim > 1:
            b_col_major = b.contiguous().transpose(-1, -2)
        else:
            b_col_major = b

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
        (
            A_val,
            A_coo_idx,
            A_shape,
            b,
            config,
        ) = inputs

        x, solver = output

        ctx.save_for_backward(A_coo_idx, x)
        ctx.solver = solver
        ctx.A_shape = A_shape

    @staticmethod
    @once_differentiable
    def backward(
        ctx, dLdx: Float[t.Tensor, "*b c *ch"], _
    ) -> tuple[
        Float[t.Tensor, " nnz"] | None,
        None,
        None,
        Float[t.Tensor, "*b *ch r"] | None,
        None,
    ]:
        needs_grad_A_val = ctx.needs_input_grad[0]
        needs_grad_b = ctx.needs_input_grad[3]

        if not (needs_grad_A_val or needs_grad_b):
            return (None,) * 5

        A_coo_idx, x = ctx.saved_tensors

        if ctx.solver is None:
            raise RuntimeError(
                "Solver was released. Calling backward() twice with retain_graph=True "
                + "is currently not supported."
            )
        else:
            solver: AutogradDirectSolver = ctx.solver

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
            A_t = _transpose_sp_csr(solver.a)
            solver.reset_operands(a=A_t, b=dLdx_col_major, stream=stream)
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
            return (None, None, None, lambda_, None)

        # dLdA will have the same sparsity pattern as A.
        if lambda_.ndim > 1:
            # Sum over the channel dimension while accounting for zero or more
            # batch dimensions.

            # Recall that A_coo_idx has shape (sp, nnz), where sp has size equal
            # to the number of batch dims plus 2 (i.e., sp = len(*b) + 2).
            n_batch = A_coo_idx.size(0) - 2

            # Extract the nonzero dLdA element row and col indices, accounting for
            # batch dimensions, and use them to extract the corresponding elements
            # from lambda_ and x to construct the nonzero outer product elements.
            r_idx: Integer[t.LongTensor, "n_batch+1 nnz"] = A_coo_idx[
                list(range(n_batch)) + [n_batch]
            ]
            c_idx: Integer[t.LongTensor, "n_batch+1 nnz"] = A_coo_idx[
                list(range(n_batch)) + [n_batch + 1]
            ]
            # Note that r_idx.unbind(0) is equivalent to *r_idx for indexing.
            dLdA_val = t.sum(-lambda_[r_idx.unbind(0)] * x[c_idx.unbind(0)], dim=-1)

        else:
            # if there are no batch dimensions, then the A_coo_idx is of shape
            # (sp=2, nnz).
            dLdA_val = -lambda_[A_coo_idx[0]] * x[A_coo_idx[1]]

        if needs_grad_A_val and not needs_grad_b:
            return (dLdA_val, None, None, None, None)

        if needs_grad_A_val and needs_grad_b:
            return (dLdA_val, None, None, lambda_, None)


# TODO: cache CCO -> CSR mapping
# TODO: cache solver for each sparsity pattern
def nvmath_direct_solver(
    A: Float[t.Tensor, "*b r c"],
    b: Float[t.Tensor, "*b *ch r"],
    *,
    sparse_system_type: Literal["general", "symmetric", "SPD"] = "general",
    config: DirectSolverConfig | None = None,
) -> Float[t.Tensor, "*b c *ch"]:
    """
    This function provides a differentiable wrapper for the `nvmath.sparse.advanced.DirectSolver`
    class in `nvmath-python` for solving sparse linear systems of the form `A@x=b`.

    Here, `A` is a coalesced sparse coo tensor and `b` is a dense tensor. The
    `DirectSolver` class supports batching in `A` and/or `b` tensors. In particular,
    there are four supported batching configurations:

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

    x, solver = _NvmathDirectSolverWrapper.apply(
        A.values(),
        A.indices(),
        A.shape,
        b,
        config,
    )

    return x

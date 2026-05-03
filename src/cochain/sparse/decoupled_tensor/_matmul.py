import torch
from jaxtyping import Float, Integer
from torch import Tensor

from ._spsp_mm_plan import SpSpMMPlan
from .pattern import SparsityPattern


class FixedTopoSpDenseMM(torch.autograd.Function):
    @staticmethod
    def forward(
        a_val: Float[Tensor, " nz"],
        a_pattern: Integer[SparsityPattern, "i j"],
        b_dense: Float[Tensor, "j k"],
    ) -> Float[Tensor, "i k"]:
        # Forwad pass with sparse csr tensor.
        a_sp = torch.sparse_csr_tensor(
            a_pattern.idx_crow,
            a_pattern.idx_col,
            a_val,
            size=a_pattern.shape,
            device=a_val.device,
        )
        c_dense = torch.sparse.mm(a_sp, b_dense)

        return c_dense

    @staticmethod
    def setup_context(ctx, inputs, output):
        a_val, a_pattern, b_dense = inputs

        ctx.save_for_backward(a_val, b_dense)

        # It is okay to attach SparsityPattern to ctx since none of its index tensors
        # require gradient.
        ctx.a_pattern = a_pattern

    @staticmethod
    def backward(
        ctx, dLdC: Float[Tensor, "i k"]
    ) -> tuple[Float[Tensor, " nz"] | None, None, Float[Tensor, "j k"] | None]:
        a_val, b_dense = ctx.saved_tensors
        a_pattern: SparsityPattern = ctx.a_pattern

        dLdA_val = None
        dLdA_pattern = None
        dLdB = None

        needs_dLdA = ctx.needs_input_grad[0]
        needs_dLdB = ctx.needs_input_grad[-1]

        # For a matrix multiplication C = A@B,
        #   dLdA_ij = sum_k[dLdC_ik * B_jk]
        #   dLdB_ij = sum_k[dLdC_kj * A_ki]

        if needs_dLdA:
            dLdA_val = torch.einsum(
                "ik,ik->i", dLdC[a_pattern.idx_coo[0]], b_dense[a_pattern.idx_coo[1]]
            )

        if needs_dLdB:
            a_sp_T = torch.sparse_csr_tensor(
                a_pattern.idx_ccol,
                a_pattern.idx_row_csc,
                a_val[a_pattern.csc_to_coo_map],
                size=a_pattern.shape[::-1],
                device=a_val.device,
            )
            dLdB = torch.sparse.mm(a_sp_T, dLdC)

        return (dLdA_val, dLdA_pattern, dLdB)


class FixedTopoDenseSpMM(torch.autograd.Function):
    @staticmethod
    def forward(
        a_val: Float[Tensor, " nz"],
        a_pattern: Integer[SparsityPattern, "j k"],
        b_dense: Float[Tensor, "i j"],
    ) -> Float[Tensor, "i k"]:
        # Forwad pass with sparse csr tensor.
        a_sp_T = torch.sparse_csr_tensor(
            a_pattern.idx_ccol,
            a_pattern.idx_row_csc,
            a_val[a_pattern.csc_to_coo_map],
            size=a_pattern.shape[::-1],
            device=a_val.device,
        )
        c_dense = torch.sparse.mm(a_sp_T, b_dense.T).T

        return c_dense

    @staticmethod
    def setup_context(ctx, inputs, output):
        a_val, a_pattern, b_dense = inputs

        ctx.save_for_backward(a_val, b_dense)

        # It is okay to attach SparsityPattern to ctx since none of its index tensors
        # require gradient.
        ctx.a_pattern = a_pattern

    @staticmethod
    def backward(
        ctx, dLdC: Float[Tensor, "i k"]
    ) -> tuple[Float[Tensor, " nz"] | None, None, Float[Tensor, "i j"] | None]:
        a_val, b_dense = ctx.saved_tensors
        a_pattern: SparsityPattern = ctx.a_pattern

        dLdA_val = None
        dLdA_pattern = None
        dLdB = None

        needs_dLdA = ctx.needs_input_grad[0]
        needs_dLdB = ctx.needs_input_grad[-1]

        # For a matrix multiplication C = B@A,
        #   dLdA_ij = sum_k[dLdC_kj * B_ki]
        #   dLdB_ij = sum_k[dLdC_ik * A_jk]

        if needs_dLdA:
            dLdA_val = torch.einsum(
                "ki,ki->i",
                dLdC[:, a_pattern.idx_coo[1]],
                b_dense[:, a_pattern.idx_coo[0]],
            )

        if needs_dLdB:
            a_sp = torch.sparse_csr_tensor(
                a_pattern.idx_crow,
                a_pattern.idx_col,
                a_val,
                size=a_pattern.shape,
                device=a_val.device,
            )
            dLdB = torch.sparse.mm(a_sp, dLdC.T).T

        return (dLdA_val, dLdA_pattern, dLdB)


class FixedTopoSpSpMM(torch.autograd.Function):
    @staticmethod
    def forward(
        a_val: Float[Tensor, " a_nnz"],
        b_val: Float[Tensor, " b_nnz"],
        plan: SpSpMMPlan,
    ) -> tuple[
        Float[Tensor, " c_nnz"],
        Integer[Tensor, "2 c_nnz"],
        torch.Size,
    ]:
        c_val = torch.zeros(plan.fwd_plan.c_nnz, dtype=a_val.dtype, device=a_val.device)
        c_val.scatter_add_(
            dim=0,
            index=plan.fwd_plan.c_idx,
            src=a_val[plan.fwd_plan.a_idx] * b_val[plan.fwd_plan.b_idx],
        )

        return c_val, plan.fwd_plan.c_idx_coo, plan.fwd_plan.c_shape

    @staticmethod
    def setup_context(ctx, inputs, output):
        a_val, b_val, plan = inputs
        c_val, c_idx_coo, c_shape = output

        ctx.mark_non_differentiable(c_idx_coo)
        ctx.save_for_backward(a_val, b_val)

        # It is okay to attach SpSpMMPlan to ctx since none of its index tensors
        # require gradient.
        ctx.plan = plan

    @staticmethod
    def backward(
        ctx,
        dLdC_val: Float[Tensor, " c_nnz"],
        _1,
        _2,
    ) -> tuple[Float[Tensor, " a_nnz"] | None, Float[Tensor, " b_nnz"] | None, None]:
        a_val, b_val = ctx.saved_tensors
        plan: SpSpMMPlan = ctx.plan

        dLdA_val = None
        dLdB_val = None
        d_plan = None

        needs_dLdA = ctx.needs_input_grad[0]
        needs_dLdB = ctx.needs_input_grad[1]

        # For a matrix multiplication C = A@B,
        #
        #   dLdA_ij = sum_k[dLdC_ik * B_jk]
        #   dLdB_ij = sum_k[dLdC_kj * A_ki]
        #
        # A naive approach to computing these gradients might be to construct
        # the sparse tensors dLdC, B, and A, and perform the sparse matrix
        # multiplications. The problem with this approach is that, e.g., the
        # product of dLdC and B will tend to be much denser than dLdA (a phenomenon
        # called "fill in"); thus, materializing the full product is wasteful
        # (since the "extra" elements in the product that doesn't correspond to
        # a nonzero element in dLdA, which shares the same sparsity pattern as
        # A, will need to be discarded).

        if needs_dLdA:
            dLdA_val = torch.zeros(
                plan.bwd_plan_A.a_nnz, dtype=a_val.dtype, device=a_val.device
            )
            dLdA_val.scatter_add_(
                dim=0,
                index=plan.bwd_plan_A.dLdA_idx,
                src=dLdC_val[plan.bwd_plan_A.dLdC_idx] * b_val[plan.bwd_plan_A.b_idx],
            )

        if needs_dLdB:
            dLdB_val = torch.zeros(
                plan.bwd_plan_B.b_nnz, dtype=b_val.dtype, device=b_val.device
            )
            dLdB_val.scatter_add_(
                dim=0,
                index=plan.bwd_plan_B.dLdB_idx,
                src=dLdC_val[plan.bwd_plan_B.dLdC_idx] * a_val[plan.bwd_plan_B.a_idx],
            )

        return dLdA_val, dLdB_val, d_plan


class FixedTopoSpMV(torch.autograd.Function):
    @staticmethod
    def forward(
        a_val: Float[Tensor, " nz"],
        a_pattern: Integer[SparsityPattern, "i j"],
        b_dense: Float[Tensor, " j"],
    ) -> Float[Tensor, " i"]:
        # Forwad pass with sparse csr tensor.
        a_sp = torch.sparse_csr_tensor(
            a_pattern.idx_crow,
            a_pattern.idx_col,
            a_val,
            size=a_pattern.shape,
            device=a_val.device,
        )
        c_dense = torch.mv(a_sp, b_dense)

        return c_dense

    @staticmethod
    def setup_context(ctx, inputs, output):
        a_val, a_pattern, b_dense = inputs

        ctx.save_for_backward(a_val, b_dense)

        # It is okay to attach SparsityPattern to ctx since none of its index tensors
        # require gradient.
        ctx.a_pattern = a_pattern

    @staticmethod
    def backward(
        ctx, dLdc: Float[Tensor, " i"]
    ) -> tuple[Float[Tensor, " nz"] | None, None, Float[Tensor, " j"] | None]:
        a_val, b_dense = ctx.saved_tensors
        a_pattern: SparsityPattern = ctx.a_pattern

        dLdA_val = None
        dLdA_pattern = None
        dLdb = None

        needs_dLdA = ctx.needs_input_grad[0]
        needs_dLdb = ctx.needs_input_grad[-1]

        # For a matrix-vector multiplication c = A@b
        #   dLdA_ij = dLdc_i * b_j
        #   dLdb_i  = sum_k[dLdc_k * A_ki]

        if needs_dLdA:
            dLdA_val = dLdc[a_pattern.idx_coo[0]] * b_dense[a_pattern.idx_coo[1]]

        if needs_dLdb:
            # This is effectively a diagonal-sparse matmul, which is equivalent
            # to scaling the k-th row of A by dLdc_k.
            dLdb_val = a_val * dLdc[a_pattern.idx_coo[0]]
            dLdb = torch.zeros_like(b_dense)
            dLdb.index_add_(0, a_pattern.idx_coo[1], dLdb_val)

        return (dLdA_val, dLdA_pattern, dLdb)


class FixedTopoSpVM(torch.autograd.Function):
    @staticmethod
    def forward(
        a_val: Float[Tensor, " nz"],
        a_pattern: Integer[SparsityPattern, "i j"],
        b_dense: Float[Tensor, " i"],
    ) -> Float[Tensor, " j"]:
        # Forwad pass with sparse csr tensor.
        a_sp_T = torch.sparse_csr_tensor(
            a_pattern.idx_ccol,
            a_pattern.idx_row_csc,
            a_val[a_pattern.csc_to_coo_map],
            size=a_pattern.shape[::-1],
            device=a_val.device,
        )
        c_dense = torch.mv(a_sp_T, b_dense)

        return c_dense

    @staticmethod
    def setup_context(ctx, inputs, output):
        a_val, a_pattern, b_dense = inputs

        ctx.save_for_backward(a_val, b_dense)

        # It is okay to attach SparsityPattern to ctx since none of its index tensors
        # require gradient.
        ctx.a_pattern = a_pattern

    @staticmethod
    def backward(
        ctx, dLdc: Float[Tensor, " j"]
    ) -> tuple[Float[Tensor, " nz"] | None, None, Float[Tensor, " i"] | None]:
        a_val, b_dense = ctx.saved_tensors
        a_pattern: SparsityPattern = ctx.a_pattern

        dLdA_val = None
        dLdA_pattern = None
        dLdb = None

        needs_dLdA = ctx.needs_input_grad[0]
        needs_dLdb = ctx.needs_input_grad[-1]

        # For a vector_matrix multiplication c = b@A
        #   dLdA_ij = dLdc_j * b_i
        #   dLdb_i  = sum_k[dLdc_k * A_ik]

        if needs_dLdA:
            dLdA_val = dLdc[a_pattern.idx_coo[1]] * b_dense[a_pattern.idx_coo[0]]

        if needs_dLdb:
            # This is effectively a sparse-diagonal matmul, which is equivalent
            # to scaling the k-th col of A by dLdc_k.
            dLdb_val = a_val * dLdc[a_pattern.idx_coo[1]]
            dLdb = torch.zeros_like(b_dense)
            dLdb.index_add_(0, a_pattern.idx_coo[0], dLdb_val)

        return (dLdA_val, dLdA_pattern, dLdb)


def sp_dense_mm(
    a_val: Float[Tensor, " nz"],
    a_pattern: Integer[SparsityPattern, "i j"],
    b_dense: Float[Tensor, "j k"],
) -> Float[Tensor, "i k"]:
    """Sparse-dense 2D matrix multiplication with fixed sparsity autograd."""
    return FixedTopoSpDenseMM.apply(a_val, a_pattern, b_dense)


def dense_sp_mm(
    b_dense: Float[Tensor, "i j"],
    a_val: Float[Tensor, " nz"],
    a_pattern: Integer[SparsityPattern, "j k"],
) -> Float[Tensor, "i k"]:
    """Dense-sparse 2D matrix multiplication with fixed sparsity autograd."""
    return FixedTopoDenseSpMM.apply(a_val, a_pattern, b_dense)


def sp_sp_mm(
    a_val: Float[Tensor, " a_nnz"],
    b_val: Float[Tensor, " b_nnz"],
    spsp_mm_plan: SpSpMMPlan,
) -> tuple[
    Integer[Tensor, " c_nnz"],
    Integer[Tensor, "2 c_nnz"],
]:
    """Sparse-Sparse 2D matrix multiplication with fixed sparsity autograd."""
    return FixedTopoSpSpMM.apply(a_val, b_val, spsp_mm_plan)


def sp_mv(
    a_val: Float[Tensor, " nz"],
    a_pattern: Integer[SparsityPattern, "i j"],
    b_dense: Float[Tensor, " j"],
) -> Float[Tensor, " i"]:
    """Sparse 2D matrix-vector multiplication with fixed sparsity autograd."""
    return FixedTopoSpMV.apply(a_val, a_pattern, b_dense)


def sp_vm(
    b_dense: Float[Tensor, " i"],
    a_val: Float[Tensor, " nz"],
    a_pattern: Integer[SparsityPattern, "i j"],
) -> Float[Tensor, " j"]:
    """
    Sparse 2D vector-matrix multiplication with fixed sparsity autograd.

    The vector argument (b_dense) is interpreted as a row vector multiplying
    the matrix from the left.
    """
    return FixedTopoSpVM.apply(a_val, a_pattern, b_dense)


# Specialized functions for diagonal operators that do not require custom backward()


def diag_sp_mm(
    diag_val: Float[Tensor, " r"],
    sp_val: Float[Tensor, " nz"],
    pattern: Integer[SparsityPattern, "r c"],
) -> tuple[Float[Tensor, " nz"], Integer[SparsityPattern, "r c"]]:
    """
    Diagonal-sparse matrix multiplication.

    `D@A` scales the `i`th row of `A` by the `i`th element of `D`.
    """
    rows = pattern.idx_coo[0]
    scaled_vals = sp_val * diag_val[rows]
    return scaled_vals, pattern


def diag_dense_mm(
    diag_val: Float[Tensor, " r"],
    dense: Float[Tensor, "r c"],
) -> Float[Tensor, "r c"]:
    """
    Diagonal-dense matrix multiplication.

    `D@A` scales the `i`th row of `A` by the `i`th element of `D`.
    """
    return diag_val.view(-1, 1) * dense


def sp_diag_mm(
    sp_val: Float[Tensor, " nz"],
    pattern: Integer[SparsityPattern, "r c"],
    diag_val: Float[Tensor, " c"],
) -> tuple[Float[Tensor, " nz"], Integer[SparsityPattern, "r c"]]:
    """
    Sparse-diagonal matrix multiplication.

    `A@D` scales the `i`th col of `A` by the `i`th element of `D`.
    """
    cols = pattern.idx_coo[1]
    scaled_vals = sp_val * diag_val[cols]

    return scaled_vals, pattern


def dense_diag_mm(
    dense: Float[Tensor, "r c"],
    diag_val: Float[Tensor, " c"],
) -> Float[Tensor, "r c"]:
    """
    Dense-diagonal matrix multiplication.

    `A@D` scales the `i`th col of `A` by the `i`th element of `D`.
    """
    return dense * diag_val.view(1, -1)

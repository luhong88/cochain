import torch as t
from jaxtyping import Float, Integer

from ._sp_topo import SparseTopology
from ._utils import project_and_extract_cnz_vals


class _FixedTopoSpDenseMM(t.autograd.Function):
    @staticmethod
    def forward(
        a_val: Float[t.Tensor, " nnz"],
        a_sp_topo: Integer[SparseTopology, "i j"],
        b_dense: Float[t.Tensor, "j k"],
    ) -> Float[t.Tensor, "i k"]:
        # Forwad pass with sparse csr tensor.
        a_sp = t.sparse_csr_tensor(
            a_sp_topo.idx_crow,
            a_sp_topo.idx_col,
            a_val,
            size=a_sp_topo.shape,
            device=a_val.device,
        )
        c_dense = t.sparse.mm(a_sp, b_dense)

        return c_dense

    @staticmethod
    def setup_context(ctx, inputs, output):
        a_val, a_sp_topo, b_dense = inputs

        ctx.save_for_backward(a_val, b_dense)

        # It is okay to attach SparseTopology to ctx since none of its index tensors
        # require gradient.
        ctx.a_sp_topo = a_sp_topo

    @staticmethod
    def backward(
        ctx, dLdC: Float[t.Tensor, "i k"]
    ) -> tuple[Float[t.Tensor, " nnz"] | None, None, Float[t.Tensor, "j k"] | None]:
        a_val, b_dense = ctx.saved_tensors
        a_sp_topo: SparseTopology = ctx.a_sp_topo

        dLdA_val = None
        dLdA_sp_topo = None
        dLdB = None

        needs_dLdA = ctx.needs_input_grad[0]
        needs_dLdB = ctx.needs_input_grad[-1]

        # For a matrix multiplication C = A@B,
        #   dLdA_ij = sum_k[dLdC_ik * B_jk]
        #   dLdB_ij = sum_k[dLdC_kj * A_ki]

        if needs_dLdA:
            dLdA_val = t.einsum(
                "ik,ik->i", dLdC[a_sp_topo.idx_row], b_dense[a_sp_topo.idx_col]
            )

        if needs_dLdB:
            a_sp_T = t.sparse_csr_tensor(
                a_sp_topo.idx_ccol,
                a_sp_topo.idx_row,
                a_val[a_sp_topo.coo_to_csc_perm],
                size=a_sp_topo.shape[::-1],
                device=a_val.device,
            )
            dLdB = t.sparse.mm(a_sp_T, dLdC)

        return (dLdA_val, dLdA_sp_topo, dLdB)


class _FixedTopoDenseSpMM(t.autograd.Function):
    @staticmethod
    def forward(
        a_val: Float[t.Tensor, " nnz"],
        a_sp_topo: Integer[SparseTopology, "j k"],
        b_dense: Float[t.Tensor, "i j"],
    ) -> Float[t.Tensor, "i k"]:
        # Forwad pass with sparse csr tensor.
        a_sp_T = t.sparse_csr_tensor(
            a_sp_topo.idx_ccol,
            a_sp_topo.idx_row,
            a_val[a_sp_topo.coo_to_csc_perm],
            size=a_sp_topo.shape[::-1],
            device=a_val.device,
        )
        c_dense = t.sparse.mm(a_sp_T, b_dense.T).T

        return c_dense

    @staticmethod
    def setup_context(ctx, inputs, output):
        a_val, a_sp_topo, b_dense = inputs

        ctx.save_for_backward(a_val, b_dense)

        # It is okay to attach SparseTopology to ctx since none of its index tensors
        # require gradient.
        ctx.a_sp_topo = a_sp_topo

    @staticmethod
    def backward(
        ctx, dLdC: Float[t.Tensor, "i k"]
    ) -> tuple[Float[t.Tensor, " nnz"] | None, None, Float[t.Tensor, "i j"] | None]:
        a_val, b_dense = ctx.saved_tensors
        a_sp_topo: SparseTopology = ctx.a_sp_topo

        dLdA_val = None
        dLdA_sp_topo = None
        dLdB = None

        needs_dLdA = ctx.needs_input_grad[0]
        needs_dLdB = ctx.needs_input_grad[-1]

        # For a matrix multiplication C = B@A,
        #   dLdA_ij = sum_k[dLdC_kj * B_ki]
        #   dLdB_ij = sum_k[dLdC_ik * A_jk]

        if needs_dLdA:
            dLdA_val = t.einsum(
                "ki,ki->i", dLdC[:, a_sp_topo.idx_col], b_dense[:, a_sp_topo.idx_row]
            )

        if needs_dLdB:
            a_sp = t.sparse_csr_tensor(
                a_sp_topo.idx_crow,
                a_sp_topo.idx_col,
                a_val,
                size=a_sp_topo.shape,
                device=a_val.device,
            )
            dLdB = t.sparse.mm(a_sp, dLdC.T).T

        return (dLdA_val, dLdA_sp_topo, dLdB)


class _FixedTopoSpSpMM(t.autograd.Function):
    @staticmethod
    def forward(
        a_val: Float[t.Tensor, " a_nnz"],
        a_sp_topo: Integer[SparseTopology, "i j"],
        b_val: Float[t.Tensor, " b_nnz"],
        b_sp_topo: Integer[SparseTopology, "j k"],
    ) -> tuple[
        Integer[t.LongTensor, " c_nnz"],
        Integer[t.LongTensor, " c_nnz"],
        Float[t.Tensor, " c_nnz"],
        t.Size,
    ]:
        # Forwad pass with sparse csr tensor.
        a_sp = t.sparse_csr_tensor(
            a_sp_topo.idx_crow,
            a_sp_topo.idx_col,
            a_val,
            size=a_sp_topo.shape,
            device=a_val.device,
        )
        b_sp = t.sparse_csr_tensor(
            b_sp_topo.idx_crow,
            b_sp_topo.idx_col,
            b_val,
            size=b_sp_topo.shape,
            device=a_val.device,
        )
        c_sp = t.sparse.mm(a_sp, b_sp)

        return c_sp.crow_indices(), c_sp.col_indices(), c_sp.values(), c_sp.shape

    @staticmethod
    def setup_context(ctx, inputs, output):
        a_val, a_sp_topo, b_val, b_sp_topo = inputs
        c_crow_idx, c_col_idx, c_val, c_shape = output

        ctx.mark_non_differentiable(c_crow_idx, c_col_idx)
        ctx.save_for_backward(a_val, b_val, c_crow_idx, c_col_idx)

        # It is okay to attach SparseTopology to ctx since none of its index tensors
        # require gradient.
        ctx.a_sp_topo = a_sp_topo
        ctx.b_sp_topo = b_sp_topo
        ctx.c_shape = c_shape

    @staticmethod
    def backward(
        ctx,
        _1,
        _2,
        dLdC_val: Float[t.Tensor, " c_nnz"],
        _3,
    ) -> tuple[
        Float[t.Tensor, " a_nnz"] | None, None, Float[t.Tensor, " b_nnz"] | None, None
    ]:
        a_val, b_val, c_crow_idx, c_col_idx = ctx.saved_tensors

        a_sp_topo: SparseTopology = ctx.a_sp_topo
        b_sp_topo: SparseTopology = ctx.b_sp_topo
        c_shape = ctx.c_shape

        dLdA_val = None
        dLdA_sp_topo = None
        dLdB_val = None
        dLdB_sp_topo = None

        needs_dLdA = ctx.needs_input_grad[0]
        needs_dLdB = ctx.needs_input_grad[2]

        # For a matrix multiplication C = A@B,
        #   dLdA_ij = sum_k[dLdC_ik * B_jk]
        #   dLdB_ij = sum_k[dLdC_kj * A_ki]

        dLdC_sp = t.sparse_csr_tensor(
            c_crow_idx,
            c_col_idx,
            dLdC_val,
            size=c_shape,
            device=dLdC_val.device,
        )

        if needs_dLdA:
            b_sp_T = t.sparse_csr_tensor(
                b_sp_topo.idx_ccol,
                b_sp_topo.idx_row,
                b_val[b_sp_topo.coo_to_csc_perm],
                size=b_sp_topo.shape[::-1],
                device=b_val.device,
            )

            # csr -> coo conversion produces coalesced tensor.
            dLdA = t.sparse.mm(dLdC_sp, b_sp_T).to_sparse_coo()
            dLdA_val = project_and_extract_cnz_vals(
                src_coo=dLdA.indices(),
                src_val=dLdA.values(),
                target_coo=a_sp_topo.idx_coo,
                target_shape=a_sp_topo.shape,
            )

        if needs_dLdB:
            a_sp_T = t.sparse_csr_tensor(
                a_sp_topo.idx_ccol,
                a_sp_topo.idx_row,
                a_val[a_sp_topo.coo_to_csc_perm],
                size=a_sp_topo.shape[::-1],
                device=a_val.device,
            )

            dLdB = t.sparse.mm(a_sp_T, dLdC_sp).to_sparse_coo()
            dLdB_val = project_and_extract_cnz_vals(
                src_coo=dLdB.indices(),
                src_val=dLdB.values(),
                target_coo=b_sp_topo.idx_coo,
                target_shape=b_sp_topo.shape,
            )

        return (dLdA_val, dLdA_sp_topo, dLdB_val, dLdB_sp_topo)


class _FixedTopoSpMV(t.autograd.Function):
    @staticmethod
    def forward(
        a_val: Float[t.Tensor, " nnz"],
        a_sp_topo: Integer[SparseTopology, "i j"],
        b_dense: Float[t.Tensor, " j"],
    ) -> Float[t.Tensor, " i"]:
        # Forwad pass with sparse csr tensor.
        a_sp = t.sparse_csr_tensor(
            a_sp_topo.idx_crow,
            a_sp_topo.idx_col,
            a_val,
            size=a_sp_topo.shape,
            device=a_val.device,
        )
        c_dense = t.mv(a_sp, b_dense)

        return c_dense

    @staticmethod
    def setup_context(ctx, inputs, output):
        a_val, a_sp_topo, b_dense = inputs

        ctx.save_for_backward(a_val, b_dense)

        # It is okay to attach SparseTopology to ctx since none of its index tensors
        # require gradient.
        ctx.a_sp_topo = a_sp_topo

    @staticmethod
    def backward(
        ctx, dLdc: Float[t.Tensor, " i"]
    ) -> tuple[Float[t.Tensor, " nnz"] | None, None, Float[t.Tensor, " j"] | None]:
        a_val, b_dense = ctx.saved_tensors
        a_sp_topo: SparseTopology = ctx.a_sp_topo

        dLdA_val = None
        dLdA_sp_topo = None
        dLdb = None

        needs_dLdA = ctx.needs_input_grad[0]
        needs_dLdb = ctx.needs_input_grad[-1]

        # For a matrix-vector multiplication c = A@b
        #   dLdA_ij = dLdc_i * b_j
        #   dLdb_i  = sum_k[dLdc_k * A_ki]

        if needs_dLdA:
            dLdA_val = dLdc[a_sp_topo.idx_row] * b_dense[a_sp_topo.idx_col]

        if needs_dLdb:
            # This is effectively a diagonal-sparse matmul, which is equivalent
            # to scaling the k-th row of A by dLdc_k.
            dLdb_val = a_val * dLdc[a_sp_topo.idx_row]
            dLdb = t.zeros_like(b_dense)
            dLdb.index_add_(0, a_sp_topo.idx_col, dLdb_val)

        return (dLdA_val, dLdA_sp_topo, dLdb)


class _FixedTopoSpVM(t.autograd.Function):
    @staticmethod
    def forward(
        a_val: Float[t.Tensor, " nnz"],
        a_sp_topo: Integer[SparseTopology, "i j"],
        b_dense: Float[t.Tensor, " i"],
    ) -> Float[t.Tensor, " j"]:
        # Forwad pass with sparse csr tensor.
        a_sp_T = t.sparse_csr_tensor(
            a_sp_topo.idx_ccol,
            a_sp_topo.idx_row,
            a_val[a_sp_topo.coo_to_csc_perm],
            size=a_sp_topo.shape[::-1],
            device=a_val.device,
        )
        c_dense = t.mv(a_sp_T, b_dense)

        return c_dense

    @staticmethod
    def setup_context(ctx, inputs, output):
        a_val, a_sp_topo, b_dense = inputs

        ctx.save_for_backward(a_val, b_dense)

        # It is okay to attach SparseTopology to ctx since none of its index tensors
        # require gradient.
        ctx.a_sp_topo = a_sp_topo

    @staticmethod
    def backward(
        ctx, dLdc: Float[t.Tensor, " j"]
    ) -> tuple[Float[t.Tensor, " nnz"] | None, None, Float[t.Tensor, " i"] | None]:
        a_val, b_dense = ctx.saved_tensors
        a_sp_topo: SparseTopology = ctx.a_sp_topo

        dLdA_val = None
        dLdA_sp_topo = None
        dLdb = None

        needs_dLdA = ctx.needs_input_grad[0]
        needs_dLdb = ctx.needs_input_grad[-1]

        # For a vector_matrix multiplication c = b@A
        #   dLdA_ij = dLdc_j * b_i
        #   dLdb_i  = sum_k[dLdc_k * A_ik]

        if needs_dLdA:
            dLdA_val = dLdc[a_sp_topo.idx_col] * b_dense[a_sp_topo.idx_row]

        if needs_dLdb:
            # This is effectively a sparse-diagonal matmul, which is equivalent
            # to scaling the k-th col of A by dLdc_k.
            dLdb_val = a_val * dLdc[a_sp_topo.idx_col]
            dLdb = t.zeros_like(b_dense)
            dLdb.index_add_(0, a_sp_topo.idx_row, dLdb_val)

        return (dLdA_val, dLdA_sp_topo, dLdb)


def sp_dense_mm(
    a_val: Float[t.Tensor, " nnz"],
    a_sp_topo: Integer[SparseTopology, "i j"],
    b_dense: Float[t.Tensor, "j k"],
) -> Float[t.Tensor, "i k"]:
    return _FixedTopoSpDenseMM.apply(a_val, a_sp_topo, b_dense)


def dense_sp_mm(
    a_val: Float[t.Tensor, " nnz"],
    a_sp_topo: Integer[SparseTopology, "j k"],
    b_dense: Float[t.Tensor, "i j"],
) -> Float[t.Tensor, "i k"]:
    return _FixedTopoDenseSpMM.apply(a_val, a_sp_topo, b_dense)


def sp_sp_mm(
    a_val: Float[t.Tensor, " a_nnz"],
    a_sp_topo: Integer[SparseTopology, "i j"],
    b_val: Float[t.Tensor, " b_nnz"],
    b_sp_topo: Integer[SparseTopology, "j k"],
) -> tuple[
    Integer[t.LongTensor, " c_nnz"],
    Integer[t.LongTensor, " c_nnz"],
    Float[t.Tensor, " c_nnz"],
    t.Size,
]:
    return _FixedTopoSpSpMM.apply(a_val, a_sp_topo, b_val, b_sp_topo)


def sp_mv(
    a_val: Float[t.Tensor, " nnz"],
    a_sp_topo: Integer[SparseTopology, "i j"],
    b_dense: Float[t.Tensor, " j"],
) -> Float[t.Tensor, " i"]:
    return _FixedTopoSpMV.apply(a_val, a_sp_topo, b_dense)


def sp_vm(
    a_val: Float[t.Tensor, " nnz"],
    a_sp_topo: Integer[SparseTopology, "i j"],
    b_dense: Float[t.Tensor, " i"],
) -> Float[t.Tensor, " j"]:
    return _FixedTopoSpVM.apply(a_val, a_sp_topo, b_dense)

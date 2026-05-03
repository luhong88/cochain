import numba
import numpy as np
import numpy.typing as npt
import torch
from jaxtyping import Float, Integer
from torch import Tensor

from ._index import csr_to_csc
from .pattern import SparsityPattern


@numba.jit(nopython=True)
def _collect_dLdA_idx(
    a_idx_coo: npt.NDArray,
    c_idx_crow: npt.NDArray,
    c_idx_col: npt.NDArray,
    b_idx_crow: npt.NDArray,
    b_idx_col: npt.NDArray,
) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
    """
    Collect the indices required to compute the gradient of A.

    Consider the sparse matrix multiplication C = A @ B. The gradient of A is
    defined as

    dLdA_ij = sum_k[dLdC_ik * B_jk]

    This function determines a set of flat indices dLdA_idx, dLdC_idx, B_idx,
    such that

    dLdA_val.scatter_add_(
        dim = 0,
        index= dLdA_idx,
        src= dLdC_val[dLdC_idx] * B_val[B_idx]
    )

    gives the coalesced nonzero values of dLdA in COO/CSR layout.
    """
    idx_dtype = a_idx_coo.dtype

    # Determine the max buffer size. In the worst-case scenario, all elements
    # on a row of dLdC and a row of B contributes to a nonzero element in dLdA.
    # Therefore, the max buffer size should be the nnz of dLdA multiplied by
    # the row length of dLdC or B, whichever is longer.
    dLdA_nnz = a_idx_coo.shape[-1]

    max_c_row = np.max(c_idx_crow[1:] - c_idx_crow[:-1])
    max_b_row = np.max(b_idx_crow[1:] - b_idx_crow[:-1])
    max_row_size = max(max_c_row, max_b_row)

    max_n_idx = dLdA_nnz * max_row_size

    dLdA_idx = np.empty(max_n_idx, dtype=idx_dtype)
    dLdC_idx = np.empty(max_n_idx, dtype=idx_dtype)
    b_idx = np.empty(max_n_idx, dtype=idx_dtype)

    buffer_ptr = 0
    for dLdA_ptr in range(dLdA_nnz):
        # Find the row and col indices for the nonzero element in dLdA.
        i = a_idx_coo[0, dLdA_ptr]
        j = a_idx_coo[1, dLdA_ptr]

        # Find the start and end of the nonzero element indices in the ith row
        # of dLdC and the jth row of B (assuming CSR/COO layout).
        dLdC_row_i_start = c_idx_crow[i]
        dLdC_row_i_end = c_idx_crow[i + 1]

        b_row_j_start = b_idx_crow[j]
        b_row_j_end = b_idx_crow[j + 1]

        # Find the overlapping subset of indices between the column indices in the
        # ith row of dLdC and the column indices in the jth row of B, which
        # contributes to the matrix product, using the two pointers technique.
        dLdC_row_i_k_ptr = dLdC_row_i_start
        b_row_j_k_ptr = b_row_j_start
        while (dLdC_row_i_k_ptr < dLdC_row_i_end) and (b_row_j_k_ptr < b_row_j_end):
            k_dLdC_row_i = c_idx_col[dLdC_row_i_k_ptr]
            k_B_row_j = b_idx_col[b_row_j_k_ptr]

            if k_dLdC_row_i == k_B_row_j:
                # If a matching column index k is found, record the index of ik
                # from dLdC and jk from B, as well as the index of ij from dLdA.
                dLdA_idx[buffer_ptr] = dLdA_ptr
                dLdC_idx[buffer_ptr] = dLdC_row_i_k_ptr
                b_idx[buffer_ptr] = b_row_j_k_ptr

                buffer_ptr += 1
                dLdC_row_i_k_ptr += 1
                b_row_j_k_ptr += 1

            elif k_dLdC_row_i < k_B_row_j:
                dLdC_row_i_k_ptr += 1

            elif k_dLdC_row_i > k_B_row_j:
                b_row_j_k_ptr += 1

    return dLdA_idx[:buffer_ptr], dLdC_idx[:buffer_ptr], b_idx[:buffer_ptr]


@numba.jit(nopython=True)
def _collect_dLdB_idx(
    b_idx_coo: npt.NDArray,
    c_idx_ccol: npt.NDArray,
    c_idx_row: npt.NDArray,
    c_csc_to_coo_map: npt.NDArray,
    a_idx_ccol: npt.NDArray,
    a_idx_row: npt.NDArray,
    a_csc_to_coo_map: npt.NDArray,
) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
    """
    Collect the indices required to compute the gradient of B.

    Consider the sparse matrix multiplication C = A @ B. The gradient of A is
    defined as

    dLdB_ij = sum_k[dLdC_kj * A_ki]

    This function determines a set of flat indices dLdB_idx, dLdC_idx, A_idx,
    such that

    dLdB_val.scatter_add_(
        dim = 0,
        index= dLdB_idx,
        src= dLdC_val[dLdC_idx] * A_val[A_idx]
    )

    gives the coalesced nonzero values of dLdB in COO/CSR layout.
    """
    idx_dtype = b_idx_coo.dtype

    # Determine the max buffer size. In the worst-case scenario, all elements
    # on a row of dLdC and a col of A contributes to a nonzero element in dLdB.
    # Therefore, the max buffer size should be the nnz of dLdB multiplied by
    # the col length of dLdC or A, whichever is longer.
    dLdB_nnz = b_idx_coo.shape[-1]

    max_c_col = np.max(c_idx_ccol[1:] - c_idx_ccol[:-1])
    max_a_col = np.max(a_idx_ccol[1:] - a_idx_ccol[:-1])
    max_dim_size = max(max_c_col, max_a_col)

    max_n_idx = dLdB_nnz * max_dim_size

    dLdB_idx = np.empty(max_n_idx, dtype=idx_dtype)
    dLdC_idx = np.empty(max_n_idx, dtype=idx_dtype)
    a_idx = np.empty(max_n_idx, dtype=idx_dtype)

    buffer_ptr = 0
    for dLdB_ptr in range(dLdB_nnz):
        # Find the row and col indices for the nonzero element in dLdB.
        i = b_idx_coo[0, dLdB_ptr]
        j = b_idx_coo[1, dLdB_ptr]

        # Find the start and end of the nonzero element indices in the jth col
        # of dLdC and the ith col of B (assuming CSC layout).
        dLdC_col_j_start = c_idx_ccol[j]
        dLdC_col_j_end = c_idx_ccol[j + 1]

        a_col_i_start = a_idx_ccol[i]
        a_col_i_end = a_idx_ccol[i + 1]

        # Find the overlapping subset of indices between the row indices in the
        # jth col of dLdC and the row indices in the ith col of B, which
        # contributes to the matrix product, using the two pointers technique.
        dLdC_col_j_k_ptr = dLdC_col_j_start
        a_col_i_k_ptr = a_col_i_start
        while (dLdC_col_j_k_ptr < dLdC_col_j_end) and (a_col_i_k_ptr < a_col_i_end):
            k_dLdC_col_j = c_idx_row[dLdC_col_j_k_ptr]
            k_A_col_i = a_idx_row[a_col_i_k_ptr]

            if k_dLdC_col_j == k_A_col_i:
                # If a matching row index k is found, record the index of kj
                # from dLdC and ki from B, as well as the index of ij from dLdB.
                dLdB_idx[buffer_ptr] = dLdB_ptr
                dLdC_idx[buffer_ptr] = dLdC_col_j_k_ptr
                a_idx[buffer_ptr] = a_col_i_k_ptr

                buffer_ptr += 1
                dLdC_col_j_k_ptr += 1
                a_col_i_k_ptr += 1

            elif k_dLdC_col_j < k_A_col_i:
                dLdC_col_j_k_ptr += 1

            elif k_dLdC_col_j > k_A_col_i:
                a_col_i_k_ptr += 1

    # Unlike the _collect_dLdA_idx() case, here we work with the CSC indices of
    # dLdC and A, which means that the resulting indices assume that the nonzero
    # elements of dLdC and A are in the CSC layout. Therefore, the CSC -> COO
    # map is required to get back the nonzero element indices in the COO/CSR layout.
    return (
        dLdB_idx[:buffer_ptr],
        c_csc_to_coo_map[dLdC_idx[:buffer_ptr]],
        a_csc_to_coo_map[a_idx[:buffer_ptr]],
    )


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
        a_pattern: Integer[SparsityPattern, "i j"],
        b_val: Float[Tensor, " b_nnz"],
        b_pattern: Integer[SparsityPattern, "j k"],
    ) -> tuple[
        Integer[Tensor, " c_nnz"],
        Integer[Tensor, " c_nnz"],
        Float[Tensor, " c_nnz"],
        torch.Size,
    ]:
        # Forwad pass with sparse csr tensor.
        a_sp = torch.sparse_csr_tensor(
            a_pattern.idx_crow,
            a_pattern.idx_col,
            a_val,
            size=a_pattern.shape,
            device=a_val.device,
        )
        b_sp = torch.sparse_csr_tensor(
            b_pattern.idx_crow,
            b_pattern.idx_col,
            b_val,
            size=b_pattern.shape,
            device=a_val.device,
        )
        c_sp = torch.sparse.mm(a_sp, b_sp)

        return c_sp.crow_indices(), c_sp.col_indices(), c_sp.values(), c_sp.shape

    @staticmethod
    def setup_context(ctx, inputs, output):
        a_val, a_pattern, b_val, b_pattern = inputs
        c_crow_idx, c_col_idx, c_val, c_shape = output

        ctx.mark_non_differentiable(c_crow_idx, c_col_idx)
        ctx.save_for_backward(a_val, b_val, c_crow_idx, c_col_idx)

        # It is okay to attach SparsityPattern to ctx since none of its index tensors
        # require gradient.
        ctx.a_pattern = a_pattern
        ctx.b_pattern = b_pattern
        ctx.c_shape = c_shape

    @staticmethod
    def backward(
        ctx,
        _1,
        _2,
        dLdC_val: Float[Tensor, " c_nnz"],
        _3,
    ) -> tuple[
        Float[Tensor, " a_nnz"] | None, None, Float[Tensor, " b_nnz"] | None, None
    ]:
        idx_device = dLdC_val.device

        a_val, b_val, c_crow_idx, c_col_idx = ctx.saved_tensors

        a_pattern: SparsityPattern = ctx.a_pattern
        b_pattern: SparsityPattern = ctx.b_pattern
        c_shape = ctx.c_shape

        dLdA_val = None
        dLdA_pattern = None
        dLdB_val = None
        dLdB_pattern = None

        needs_dLdA = ctx.needs_input_grad[0]
        needs_dLdB = ctx.needs_input_grad[2]

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
            a_idx_coo_np = a_pattern.idx_coo.detach().cpu().numpy()
            c_idx_crow_np = c_crow_idx.detach().cpu().numpy()
            c_idx_col_np = c_col_idx.detach().cpu().numpy()
            b_idx_crow_np = b_pattern.idx_crow.detach().cpu().numpy()
            b_idx_col_np = b_pattern.idx_col.detach().cpu().numpy()

            dLdA_idx_np, dLdC_idx_np, b_idx_np = _collect_dLdA_idx(
                a_idx_coo_np,
                c_idx_crow_np,
                c_idx_col_np,
                b_idx_crow_np,
                b_idx_col_np,
            )

            dLdA_idx = torch.from_numpy(dLdA_idx_np).to(idx_device)
            dLdC_idx = torch.from_numpy(dLdC_idx_np).to(idx_device)
            b_idx = torch.from_numpy(b_idx_np).to(idx_device)

            dLdA_val = torch.zeros(
                a_pattern._nnz(), dtype=a_val.dtype, device=a_val.device
            )
            dLdA_val.scatter_add_(
                dim=0, index=dLdA_idx, src=dLdC_val[dLdC_idx] * b_val[b_idx]
            )

        if needs_dLdB:
            c_idx_ccol, c_idx_row_csc, c_csc_to_coo_map = csr_to_csc(
                idx_crow=c_crow_idx, idx_col=c_col_idx, n_cols=c_shape[-1]
            )

            b_idx_coo_np = b_pattern.idx_coo.detach().cpu().numpy()
            c_idx_ccol_np = c_idx_ccol.detach().cpu().numpy()
            c_idx_row_np = c_idx_row_csc.detach().cpu().numpy()
            c_csc_to_coo_map_np = c_csc_to_coo_map.detach().cpu().numpy()
            a_idx_ccol_np = a_pattern.idx_ccol.detach().cpu().numpy()
            a_idx_row_np = a_pattern.idx_row_csc.detach().cpu().numpy()
            a_csc_to_coo_map_np = a_pattern.csc_to_coo_map.detach().cpu().numpy()

            dLdB_idx_np, dLdC_idx_np, a_idx_np = _collect_dLdB_idx(
                b_idx_coo_np,
                c_idx_ccol_np,
                c_idx_row_np,
                c_csc_to_coo_map_np,
                a_idx_ccol_np,
                a_idx_row_np,
                a_csc_to_coo_map_np,
            )

            dLdB_idx = torch.from_numpy(dLdB_idx_np).to(idx_device)
            dLdC_idx = torch.from_numpy(dLdC_idx_np).to(idx_device)
            a_idx = torch.from_numpy(a_idx_np).to(idx_device)

            dLdB_val = torch.zeros(
                b_pattern._nnz(), dtype=b_val.dtype, device=b_val.device
            )
            dLdB_val.scatter_add_(
                dim=0, index=dLdB_idx, src=dLdC_val[dLdC_idx] * a_val[a_idx]
            )

        return (dLdA_val, dLdA_pattern, dLdB_val, dLdB_pattern)


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
    a_pattern: Integer[SparsityPattern, "i j"],
    b_val: Float[Tensor, " b_nnz"],
    b_pattern: Integer[SparsityPattern, "j k"],
) -> tuple[
    Integer[Tensor, " c_nnz"],
    Integer[Tensor, " c_nnz"],
    Float[Tensor, " c_nnz"],
    torch.Size,
]:
    """Sparse-Sparse 2D matrix multiplication with fixed sparsity autograd."""
    return FixedTopoSpSpMM.apply(a_val, a_pattern, b_val, b_pattern)


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

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numba
import numpy as np
import numpy.typing as npt
import scipy.sparse
import torch
from jaxtyping import Int64, Integer
from torch import Tensor

try:
    import cupy as cp
    import cupyx.scipy.sparse as cp_sp

    _HAS_CUPY = True

except ImportError:
    _HAS_CUPY = False

if TYPE_CHECKING:
    import cupy as cp
    import cupyx.scipy.sparse as cp_sp


from ...utils.parsing import to_np


@dataclass
class SpSpMMFwdPlan:
    c_nnz: int
    c_idx_coo: Int64[Tensor, " 2 c_nz"]
    c_idx_crow: Integer[Tensor, " c_r+1"]
    c_idx_col: Integer[Tensor, " c_nz"]
    c_shape: torch.Size
    c_idx: Int64[Tensor, " c_idx"]
    a_idx: Int64[Tensor, " a_idx"]
    b_idx: Int64[Tensor, " b_idx"]

    def to(self, *args, **kwargs) -> "SpSpMMFwdPlan":
        return SpSpMMFwdPlan(
            self.c_nnz,
            self.c_idx_coo.to(*args, **kwargs),
            self.c_idx_crow.to(*args, **kwargs),
            self.c_idx_col.to(*args, **kwargs),
            self.c_shape,
            self.c_idx.to(*args, **kwargs),
            self.a_idx.to(*args, **kwargs),
            self.b_idx.to(*args, **kwargs),
        )


@dataclass
class SpSpMMBwdPlanA:
    a_nnz: int
    dLdA_idx: Int64[Tensor, " a_idx"]
    dLdC_idx: Int64[Tensor, " c_idx"]
    b_idx: Int64[Tensor, " b_idx"]

    def to(self, *args, **kwargs) -> "SpSpMMBwdPlanA":
        return SpSpMMBwdPlanA(
            self.a_nnz,
            self.dLdA_idx.to(*args, **kwargs),
            self.dLdC_idx.to(*args, **kwargs),
            self.b_idx.to(*args, **kwargs),
        )


@dataclass
class SpSpMMBwdPlanB:
    b_nnz: int
    dLdB_idx: Int64[Tensor, " b_idx"]
    dLdC_idx: Int64[Tensor, " c_idx"]
    a_idx: Int64[Tensor, " a_idx"]

    def to(self, *args, **kwargs) -> "SpSpMMBwdPlanB":
        return SpSpMMBwdPlanB(
            self.b_nnz,
            self.dLdB_idx.to(*args, **kwargs),
            self.dLdC_idx.to(*args, **kwargs),
            self.a_idx.to(*args, **kwargs),
        )


@dataclass
class SpSpMMPlan:
    fwd_plan: SpSpMMFwdPlan
    bwd_plan_A: SpSpMMBwdPlanA | None
    bwd_plan_B: SpSpMMBwdPlanB | None

    def to(self, *args, **kwargs) -> "SpSpMMPlan":
        return SpSpMMPlan(
            self.fwd_plan.to(*args, **kwargs),
            self.bwd_plan_A.to(*args, **kwargs) if self.bwd_plan_A else None,
            self.bwd_plan_B.to(*args, **kwargs) if self.bwd_plan_B else None,
        )


def _csr_to_csc(
    idx_crow: Integer[Tensor, " r+1"],
    idx_col: Integer[Tensor, " nz"],
    n_cols: int,
) -> tuple[
    Integer[Tensor, " c+1"],
    Integer[Tensor, " nz"],
    Int64[Tensor, " nz"],
]:
    """
    Convert CSR index tensors to CSC and return the CSR->CSC mapping.

    Note that this function only works on 2D sparse tensors with no batch dimensions.
    The output inverse permutation tensor is always of dtype `int64`, while the
    output `idx_ccol` and `idx_row_csc` tensors follow the dtype of `idx_crow`.
    All output tensors are guaranteed to be contiguous.
    """
    # Compute the forward permutation (CSR -> CSC)
    csc_to_coo_map = torch.argsort(idx_col, dim=0, stable=True)

    # Compute CSC ccol index
    idx_ccol = torch.zeros(n_cols + 1, dtype=idx_crow.dtype, device=idx_crow.device)
    idx_ccol[1:] = torch.bincount(idx_col, minlength=n_cols).cumsum(dim=0)

    # Compute CSC row index
    n_rows = idx_crow.size(0) - 1
    row_idx = torch.arange(n_rows, dtype=idx_crow.dtype, device=idx_crow.device)
    idx_row_csr = row_idx.repeat_interleave(idx_crow[1:] - idx_crow[:-1])

    # Permute the uncompressed CSR row index into CSC order
    idx_row_csc = idx_row_csr[csc_to_coo_map]

    return idx_ccol.contiguous(), idx_row_csc.contiguous(), csc_to_coo_map.contiguous()


@numba.jit(nopython=True)
def _collect_dLdA_idx(
    a_idx_coo: npt.NDArray,
    c_idx_crow: npt.NDArray,
    c_idx_col: npt.NDArray,
    b_idx_crow: npt.NDArray,
    b_idx_col: npt.NDArray,
) -> tuple[int, npt.NDArray, npt.NDArray, npt.NDArray]:
    """
    Collect the indices required to compute the gradient of A over A@B matmul.

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

    # Determine the nnz of dLdA and exit early if dLdA is empty.
    dLdA_nnz = a_idx_coo.shape[-1]

    if dLdA_nnz == 0:
        return (
            0,
            np.empty(0, dtype=idx_dtype),
            np.empty(0, dtype=idx_dtype),
            np.empty(0, dtype=idx_dtype),
        )

    # Determine the max buffer size. In the worst-case scenario, all elements
    # on a row of dLdC and a row of B contributes to a nonzero element in dLdA.
    # Therefore, the max buffer size should be the nnz of dLdA multiplied by
    # the row length of dLdC or B, whichever is longer.

    # Note that the np.max() call needs to be guarded; if a tensor has no rows,
    # then a_idx_crow = [0] and a_idx_crow[1:] - a_idx_crow[:-1] gives an empty
    # array.
    max_c_row = 0
    if c_idx_crow.shape[0] > 1:
        max_c_row = np.max(c_idx_crow[1:] - c_idx_crow[:-1])

    max_b_row = 0
    if b_idx_crow.shape[0] > 1:
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

    return (
        dLdA_nnz,
        dLdA_idx[:buffer_ptr],
        dLdC_idx[:buffer_ptr],
        b_idx[:buffer_ptr],
    )


@numba.jit(nopython=True)
def _collect_dLdB_idx(
    b_idx_coo: npt.NDArray,
    c_idx_ccol: npt.NDArray,
    c_idx_row: npt.NDArray,
    c_csc_to_coo_map: npt.NDArray,
    a_idx_ccol: npt.NDArray,
    a_idx_row: npt.NDArray,
    a_csc_to_coo_map: npt.NDArray,
) -> tuple[int, npt.NDArray, npt.NDArray, npt.NDArray]:
    """
    Collect the indices required to compute the gradient of B over A@B matmul.

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

    # Determine the nnz of dLdB and exit early if dLdB is empty.
    dLdB_nnz = b_idx_coo.shape[-1]
    if dLdB_nnz == 0:
        return (
            0,
            np.empty(0, dtype=idx_dtype),
            np.empty(0, dtype=idx_dtype),
            np.empty(0, dtype=idx_dtype),
        )

    # Determine the max buffer size. In the worst-case scenario, all elements
    # on a row of dLdC and a col of A contributes to a nonzero element in dLdB.
    # Therefore, the max buffer size should be the nnz of dLdB multiplied by
    # the col length of dLdC or A, whichever is longer.

    # Note that the np.max() call needs to be guarded; if a tensor has no cols,
    # then c_idx_ccol = [0] and c_idx_ccol[1:] - c_idx_ccol[:-1] gives an empty
    # array.
    max_c_col = 0
    if c_idx_ccol.shape[0] > 1:
        max_c_col = np.max(c_idx_ccol[1:] - c_idx_ccol[:-1])

    max_a_col = 0
    if a_idx_ccol.shape[0] > 1:
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
        dLdB_nnz,
        dLdB_idx[:buffer_ptr],
        c_csc_to_coo_map[dLdC_idx[:buffer_ptr]],
        a_csc_to_coo_map[a_idx[:buffer_ptr]],
    )


@numba.jit(nopython=True)
def _collect_C_idx(
    c_idx_coo: npt.NDArray,
    a_idx_crow: npt.NDArray,
    a_idx_col: npt.NDArray,
    b_idx_ccol: npt.NDArray,
    b_idx_row: npt.NDArray,
    b_csc_to_coo_map: npt.NDArray,
) -> tuple[int, int, int, npt.NDArray, npt.NDArray, npt.NDArray]:
    """
    Collect the indices required to compute the A@B matmul.

    Consider the sparse matrix multiplication C = A @ B. This function determines
    a set of flat indices A_idx, B_idx, and C_idx, such that

    C_val.scatter_add_(
        dim = 0,
        index= C_idx,
        src= A_val[A_idx] * B_val[B_idx]
    )

    gives the coalesced nonzero values of C in COO/CSR layout.
    """
    idx_dtype = c_idx_coo.dtype

    # Determine the nnz of C and exit early if C is empty.
    c_nnz = c_idx_coo.shape[-1]

    c_n_row = a_idx_crow.shape[0] - 1
    c_n_col = b_idx_ccol.shape[0] - 1

    if c_nnz == 0:
        return (
            0,
            c_n_row,
            c_n_col,
            np.empty(0, dtype=idx_dtype),
            np.empty(0, dtype=idx_dtype),
            np.empty(0, dtype=idx_dtype),
        )

    # Determine the max buffer size. In the worst-case scenario, all elements
    # on a row of A and a col of B contributes to a nonzero element in C.
    # Therefore, the max buffer size should be the nnz of C multiplied by
    # the row length of A or col length of B, whichever is longer.

    # Note that the np.max() call needs to be guarded; if a tensor has no rows,
    # then a_idx_crow = [0] and a_idx_crow[1:] - a_idx_crow[:-1] gives an empty
    # array.
    max_a_row = 0
    if c_n_row > 0:
        max_a_row = np.max(a_idx_crow[1:] - a_idx_crow[:-1])

    max_b_col = 0
    if c_n_col > 0:
        max_b_col = np.max(b_idx_ccol[1:] - b_idx_ccol[:-1])

    max_row_size = max(max_a_row, max_b_col)

    max_n_idx = c_nnz * max_row_size

    c_idx = np.empty(max_n_idx, dtype=idx_dtype)
    a_idx = np.empty(max_n_idx, dtype=idx_dtype)
    b_idx = np.empty(max_n_idx, dtype=idx_dtype)

    buffer_ptr = 0
    for c_ptr in range(c_nnz):
        # Find the row and col indices for the nonzero element in dLdA.
        i = c_idx_coo[0, c_ptr]
        j = c_idx_coo[1, c_ptr]

        # Find the start and end of the nonzero element indices in the ith row
        # of dLdC and the jth row of B (assuming CSR/COO layout).
        a_row_i_start = a_idx_crow[i]
        a_row_i_end = a_idx_crow[i + 1]

        b_col_j_start = b_idx_ccol[j]
        b_col_j_end = b_idx_ccol[j + 1]

        # Find the overlapping subset of indices between the column indices in the
        # ith row of A and the row indices in the jth col of B, which contributes
        # to the matrix product, using the two pointers technique.
        a_row_i_k_ptr = a_row_i_start
        b_col_j_k_ptr = b_col_j_start
        while (a_row_i_k_ptr < a_row_i_end) and (b_col_j_k_ptr < b_col_j_end):
            k_A_row_i = a_idx_col[a_row_i_k_ptr]
            k_B_col_j = b_idx_row[b_col_j_k_ptr]

            if k_A_row_i == k_B_col_j:
                # If a matching index k is found, record the index of ik from A
                # and kj from B, as well as the index of ij from C.
                c_idx[buffer_ptr] = c_ptr
                a_idx[buffer_ptr] = a_row_i_k_ptr
                b_idx[buffer_ptr] = b_col_j_k_ptr

                buffer_ptr += 1
                a_row_i_k_ptr += 1
                b_col_j_k_ptr += 1

            elif k_A_row_i < k_B_col_j:
                a_row_i_k_ptr += 1

            elif k_A_row_i > k_B_col_j:
                b_col_j_k_ptr += 1

    return (
        c_nnz,
        c_n_row,
        c_n_col,
        c_idx[:buffer_ptr],
        a_idx[:buffer_ptr],
        b_csc_to_coo_map[b_idx[:buffer_ptr]],
    )


def get_bwd_plan_A(
    a_idx_coo: Int64[Tensor, "2 a_nz"],
    c_idx_crow: Integer[Tensor, " a_r+1"],
    c_idx_col: Integer[Tensor, " a_nz"],
    b_idx_crow: Integer[Tensor, " b_r+1"],
    b_idx_col: Integer[Tensor, " b_nz"],
    dtype: torch.dtype = torch.int64,
    device: torch.device | None = None,
) -> SpSpMMBwdPlanA:
    """Wrap for _collect_dLdA_idx()."""
    a_idx_coo_np = to_np(a_idx_coo)
    c_idx_crow_np = to_np(c_idx_crow)
    c_idx_col_np = to_np(c_idx_col)
    b_idx_crow_np = to_np(b_idx_crow)
    b_idx_col_np = to_np(b_idx_col)

    a_nnz, dLdA_idx_np, dLdC_idx_np, b_idx_np = _collect_dLdA_idx(
        a_idx_coo_np,
        c_idx_crow_np,
        c_idx_col_np,
        b_idx_crow_np,
        b_idx_col_np,
    )

    if device is None:
        device = a_idx_coo.device

    dLdA_idx = torch.from_numpy(dLdA_idx_np).to(dtype=dtype, device=device)
    dLdC_idx = torch.from_numpy(dLdC_idx_np).to(dtype=dtype, device=device)
    b_idx = torch.from_numpy(b_idx_np).to(dtype=dtype, device=device)

    return SpSpMMBwdPlanA(a_nnz, dLdA_idx, dLdC_idx, b_idx)


def get_bwd_plan_B(
    b_idx_coo: Int64[Tensor, "2 b_nz"],
    c_idx_crow: Integer[Tensor, " b_r+1"],
    c_idx_col: Integer[Tensor, " b_nz"],
    c_shape: tuple[int, int] | torch.Size,
    a_idx_ccol: Integer[Tensor, " c_c+1"],
    a_idx_row: Integer[Tensor, " c_nz"],
    a_csc_to_coo_map: Int64[Tensor, " a_nz"],
    dtype: torch.dtype = torch.int64,
    device: torch.device | None = None,
) -> SpSpMMBwdPlanB:
    """Wrap for _collect_dLdB_idx()."""
    c_idx_ccol, c_idx_row_csc, c_csc_to_coo_map = _csr_to_csc(
        idx_crow=c_idx_crow, idx_col=c_idx_col, n_cols=c_shape[-1]
    )

    b_idx_coo_np = to_np(b_idx_coo)
    c_idx_ccol_np = to_np(c_idx_ccol)
    c_idx_row_np = to_np(c_idx_row_csc)
    c_csc_to_coo_map_np = to_np(c_csc_to_coo_map)
    a_idx_ccol_np = to_np(a_idx_ccol)
    a_idx_row_np = to_np(a_idx_row)
    a_csc_to_coo_map_np = to_np(a_csc_to_coo_map)

    b_nnz, dLdB_idx_np, dLdC_idx_np, a_idx_np = _collect_dLdB_idx(
        b_idx_coo_np,
        c_idx_ccol_np,
        c_idx_row_np,
        c_csc_to_coo_map_np,
        a_idx_ccol_np,
        a_idx_row_np,
        a_csc_to_coo_map_np,
    )

    if device is None:
        device = b_idx_coo.device

    dLdB_idx = torch.from_numpy(dLdB_idx_np).to(dtype=dtype, device=device)
    dLdC_idx = torch.from_numpy(dLdC_idx_np).to(dtype=dtype, device=device)
    a_idx = torch.from_numpy(a_idx_np).to(dtype=dtype, device=device)

    return SpSpMMBwdPlanB(b_nnz, dLdB_idx, dLdC_idx, a_idx)


def get_fwd_plan(
    c_idx_coo: Int64[Tensor, "2 c_nz"],
    c_idx_crow: Integer[Tensor, " c_r+1"],
    c_idx_col: Integer[Tensor, " c_nz"],
    a_idx_crow: Integer[Tensor, " a_r+1"],
    a_idx_col: Integer[Tensor, " a_nz"],
    b_idx_ccol: Integer[Tensor, " b_c+1"],
    b_idx_row: Integer[Tensor, " b_nz"],
    b_csc_to_coo_map: Integer[Tensor, " b_nz"],
    dtype: torch.dtype = torch.int64,
    device: torch.device | None = None,
) -> SpSpMMFwdPlan:
    """Wrap for _collect_C_idx()."""
    c_idx_coo_np = to_np(c_idx_coo)
    a_idx_crow_np = to_np(a_idx_crow)
    a_idx_col_np = to_np(a_idx_col)
    b_idx_ccol_np = to_np(b_idx_ccol)
    b_idx_row_np = to_np(b_idx_row)
    b_csc_to_coo_map_np = to_np(b_csc_to_coo_map)

    c_nnz, c_n_row, c_n_col, c_idx_np, a_idx_np, b_idx_np = _collect_C_idx(
        c_idx_coo_np,
        a_idx_crow_np,
        a_idx_col_np,
        b_idx_ccol_np,
        b_idx_row_np,
        b_csc_to_coo_map_np,
    )

    if device is None:
        device = c_idx_coo.device

    c_idx = torch.from_numpy(c_idx_np).to(dtype=dtype, device=device)
    a_idx = torch.from_numpy(a_idx_np).to(dtype=dtype, device=device)
    b_idx = torch.from_numpy(b_idx_np).to(dtype=dtype, device=device)

    c_shape = torch.Size([c_n_row, c_n_col])

    return SpSpMMFwdPlan(
        c_nnz, c_idx_coo, c_idx_crow, c_idx_col, c_shape, c_idx, a_idx, b_idx
    )


def discover_matmul_pattern(
    a_shape: torch.Size,
    a_idx_crow: Integer[Tensor, " a_r+1"],
    a_idx_col: Integer[Tensor, " a_nz"],
    b_shape: torch.Size,
    b_idx_crow: Integer[Tensor, " b_r+1"],
    b_idx_col: Integer[Tensor, " b_nz"],
) -> tuple[
    Int64[Tensor, "2 c_nz"],
    Integer[Tensor, " c_r+1"],
    Integer[Tensor, " c_nz"],
]:
    """
    Discover the sparsity pattern of a 2D matrix multiplication.

    To determine the plan for the forward pass with `get_fwd_plan()` requires
    knowing the sparse COO and CSR index tensors matmul result before the
    actual matmul is carried out. While it is possible to determine these
    indices with `torch.sparse.mm()`, an issue with PyTorch is that the sparse
    CSR `col_indices` of the matmul output are not always guaranteed to be sorted,
    depending on which sparse linalg backend is invoked. This function attempts
    to bypass this issue by determining the sparse index tensors with SciPy
    (if the tensors are on CPU) or CuPy (if the tensors are on GPU) before
    falling back to PyTorch; the advantage of SciPy and CuPy is that they
    expose the flag `has_sorted_indices` and method `sort_indices()` that
    specifically address the sorting issue.
    """
    # Use a_idx_crow as a representative to determine the target device and
    # dtype; note that the COO index will always use int64 dtype.
    device = a_idx_crow.device
    dtype = a_idx_crow.dtype

    if device.type == "cpu":
        a_idx_crow_np = to_np(a_idx_crow, contiguous=True)
        a_idx_col_np = to_np(a_idx_col, contiguous=True)
        b_idx_crow_np = to_np(b_idx_crow, contiguous=True)
        b_idx_col_np = to_np(b_idx_col, contiguous=True)

        a_val_dummy = np.ones_like(a_idx_col, dtype=np.float32)
        b_val_dummy = np.ones_like(b_idx_col, dtype=np.float32)

        a_csr_scipy = scipy.sparse.csr_array(
            (a_val_dummy, a_idx_col_np, a_idx_crow_np),
            dtype=np.bool,
            shape=tuple(a_shape),
        )
        b_csr_scipy = scipy.sparse.csr_array(
            (b_val_dummy, b_idx_col_np, b_idx_crow_np),
            dtype=np.bool,
            shape=tuple(b_shape),
        )

        c_csr_scipy = a_csr_scipy @ b_csr_scipy

        # Call sort_indices() to guarnatee that the col indices are sorted.
        if not c_csr_scipy.has_sorted_indices:
            c_csr_scipy.sort_indices()

        c_idx_crow_np = c_csr_scipy.indptr
        c_idx_col_np = c_csr_scipy.indices

        c_coo_scipy = c_csr_scipy.tocoo()
        c_idx_coo_np = np.stack((c_coo_scipy.row, c_coo_scipy.col))

        c_idx_coo = torch.from_numpy(c_idx_coo_np).to(dtype=torch.int64, device=device)
        c_idx_crow = torch.from_numpy(c_idx_crow_np).to(dtype=dtype, device=device)
        c_idx_col = torch.from_numpy(c_idx_col_np).to(dtype=dtype, device=device)

    elif device.type == "cuda" and _HAS_CUPY:
        a_val_dummy = torch.ones_like(a_idx_col, dtype=torch.float32)
        b_val_dummy = torch.ones_like(b_idx_col, dtype=torch.float32)

        stream = torch.cuda.current_stream()
        with cp.cuda.ExternalStream(stream.cuda_stream, stream.device_index):
            a_csr_cp = cp_sp.csr_matrix(
                (
                    cp.from_dlpack(a_val_dummy),
                    cp.from_dlpack(a_idx_col),
                    cp.from_dlpack(a_idx_crow),
                ),
                shape=tuple(a_shape),
            )
            b_csr_cp = cp_sp.csr_matrix(
                (
                    cp.from_dlpack(b_val_dummy),
                    cp.from_dlpack(b_idx_col),
                    cp.from_dlpack(b_idx_crow),
                ),
                shape=tuple(b_shape),
            )

            c_csr_cp = a_csr_cp @ b_csr_cp

            if not c_csr_cp.has_sorted_indices:
                c_csr_cp.sort_indices()

            c_idx_crow_cp = c_csr_cp.indptr
            c_idx_col_cp = c_csr_cp.indices

            c_coo_cp = c_csr_cp.tocoo()
            c_idx_coo_cp = cp.stack((c_coo_cp.row, c_coo_cp.col))

            c_idx_coo = torch.from_dlpack(c_idx_coo_cp, device=device).to(
                dtype=torch.int64
            )
            c_idx_crow = torch.from_dlpack(c_idx_crow_cp, device=device).to(dtype=dtype)
            c_idx_col = torch.from_dlpack(c_idx_col_cp, device=device).to(dtype=dtype)

    else:
        with torch.no_grad():
            a_val_dummy = torch.ones_like(a_idx_col, dtype=torch.float32)
            b_val_dummy = torch.ones_like(b_idx_col, dtype=torch.float32)

            a_csr = torch.sparse_csr_tensor(a_idx_crow, a_idx_col, a_val_dummy, a_shape)
            b_csr = torch.sparse_csr_tensor(b_idx_crow, b_idx_col, b_val_dummy, b_shape)

            # _coalesced_(False) forces the is_coalesced flag to False so that
            # calling coalesce() actually triggers an index sorting.
            c_sp_coo = (
                torch.sparse.mm(
                    a_csr,
                    b_csr,
                )
                .to_sparse_coo()
                ._coalesced_(False)
                .coalesce()
            )

            c_idx_coo = c_sp_coo.indices().to(dtype=torch.int64)

            c_sp_csr = c_sp_coo.to_sparse_csr()
            c_idx_crow = c_sp_csr.crow_indices().to(dtype=dtype)
            c_idx_col = c_sp_csr.col_indices().to(dtype=dtype)

    return c_idx_coo, c_idx_crow, c_idx_col

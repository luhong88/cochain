from typing import Literal

import torch
from jaxtyping import Int64, Integer
from torch import Tensor


def coalesced_coo_to_compressed_idx(
    coo_idx: Integer[Tensor, "sp nz"],
    shape: torch.Size,
    *,
    format: Literal["crow", "ccol"],
    dtype: torch.dtype | None = None,
) -> Integer[Tensor, "*b nz_per_b"]:
    """
    Convert COO indices to compressed formats.

    The input `coo_idx` is assumed to be coalesced and allowed to have either
    one or zero batch dim. The output compressed index tensor is guaranteed to
    be contiguous.
    """
    if dtype is None:
        dtype = coo_idx.dtype

    device = coo_idx.device

    # The following code is written with variable names assuming format = 'crow',
    # but the same logic applies for ccol by switching the target_idx.
    match format:
        case "crow":
            target_dim = -2
        case "ccol":
            target_dim = -1
        case _:
            raise ValueError(f"Unknown format argument '{format}'.")

    match len(shape):
        case 2:
            n_row = shape[target_dim]
            row_idx = coo_idx[target_dim].to(dtype)
            # Compress row idx using bincount; minlength=rows accounts for empty rows.
            # Note that this works even if row_idx is not sorted.
            counts = torch.bincount(row_idx, minlength=n_row)
            crow_idx = torch.zeros(n_row + 1, dtype=dtype, device=device)
            torch.cumsum(counts, dim=0, out=crow_idx[1:])

            return crow_idx.contiguous()

        case 3:
            n_row = shape[target_dim]
            n_batch = shape[0]
            nnz = coo_idx.size(1)
            nnz_per_batch = nnz // n_batch

            row_idx_batched = (
                coo_idx[target_dim].view(n_batch, nnz_per_batch).to(dtype=dtype)
            )

            # Compute histogram of row counts per batch. The scatter_add_() function
            # effectively counts how many times the k-th row idx shows up per batch,
            # and then put this count in the k-th position. The +1 is needed to
            # account for the fact that compressed row idx always starts at 0. The
            # cumsum() converts the row counts to compressed row idx.
            counts = torch.zeros(n_batch, n_row + 1, dtype=dtype, device=device)
            ones = torch.tensor(1, dtype=dtype, device=device).expand_as(
                row_idx_batched
            )
            # counts[b][row_idx_batched[b][r]] = ones[b][r]
            counts.scatter_add_(dim=1, index=row_idx_batched + 1, src=ones)
            crow_idx = counts.cumsum(dim=1, dtype=dtype)

            return crow_idx.contiguous()


def coalesced_coo_to_csr_col_idx(
    coo_idx: Integer[Tensor, "sp nz"],
    shape: torch.Size,
    *,
    dtype: torch.dtype | None = None,
) -> Integer[Tensor, "*b nz_per_b"]:
    """
    Extract the column indices from a COO index tensor.

    The input `coo_idx` is assumed to be coalesced and allowed to have either
    one or zero batch dim. The output column index tensor is guaranteed to
    be contiguous.
    """
    if dtype is None:
        dtype = coo_idx.dtype

    match len(shape):
        case 2:
            col_idx = coo_idx[1].to(dtype)
            return col_idx.contiguous()

        case 3:
            n_batch = shape[0]
            nnz = coo_idx.size(1)
            nnz_per_batch = nnz_per_batch = nnz // n_batch

            col_idx_batched = coo_idx[2].view(n_batch, nnz_per_batch).to(dtype=dtype)

            return col_idx_batched.contiguous()


def get_csc_to_coo_map(
    coo_idx: Integer[Tensor, "sp nz"],
    shape: torch.Size,
) -> Int64[Tensor, " nz"]:
    """
    Compute the permutation to convert from col-major to row-major order.

    The input `coo_idx` is assumed to be coalesced and allowed to have either
    one or zero batch dim. The output permutation tensor is guaranteed to
    be contiguous and always in int64 dtype.
    """
    match len(shape):
        case 2:
            col_idx = coo_idx[1]
            # stable=True to preserve the existing row ordering in the same col.
            perm = torch.argsort(col_idx, dim=0, stable=True)

            return perm.contiguous()

        case 3:
            n_batch = shape[0]
            nnz = coo_idx.size(1)
            nnz_per_batch = nnz // n_batch

            # View cols as (batch, nnz_per_batch) to sort per-batch independently
            col_idx_batched = coo_idx[2].view(n_batch, nnz_per_batch)

            # Sort per-batch with dim=-1
            perm_per_batch = torch.argsort(col_idx_batched, dim=-1, stable=True)

            # Convert local per-batch indices to global indices
            batch_offset = torch.arange(
                0, nnz, step=nnz_per_batch, device=coo_idx.device
            ).view(-1, 1)

            perm = (perm_per_batch + batch_offset).view(-1)

            return perm.contiguous()


def coalesced_coo_to_csc_row_idx(
    coo_idx: Integer[Tensor, "sp nz"],
    shape: torch.Size,
    perm: Integer[Tensor, " nz"],
    *,
    dtype: torch.dtype | None = None,
) -> Integer[Tensor, "*b nz_per_b"]:
    """
    Extract the row indices from a COO index tensor.

    The input `coo_idx` is assumed to be coalesced and allowed to have either
    one or zero batch dim. The output row index tensor is guaranteed to be
    contiguous and sorted for the CSC format.
    """
    if dtype is None:
        dtype = coo_idx.dtype

    match len(shape):
        case 2:
            return coo_idx[0][perm].to(dtype).contiguous()

        case 3:
            n_batch = shape[0]
            nnz = coo_idx.size(1)
            nnz_per_batch = nnz_per_batch = nnz // n_batch

            row_idx_sorted = coo_idx[1][perm].view(n_batch, nnz_per_batch).to(dtype)

            return row_idx_sorted.contiguous()


def csr_to_csc(
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
    coo_to_csc_map = torch.argsort(idx_col, dim=0, stable=True)

    # Compute CSC ccol index
    idx_ccol = torch.zeros(n_cols + 1, dtype=idx_crow.dtype, device=idx_crow.device)
    idx_ccol[1:] = torch.bincount(idx_col, minlength=n_cols).cumsum(dim=0)

    # Compute CSC row index
    n_rows = idx_crow.size(0) - 1
    row_idx = torch.arange(n_rows, dtype=idx_crow.dtype, device=idx_crow.device)
    idx_row_csr = row_idx.repeat_interleave(idx_crow[1:] - idx_crow[:-1])

    # Permute the uncompressed CSR row index into CSC order
    idx_row_csc = idx_row_csr[coo_to_csc_map]

    return idx_ccol.contiguous(), idx_row_csc.contiguous(), coo_to_csc_map.contiguous()

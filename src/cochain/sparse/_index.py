from typing import Literal

import torch as t
from jaxtyping import Float, Integer


def coalesced_coo_to_compressed_idx(
    coo_idx: Integer[t.LongTensor, "sp nnz"],
    shape: t.Size,
    *,
    format: Literal["crow", "ccol"],
    dtype: t.dtype | None = None,
) -> Integer[t.LongTensor, "*b nnz/b"]:
    """
    Convert a coalesced, sparse coo index tensor to a compressed row idx (crow)
    tensor or a compressed col idx (ccol) tensor, depending on the 'format' argument.
    """
    if dtype is None:
        dtype = coo_idx.dtype

    device = coo_idx.device

    # The following code is written with variable names assuming format = 'crow',
    # but the same logic applies for ccol by switching the target_idx.
    match format:
        case "crow":
            target_idx = -2
        case "ccol":
            target_idx = -1
        case _:
            raise ValueError(f"Unknown format argument '{format}'.")

    match len(shape):
        case 2:
            n_row = shape[target_idx]
            row_idx = coo_idx[target_idx].to(dtype)
            # Compress row idx using bincount; minlength=rows accounts for empty rows.
            # Note that this works even if row_idx is not sorted.
            counts = t.bincount(row_idx, minlength=n_row)
            crow_idx = t.zeros(n_row + 1, dtype=dtype, device=device)
            t.cumsum(counts, dim=0, out=crow_idx[1:])

            return crow_idx.contiguous()

        case 3:
            n_row = shape[target_idx]
            n_batch = shape[0]
            nnz = coo_idx.size(1)
            nnz_per_batch = nnz_per_batch = nnz // n_batch

            row_idx_batched = (
                coo_idx[target_idx].view(n_batch, nnz_per_batch).to(dtype=dtype)
            )

            # Compute histogram of row counts per batch. The scatter_add_() function
            # effectively counts how many times the k-th row idx shows up per batch,
            # and then put this count in the k-th position. The +1 is needed to
            # account for the fact that compressed row idx always starts at 0. The
            # cumsum() converts the row counts to compressed row idx.
            counts = t.zeros(n_batch, n_row + 1, dtype=dtype, device=device)
            ones = t.tensor(1, dtype=dtype, device=device).expand_as(row_idx_batched)
            counts.scatter_add_(1, row_idx_batched + 1, ones)
            crow_idx = counts.cumsum(dim=1, dtype=dtype)

            return crow_idx.contiguous()


def coalesced_coo_to_col_idx(
    coo_idx: Integer[t.LongTensor, "sp nnz"],
    shape: t.Size,
    *,
    dtype: t.dtype | None = None,
) -> Integer[t.LongTensor, "*b nnz/b"]:
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


def get_csc_sort_perm(
    coo_idx: Integer[t.LongTensor, "sp nnz"],
    shape: t.Size,
) -> Integer[t.LongTensor, " nnz"]:
    """
    Returns a permutation that reorders a row-sorted coo tensor into a col-sorted,
    csc-ready order.
    """
    match len(shape):
        case 2:
            col_idx = coo_idx[1]
            # stable=True to preserve the existing row ordering in the same col.
            perm = t.argsort(col_idx, dim=0, stable=True)

            return perm.contiguous()

        case 3:
            n_batch = shape[0]
            nnz = coo_idx.size(1)
            nnz_per_batch = nnz_per_batch = nnz // n_batch

            # View cols as (batch, nnz_per_batch) to sort per-batch independently
            col_idx_batched = coo_idx[2].view(n_batch, nnz_per_batch)

            # Sort per-batch with dim=-1
            perm_per_batch = t.argsort(col_idx_batched, dim=-1, stable=True)

            # Convert local per-batch indices to global indices
            batch_offset = t.arange(
                0, nnz, step=nnz_per_batch, device=coo_idx.device
            ).view(-1, 1)

            perm = (perm_per_batch + batch_offset).view(-1)

            return perm.contiguous()


def coalesced_coo_to_row_idx(
    coo_idx: Integer[t.LongTensor, "sp nnz"],
    shape: t.Size,
    perm: Integer[t.LongTensor, " nnz"],
    *,
    dtype: t.dtype | None = None,
) -> Integer[t.LongTensor, "*b nnz/b"]:
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


def project_and_extract_cnz_vals(
    src_coo: Integer[t.LongTensor, "2 src_nnz"],
    src_val: Float[t.Tensor, " source_nnz"],
    target_coo: Integer[t.LongTensor, "2 target_nnz"],
    target_shape: t.Size,
) -> Integer[t.LongTensor, " target_nnz"]:
    """
    For two coalesced sparse coo tensors "source" and "target", find the indices
    of the nonzero (r, c) index pairs of the source that is also present in the
    target, and then return a target-shaped value tensor filled with source values
    at these index pair locations.

    This function is most performant if the source has more nnz than the target.
    """
    target_nnz = target_coo.size(1)
    src_nnz = src_coo.size(1)

    if target_nnz == 0 or src_nnz == 0:
        return t.empty(0, dtype=src_val.dtype, device=src_coo.device)

    # Perform a radix packing to index the nonzero index pairs in the source
    # and target coo index tensors.
    target_idx_packed = target_coo[0] * target_shape[1] + target_coo[1]
    src_idx_packed = src_coo[0] * target_shape[1] + src_coo[1]

    # Use searchsorted() to find the insertion location of target index pairs
    # into the list of source index pairs.
    target_idx_insert_loc = t.searchsorted(src_idx_packed, target_idx_packed)
    target_idx_insert_loc_clipped = t.clip(target_idx_insert_loc, 0, src_nnz - 1)

    # If a target index pair has a matching source index pair, then its packed
    # index matches the packed index of the source index pair at its insertion
    # location. Checking this gives a "common nonzero" mask for the target index pairs.
    cnz_target_idx_mask = (
        src_idx_packed[target_idx_insert_loc_clipped] == target_idx_packed
    )

    # For each nonzero index pair in the target, if it has a matching index pair
    # in the source, find the source value at this index pair, otherwise returns 0.
    cnz_src_val = t.where(
        cnz_target_idx_mask,
        src_val[target_idx_insert_loc_clipped],
        t.tensor(0.0, dtype=src_val.dtype, device=src_val.device),
    )

    return cnz_src_val.contiguous()

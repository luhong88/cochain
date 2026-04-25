from typing import Literal

import torch
from jaxtyping import Float, Integer
from torch import LongTensor, Tensor

from ...utils.search import splx_search


def coalesced_coo_to_compressed_idx(
    coo_idx: Integer[LongTensor, "sp nz"],
    shape: torch.Size,
    *,
    format: Literal["crow", "ccol"],
    dtype: torch.dtype | None = None,
) -> Integer[LongTensor, "*b nz_per_b"]:
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
    coo_idx: Integer[LongTensor, "sp nz"],
    shape: torch.Size,
    *,
    dtype: torch.dtype | None = None,
) -> Integer[LongTensor, "*b nz_per_b"]:
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


def get_csc_sort_perm(
    coo_idx: Integer[LongTensor, "sp nz"],
    shape: torch.Size,
) -> Integer[LongTensor, " nz"]:
    """
    Compute the permutation to convert from row-major to column-major order.

    The input `coo_idx` is assumed to be coalesced and allowed to have either
    one or zero batch dim. The output permutation tensor is guaranteed to
    be contiguous.
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
    coo_idx: Integer[LongTensor, "sp nnz"],
    shape: torch.Size,
    perm: Integer[LongTensor, " nnz"],
    *,
    dtype: torch.dtype | None = None,
) -> Integer[LongTensor, "*b nnz/b"]:
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


def project_and_extract_cnz_vals(
    src_coo: Integer[LongTensor, "2 src_nz"],
    src_val: Float[Tensor, " src_nz"],
    template_coo: Integer[LongTensor, "2 target_nz"],
) -> Float[Tensor, " target_nz"]:
    """
    Extract common nonzero elements by comparing two sparse COO tensors.

    For two coalesced sparse COO tensors "source" and "template", find the indices
    of the nonzero (r, c) index pairs of the source that is also present in the
    template, and then return a template-nnz-shaped value tensor filled with source
    values at these index pair locations.

    This function is most performant if the source has more nnz than the target.
    Batching is not supported, and both the source and template COO index tensors
    are assumed to have exactly two rows.
    """
    target_nnz = template_coo.size(1)
    src_nnz = src_coo.size(1)

    if target_nnz == 0 or src_nnz == 0:
        return torch.empty(0, dtype=src_val.dtype, device=src_coo.device)

    # We can interpret the (r, c) coordinates of the nonzero elements in the template
    # and source tensors as "edges", and use splx_search() to identify the
    # indices of the template nonzeros in the source nonzeros. Since both the
    # source and template COO indices are coalesced, they are already lex-sorted.
    # subset="check" ensures that template nonzeros that are not present in the
    # source nonzeros get an index of -1.
    template_idx = splx_search(
        key_splx=src_coo.T,
        query_splx=template_coo.T,
        sort_key_splx=False,
        sort_key_vert=False,
        sort_query_vert=False,
        method="lex_sort",
        subset="check",
    )

    # For each nonzero index pair in the target, if it has a matching index pair
    # in the source, find the source value at this index pair, otherwise returns 0.
    # Here, template_idx >= 0 filters out the -1 sentinel index values corresponding
    # to source nonzeros that are not mapped to a template nonzero.
    zero = torch.tensor(0.0, dtype=src_val.dtype, device=src_val.device)
    cnz_src_val = torch.where(template_idx >= 0, src_val[template_idx], zero)

    return cnz_src_val.contiguous()

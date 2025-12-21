from typing import Literal

import torch as t
from jaxtyping import Float, Integer


def validate_coo_idx_shape(
    coo_idx: Integer[t.LongTensor, "sp nnz"], shape: tuple[int, ...] | t.Size
):
    match len(shape):
        case 1:
            raise ValueError(
                "The 'idx_coo' tensor must have at least two sparse dimensions."
            )

        case 2:
            if coo_idx.size(0) != 2:
                raise ValueError(
                    "For a 2D sparse coo tensor, 'idx_coo' must be of shape (2, nnz)."
                )

        case 3:
            if coo_idx.size(0) != 3:
                raise ValueError(
                    "For a sparse coo tensor with a batch dimension, 'idx_coo' "
                    + "must be of shape (3, nnz)."
                )

        case _:
            raise NotImplementedError(
                "More than one batch dimensions is not supported."
            )


def _get_nnz_per_batch(
    nnz: int,
    n_batch: int,
    batch_idx: Integer[t.Tensor, " nnz"] | None = None,
) -> int:
    # If the input tensor has equal nnz along the batch dimension, then the nnz
    # per tensor in the batch is given by nnz // batch.
    if nnz % n_batch != 0:
        raise ValueError(
            f"Total nnz ({nnz}) is not divisible by batch size ({n_batch})."
        )

    nnz_per_batch = nnz // n_batch

    # It is possible for a tensor to have non-equal nnz along the batch dimension
    # but still satisfies nnz % batch = 0 (e.g., if the first tensor has 6 nnz,
    # while the second has 2). This optional (but somewhat expensive) check rules
    # out this possibility.
    if batch_idx is not None:
        batch_counts = t.bincount(batch_idx, minlength=n_batch)
        if not (batch_counts == nnz_per_batch).all():
            raise ValueError("The equal nnz per batch item condition is not met.")

    return nnz_per_batch


def coalesced_coo_to_compressed_idx(
    coo_idx: Integer[t.LongTensor, "sp nnz"],
    shape: tuple[int, ...] | t.Size,
    *,
    target: Literal["crow", "ccol"],
    dtype: t.dtype | None,
    strict_batch_nnz_check: bool = False,
):
    """
    Convert a coalesced, sparse coo index tensor to a compressed row idx (crow)
    tensor or a compressed col idx (ccol) tensor, depending on the 'target' argument.
    """
    if dtype is None:
        dtype = coo_idx.dtype

    device = coo_idx.device

    # The following code is written with variable names assuming target = 'crow',
    # but the same logic applies for ccol by switching the target_idx.
    match target:
        case "crow":
            target_idx = -2
        case "ccol":
            target_idx = -1
        case _:
            raise ValueError(f"Unknown target argument '{target}'.")

    match len(shape):
        case 2:
            n_row = shape[target_idx]
            row_idx = coo_idx[target_idx].to(dtype)
            # Compress row idx using bincount; minlength=rows accounts for empty rows.
            # Note that this works even if row_idx is not sorted.
            counts = t.bincount(row_idx, minlength=n_row)
            crow_idx = t.zeros(n_row + 1, dtype=dtype, device=device)
            t.cumsum(counts, dim=0, out=crow_idx[1:])

            return crow_idx

        case 3:
            n_row = shape[target_idx]
            n_batch = shape[0]
            nnz = coo_idx.size(1)
            nnz_per_batch = _get_nnz_per_batch(
                nnz, n_batch, coo_idx[0] if strict_batch_nnz_check else None
            )

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

            return crow_idx


def coalesced_coo_to_col_idx(
    coo_idx: Integer[t.LongTensor, "sp nnz"],
    shape: tuple[int, ...] | t.Size,
    *,
    dtype: t.dtype | None,
    strict_batch_nnz_check: bool = False,
) -> Integer[t.LongTensor, "*b nnz/b"]:
    if dtype is None:
        dtype = coo_idx.dtype

    match len(shape):
        case 2:
            col_idx = coo_idx[1].to(dtype)
            return col_idx

        case 3:
            n_batch = shape[0]
            nnz = coo_idx.size(1)
            nnz_per_batch = _get_nnz_per_batch(
                nnz, n_batch, coo_idx[0] if strict_batch_nnz_check else None
            )

            col_idx_batched = coo_idx[2].view(n_batch, nnz_per_batch).to(dtype=dtype)

            return col_idx_batched


def get_csc_sort_perm(
    coo_idx: Integer[t.LongTensor, "sp nnz"],
    shape: tuple[int, ...] | t.Size,
    strict_batch_nnz_check: bool = False,
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

            return perm

        case 3:
            n_batch = shape[0]
            nnz = coo_idx.size(1)
            nnz_per_batch = _get_nnz_per_batch(
                nnz, n_batch, coo_idx[0] if strict_batch_nnz_check else None
            )

            # View cols as (batch, nnz_per_batch) to sort per-batch independently
            col_idx_batched = coo_idx[2].view(n_batch, nnz_per_batch)

            # Sort per-batch with dim=-1
            perm_per_batch = t.argsort(col_idx_batched, dim=-1, stable=True)

            # Convert local per-batch indices to global indices
            batch_offset = t.arange(
                0, nnz, step=nnz_per_batch, device=coo_idx.device
            ).view(-1, 1)

            perm = (perm_per_batch + batch_offset).view(-1)

            return perm


def coalesced_coo_to_row_idx(
    coo_idx: Integer[t.LongTensor, "sp nnz"],
    shape: tuple[int, ...] | t.Size,
    perm: Integer[t.LongTensor, " nnz"],
    *,
    dtype: t.dtype | None,
    strict_batch_nnz_check: bool = False,
) -> Integer[t.LongTensor, "*b nnz/b"]:
    if dtype is None:
        dtype = coo_idx.dtype

    match len(shape):
        case 2:
            return coo_idx[0][perm].to(dtype)

        case 3:
            n_batch = shape[0]
            nnz = coo_idx.size(1)
            nnz_per_batch = _get_nnz_per_batch(
                nnz, n_batch, coo_idx[0] if strict_batch_nnz_check else None
            )

            row_idx_sorted = coo_idx[1][perm].view(n_batch, nnz_per_batch).to(dtype)

            return row_idx_sorted


# TODO: depreciate
def coalesced_coo_to_int32_csr(
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


def transpose_sp_csr(sp_csr: Float[t.Tensor, "*b r c"]) -> Float[t.Tensor, "*b c r"]:
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

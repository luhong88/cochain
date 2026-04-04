from typing import Literal

import torch
from jaxtyping import Integer
from torch import LongTensor


def _polynomial_hash_splx_search(
    key_splx: Integer[LongTensor, "key_splx vert"],
    query_splx: Integer[LongTensor, "query_splx vert"],
    *,
    sort_key_splx: bool,
    sort_key_vert: bool,
    sort_query_vert: bool,
) -> Integer[LongTensor, " query_splx"]:
    dtype = torch.int64
    device = key_splx.device

    # Note that a k-simplex has k + 1 vertices.
    splx_dim = key_splx.size(1) - 1
    if key_splx.size(1) != query_splx.size(1):
        raise ValueError(
            "The 'key_splx' and 'query_splx' must have the same dimension."
        )

    key_splx_canon = key_splx.sort(dim=-1).values if sort_key_vert else key_splx
    query_splx_canon = query_splx.sort(dim=-1).values if sort_query_vert else query_splx

    min_vert_idx = min(key_splx_canon.min(), query_splx_canon.min())
    key_splx_reduced = key_splx_canon - min_vert_idx
    query_splx_reduced = query_splx_canon - min_vert_idx
    # +1 to ensure base > max digit
    max_vert_idx = max(key_splx_reduced.max(), query_splx_reduced.max()).to(dtype) + 1

    exponent = torch.arange(splx_dim, -1, -1, device=device, dtype=dtype)
    coef = torch.pow(max_vert_idx, exponent).view(1, -1)

    max_allowed_hash_val = torch.iinfo(dtype).max
    worst_case_hash_val = float(max_vert_idx) ** (splx_dim + 1)
    if worst_case_hash_val > max_allowed_hash_val:
        raise RuntimeError(
            "Potential polynomial hash overflow detected. Use method='lex_sort' instead."
        )

    key_splx_packed = torch.sum(key_splx_reduced * coef, dim=-1)
    query_splx_packed = torch.sum(query_splx_reduced * coef, dim=-1)

    if sort_key_splx:
        key_splx_packed_sorted, key_splx_packed_sort_idx = key_splx_packed.sort()
        query_splx_idx_sorted = torch.searchsorted(
            key_splx_packed_sorted, query_splx_packed
        )
        query_splx_idx = key_splx_packed_sort_idx[query_splx_idx_sorted]

    else:
        query_splx_idx = torch.searchsorted(key_splx_packed, query_splx_packed)

    return query_splx_idx


def _lex_splx_search(
    key_splx: Integer[LongTensor, "key_splx vert"],
    query_splx: Integer[LongTensor, "query_splx vert"],
    *,
    sort_key_splx: bool,
    sort_key_vert: bool,
    sort_query_vert: bool,
) -> Integer[LongTensor, " query_splx"]:
    if key_splx.size(1) != query_splx.size(1):
        raise ValueError(
            "The 'key_splx' and 'query_splx' must have the same dimension."
        )

    key_splx_canon = key_splx.sort(dim=-1).values if sort_key_vert else key_splx
    query_splx_canon = query_splx.sort(dim=-1).values if sort_query_vert else query_splx

    combined_splx = torch.vstack((key_splx_canon, query_splx_canon))
    _, combined_splx_ids = combined_splx.unique(dim=0, sorted=True, return_inverse=True)

    n_key_splx = key_splx_canon.size(0)
    key_splx_ids = combined_splx_ids[:n_key_splx]
    query_splx_ids = combined_splx_ids[n_key_splx:]

    if sort_key_splx:
        key_splx_ids_sorted, key_splx_id_sort_idx = key_splx_ids.sort()
        query_splx_idx_sorted = torch.searchsorted(key_splx_ids_sorted, query_splx_ids)
        query_splx_idx = key_splx_id_sort_idx[query_splx_idx_sorted]

    else:
        query_splx_idx = torch.searchsorted(key_splx_ids, query_splx_ids)

    return query_splx_idx


def splx_search(
    key_splx: Integer[LongTensor, "key_splx vert"],
    query_splx: Integer[LongTensor, "*b query_splx vert"],
    *,
    sort_key_splx: bool,
    sort_key_vert: bool,
    sort_query_vert: bool,
    method: Literal["lex_sort", "polynomial_hash", "auto"] = "auto",
) -> Integer[LongTensor, "*b query_splx"]:
    """
    Search for simplices in `query_splx` in the `key_splx`. It uses polynomial
    rolling hash to convert the simplex vert index tuples into integers and use
    `searchsorted()` to perform the search.

    This function supports two search methods based on either lex sort or
    polynomial hashing. In general, polynomial hashing is faster than lex sort,
    but is not as overflow-safe as lex sort. As such, polynomial hashing is not
    recommended for searching over k-simplices for k >= 3. If `method='auto'`,
    use polynomial hashing if k < 2 and lex sort if k >= 2.

    This function has the following requirements:
    * `query_splx` must be a subset of `key_splx` (up to vertex permutation).
    * `key_splx` cannot contain duplicates, but `query_splx` can.
    * Each simplex in `key_splx` and `query_splx` must be represented by vertex
    indices in ascending order; if this is not true, set `sort_key_vert=True` and/or
    `sort_query_vert=True`.
    * The simplices in `key_splx` must be sorted in lex order; i.e., the simplices
    in `key_splx` need to be aranged (in ascending order) based on the first vertex
    index, with tie breaks based on the second vertex index, and so on. If this is
    not true, set `sort_key_splx=True`.
    """
    if query_splx.size(-2) == 0:
        return query_splx[..., 0]

    if method == "auto":
        splx_dim = key_splx.size(-1) - 1

        if splx_dim < 2:
            method = "polynomial_hash"
        else:
            method = "lex_sort"

    query_splx_flat = query_splx.flatten(end_dim=-2)

    match method:
        case "lex_sort":
            splx_idx_flat = _lex_splx_search(
                key_splx,
                query_splx_flat,
                sort_key_splx=sort_key_splx,
                sort_key_vert=sort_key_vert,
                sort_query_vert=sort_query_vert,
            )
        case "polynomial_hash":
            splx_idx_flat = _polynomial_hash_splx_search(
                key_splx,
                query_splx_flat,
                sort_key_splx=sort_key_splx,
                sort_key_vert=sort_key_vert,
                sort_query_vert=sort_query_vert,
            )
        case _:
            raise ValueError("Unrecognized 'method' argument.")

    splx_idx_shaped = splx_idx_flat.view(*query_splx.shape[:-1])

    return splx_idx_shaped

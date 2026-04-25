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
    subset: Literal["assume", "check", "assert"],
) -> Integer[LongTensor, " query_splx"]:
    dtype = torch.int64
    device = key_splx.device

    # Note that a k-simplex has k + 1 vertices.
    splx_dim = key_splx.size(1) - 1
    if key_splx.size(1) != query_splx.size(1):
        raise ValueError(
            "The 'key_splx' and 'query_splx' must have the same dimension."
        )

    n_key_splx = key_splx.size(0)

    key_splx_canon = key_splx.sort(dim=-1).values if sort_key_vert else key_splx
    query_splx_canon = query_splx.sort(dim=-1).values if sort_query_vert else query_splx

    # Shift the vert indices so that they are 0-indexed to save on some integers.
    min_vert_idx = min(key_splx_canon.min(), query_splx_canon.min())
    key_splx_reduced = key_splx_canon - min_vert_idx
    query_splx_reduced = query_splx_canon - min_vert_idx
    # +1 to ensure base > max digit
    max_vert_idx = max(key_splx_reduced.max(), query_splx_reduced.max()).to(dtype) + 1

    # Simplex (v0, v1, v2, v3) -> polynomial v0*V^3 + v1*V^2 + v2*V^1 + v1*V^0
    # where V is the max_vert_idx.
    exponent = torch.arange(splx_dim, -1, -1, device=device, dtype=dtype)
    coef = torch.pow(max_vert_idx, exponent).view(1, -1)

    # Check for overflow by checking if the max possible leading term in the
    # polynomial (V*V^dim) is above the dtype limit.
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
        query_splx_idx_clamped = torch.clamp_max(query_splx_idx_sorted, n_key_splx - 1)
        query_splx_idx = key_splx_packed_sort_idx[query_splx_idx_clamped]

    else:
        query_splx_idx = torch.searchsorted(key_splx_packed, query_splx_packed)

    match subset:
        case "assume":
            return query_splx_idx

        case "check":
            # If a query splx is in the key splx set, then the key splx at the
            # insertion index (by searchsorted()) should match the query splx.
            # One caveat is that if the query splx value is strictly larger than
            # the largest key splx value, then its idx will place it out of
            # bound of the key splx list; therefore, the query_splx_idx needs to
            # be clamped first.
            query_splx_idx_clamped = torch.clamp_max(query_splx_idx, n_key_splx - 1)
            query_splx_is_found = (
                key_splx_packed[query_splx_idx_clamped] == query_splx_packed
            )
            sentinel = torch.tensor(
                -1, dtype=query_splx.dtype, device=query_splx.device
            )
            return torch.where(
                query_splx_is_found,
                query_splx_idx,
                sentinel,
            )

        case "assert":
            query_splx_idx_clamped = torch.clamp_max(query_splx_idx, n_key_splx - 1)
            query_splx_is_found = (
                key_splx_packed[query_splx_idx_clamped] == query_splx_packed
            )
            if query_splx_is_found.all():
                return query_splx_idx
            else:
                raise ValueError("The 'query_splx' is not a subset of the 'key_splx'.")

        case _:
            raise ValueError(f"Unknown subset argument '{subset}'.")


def _lex_splx_search(
    key_splx: Integer[LongTensor, "key_splx vert"],
    query_splx: Integer[LongTensor, "query_splx vert"],
    *,
    sort_key_splx: bool,
    sort_key_vert: bool,
    sort_query_vert: bool,
    subset: Literal["assume", "check", "assert"],
) -> Integer[LongTensor, " query_splx"]:
    if key_splx.size(1) != query_splx.size(1):
        raise ValueError(
            "The 'key_splx' and 'query_splx' must have the same dimension."
        )

    key_splx_canon = key_splx.sort(dim=-1).values if sort_key_vert else key_splx
    query_splx_canon = query_splx.sort(dim=-1).values if sort_query_vert else query_splx

    # Put all key and query splx on a single list, and use unique() to determine
    # the unique splx on the combined list and the indices (IDs) of the key and
    # query splx on this unique list.
    combined_splx = torch.vstack((key_splx_canon, query_splx_canon))
    _, combined_splx_ids = combined_splx.unique(dim=0, sorted=True, return_inverse=True)

    n_key_splx = key_splx_canon.size(0)
    key_splx_ids = combined_splx_ids[:n_key_splx]
    query_splx_ids = combined_splx_ids[n_key_splx:]

    if sort_key_splx:
        key_splx_ids_sorted, key_splx_id_sort_idx = key_splx_ids.sort()
        # If a query splx is in the key splx set, then the key splx at the insertion
        # index (by searchsorted()) should match the query splx. One caveat is that
        # if the query splx ID is strictly larger than the largest key splx ID,
        # then its idx will place it out of bound of the key splx list; therefore,
        # the query_splx_idx_sorted needs to be clamped first.
        query_splx_idx_sorted = torch.searchsorted(key_splx_ids_sorted, query_splx_ids)
        query_splx_idx_clamped = torch.clamp_max(query_splx_idx_sorted, n_key_splx - 1)
        query_splx_idx = key_splx_id_sort_idx[query_splx_idx_clamped]

    else:
        query_splx_idx = torch.searchsorted(key_splx_ids, query_splx_ids)

    match subset:
        case "assume":
            return query_splx_idx

        case "check":
            query_splx_idx_clamped = torch.clamp_max(query_splx_idx, n_key_splx - 1)
            query_splx_is_found = key_splx_ids[query_splx_idx_clamped] == query_splx_ids
            sentinel = torch.tensor(
                -1, dtype=query_splx.dtype, device=query_splx.device
            )
            return torch.where(
                query_splx_is_found,
                query_splx_idx,
                sentinel,
            )

        case "assert":
            query_splx_idx_clamped = torch.clamp_max(query_splx_idx, n_key_splx - 1)
            query_splx_is_found = key_splx_ids[query_splx_idx_clamped] == query_splx_ids
            if query_splx_is_found.all():
                return query_splx_idx
            else:
                raise ValueError("The 'query_splx' is not a subset of the 'key_splx'.")

        case _:
            raise ValueError(f"Unknown subset argument '{subset}'.")


def splx_search(
    key_splx: Integer[LongTensor, "key_splx vert"],
    query_splx: Integer[LongTensor, "*b query_splx vert"],
    *,
    sort_key_splx: bool,
    sort_key_vert: bool,
    sort_query_vert: bool,
    method: Literal["lex_sort", "polynomial_hash", "auto"] = "auto",
    subset: Literal["assume", "check", "assert"] = "assume",
) -> Integer[LongTensor, "*b query_splx"]:
    """
    Find the indices of a set of query simplices on a list of key simplices.

    Parameters
    ----------
    key_splx : [key_splx, vert]
        A list of unique key simplices. No leading batch dimensions are allowed.
        No duplicate key simplices are allowed, but this uniqueness condition is
        not checked in this function.
    query_splx : [*b, query_splx, vert]
        A list of query simplices. Arbitrary leading batch dimensions are allowed,
        and duplicate simplices are allowed.
    sort_key_splx
        Whether to sort the key simplices in lex order; must be True if the key
        simplices are not already lex-sorted. The key simplices are lex-sorted if
        they are arranged (in ascending order) based on the first vertex index,
        with tie breaks based on the second vertex index, and so on.
    sort_key_vert
        Whether to sort the vert indices of the key simplices in ascending order;
        must be True if the key simplices are not already put in such "canonical"
        permutation. This sort ensures that the simplex search is invariant to
        vertex permutations.
    sort_query_vert
        Whether to sort the vert indices of the query simplices in ascending
        order; must be True if the query simplices are not already put in such
        "canonical" permutation. This sort ensures that the simplex search is
        invariant to vertex permutations.
    method
        This function supports two search methods based on either lex sort ("lex_sort")
        or polynomial hashing ("polynomial_hash"). In general, polynomial hashing
        is faster than lex sort, but is not as overflow-safe as lex sort. As such,
        polynomial hashing is not recommended for searching over k-simplices for
        k >= 3. If set to "auto:, use polynomial hashing if k < 2 and lex sort
        if k >= 2.
    subset
        How to handle subset membership. If "assume", then assume that the given
        query simplex set is a subset of the key simplex set; if "check", then
        an additional subset check is performed after the search, and query simplices
        that are not in the key simplex set will be given an index value of -1;
        if "assert", then the same subset check is performed after the search,
        but an exception is raised if any non-subset membership is detected.

    Returns
    -------
    [*b, query_splx]
        The indices of the query simplices in the key simplex set.
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
                subset=subset,
            )
        case "polynomial_hash":
            splx_idx_flat = _polynomial_hash_splx_search(
                key_splx,
                query_splx_flat,
                sort_key_splx=sort_key_splx,
                sort_key_vert=sort_key_vert,
                sort_query_vert=sort_query_vert,
                subset=subset,
            )
        case _:
            raise ValueError("Unrecognized 'method' argument.")

    splx_idx_shaped = splx_idx_flat.view(*query_splx.shape[:-1])

    return splx_idx_shaped

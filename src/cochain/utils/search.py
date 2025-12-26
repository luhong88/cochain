from typing import Literal

import torch as t
from jaxtyping import Integer


def _polynomial_hash_simplex_search(
    key_simps: Integer[t.Tensor, "key_simp vert"],
    query_simps: Integer[t.Tensor, "query_simp vert"],
    *,
    sort_key_simp: bool,
    sort_key_vert: bool,
    sort_query_vert: bool,
) -> Integer[t.Tensor, " query_simp"]:
    dtype = t.int64
    device = key_simps.device

    # Note that a k-simplex has k + 1 vertices.
    simp_dim = key_simps.size(1) - 1
    if key_simps.size(1) != query_simps.size(1):
        raise ValueError(
            "The 'key_simps' and 'query_simps' must have the same dimension."
        )

    key_simps_canon = key_simps.sort(dim=-1).values if sort_key_vert else key_simps
    query_simps_canon = (
        query_simps.sort(dim=-1).values if sort_query_vert else query_simps
    )

    min_vert_idx = min(key_simps_canon.min(), query_simps_canon.min())
    key_simps_reduced = key_simps_canon - min_vert_idx
    query_simps_reduced = query_simps_canon - min_vert_idx
    # +1 to ensure base > max digit
    max_vert_idx = max(key_simps_reduced.max(), query_simps_reduced.max()).to(dtype) + 1

    exponent = t.arange(simp_dim, -1, -1, device=device, dtype=dtype)
    coef = t.pow(max_vert_idx, exponent)

    max_allowed_hash_val = t.iinfo(dtype).max
    worst_case_hash_val = float(max_vert_idx) ** (simp_dim + 1)
    if worst_case_hash_val > max_allowed_hash_val:
        raise RuntimeError(
            "Potential polynomial hash overflow detected. Use method='lex_sort' instead."
        )

    key_simps_packed = t.einsum("sv,v->s", key_simps_reduced, coef)
    query_simps_packed = t.einsum("sv,v->s", query_simps_reduced, coef)

    if sort_key_simp:
        key_simps_packed_sorted, key_simps_packed_sort_idx = key_simps_packed.sort()
        query_simps_idx_sorted = t.searchsorted(
            key_simps_packed_sorted, query_simps_packed
        )
        query_simps_idx = key_simps_packed_sort_idx[query_simps_idx_sorted]

    else:
        query_simps_idx = t.searchsorted(key_simps_packed, query_simps_packed)

    return query_simps_idx


def _lex_simplex_search(
    key_simps: Integer[t.Tensor, "key_simp vert"],
    query_simps: Integer[t.Tensor, "query_simp vert"],
    *,
    sort_key_simp: bool,
    sort_key_vert: bool,
    sort_query_vert: bool,
) -> Integer[t.Tensor, " query_simp"]:
    if key_simps.size(1) != query_simps.size(1):
        raise ValueError(
            "The 'key_simps' and 'query_simps' must have the same dimension."
        )

    key_simps_canon = key_simps.sort(dim=-1).values if sort_key_vert else key_simps
    query_simps_canon = (
        query_simps.sort(dim=-1).values if sort_query_vert else query_simps
    )

    combined_simps = t.vstack((key_simps_canon, query_simps_canon))
    _, combined_simp_ids = combined_simps.unique(
        dim=0, sorted=True, return_inverse=True
    )

    n_key_simps = key_simps_canon.size(0)
    key_simp_ids = combined_simp_ids[:n_key_simps]
    query_simp_ids = combined_simp_ids[n_key_simps:]

    if sort_key_simp:
        key_simp_ids_sorted, key_simps_id_sort_idx = key_simp_ids.sort()
        query_simps_idx_sorted = t.searchsorted(key_simp_ids_sorted, query_simp_ids)
        query_simps_idx = key_simps_id_sort_idx[query_simps_idx_sorted]

    else:
        query_simps_idx = t.searchsorted(key_simp_ids, query_simp_ids)

    return query_simps_idx


def simplex_search(
    key_simps: Integer[t.Tensor, "key_simp vert"],
    query_simps: Integer[t.Tensor, "query_simp vert"],
    *,
    sort_key_simp: bool,
    sort_key_vert: bool,
    sort_query_vert: bool,
    method: Literal["lex_sort", "polynomial_hash"] = "lex_sort",
) -> Integer[t.Tensor, " query_simp"]:
    """
    Search for simplices in `query_simps` in the `key_simps`. It uses polynomial
    rolling hash to convert the simplex vert index tuples into integers and use
    `searchsorted()` to perform the search.

    This function supports two search methods based on either lex sort or
    polynomial hashing. In general, polynomial hashing is faster than lex sort,
    but is not as overflow-safe as lex sort. As such, polynomial hashing is not
    recommended for searching over k-simplices for k >= 3.

    This function has the following requirements:
    * `query_simps` must be a subset of `key_simps` (up to vertex permutation).
    * `key_simps` cannot contain duplicates, but `query_simps` can.
    * Each simplex in `key_simps` and `query_simps` must be represented by vertex
    indices in ascending order; if this is not true, set `sort_key_vert=True` and/or
    `sort_query_vert=True`.
    * The simplices in `key_simps` must be sorted in lex order; i.e., the simplices
    in `key_simps` need to be aranged (in ascending order) based on the first vertex
    index, with tie breaks based on the second vertex index, and so on. If this is
    not true, set `sort_key_simp=True`.
    """
    match method:
        case "lex_sort":
            return _lex_simplex_search(
                key_simps,
                query_simps,
                sort_key_simp=sort_key_simp,
                sort_key_vert=sort_key_vert,
                sort_query_vert=sort_query_vert,
            )
        case "polynomial_hash":
            return _polynomial_hash_simplex_search(
                key_simps,
                query_simps,
                sort_key_simp=sort_key_simp,
                sort_key_vert=sort_key_vert,
                sort_query_vert=sort_query_vert,
            )
        case _:
            raise ValueError("Unrecognized 'method' argument.")

from collections import defaultdict

import torch as t
from jaxtyping import Float

from ..complex import SimplicialComplex


def laplacian_k(
    sc: SimplicialComplex, k: int
) -> tuple[
    Float[t.Tensor, "k_simp k_simp"],
    Float[t.Tensor, "k_simp k_simp"],
    Float[t.Tensor, "k_simp k_simp"],
]:
    """
    Laplacian_k = d_j @ d_j.T + d_k.T @ d_k, where d_k is the k-coboundary
    operator, d_k.T is the k-boundary operator, and j = k - 1.
    """
    # Get the default dtype and device from d_0
    dtype = getattr(sc, "coboundary_0").dtype
    device = getattr(sc, "coboundary_0").device

    # Generate a dictionary that maps the kth-dimension to the number of k-simplices,
    # with default value set to 0.
    dim_dict = defaultdict(int)
    for dim, n_simplices in enumerate([sc.n_verts, sc.n_edges, sc.n_tris, sc.n_tets]):
        dim_dict[dim] = n_simplices

    # Get the k-th and (k-1)th- coboundary operator (or, generate an empty one with
    # the appropriate dimensions for "out-of-bound" values of k), and use them to
    # construct the Laplacian.
    d_k = getattr(
        sc,
        f"coboundary_{k}",
        t.sparse_coo_tensor(
            indices=t.empty((2, 0), dtype=t.long, device=device),
            values=t.empty((0,), dtype=dtype, device=device),
            size=(dim_dict[k + 1], dim_dict[k]),
        ),
    )
    d_k_T = d_k.transpose(0, 1).coalesce()
    up_laplacian = (d_k_T @ d_k).coalesce()

    d_j = getattr(
        sc,
        f"coboundary_{k - 1}",
        t.sparse_coo_tensor(
            indices=t.empty((2, 0), dtype=t.long, device=device),
            values=t.empty((0,), dtype=dtype, device=device),
            size=(dim_dict[k], dim_dict[k - 1]),
        ),
    )
    d_j_T = d_j.transpose(0, 1).coalesce()
    down_laplacian = (d_j @ d_j_T).coalesce()

    laplacian_k = (up_laplacian + down_laplacian).coalesce()

    return down_laplacian, up_laplacian, laplacian_k

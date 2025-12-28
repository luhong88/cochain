import torch as t
from jaxtyping import Float, Integer

from ..utils.search import simplex_search


def get_edge_face_idx(
    tris: Integer[t.LongTensor, "tri 3"],
    edges: Integer[t.LongTensor, "edge 2"],
) -> Integer[t.LongTensor, "tri 3"]:
    """
    Enumerate all edges for each tri and find their orientations and indices on
    the tri_mesh.edges list.
    """
    # Enumerate all unique edges via their vertex position in the tris.
    i, j, k = 0, 1, 2

    # For each tri and each unique edge pair, find the orientations of the edges
    # and their indices on the list of unique, canonical edges (tri_mesh.edges).
    all_edges: Float[t.Tensor, "tri*3 2"] = tris[:, [[i, j], [i, k], [j, k]]].flatten(
        end_dim=-2
    )

    canon_edges_idx = simplex_search(
        key_simps=edges,
        query_simps=all_edges,
        sort_key_simp=False,
        sort_key_vert=False,
        sort_query_vert=True,
        method="polynomial_hash",
    ).view(-1, 3)

    return canon_edges_idx


def get_edge_face_orientations(
    tris: Integer[t.LongTensor, "tri 3"],
) -> Float[t.Tensor, "tri 3"]:
    """
    Enumerate all edges for each tri and find their orientations and indices on
    the tri_mesh.edges list.
    """
    # Enumerate all unique edges via their vertex position in the tris.
    i, j, k = 0, 1, 2

    # For each tri and each unique edge pair, find the orientations of the edges
    # and their indices on the list of unique, canonical edges (tri_mesh.edges).
    all_edges: Float[t.Tensor, "tri*3 2"] = tris[:, [[i, j], [i, k], [j, k]]].flatten(
        end_dim=-2
    )

    # Same method as used in the construction of coboundary operators to use
    # sort() to identify edge orientations.
    canon_edge_orientations = all_edges.sort(dim=-1).indices
    canon_edge_signs = t.where(
        canon_edge_orientations[:, 1] > 0, canon_edge_orientations[:, 1], -1
    ).view(-1, 3)

    return canon_edge_signs

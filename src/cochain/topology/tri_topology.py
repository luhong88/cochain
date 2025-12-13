import torch as t
from jaxtyping import Float, Integer


def get_edge_face_idx(
    tris: Integer[t.LongTensor, "tri 3"],
    edges: Integer[t.LongTensor, "edge 2"],
    n_verts: int,
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

    # Same method as used in the construction of coboundary operators to use
    # sort() to identify edge orientations.
    all_canon_edges = all_edges.sort(dim=-1).values

    # This assumes that the edge indices in tri_mesh.edges are already in canonical
    # orders.
    unique_canon_edges_packed = edges[:, 0] * n_verts + edges[:, 1]
    canon_edges_packed_sorted, canon_edges_idx = t.sort(unique_canon_edges_packed)

    canon_edges_packed = all_canon_edges[:, 0] * n_verts + all_canon_edges[:, 1]
    canon_edges_idx: Float[t.Tensor, "tri 3"] = canon_edges_idx[
        t.searchsorted(canon_edges_packed_sorted, canon_edges_packed)
    ].view(-1, 3)

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

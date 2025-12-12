import torch as t
from jaxtyping import Float, Integer


def get_edge_face_idx(
    tets: Integer[t.LongTensor, "tet 4"],
    edges: Integer[t.LongTensor, "edge 2"],
    n_verts: int,
) -> Integer[t.LongTensor, "tet 6"]:
    """
    Enumerate all edges for each tet and find their indices on the tet_mesh.edges list.
    """

    i, j, k, l = 0, 1, 2, 3

    # For each tet and each unique edge pair, find the orientations of the edges
    # and their indices on the list of unique, canonical edges (tet_mesh.edges).
    all_edges: Float[t.Tensor, "tet*6 2"] = tets[
        :, [[i, j], [i, k], [j, k], [j, l], [k, l], [i, l]]
    ].flatten(end_dim=-2)

    # Same method as used in the construction of coboundary operators to use
    # sort() to identify edge orientations.
    all_canon_edges = all_edges.sort(dim=-1).values

    # This assumes that the edge indices in tet_mesh.edges are already in canonical
    # orders.
    unique_canon_edges_packed = edges[:, 0] * n_verts + edges[:, 1]
    canon_edges_packed_sorted, canon_edges_idx = t.sort(unique_canon_edges_packed)

    canon_edges_packed = all_canon_edges[:, 0] * n_verts + all_canon_edges[:, 1]
    canon_edges_idx: Float[t.Tensor, "tet 6"] = canon_edges_idx[
        t.searchsorted(canon_edges_packed_sorted, canon_edges_packed)
    ].view(-1, 6)

    return canon_edges_idx


def get_edge_face_orientations(
    tets: Integer[t.LongTensor, "tet 4"],
) -> Float[t.Tensor, "tet 6"]:
    """
    Enumerate all edges for each tet and find their orientations relative to the
    canonical edges on the tet_mesh.edges list.
    """
    i, j, k, l = 0, 1, 2, 3

    # For each tet and each unique edge pair, find the orientations of the edges
    # and their indices on the list of unique, canonical edges (tet_mesh.edges).
    all_edges: Float[t.Tensor, "tet*6 2"] = tets[
        :, [[i, j], [i, k], [j, k], [j, l], [k, l], [i, l]]
    ].flatten(end_dim=-2)

    # Same method as used in the construction of coboundary operators to use
    # sort() to identify edge orientations.
    canon_edge_orientations = all_edges.sort(dim=-1).indices
    canon_edge_signs = t.where(
        canon_edge_orientations[:, 1] > 0, canon_edge_orientations[:, 1], -1
    ).view(-1, 6)

    return canon_edge_signs


def get_tri_face_idx(
    tets: Integer[t.LongTensor, "tet 4"],
    tris: Integer[t.LongTensor, "tri 3"],
    n_verts: int,
) -> Integer[t.LongTensor, "tet 4"]:
    """
    For each tet and each of its vertices, find the triangle face opposite to the
    vertex and its index in the tet_mesh.tris list.
    """
    i, j, k, l = 0, 1, 2, 3

    # For each tet and each vertex, triangle opposite to the vertex.
    all_tris: Integer[t.LongTensor, "tet 4 3"] = tets[
        :, [[j, k, l], [i, l, k], [i, j, l], [i, k, j]]
    ]

    all_canon_tris = all_tris.sort(dim=-1).values

    # Find the indices of the triangles on the list of unique, canonical triangles
    # (tet_mesh.tris) by radix encoding and searchsorted(). Because each triangle
    # ijk is encoded as i*n_verts^2 + j*n_verts + k and the max value of t.int64
    # is ~ 2^63, the max number of vertices this method can accommodate is
    # ~ n_verts < 2^21. Note that this method assumes that the triangle indices
    # in tet_mesh.tris are already in canonical orders.
    unique_canon_tris_packed = (
        tris[:, 0] * n_verts**2 + tris[:, 1] * n_verts + tris[:, 2]
    )
    unique_canon_tris_packed_sorted, unique_canon_tris_idx = t.sort(
        unique_canon_tris_packed
    )

    all_canon_tris_flat: Integer[t.LongTensor, "tet*4 3"] = all_canon_tris.flatten(
        end_dim=-2
    )
    all_canon_tris_packed = (
        all_canon_tris_flat[:, 0] * n_verts**2
        + all_canon_tris_flat[:, 1] * n_verts
        + all_canon_tris_flat[:, 2]
    )

    all_canon_tris_idx: Integer[t.LongTensor, "tet 4"] = unique_canon_tris_idx[
        t.searchsorted(unique_canon_tris_packed_sorted, all_canon_tris_packed)
    ].view(-1, 4)

    return all_canon_tris_idx

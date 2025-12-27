import torch as t
from jaxtyping import Float, Integer

from ..utils.perm_parity import compute_lex_rel_orient
from ..utils.search import simplex_search


def get_edge_face_idx(
    tets: Integer[t.LongTensor, "tet 4"],
    edges: Integer[t.LongTensor, "edge 2"],
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

    canon_edges_idx = simplex_search(
        key_simps=edges,
        query_simps=all_edges,
        sort_key_simp=False,
        sort_key_vert=False,
        sort_query_vert=True,
        method="polynomial_hash",
    ).view(-1, 6)

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
) -> Integer[t.LongTensor, "tet 4"]:
    """
    For each tet and each of its vertices, find the triangle face opposite to the
    vertex and its index in the tet_mesh.tris list.
    """
    i, j, k, l = 0, 1, 2, 3

    # For each tet and each vertex, triangle opposite to the vertex.
    all_tris: Integer[t.LongTensor, "tet*4 3"] = tets[
        :, [[j, k, l], [i, l, k], [i, j, l], [i, k, j]]
    ].flatten(end_dim=-2)

    all_canon_tris_idx = simplex_search(
        key_simps=tris,
        query_simps=all_tris,
        sort_key_simp=False,
        sort_key_vert=False,
        sort_query_vert=True,
        method="lex_sort",
    ).view(-1, 4)

    return all_canon_tris_idx


def get_tri_face_orientations(
    tets: Integer[t.LongTensor, "tet 4"],
) -> Float[t.Tensor, "tet 4"]:
    i, j, k, l = 0, 1, 2, 3

    # For each tet and each vertex, find the outward-facing triangle opposite
    # to the vertex (note that the way the triangles are indexed here satisfies
    # the right-hand rule for positively oriented tets).
    all_tris: Integer[t.LongTensor, "tet 4 3"] = tets[
        :, [[j, k, l], [i, l, k], [i, j, l], [i, k, j]]
    ]

    canon_pos_orientation = t.tensor([0, 1, 2], dtype=t.long, device=tets.device)

    all_tris_orientations = all_tris.sort(dim=-1).indices
    # Same method as used in the construction of coboundary operators to use
    # sort() to identify triangle orientations.
    all_tris_signs: Float[t.Tensor, "tet 4"] = t.where(
        condition=t.sum(all_tris_orientations == canon_pos_orientation, dim=-1) == 1,
        self=-1.0,
        other=1.0,
    )

    return all_tris_signs

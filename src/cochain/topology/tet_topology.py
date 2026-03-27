import torch as t
from jaxtyping import Float, Integer

from ..utils.faces import enumerate_local_faces
from ..utils.perm_parity import compute_lex_rel_orient
from ..utils.search import splx_search


def get_edge_face_idx(
    tets: Integer[t.LongTensor, "tet 4"],
    edges: Integer[t.LongTensor, "edge 2"],
) -> Integer[t.LongTensor, "tet 6"]:
    """
    Enumerate all edges for each tet and find their indices on the tet_mesh.edges list.
    """
    local_edge_idx = enumerate_local_faces(splx_dim=3, face_dim=1, device=tets.device)

    # For each tet and each unique edge pair, find the orientations of the edges
    # and their indices on the list of unique, canonical edges (tet_mesh.edges).
    all_edges: Float[t.Tensor, "tet 6 2"] = tets[:, local_edge_idx]

    canon_edges_idx = splx_search(
        key_splx=edges,
        query_splx=all_edges,
        sort_key_splx=False,
        sort_key_vert=False,
        sort_query_vert=True,
        method="polynomial_hash",
    )

    return canon_edges_idx


def get_edge_face_orientations(
    tets: Integer[t.LongTensor, "tet 4"],
) -> Float[t.Tensor, "tet 6"]:
    """
    Enumerate all edges for each tet and find their orientations relative to the
    canonical edges on the tet_mesh.edges list.
    """
    local_edge_idx = enumerate_local_faces(splx_dim=3, face_dim=1, device=tets.device)

    all_edges: Float[t.Tensor, "tet 6 2"] = tets[:, local_edge_idx]

    edge_signs = compute_lex_rel_orient(all_edges)

    return edge_signs


def get_tri_face_idx(
    tets: Integer[t.LongTensor, "tet 4"],
    tris: Integer[t.LongTensor, "tri 3"],
) -> Integer[t.LongTensor, "tet 4"]:
    """
    For each tet and each of its vertices, find the triangle face opposite to the
    vertex and its index in the tet_mesh.tris list.
    """
    local_tri_idx = enumerate_local_faces(splx_dim=3, face_dim=2, device=tets.device)

    all_tris: Integer[t.LongTensor, "tet 4 3"] = tets[:, local_tri_idx]

    all_canon_tris_idx = splx_search(
        key_splx=tris,
        query_splx=all_tris,
        sort_key_splx=False,
        sort_key_vert=False,
        sort_query_vert=True,
        method="lex_sort",
    )

    return all_canon_tris_idx


def get_tri_face_orientations(
    tets: Integer[t.LongTensor, "tet 4"],
) -> Float[t.Tensor, "tet 4"]:
    local_tri_idx = enumerate_local_faces(splx_dim=3, face_dim=2, device=tets.device)

    all_tris: Integer[t.LongTensor, "tet 4 3"] = tets[:, local_tri_idx]

    tris_signs = compute_lex_rel_orient(all_tris)

    return tris_signs

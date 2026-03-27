import torch as t
from jaxtyping import Float, Integer

from ..utils.faces import enumerate_faces
from ..utils.perm_parity import compute_lex_rel_orient
from ..utils.search import splx_search


def get_edge_face_idx(
    tris: Integer[t.LongTensor, "tri 3"],
    edges: Integer[t.LongTensor, "edge 2"],
) -> Integer[t.LongTensor, "tri 3"]:
    """
    Enumerate all edges for each tri and find their orientations and indices on
    the tri_mesh.edges list.
    """
    # Enumerate all unique edges via their vertex position in the tris.
    local_edge_idx = enumerate_faces(simp_dim=2, face_dim=1, device=tris.device)

    all_edges: Float[t.Tensor, "tri*3 2"] = tris[:, local_edge_idx].flatten(end_dim=-2)

    canon_edges_idx = splx_search(
        key_splx=edges,
        query_splx=all_edges,
        sort_key_splx=False,
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
    local_edge_idx = enumerate_faces(simp_dim=2, face_dim=1, device=tris.device)

    all_edges: Float[t.Tensor, "tri*3 2"] = tris[:, local_edge_idx].flatten(end_dim=-2)

    edge_signs = compute_lex_rel_orient(all_edges).view(-1, 3)

    return edge_signs

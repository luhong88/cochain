import itertools
from typing import NamedTuple

import torch
from jaxtyping import Float, Integer
from torch import Tensor
from torch.utils._pytree import register_pytree_node

from .perm_parity import compute_lex_rel_orient
from .search import splx_search


class GlobalFaces(NamedTuple):
    idx: Integer[Tensor, "splx face"]
    parity: Float[Tensor, "splx face"]


# Register GlobalFaces in PyTree
def _flatten_global_faces(faces: GlobalFaces):
    leaves = [faces.idx, faces.parity]
    context = None  # There is no non-tensor metadata (e.g., str, int)
    return leaves, context


def _unflatten_global_faces(leaves, context) -> GlobalFaces:
    return GlobalFaces(idx=leaves[0], parity=leaves[1])


register_pytree_node(GlobalFaces, _flatten_global_faces, _unflatten_global_faces)


def enumerate_local_faces(
    splx_dim: int, face_dim: int, device: torch.device
) -> Integer[Tensor, "face vert"]:
    """
    For a simplex of dimension `splx_dim`, enumerate all faces of dimension
    `face_dim` (up to vertex index permutation) in local index lex order.
    """
    if face_dim > splx_dim:
        raise ValueError()

    return torch.tensor(
        list(itertools.combinations(list(range(splx_dim + 1)), face_dim + 1)),
        device=device,
    )


def enumerate_global_faces(
    m_splx: Integer[Tensor, "m_splx m_vert"],
    k_splx: Integer[Tensor, "k_splx k_vert"],
    float_dtype: torch.dtype = torch.float32,
) -> GlobalFaces:
    """
    Given a simplicial m-complex, for each top level m-simplex, find all of its
    k-faces; then, find the indices of the k-faces on the list of canonical
    k-simplices in the mesh, and compute their permutation sign/parity relative
    to the canonical k-simplices.
    """
    k = k_splx.size(-1) - 1
    m = m_splx.size(-1) - 1
    device = m_splx.device

    if k > m:
        raise ValueError()

    k_faces: Float[Tensor, "m_splx k_face k+1"] = m_splx[
        :, enumerate_local_faces(splx_dim=m, face_dim=k, device=device)
    ]
    # If m is the mesh dimension, then the key splx/vert requires sorting only
    # if k == m, because all but the top-level simplices are already lex-sorted.
    # If m is less than the mesh dimension, then the key splx/vert never requires
    # sorting (and the if-else ternary expression is unnecessary and potentially
    # wasteful).
    k_faces_idx: Integer[Tensor, "m_splx k_face"] = splx_search(
        key_splx=k_splx,
        query_splx=k_faces,
        sort_key_splx=True if k == m else False,
        sort_key_vert=True if k == m else False,
        sort_query_vert=True,
    )

    if k < m:
        k_face_parity = compute_lex_rel_orient(k_faces, dtype=float_dtype)
    else:
        k_face_parity = torch.ones_like(k_faces_idx, dtype=float_dtype, device=device)

    return GlobalFaces(idx=k_faces_idx, parity=k_face_parity)

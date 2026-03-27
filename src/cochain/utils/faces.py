import itertools
from typing import NamedTuple

import torch as t
from jaxtyping import Float, Integer

from .perm_parity import compute_lex_rel_orient
from .search import splx_search


def enumerate_local_faces(
    splx_dim: int, face_dim: int, device: t.device
) -> Integer[t.LongTensor, "face vert"]:
    """
    For a simplex of dimension `splx_dim`, enumerate all faces of dimension
    `face_dim` (up to vertex index permutation) in local index lex order.
    """
    if face_dim > splx_dim:
        raise ValueError()

    return t.tensor(
        list(itertools.combinations(list(range(splx_dim + 1)), face_dim + 1)),
        device=device,
    )


class GlobalFaces(NamedTuple):
    idx: Integer[t.LongTensor, "splx face"]
    parity: Float[t.Tensor, "splx face"]


def enumerate_global_faces(
    m_splx: Integer[t.LongTensor, "m_splx m_vert"],
    k_splx: Integer[t.LongTensor, "k_splx k_vert"],
    float_dtype: t.dtype = t.float32,
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

    k_faces: Float[t.Tensor, "m_splx k_face k+1"] = m_splx[
        :, enumerate_local_faces(splx_dim=m, face_dim=k, device=device)
    ]
    k_faces_idx: Integer[t.LongTensor, "m_splx k_face"] = splx_search(
        key_splx=k_splx,
        query_splx=k_faces,
        sort_key_splx=True if k == m else False,
        sort_key_vert=True if k == m else False,
        sort_query_vert=True,
    )

    if k < m:
        k_face_parity = compute_lex_rel_orient(k_faces, dtype=float_dtype)
    else:
        k_face_parity = t.ones_like(k_faces_idx, dtype=float_dtype, device=device)

    return GlobalFaces(idx=k_faces_idx, parity=k_face_parity)

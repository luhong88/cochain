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
    Enumerate all faces using local vertex indices.

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
    is_k_lex_sorted: bool,
    float_dtype: torch.dtype = torch.float32,
) -> GlobalFaces:
    """
    Find the global indices and parity of all faces of a given dimension.

    Parameters
    ----------
    m_splx : [m_splx, m_vert]
        The list of m-simplices whose k-faces will be enumerated.
    k_splx : [k_splx, k_vert]
        The list of canonical k-simplices.
    is_k_lex_sorted
        Whether the canonical k-simplices are lex sorted.
    float_dtype
        The dtype for the permutation sign/parity of the k-faces.

    Returns
    -------
    A `GlobalFaces` named tuple with the following attributes

    idx : [m_splx, k_face]
        The indices of the k-faces of each m-simplex on the list of canonical
        k-simplices; note that identity between a k-face and canonical
        k-simplex is determined up to vertex permutation.
    parity : [m_splx, k_face]
        The permutation sign/parity of the k-faces relative to the lex-sorted
        canonical k-simplices.

    Notes
    -----
    In general, we assume that the canonical k-simplices attached to the
    SimplicialMesh are lex-sorted, except for the top-level simplices, which
    can carry geometric orientation information and may not be lex-sorted.
    Therefore, `is_k_lex_sorted` is typically True whenever `k < mesh.dim`.
    """
    k = k_splx.size(-1) - 1
    m = m_splx.size(-1) - 1
    n_m_splx = m_splx.size(0)

    int_dtype = m_splx.dtype
    device = m_splx.device

    match k:
        case 0:
            return GlobalFaces(
                idx=m_splx,
                parity=torch.ones_like(m_splx, dtype=float_dtype, device=device),
            )

        case _ if k == m and m_splx is k_splx:
            # When m_splx and k_splx are identical, the GlobalFaces is trivial.
            m_face_idx = torch.arange(
                n_m_splx,
                dtype=int_dtype,
                device=device,
            ).view(-1, 1)
            m_face_parity = torch.ones_like(
                m_face_idx, dtype=float_dtype, device=device
            )
            return GlobalFaces(idx=m_face_idx, parity=m_face_parity)

        case _ if k <= m:
            k_faces: Integer[Tensor, "m_splx k_face k+1"] = m_splx[
                :, enumerate_local_faces(splx_dim=m, face_dim=k, device=device)
            ]
            k_face_idx: Integer[Tensor, "m_splx k_face"] = splx_search(
                key_splx=k_splx,
                query_splx=k_faces,
                sort_key_splx=not is_k_lex_sorted,
                sort_key_vert=not is_k_lex_sorted,
                sort_query_vert=True,
            )

            # Compute orientation of k_faces relative to the lex-sorted canonical
            # k_splx basis.
            k_face_parity = compute_lex_rel_orient(k_faces, dtype=float_dtype)

            # If the input k_splx is not lex-sorted (e.g. they possess a geometric
            # orientation), then two parity calculations are required to find the
            # relative orientation: one for the permutation parity of the queried
            # faces (induced parity), and one for the permutation parity of the
            # target canonical k-simplices (global parity).
            if not is_k_lex_sorted:
                k_splx_parity = compute_lex_rel_orient(
                    k_splx[k_face_idx], dtype=float_dtype
                )
                k_face_parity = k_face_parity * k_splx_parity

            return GlobalFaces(idx=k_face_idx, parity=k_face_parity)

        case _:
            # If k > m, then return an empty GlobalFaces object.
            return GlobalFaces(
                idx=torch.empty((n_m_splx, 0), dtype=int_dtype, device=device),
                parity=torch.empty((n_m_splx, 0), dtype=float_dtype, device=device),
            )

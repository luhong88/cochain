__all__ = ["detect_mesh_boundaries"]

import torch
from jaxtyping import Bool, Float
from torch import Tensor

from ..sparse.decoupled_tensor import SparseDecoupledTensor


def detect_mesh_boundaries(
    cbd: tuple[
        Float[SparseDecoupledTensor, "edge vert"],
        Float[SparseDecoupledTensor, "tri edge"],
        Float[SparseDecoupledTensor, "tet tri"],
    ],
) -> tuple[
    Bool[Tensor, " vert"],
    Bool[Tensor, " edge"],
    Bool[Tensor, " tri"],
    Bool[Tensor, " tet"],
]:
    """
    Detect the boundary simplices on a mesh.

    Parameters
    ----------
    cbd
        A tuple containing the 0-, 1-, and 2-coboundary operators. For a tri mesh,
        the 2-coboundary operator is an empty SparseDecoupledTensor of shape (0, tri).

    Returns
    -------
    vert_bd_mask : (vert,)
        A boolean mask for the mesh verts where `True` marks the boundary verts.
    edge_bd_mask : (vert,)
        A boolean mask for the mesh edges where `True` marks the boundary edges.
    tri_bd_mask : (vert,)
        A boolean mask for the mesh tris where `True` marks the boundary tris.
        For a tri mesh, this mask contains `False` only.
    tet_bd_mask : (vert,)
        A boolean mask for the mesh tets where `True` marks the boundary tets.
        For a tri mesh, this is an empty Tensor; for a tet mesh, this mask contains
        `False` only.

    Notes
    -----
    The logic implemented in this function is only valid for pure simplicial
    complexes.
    """
    cbd_ops = [cbd[dim] for dim in [2, 1, 0]]

    # The top-level simplies by definition cannot be boundaries; populate the
    # tet boundary mask with False.
    bd_masks = [
        torch.zeros(
            cbd_ops[0].size(0),
            dtype=torch.bool,
            device=cbd_ops[0].device,
        )
    ]

    is_top_level = True
    for cbd in cbd_ops:
        if cbd._nnz() == 0:
            # Only the 2-coboundary operator can be zero among tri and tet meshes.
            # If the 2-coboundary operator is empty, then there are no tets in
            # the mesh and the tris are top-level and cannot be boundaries.
            n_faces = cbd.size(-1)
            face_is_boundary = torch.zeros(n_faces, dtype=torch.bool, device=cbd.device)
            bd_masks.append(face_is_boundary)

        else:
            # The first non-empty cbd encodes the relation between the top-level
            # simplices and their codim 1 faces.
            if is_top_level:
                # A face of a top-level simplex is on the boundary if it is the
                # face of exactly one top-level simplex. This can be checked by
                # summing over the rows of the absolute values of the coboundary
                # operator, which counts the number of cofaces.
                face_relation_count = cbd.to_sparse_coo().abs().sum(dim=0).to_dense()
                face_is_boundary = torch.isclose(
                    face_relation_count, torch.ones_like(face_relation_count)
                )
                bd_masks.append(face_is_boundary)
                is_top_level = False

            else:
                # If a simplex is on the boundary, then all of its faces are also
                # boundary simplices. The matrix-vector multiplication effectively
                # counts, for each k-simplex, how many boundary (k+1)-simplices
                # it shares a face relation with.
                boundary_face_relation_count = cbd.T.abs() @ bd_masks[-1].to(
                    dtype=cbd.dtype
                )
                face_is_boundary = ~torch.isclose(
                    boundary_face_relation_count,
                    torch.zeros_like(boundary_face_relation_count),
                )
                bd_masks.append(face_is_boundary)

    bd_masks.reverse()

    return bd_masks

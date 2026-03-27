import torch as t
from jaxtyping import Bool, Float

from ..sparse.decoupled_tensor import SparseDecoupledTensor


def detect_mesh_boundaries(
    cbd: tuple[
        Float[SparseDecoupledTensor, "edge vert"],
        Float[SparseDecoupledTensor, "tri edge"],
        Float[SparseDecoupledTensor, "tet tri"],
    ],
) -> tuple[
    Bool[t.Tensor, " vert"],
    Bool[t.Tensor, " edge"],
    Bool[t.Tensor, " tri"],
    Bool[t.Tensor, " tet"],
]:
    """
    Return 4 tensor boolean masks that mark the vert, edge, tri, and tet that are
    at the boundary of the simplicial complex.

    The logic implemented in this function is only valid for pure simplicial
    complexes.
    """
    cbd_ops = [cbd[dim] for dim in [2, 1, 0]]

    # The top-level simplies by definition cannot be boundaries.
    boundary_masks = [
        t.zeros(
            cbd_ops[0].size(0),
            dtype=t.bool,
            device=cbd_ops[0].device,
        )
    ]

    is_top_level = True
    for cbd in cbd_ops:
        if cbd._nnz() == 0:
            # If the k-th coboundary operator is empty, then either there is no
            # (k-1)-simplices, or the (k-1)-simplices are at the top level.
            n_faces = cbd.size(-1)
            face_is_boundary = t.zeros(n_faces, dtype=t.bool, device=cbd.device)
            boundary_masks.append(face_is_boundary)

        else:
            if is_top_level:
                # A face of a top-level simplex is on the boundary if it is the face
                # of exactly one top-level simplex.
                face_relation_count = cbd.to_sparse_coo().abs().sum(dim=0).to_dense()
                face_is_boundary = t.isclose(
                    face_relation_count,
                    t.tensor(
                        1.0,
                        dtype=face_relation_count.dtype,
                        device=face_relation_count.device,
                    ),
                )
                boundary_masks.append(face_is_boundary)
                is_top_level = False

            else:
                # If a simplex is on the boundary, then all of its faces are also
                # boundary simplices. The matrix-vector multiplication effectively
                # counts, for each k-simplex, how many boundary (k+1)-simplices
                # it shares a face relation with.
                boundary_face_relation_count = cbd.T.abs() @ boundary_masks[-1].to(
                    dtype=cbd.dtype
                )
                face_is_boundary = ~t.isclose(
                    boundary_face_relation_count,
                    t.tensor(
                        0.0,
                        dtype=boundary_face_relation_count.dtype,
                        device=boundary_face_relation_count.device,
                    ),
                )
                boundary_masks.append(face_is_boundary)

    boundary_masks.reverse()

    return boundary_masks

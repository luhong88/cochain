__all__ = ["stiffness_matrix"]

import torch
from jaxtyping import Float

from ...complex import SimplicialMesh
from ...sparse.decoupled_tensor import SparseDecoupledTensor
from ._tet_geometry import cotan_weights


def stiffness_matrix(
    tet_mesh: SimplicialMesh,
) -> Float[SparseDecoupledTensor, "vert vert"]:
    """
    Computes the stiffness matrix for a 3D mesh, sometimes also known as the "cotan
    Laplacian".
    """
    # The cotan weight matrix W gives the stiffness matrix except for the diagonal
    # elements.
    sym_stiffness = cotan_weights(tet_mesh.vert_coords, tet_mesh.tets, tet_mesh.n_verts)

    # Compute the diagonal elements of the stiffness matrix.
    stiffness_diag = torch.sparse.sum(sym_stiffness, dim=-1)
    # laplacian_diag.indices() has shape (1, nnz_diag)
    diag_idx = torch.concatenate([stiffness_diag.indices(), stiffness_diag.indices()])

    # Generate the final, complete stiffness matrix.
    stiffness = torch.sparse_coo_tensor(
        torch.hstack((sym_stiffness.indices(), diag_idx)),
        torch.concatenate((sym_stiffness.values(), -stiffness_diag.values())),
    ).coalesce()

    return SparseDecoupledTensor.from_tensor(stiffness)

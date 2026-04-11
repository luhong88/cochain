__all__ = ["stiffness_matrix"]

import torch
from jaxtyping import Float

from ...complex import SimplicialMesh
from ...sparse.decoupled_tensor import SparseDecoupledTensor
from ._tri_geometry import compute_cotan_weights


def stiffness_matrix(
    tri_mesh: SimplicialMesh,
) -> Float[SparseDecoupledTensor, "vert vert"]:
    """Compute the stiffness matrix/cotan Laplacian for a tri mesh."""
    # The cotan weight matrix W gives the stiffness matrix except for the diagonal
    # elements.
    sym_stiffness = compute_cotan_weights(tri_mesh.vert_coords, tri_mesh.tris)

    # Compute the diagonal elements of the stiffness matrix, which is the negative
    # of the corresponding row/column sum.
    stiffness_diag = torch.sparse.sum(sym_stiffness, dim=-1)
    # laplacian_diag.indices() has shape (1, nnz_diag)
    diag_idx = torch.concatenate([stiffness_diag.indices(), stiffness_diag.indices()])

    # Generate the final, complete stiffness matrix.
    stiffness = torch.sparse_coo_tensor(
        torch.hstack((sym_stiffness.indices(), diag_idx)),
        torch.concatenate((sym_stiffness.values(), -stiffness_diag.values())),
    ).coalesce()

    return SparseDecoupledTensor.from_tensor(stiffness)

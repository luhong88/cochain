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
    r_idx, c_idx, vals_off_diag = compute_cotan_weights(tri_mesh)

    # Each term W_ij in the (asymmetric) weight matrix contributes four times
    # to the stiffness matrix S_ij:
    #
    # S_ij += W_ij
    # S_ji += W_ij
    # S_ii -= W_ij
    # W_jj -= W_ij
    #
    # To understand the last two terms, note that S_ii = Σ_j W_ij and
    # S_jj = Σ_i W_ji; since the asymmetric weight matrix only contains W_ij
    # or W_ji, this terms contributes to the diagonal term of both its row and col.
    idx_coo = torch.stack(
        (
            torch.cat((r_idx, c_idx, r_idx, c_idx)),
            torch.cat((c_idx, r_idx, r_idx, c_idx)),
        )
    )

    vals = torch.cat((vals_off_diag, vals_off_diag, -vals_off_diag, -vals_off_diag))

    stiffness = tri_mesh._sparse_coalesced_matrix(
        operator="tri_stiffness_matrix",
        indices=idx_coo,
        values=vals,
        size=(tri_mesh.n_verts, tri_mesh.n_verts),
    )

    return stiffness

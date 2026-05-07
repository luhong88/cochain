__all__ = ["stiffness_matrix"]

import torch
from einops import einsum, rearrange
from jaxtyping import Float
from torch import Tensor

from ...complex import SimplicialMesh
from ...sparse.decoupled_tensor import SparseDecoupledTensor
from ._tet_geometry import compute_tet_signed_vols


def stiffness_matrix(
    tet_mesh: SimplicialMesh,
) -> Float[SparseDecoupledTensor, "vert vert"]:
    r"""
    Compute the stiffness matrix/cotan Laplacian for a tet mesh.

    For edge $e_{ij}$, the cotan weight is given by

    $$W_{ij} = \frac 1 6 \sum_{kl} \|e_{kl}\| \cot\theta^{ij}_{kl}$$

    where $kl$ sums over all vertices $k$ and $l$ such that $ijkl$ forms a tet,
    $\|e_{kl}\|$ is the length of the edge $kl$, and $\theta^{ij}_{kl}$ is the
    interior dihedral angle formed by the two triangles $ikl$ and $jkl$ that
    shares $kl$ as an edge face.

    Note that the weight matrix is symmetric, and $W_{ij} = W_{ji}$.

    Given the weight matrix $W_{ij}$, the stiffness matrix $S_{ij}$ can be
    computed by populating the diagonal of $W_{ij}$ with the negative row or
    column sums.
    """
    i, j, k, l = 0, 1, 2, 3

    tet_vols = torch.abs(compute_tet_signed_vols(tet_mesh.vert_coords, tet_mesh.tets))
    tet_vert_coords = tet_mesh.vert_coords[tet_mesh.tets]

    # For each tet ijkl and each edge s, compute the (outward) normal on the two
    # triangles with o as the shared edge (i.e., th x o and hh x o).
    th_cross_o: Float[Tensor, "tet 6 3"] = torch.cross(
        tet_vert_coords[:, [k, i, l, i, j, i]] - tet_vert_coords[:, [l, l, k, l, k, j]],
        tet_vert_coords[:, [i, j, j, j, i, k]] - tet_vert_coords[:, [l, l, k, l, k, j]],
        dim=-1,
    )
    hh_cross_o: Float[Tensor, "tet 6 3"] = torch.cross(
        tet_vert_coords[:, [j, j, j, k, i, l]] - tet_vert_coords[:, [l, l, k, l, k, j]],
        tet_vert_coords[:, [k, k, i, i, l, i]] - tet_vert_coords[:, [l, l, k, l, k, j]],
        dim=-1,
    )

    # For each tet ijkl and each edge s, compute the cotan weight associated
    # with edge s, |o| * cot(θ_o) / 6, where theta_o is the interior dihedral
    # angle formed by the two triangles with o as the shared edge. This contribution
    # can also be written as <th x o, hh x o> / 36V, where V is the unsigned
    # volume of the tet. To see this, use the fact that cot(θ_o) = <th x o, hh x o>/
    # |(th x o) x (hh x o)| and the quadruple cross product term simplifies to
    # |<th, o x hh>| |o| because of the identity (axb)x(cxd) = (a⋅(bxd))c-(a⋅(bxc)d).
    weight_o: Float[Tensor, "tet 6"] = einsum(
        th_cross_o,
        hh_cross_o,
        1.0 / (36.0 * tet_vols),
        "tet edge coord, tet edge coord, tet -> tet edge",
    )

    # Scatter the local edge s contributions to form the global stiffness matrix.
    r_idx = tet_mesh.tets[:, [i, i, i, j, j, k]].flatten()
    c_idx = tet_mesh.tets[:, [j, k, l, k, l, l]].flatten()

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

    vals_off_diag = rearrange(weight_o, "tet edge -> (tet edge)")
    vals = torch.cat((vals_off_diag, vals_off_diag, -vals_off_diag, -vals_off_diag))

    stiffness = tet_mesh._sparse_coalesced_matrix(
        operator="tet_stiffness_matrix",
        indices=idx_coo,
        values=vals,
        size=(tet_mesh.n_verts, tet_mesh.n_verts),
    )

    return stiffness

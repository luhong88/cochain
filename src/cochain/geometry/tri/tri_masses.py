import torch
from jaxtyping import Float, Integer
from torch import LongTensor, Tensor

from ...complex import SimplicialMesh
from ...sparse.decoupled_tensor import SparseDecoupledTensor
from ._tri_geometry import (
    compute_bc_grad_dots,
    compute_tri_areas,
)
from .tri_hodge_stars import star_2


def mass_0(tri_mesh) -> Float[SparseDecoupledTensor, "vert vert"]:
    """
    Compute the "consistent" Galerkin vertex/0-form mass matrix.
    """
    tri_areas = compute_tri_areas(tri_mesh.vert_coords, tri_mesh.tris)

    ref_local_mass_0 = ((torch.ones(3, 3) + torch.eye(3)) / 12.0).to(
        dtype=tri_mesh.vert_coords.dtype, device=tri_mesh.vert_coords.device
    )
    local_mass_0: Float[Tensor, "tri 3 3"] = tri_areas.view(
        -1, 1, 1
    ) * ref_local_mass_0.view(1, 3, 3)

    r_idx = tri_mesh.tris.view(-1, 3, 1).expand(-1, 3, 3)
    c_idx = tri_mesh.tris.view(-1, 1, 3).expand(-1, 3, 3)

    mass = torch.sparse_coo_tensor(
        indices=torch.vstack((r_idx.flatten(), c_idx.flatten())),
        values=local_mass_0.flatten(),
        size=(tri_mesh.n_verts, tri_mesh.n_verts),
    ).coalesce()

    return SparseDecoupledTensor.from_tensor(mass)


def mass_1(tri_mesh: SimplicialMesh) -> Float[SparseDecoupledTensor, "edge edge"]:
    """
    Compute the Galerkin edge/1-form mass matrix.

    For each tri, each (canonical) edge pair xy and pq and their associated
    Whitney 1-form basis functions W_xy and W_pq contribute the inner product
    term int[W_xy*W_pq*dV] to the mass matrix element M[xy, pq], where

    W_xy(p) = lambda_x(p)*grad_p(lambda_y(p)) - lambda_y(p)*grad_p(lambda_x(p))

    Here, p is a position vector inside the tri and lambda_x(p) is the barycentric
    coordinate function for p wrt the vertex x.
    """
    vert_coords: Float[Tensor, "vert 3"] = tri_mesh.vert_coords
    tris: Integer[LongTensor, "tri 3"] = tri_mesh.tris

    dtype = vert_coords.dtype
    device = vert_coords.device

    n_tris = tri_mesh.n_tris
    n_edges = tri_mesh.n_edges

    # For each tri ijk, compute all pairwise inner products of the barycentric
    # coordinate gradients wrt each pair of vertices.
    tri_areas, bary_coords_grad_dot = compute_bc_grad_dots(vert_coords, tris)

    # For each tri ijk, compute all pairwise integrals of the barycentric coordinates;
    # i.e., int[lambda_i(p)lambda_j(p)dA_ijk]. Using the "magic formula", this
    # integral is A_ijk*(1 + delta_ij)/12, where delta is the Kronecker delta
    # function.
    bary_coords_int: Float[Tensor, "tri 3 3"] = torch.abs(
        tri_areas.view(-1, 1, 1) / 12.0
    ) * (
        torch.ones((n_tris, 3, 3), dtype=dtype, device=device)
        + torch.eye(3, dtype=dtype, device=device).view(1, 3, 3)
    )

    # For each tri ijk, each pair of its edges e1=xy and e2=pq contributes the
    # following term to the mass matrix element M[e1,e2]:
    #
    #         W_xy,pq = I_xp*D_yq - I_xq*D_yp - I_yp*D_xq + I_yq*D_xp
    #
    # Here, I is the barycentric integral (bary_coords_int) and D is the barycentric
    # gradient inner product (barry_coords_grad_dot). Note that, since this expression
    # is "skew-symmetric" wrt the edge orientations (W_yx,pq = -W_xy,pq and
    # W_xy,qp = -W_xy,pq), each non-canonical edge orientation also contributes
    # an overall negative sign.

    # Enumerate all unique edges via their vertex position in the tri.
    i, j, k = 0, 1, 2
    unique_edges = torch.tensor(
        [[i, j], [i, k], [j, k]], dtype=torch.long, device=device
    )

    x_idx = unique_edges[:, 0][:, None]
    y_idx = unique_edges[:, 1][:, None]
    p_idx = unique_edges[:, 0][None, :]
    q_idx = unique_edges[:, 1][None, :]

    # For each tri, find all pairs of Whitney 1-form basis function inner products,
    # i.e., the W_xy,pq.
    whitney_inner_prod: Float[Tensor, "tri 3 3"] = (
        bary_coords_int[:, x_idx, p_idx] * bary_coords_grad_dot[:, y_idx, q_idx]
        - bary_coords_int[:, x_idx, q_idx] * bary_coords_grad_dot[:, y_idx, p_idx]
        - bary_coords_int[:, y_idx, p_idx] * bary_coords_grad_dot[:, x_idx, q_idx]
        + bary_coords_int[:, y_idx, q_idx] * bary_coords_grad_dot[:, x_idx, p_idx]
    )

    # For each tri and each unique edge pair, find the orientations of the edges
    # and their indices on the list of unique, canonical edges (tri_mesh.edges).
    whitney_edge_signs = tri_mesh.edge_faces.parity
    whitney_edges_idx = tri_mesh.edge_faces.idx

    # Multiply the Whitney 1-form inner product by the edge orientation signs
    # to get the contribution from canonical edges.
    whitney_flat_signed: Float[Tensor, "tri 9"] = (
        whitney_inner_prod
        * whitney_edge_signs.view(-1, 1, 3)
        * whitney_edge_signs.view(-1, 3, 1)
    ).flatten(start_dim=-2)

    # Get the canonical edge index pairs for the Whitney 1-form inner products of
    # all 9 edge pairs per tri.
    whitney_flat_r_idx: Float[Tensor, " tri*9"] = (
        whitney_edges_idx.view(-1, 3, 1).expand(-1, 3, 3).flatten()
    )
    whitney_flat_c_idx: Float[Tensor, " tri*9"] = (
        whitney_edges_idx.view(-1, 1, 3).expand(-1, 3, 3).flatten()
    )

    # Assemble the mass matrix.
    mass = torch.sparse_coo_tensor(
        torch.vstack((whitney_flat_r_idx, whitney_flat_c_idx)),
        whitney_flat_signed.flatten(),
        (n_edges, n_edges),
    ).coalesce()

    return SparseDecoupledTensor.from_tensor(mass)


mass_2 = star_2

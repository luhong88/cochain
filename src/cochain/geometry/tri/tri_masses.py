import torch as t
from jaxtyping import Float, Integer

from ...complex import SimplicialComplex
from .tri_geometry import (
    _d2_tri_areas_d2_vert_coords,
    _d_tri_areas_d_vert_coords,
    _tri_areas,
)


def _bary_coord_grad_inner_prods(
    tri_areas: Float[t.Tensor, "tri"],
    d_tri_areas_d_vert_coords: Float[t.Tensor, "tri 3 3"],
) -> Float[t.Tensor, "tri 3 3"]:
    """
    For a tri, let lambda_x(p) be the barycentric coordinate function for p wrt
    a vertex x of the tri. This function computes all pairwise inner products
    of the barycentric coordinate gradients wrt each pair of vertices; i.e., it
    computes <grad_p[lambda_x(p)], grad_p[lambda_y(p)]> for all vertices x and y.
    """
    # The gradient of lambda_i(p) wrt p is given by grad_i(area_ijk)/area_ijk, a
    # constant wrt p.
    bary_coords_grad: Float[t.Tensor, "tri 3 3"] = d_tri_areas_d_vert_coords / tri_areas

    bary_coords_grad_dot: Float[t.Tensor, "tri 3 3"] = t.einsum(
        "tic,tjc->tij", bary_coords_grad, bary_coords_grad
    )

    return bary_coords_grad_dot


def mass_1(tri_mesh: SimplicialComplex) -> Float[t.Tensor, "edge edge"]:
    """
    Compute the Galerkin edge/1-form mass matrix.

    For each tri, each (canonical) edge pair xy and pq and their associated
    Whitney 1-form basis functions W_xy and W_pq contribute the inner product
    term int[W_xy*W_pq*dV] to the mass matrix element M[xy, pq], where

    W_xy(p) = lambda_x(p)*grad_p(lambda_y(p)) - lambda_y(p)*grad_p(lambda_x(p))

    Here, p is a position vector inside the tri and lambda_x(p) is the barycentric
    coordinate function for p wrt the vertex x.
    """
    vert_coords: Float[t.Tensor, "vert 3"] = tri_mesh.vert_coords
    tris: Integer[t.LongTensor, "tri 3"] = tri_mesh.tris

    dtype = vert_coords.dtype
    device = vert_coords.device

    n_tris = tri_mesh.n_tris
    n_edges = tri_mesh.n_edges

    tri_areas = _tri_areas(vert_coords, tris).view(-1, 1, 1)
    d_tri_areas_d_vert_coords = _d_tri_areas_d_vert_coords(vert_coords, tris)

    # For each tri ijk, compute all pairwise inner products of the barycentric
    # coordinate gradients wrt each pair of vertices.
    bary_coords_grad_dot: Float[t.Tensor, "tri 3 3"] = _bary_coord_grad_inner_prods(
        tri_areas, d_tri_areas_d_vert_coords
    )

    # For each tri ijk, compute all pairwise integrals of the barycentric coordinates;
    # i.e., int[lambda_i(p)lambda_j(p)dA_ijk]. Using the "magic formula", this
    # integral is A_ijk*(1 + delta_ij)/12, where delta is the Kronecker delta
    # function.
    bary_coords_int: Float[t.Tensor, "tri 3 3"] = t.abs(tri_areas / 12.0) * (
        t.ones((n_tris, 3, 3), dtype=dtype, device=device)
        + t.eye(3, dtype=dtype, device=device).view(1, 3, 3)
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
    unique_edges = t.tensor([[i, j], [i, k], [j, k]], dtype=t.long, device=device)

    x_idx = unique_edges[:, 0][:, None]
    y_idx = unique_edges[:, 1][:, None]
    p_idx = unique_edges[:, 0][None, :]
    q_idx = unique_edges[:, 1][None, :]

    # For each tri, find all pairs of Whitney 1-form basis function inner products,
    # i.e., the W_xy,pq.
    whitney_inner_prod: Float[t.Tensor, "tri 3 3"] = (
        bary_coords_int[:, x_idx, p_idx] * bary_coords_grad_dot[:, y_idx, q_idx]
        - bary_coords_int[:, x_idx, q_idx] * bary_coords_grad_dot[:, y_idx, p_idx]
        - bary_coords_int[:, y_idx, p_idx] * bary_coords_grad_dot[:, x_idx, q_idx]
        + bary_coords_int[:, y_idx, q_idx] * bary_coords_grad_dot[:, x_idx, p_idx]
    )

    # For each tri and each unique edge pair, find the orientations of the edges
    # and their indices on the list of unique, canonical edges (tri_mesh.edges).
    whitney_edge_signs = tri_mesh.tri_edge_orientations
    whitney_edges_idx = tri_mesh.tri_edge_idx

    # Multiply the Whitney 1-form inner product by the edge orientation signs
    # to get the contribution from canonical edges.
    whitney_flat_signed: Float[t.Tensor, "tri 9"] = (
        whitney_inner_prod
        * whitney_edge_signs.view(-1, 1, 3)
        * whitney_edge_signs.view(-1, 3, 1)
    ).flatten(start_dim=-2)

    # Get the canonical edge index pairs for the Whitney 1-form inner products of
    # all 9 edge pairs per tri.
    whitney_flat_r_idx: Float[t.Tensor, "tri*9"] = (
        whitney_edges_idx.view(-1, 3, 1).expand(-1, 3, 3).flatten()
    )
    whitney_flat_c_idx: Float[t.Tensor, "tri*9"] = (
        whitney_edges_idx.view(-1, 1, 3).expand(-1, 3, 3).flatten()
    )

    # Assemble the mass matrix.
    mass = t.sparse_coo_tensor(
        t.vstack((whitney_flat_r_idx, whitney_flat_c_idx)),
        whitney_flat_signed.flatten(),
        (n_edges, n_edges),
    ).coalesce()

    return mass


def d_mass_1_d_vert_coords(
    tri_mesh: SimplicialComplex,
) -> Float[t.Tensor, "edge edge vert 3"]:
    """
    Compute the Jacobian of the 1-form mass matrix wrt the vertex coordinates.
    """
    vert_coords: Float[t.Tensor, "vert 3"] = tri_mesh.vert_coords
    tris: Integer[t.LongTensor, "tet 4"] = tri_mesh.tris

    dtype = vert_coords.dtype
    device = vert_coords.device

    n_tris = tri_mesh.n_tris
    n_edges = tri_mesh.n_edges
    n_verts = tri_mesh.n_verts

    # For D_xy, the inner products of the gradients of the barycentric coordinates,
    # its Jacobian wrt vertex p is given by
    #     grad_p[D_xy] = (hess_xp[V]*grad_y[V] + hess_yp[V]*grad_x[V])/V**2
    #                    - 2*D_xy*grad_p[V])/V
    tri_areas: Float[t.Tensor, "tri"] = _tri_areas(vert_coords, tris)
    d_tri_areas_d_vert_coords: Float[t.Tensor, "tri 3 3"] = _d_tri_areas_d_vert_coords(
        vert_coords, tris
    )

    tri_area_vhp: Float[t.Tensor, "tri x=3 y=3 p=3 3"] = _d2_tri_areas_d2_vert_coords(
        vert_coords, tris, d_tri_areas_d_vert_coords
    )

    bary_coords_grad_dot: Float[t.Tensor, "tri 3 3"] = _bary_coord_grad_inner_prods(
        tri_areas, d_tri_areas_d_vert_coords
    )

    bary_coords_grad_dot_grad: Float[t.Tensor, "tri x=3 y=3 p=3 3"] = (
        tri_area_vhp + tri_area_vhp.transpose(1, 2)
    ) / tri_areas.pow(2).view(
        -1, 1, 1, 1, 1
    ) - 2 * bary_coords_grad_dot * d_tri_areas_d_vert_coords.view(
        -1, 1, 1, 3, 3
    ) / tri_areas.view(-1, 1, 1, 1, 1)

    # For I_xy, the pairwise integrals of the barycentric coordinates, its gradient
    # wrt vertex p is given by grad_p[I_xy] =  grad_p[V]*(1 + delta_xy)/12
    bary_coords_int: Float[t.Tensor, "tri x=3 y=3 1 1"] = t.abs(tri_areas / 12.0) * (
        t.ones((n_tris, 3, 3), dtype=dtype, device=device)
        + t.eye(3, dtype=dtype, device=device).view(1, 3, 3)
    ).view(-1, 3, 3, 1, 1)

    bary_coords_int_grad: Float[t.Tensor, "tri x=3 y=3 p=3 3"] = (
        d_tri_areas_d_vert_coords.view(-1, 1, 1, 3, 3)
        * (
            t.ones((n_tris, 3, 3), dtype=dtype, device=device)
            + t.eye(3, dtype=dtype, device=device).view(1, 3, 3)
        ).view(-1, 3, 3, 1, 1)
        / 12.0
    )

    i, j, k = 0, 1, 2
    unique_edges = t.tensor([[i, j], [i, k], [j, k]], dtype=t.long, device=device)

    x_idx = unique_edges[:, 0][:, None]
    y_idx = unique_edges[:, 1][:, None]
    r_idx = unique_edges[:, 0][None, :]
    s_idx = unique_edges[:, 1][None, :]

    # Find the gradient of the mass matrix element W_xy,rs using the product rule
    whitney_inner_prods_grad: Float[t.Tensor, "tri xy=3 rs=3 p=3 3"] = t.zeros(
        (n_tris, 3, 3, 3, 3), dtype=dtype, device=device
    )

    # Use inplace operations for better peak memory usage.
    whitney_inner_prods_grad.add_(
        bary_coords_int_grad[:, x_idx, r_idx] * bary_coords_grad_dot[:, y_idx, s_idx]
    )
    whitney_inner_prods_grad.add_(
        bary_coords_int[:, x_idx, r_idx] * bary_coords_grad_dot_grad[:, y_idx, s_idx]
    )
    whitney_inner_prods_grad.subtract_(
        bary_coords_int_grad[:, x_idx, s_idx] * bary_coords_grad_dot[:, y_idx, r_idx]
    )
    whitney_inner_prods_grad.subtract_(
        bary_coords_int[:, x_idx, s_idx] * bary_coords_grad_dot_grad[:, y_idx, r_idx]
    )
    whitney_inner_prods_grad.subtract_(
        bary_coords_int_grad[:, y_idx, r_idx] * bary_coords_grad_dot[:, x_idx, s_idx]
    )
    whitney_inner_prods_grad.subtract_(
        bary_coords_int[:, y_idx, r_idx] * bary_coords_grad_dot_grad[:, x_idx, s_idx]
    )
    whitney_inner_prods_grad.add_(
        bary_coords_int_grad[:, y_idx, s_idx] * bary_coords_grad_dot[:, x_idx, r_idx]
    )
    whitney_inner_prods_grad.add_(
        bary_coords_int[:, y_idx, s_idx] * bary_coords_grad_dot_grad[:, x_idx, r_idx]
    )

    # Scatter the gradients to a sparse tensor.
    whitney_edge_signs = tri_mesh.tri_edge_orientations
    whitney_edges_idx = tri_mesh.tri_edge_idx

    whitney_inner_prods_grad_flat_signed: Float[t.Tensor, "tri 27"] = (
        whitney_inner_prods_grad
        * whitney_edge_signs.view(-1, 1, 3, 1, 1)
        * whitney_edge_signs.view(-1, 3, 1, 1, 1)
    ).flatten(start_dim=-2)

    dMdV_idx_xy = whitney_edges_idx.view(-1, 3, 1, 1).expand(-1, 3, 3, 3).flatten()
    dMdV_idx_rs = whitney_edges_idx.view(-1, 1, 3, 1).expand(-1, 3, 3, 3).flatten()
    dMdV_idx_p = tri_mesh.tris.view(-1, 1, 1, 3).expand(-1, 3, 3, 3).flatten()

    dMdV = t.sparse_coo_tensor(
        t.vstack((dMdV_idx_xy, dMdV_idx_rs, dMdV_idx_p)),
        whitney_inner_prods_grad_flat_signed.flatten(end_dim=-2),
        (n_edges, n_edges, n_verts, 3),
    ).coalesce()

    return dMdV

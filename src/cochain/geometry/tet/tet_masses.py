import torch as t
from jaxtyping import Float, Integer

from ...complex import SimplicialComplex
from .tet_geometry import (
    _d2_tet_signed_vols_d2_vert_coords,
    _d_tet_signed_vols_d_vert_coords,
    _tet_signed_vols,
)
from .tet_mass_2 import d_mass_2_d_vert_coords, mass_2


def mass_0(tet_mesh: SimplicialComplex) -> Float[t.Tensor, "vert"]:
    """
    Compute the "lumped" vertex/0-form mass matrix, which is equivalent to the
    barycentric 0-star. Since the lumped vertex mass matrix is diagonal, this
    function returns the diagonal elements.

    The barycentric dual volume for each vertex is the sum of 1/4 of the volumes
    of all tetrahedra that share the vertex as a face.
    """
    n_verts = tet_mesh.n_verts

    tet_vol = t.abs(_tet_signed_vols(tet_mesh.vert_coords, tet_mesh.tets))

    diag = t.zeros(n_verts, device=tet_mesh.vert_coords.device)
    diag.scatter_add_(
        dim=0,
        index=tet_mesh.tets.flatten(),
        src=t.repeat_interleave(tet_vol / 4.0, 4),
    )

    return diag


def _bary_coord_grad_inner_prods(
    tet_signed_vols: Float[t.Tensor, "tet"],
    d_signed_vols_d_vert_coords: Float[t.Tensor, "tet 4 3"],
) -> tuple[Float[t.Tensor, "tet 1"], Float[t.Tensor, "tet 4 4"]]:
    """
    For a tet, let lambda_x(p) be the barycentric coordinate function for p wrt
    a vertex x of the tet. This function computes all pairwise inner products
    of the barycentric coordinate gradients wrt each pair of vertices; i.e., it
    computes <grad_p[lambda_x(p)], grad_p[lambda_y(p)]> for all vertices x and y.
    """
    # The gradient of lambda_i(p) wrt p is given by grad_i(vol_ijkl)/vol_ijkl, a
    # constant wrt p.
    bary_coords_grad: Float[t.Tensor, "tet 4 3"] = (
        d_signed_vols_d_vert_coords / tet_signed_vols
    )

    bary_coords_grad_dot: Float[t.Tensor, "tet 4 4"] = t.einsum(
        "tic,tjc->tij", bary_coords_grad, bary_coords_grad
    )

    return bary_coords_grad_dot


def _tet_edge_faces(
    tet_mesh: SimplicialComplex,
) -> tuple[Float[t.Tensor, "tet 6"], Integer[t.LongTensor, "tet 6"]]:
    """
    Enumerate all edges for each tet and find their orientations and indices on
    the tet_mesh.edges list.
    """
    device = tet_mesh.vert_coords.device

    n_verts = tet_mesh.n_verts

    # Enumerate all unique edges via their vertex position in the etet.
    i, j, k, l = 0, 1, 2, 3
    unique_edges = t.tensor(
        [[i, j], [i, k], [j, k], [j, l], [k, l], [i, l]], dtype=t.long, device=device
    )

    # For each tet and each unique edge pair, find the orientations of the edges
    # and their indices on the list of unique, canonical edges (tet_mesh.edges).
    whitney_edges: Float[t.Tensor, "tet*6 2"] = tet_mesh.tets[:, unique_edges].flatten(
        end_dim=-2
    )

    # Same method as used in the construction of coboundary operators to use
    # sort() to identify edge orientations.
    whitney_canon_edges, whitney_edge_orientations = whitney_edges.sort(dim=-1)
    whitney_edge_signs: Float[t.Tensor, "tet 6"] = t.where(
        whitney_edge_orientations[:, 1] > 0, whitney_edge_orientations[:, 1], -1
    ).view(-1, 6)

    # This assumes that the edge indices in tet_mesh.edges are already in canonical
    # orders.
    unique_canon_edges_packed = tet_mesh.edges[:, 0] * n_verts + tet_mesh.edges[:, 1]
    canon_edges_packed_sorted, canon_edges_idx = t.sort(unique_canon_edges_packed)

    whitney_edges_packed = (
        whitney_canon_edges[:, 0] * n_verts + whitney_canon_edges[:, 1]
    )
    whitney_edges_idx: Float[t.Tensor, "tet 6"] = canon_edges_idx[
        t.searchsorted(canon_edges_packed_sorted, whitney_edges_packed)
    ].view(-1, 6)

    return whitney_edge_signs, whitney_edges_idx


def mass_1(tet_mesh: SimplicialComplex) -> Float[t.Tensor, "edge edge"]:
    """
    Compute the Galerkin edge/1-form mass matrix.

    For each tet, each (canonical) edge pair xy and pq and their associated
    Whitney 1-form basis functions W_xy and W_pq contribute the inner product
    term int[W_xy*W_pq*dV] to the mass matrix element M[xy, pq], where

    W_xy(p) = lambda_x(p)*grad_p(lambda_y(p)) - lambda_y(p)*grad_p(lambda_x(p))

    Here, p is a position vector inside the tet and lambda_x(p) is the barycentric
    coordinate function for p wrt the vertex x.
    """
    vert_coords: Float[t.Tensor, "vert 3"] = tet_mesh.vert_coords
    tets: Integer[t.LongTensor, "tet 4"] = tet_mesh.tets

    dtype = vert_coords.dtype
    device = vert_coords.device

    n_tets = tet_mesh.n_tets
    n_edges = tet_mesh.n_edges

    tet_signed_vols = _tet_signed_vols(vert_coords, tets).view(-1, 1, 1)
    d_signed_vols_d_vert_coords = _d_tet_signed_vols_d_vert_coords(vert_coords, tets)

    # For each tet ijkl, compute all pairwise inner products of the barycentric
    # coordinate gradients wrt each pair of vertices.
    bary_coords_grad_dot: Float[t.Tensor, "tet 4 4"] = _bary_coord_grad_inner_prods(
        tet_signed_vols, d_signed_vols_d_vert_coords
    )

    # For each tet ijkl, compute all pairwise integrals of the barycentric coordinates;
    # i.e., int[lambda_i(p)lambda_j(p)dvol_ijkl]. Using the "magic formula", this
    # integral is vol_ijkl*(1 + delta_ij)/20, where delta is the Kronecker delta
    # function.
    bary_coords_int: Float[t.Tensor, "tet 4 4"] = t.abs(tet_signed_vols / 20.0) * (
        t.ones((n_tets, 4, 4), dtype=dtype, device=device)
        + t.eye(4, dtype=dtype, device=device).view(1, 4, 4)
    )

    # For each tet ijkl, each pair of its edges e1=xy and e2=pq contributes the
    # following term to the mass matrix element M[e1,e2]:
    #
    #         W_xy,pq = I_xp*D_yq - I_xq*D_yp - I_yp*D_xq + I_yq*D_xp
    #
    # Here, I is the barycentric integral (bary_coords_int) and D is the barycentric
    # gradient inner product (barry_coords_grad_dot). Note that, since this expression
    # is "skew-symmetric" wrt the edge orientations (W_yx,pq = -W_xy,pq and
    # W_xy,qp = -W_xy,pq), each non-canonical edge orientation also contributes
    # an overall negative sign.

    # Enumerate all unique edges via their vertex position in the etet.
    i, j, k, l = 0, 1, 2, 3
    unique_edges = t.tensor(
        [[i, j], [i, k], [j, k], [j, l], [k, l], [i, l]], dtype=t.long, device=device
    )

    x_idx = unique_edges[:, 0][:, None]
    y_idx = unique_edges[:, 1][:, None]
    p_idx = unique_edges[:, 0][None, :]
    q_idx = unique_edges[:, 1][None, :]

    # For each tet, find all pairs of Whitney 1-form basis function inner products,
    # i.e., the W_xy,pq.
    whitney_inner_prod: Float[t.Tensor, "tet 6 6"] = (
        bary_coords_int[:, x_idx, p_idx] * bary_coords_grad_dot[:, y_idx, q_idx]
        - bary_coords_int[:, x_idx, q_idx] * bary_coords_grad_dot[:, y_idx, p_idx]
        - bary_coords_int[:, y_idx, p_idx] * bary_coords_grad_dot[:, x_idx, q_idx]
        + bary_coords_int[:, y_idx, q_idx] * bary_coords_grad_dot[:, x_idx, p_idx]
    )

    # For each tet and each unique edge pair, find the orientations of the edges
    # and their indices on the list of unique, canonical edges (tet_mesh.edges).
    whitney_edge_signs, whitney_edges_idx = _tet_edge_faces(tet_mesh)

    # Multiply the Whitney 1-form inner product by the edge orientation signs
    # to get the contribution from canonical edges.
    whitney_flat_signed: Float[t.Tensor, "tet 36"] = (
        whitney_inner_prod
        * whitney_edge_signs.view(-1, 1, 6)
        * whitney_edge_signs.view(-1, 6, 1)
    ).flatten(start_dim=-2)

    # Get the canonical edge index pairs for the Whitney 1-form inner products of
    # all 36 edge pairs per tet.
    whitney_flat_r_idx: Float[t.Tensor, "tet*36"] = (
        whitney_edges_idx.view(-1, 6, 1).expand(-1, 6, 6).flatten()
    )
    whitney_flat_c_idx: Float[t.Tensor, "tet*36"] = (
        whitney_edges_idx.view(-1, 1, 6).expand(-1, 6, 6).flatten()
    )

    # Assemble the mass matrix.
    mass = t.sparse_coo_tensor(
        t.vstack((whitney_flat_r_idx, whitney_flat_c_idx)),
        whitney_flat_signed.flatten(),
        (n_edges, n_edges),
    ).coalesce()

    return mass


def d_mass_1_d_vert_coords(
    tet_mesh: SimplicialComplex,
) -> Float[t.Tensor, "edge edge vert 3"]:
    """
    Compute the Jacobian of the 1-form mass matrix wrt the vertex coordinates.
    """
    vert_coords: Float[t.Tensor, "vert 3"] = tet_mesh.vert_coords
    tets: Integer[t.LongTensor, "tet 4"] = tet_mesh.tets

    dtype = vert_coords.dtype
    device = vert_coords.device

    n_tets = tet_mesh.n_tets
    n_edges = tet_mesh.n_edges
    n_verts = tet_mesh.n_verts

    # For D_xy, the inner products of the gradients of the barycentric coordinates,
    # its Jacobian wrt vertex p is given by
    #     grad_p[D_xy] = (hess_xp[V]*grad_y[V] + hess_yp[V]*grad_x[V])/V**2
    #                    - 2*D_xy*grad_p[V])/V
    tet_signed_vols: Float[t.Tensor, "tet"] = _tet_signed_vols(vert_coords, tets)
    tet_signs = tet_signed_vols.sign()
    d_signed_vols_d_vert_coords: Float[t.Tensor, "tet 4 3"] = (
        _d_tet_signed_vols_d_vert_coords(vert_coords, tets)
    )

    tet_vol_vhp: Float[t.Tensor, "tet x=4 y=4 p=4 3"] = (
        _d2_tet_signed_vols_d2_vert_coords(
            vert_coords, tets, d_signed_vols_d_vert_coords
        )
    )

    bary_coords_grad_dot: Float[t.Tensor, "tet 4 4"] = _bary_coord_grad_inner_prods(
        tet_signed_vols, d_signed_vols_d_vert_coords
    )

    bary_coords_grad_dot_grad: Float[t.Tensor, "tet x=4 y=4 p=4 3"] = (
        tet_vol_vhp + tet_vol_vhp.transpose(1, 2)
    ) / tet_signed_vols.pow(2).view(
        -1, 1, 1, 1, 1
    ) - 2 * bary_coords_grad_dot * d_signed_vols_d_vert_coords.view(
        -1, 1, 1, 4, 3
    ) / tet_signed_vols.view(-1, 1, 1, 1, 1)

    # For I_xy, the pairwise integrals of the barycentric coordinates, its gradient
    # wrt vertex p is given by grad_p[I_xy] =  grad_p[V]*(1 + delta_xy)/20
    bary_coords_int: Float[t.Tensor, "tet x=4 y=4 1 1"] = t.abs(
        tet_signed_vols / 20.0
    ) * (
        t.ones((n_tets, 4, 4), dtype=dtype, device=device)
        + t.eye(4, dtype=dtype, device=device).view(1, 4, 4)
    ).view(-1, 4, 4, 1, 1)

    bary_coords_int_grad: Float[t.Tensor, "tet x=4 y=4 p=4 3"] = (
        d_signed_vols_d_vert_coords.view(-1, 1, 1, 4, 3)
        * tet_signs.view(-1, 1, 1, 1, 1)
        * (
            t.ones((n_tets, 4, 4), dtype=dtype, device=device)
            + t.eye(4, dtype=dtype, device=device).view(1, 4, 4)
        ).view(-1, 4, 4, 1, 1)
        / 20.0
    )

    i, j, k, l = 0, 1, 2, 3
    unique_edges = t.tensor(
        [[i, j], [i, k], [j, k], [j, l], [k, l], [i, l]], dtype=t.long, device=device
    )

    x_idx = unique_edges[:, 0][:, None]
    y_idx = unique_edges[:, 1][:, None]
    r_idx = unique_edges[:, 0][None, :]
    s_idx = unique_edges[:, 1][None, :]

    # Find the gradient of the mass matrix element W_xy,rs using the product rule
    whitney_inner_prods_grad: Float[t.Tensor, "tet xy=6 rs=6 p=4 3"] = t.zeros(
        (n_tets, 6, 6, 4, 3), dtype=dtype, device=device
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
    whitney_edge_signs, whitney_edges_idx = _tet_edge_faces(tet_mesh)

    whitney_inner_prods_grad_flat_signed: Float[t.Tensor, "tet 144"] = (
        whitney_inner_prods_grad
        * whitney_edge_signs.view(-1, 1, 6, 1, 1)
        * whitney_edge_signs.view(-1, 6, 1, 1, 1)
    ).flatten(start_dim=-2)

    dMdV_idx_xy = whitney_edges_idx.view(-1, 6, 1, 1).expand(-1, 6, 6, 4).flatten()
    dMdV_idx_rs = whitney_edges_idx.view(-1, 1, 6, 1).expand(-1, 6, 6, 4).flatten()
    dMdV_idx_p = tet_mesh.tets.view(-1, 1, 1, 4).expand(-1, 6, 6, 4).flatten()

    dMdV = t.sparse_coo_tensor(
        t.vstack((dMdV_idx_xy, dMdV_idx_rs, dMdV_idx_p)),
        whitney_inner_prods_grad_flat_signed.flatten(end_dim=-2),
        (n_edges, n_edges, n_verts, 3),
    ).coalesce()

    return dMdV


def mass_3(tet_mesh: SimplicialComplex) -> Float[t.Tensor, "tet"]:
    """
    Compute the diagonal of the tet/3-form mass matrix, which is equivalent to
    the inverse of 3-star.
    """
    return t.abs(_tet_signed_vols(tet_mesh.vert_coords, tet_mesh.tets))

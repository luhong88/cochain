import torch as t
from jaxtyping import Float, Integer

from ...complex import SimplicialComplex
from ...sparse.operators import DiagOperator, SparseOperator
from .tet_geometry import (
    bary_coord_grad_inner_prods,
    d_tet_signed_vols_d_vert_coords,
    get_tet_signed_vols,
    whitney_2_form_inner_prods,
)


def mass_0(tet_mesh: SimplicialComplex) -> Float[DiagOperator, "vert vert"]:
    """
    Compute the "lumped" vertex/0-form mass matrix, which is equivalent to the
    barycentric 0-star. Since the lumped vertex mass matrix is diagonal, this
    function returns the diagonal elements.

    The barycentric dual volume for each vertex is the sum of 1/4 of the volumes
    of all tetrahedra that share the vertex as a face.
    """
    n_verts = tet_mesh.n_verts

    tet_vol = t.abs(get_tet_signed_vols(tet_mesh.vert_coords, tet_mesh.tets))

    diag = t.zeros(n_verts, device=tet_mesh.vert_coords.device)
    diag.scatter_add_(
        dim=0,
        index=tet_mesh.tets.flatten(),
        src=t.repeat_interleave(tet_vol / 4.0, 4),
    )

    return DiagOperator.from_tensor(diag)


def mass_1(tet_mesh: SimplicialComplex) -> Float[SparseOperator, "edge edge"]:
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

    tet_signed_vols = get_tet_signed_vols(vert_coords, tets).view(-1, 1, 1)
    d_signed_vols_d_vert_coords = d_tet_signed_vols_d_vert_coords(vert_coords, tets)

    # For each tet ijkl, compute all pairwise inner products of the barycentric
    # coordinate gradients wrt each pair of vertices.
    bary_coords_grad_dot: Float[t.Tensor, "tet 4 4"] = bary_coord_grad_inner_prods(
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

    # Enumerate all unique edges via their vertex position in the tet.
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
    whitney_edge_signs = tet_mesh.tet_edge_orientations
    whitney_edges_idx = tet_mesh.tet_edge_idx

    # Multiply the Whitney 1-form inner product by the edge orientation signs
    # to get the contribution from canonical edges.
    whitney_flat_signed: Float[t.Tensor, "tet 36"] = (
        whitney_inner_prod
        * whitney_edge_signs.view(-1, 1, 6)
        * whitney_edge_signs.view(-1, 6, 1)
    ).flatten(start_dim=-2)

    # Get the canonical edge index pairs for the Whitney 1-form inner products of
    # all 36 edge pairs per tet.
    whitney_flat_r_idx: Float[t.Tensor, " tet*36"] = (
        whitney_edges_idx.view(-1, 6, 1).expand(-1, 6, 6).flatten()
    )
    whitney_flat_c_idx: Float[t.Tensor, " tet*36"] = (
        whitney_edges_idx.view(-1, 1, 6).expand(-1, 6, 6).flatten()
    )

    # Assemble the mass matrix.
    mass = t.sparse_coo_tensor(
        t.vstack((whitney_flat_r_idx, whitney_flat_c_idx)),
        whitney_flat_signed.flatten(),
        (n_edges, n_edges),
    ).coalesce()

    return SparseOperator.from_tensor(mass)


def mass_2(tet_mesh: SimplicialComplex) -> Float[SparseOperator, "tri tri"]:
    """
    Compute the Galerkin triangle/2-form mass matrix.

    For each tet, each (canonical) triangle pairs xyz and rst and their associated
    Whitney 2-form basis functions W_xyz and W_rst contribute the inner product
    term int[W_xyz*W_rst*dV] to the mass matrix element M[xyz, rst], where

    W_xyz(p) = sign[xyz]*(p - v[-xyz])/3V

    Here, v[-xyz] is the coordinate vector of the vertex opposite to xyz. sign[xyz]
    is +1 whenever the triangle xyz satisfies the right-hand rule (i.e., the normal
    vector formed by the right hand points out of the tet), and -1 if not.
    """
    n_tris = tet_mesh.n_tris

    # First, compute the inner products of the Whitney 2-form basis functions.
    _, whitney_inner_prod_signed = whitney_2_form_inner_prods(
        tet_mesh.vert_coords, tet_mesh.tets, tet_mesh.tet_tri_orientations
    )

    # Then, find the indices of the tet triangle faces associated with the basis
    # functions.
    all_canon_tris_idx = tet_mesh.tet_tri_idx

    # Assemble the mass matrix by scattering the inner products according to the
    # triangle indices.
    mass_idx = t.vstack(
        (
            all_canon_tris_idx.view(-1, 4, 1).expand(-1, 4, 4).flatten(),
            all_canon_tris_idx.view(-1, 1, 4).expand(-1, 4, 4).flatten(),
        )
    )
    mass_val = whitney_inner_prod_signed.flatten()
    mass = t.sparse_coo_tensor(mass_idx, mass_val, (n_tris, n_tris)).coalesce()

    return SparseOperator.from_tensor(mass)


def mass_3(tet_mesh: SimplicialComplex) -> Float[DiagOperator, "tet tet"]:
    """
    Compute the diagonal of the tet/3-form mass matrix, which is equivalent to
    the inverse of 3-star.
    """
    return DiagOperator.from_tensor(
        t.abs(get_tet_signed_vols(tet_mesh.vert_coords, tet_mesh.tets))
    )

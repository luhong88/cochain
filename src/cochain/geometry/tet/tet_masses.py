import torch as t
from jaxtyping import Float, Integer

from ...complex import SimplicialComplex
from .tet_geometry import (
    _d2_tet_signed_vols_d2_vert_coords,
    _d_tet_signed_vols_d_vert_coords,
    _tet_signed_vols,
)


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
) -> Float[t.Tensor, "tet 4 4"]:
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
    whitney_edge_signs = tet_mesh.tet_edge_orientations
    whitney_edges_idx = tet_mesh.tet_edge_idx

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


def _whitney_2_form_inner_prods(
    tet_mesh: SimplicialComplex,
) -> tuple[Float[t.Tensor, "tet 1"], Float[t.Tensor, "tet 4 4"]]:
    """
    For each tet, compute the pairwise inner product of the Whitney 2-form basis
    functions associated with the faces of the tet, and correct for the face and
    tet orientation.
    """
    i, j, k, l = 0, 1, 2, 3

    vert_coords: Float[t.Tensor, "vert 3"] = tet_mesh.vert_coords
    tets: Integer[t.LongTensor, "tet 4"] = tet_mesh.tets
    tet_vert_coords: Float[t.Tensor, "tet 4 3"] = vert_coords[tets]

    tet_signed_vols: Float[t.Tensor, "tet"] = _tet_signed_vols(vert_coords, tets)
    tet_vols = t.abs(tet_signed_vols)
    tet_signs = t.sign(tet_signed_vols)

    # For each tet, associate the 2-form basis function with the opposite vertex.
    # Then, the inner product between the basis functions is given by
    #
    #               int[W_i*W_j*dV] = sum_k,l[C_kl*<ik,jl>]/(180*V)
    #
    # Where C_kl = 1 + delta_kl (delta is the Kronecker delta function). Here,
    # the summation represents the inner products between all edge vectors emanating
    # from vertices i and j.
    #
    # Let G_ij = <i,j> be the symmetric, local "Gram" matrix of vertex coordinates.
    # Since <ik,jl> can be written as G_kl - G_kj - G_il + G_ij, the inner product
    # can be further simplified as
    #
    # int_ij = (20*G_ij - 5*(R_i + R_j) + (S + Tr[G]))/(180*V)
    #
    # here, R_i = sum_j[G_ij], S = sum_ij[G_ij], and Tr[G] is the trace of G.

    gram: Float[t.Tensor, "tet 4 4"] = t.sum(
        tet_vert_coords.view(-1, 4, 1, 3) * tet_vert_coords.view(-1, 1, 4, 3), dim=-1
    )

    # Compute R_i + R_j
    gram_partial_sum: Float[t.Tensor, "tet 4 4"] = t.sum(
        gram, dim=-1, keepdim=True
    ) + t.sum(gram, dim=-2, keepdim=True)

    # Compute S + Tr[G]
    gram_sum: Float[t.Tensor, "tet 1 1"] = (
        t.sum(gram, dim=(-1, -2)) + t.einsum("tii->t", gram)
    ).view(-1, 1, 1)

    whitney_inner_prod: Float[t.Tensor, "tet 4 4"] = (
        20.0 * gram - 5.0 * gram_partial_sum + gram_sum
    ) / (180.0 * tet_vols.view(-1, 1, 1))

    # For each tet and each vertex, find the outward-facing triangle opposite
    # to the vertex (note that the way the triangles are indexed here satisfies
    # the right-hand rule for positively oriented tets).
    all_tris: Integer[t.LongTensor, "tet 4 3"] = tets[
        :, [[j, k, l], [i, l, k], [i, j, l], [i, k, j]]
    ]

    canon_pos_orientation = t.tensor([0, 1, 2], dtype=t.long, device=tets.device)

    all_tris_orientations = all_tris.sort(dim=-1).indices
    # Same method as used in the construction of coboundary operators to use
    # sort() to identify triangle orientations.
    all_tris_signs: Float[t.Tensor, "tet 4"] = t.where(
        condition=t.sum(all_tris_orientations == canon_pos_orientation, dim=-1) == 1,
        self=-1.0,
        other=1.0,
    ).to(dtype=vert_coords.dtype)

    # Mapping the local basis function to the global basis function requires
    # correction of both the triangle face orientation as well as the tet orientations
    # (to account for negatively oriented tets, for which all_tris no longer satisfies
    # the right-hand rule).
    sign_corrections = all_tris_signs * tet_signs.view(-1, 1)

    whitney_inner_prod_signed: Float[t.Tensor, "tet 4 4"] = (
        whitney_inner_prod
        * sign_corrections.view(-1, 1, 4)
        * sign_corrections.view(-1, 4, 1)
    )

    return sign_corrections, whitney_inner_prod_signed


def mass_2(tet_mesh: SimplicialComplex) -> Float[t.Tensor, "tri tri"]:
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
    _, whitney_inner_prod_signed = _whitney_2_form_inner_prods(tet_mesh)

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

    return mass


def d_mass_2_d_vert_coords(
    tet_mesh: SimplicialComplex,
) -> Float[t.Tensor, "tri tri vert 3"]:
    """
    Compute the Jacobian of the 2-form mass matrix wrt the vertex coordinates.
    """
    vert_coords: Float[t.Tensor, "vert 3"] = tet_mesh.vert_coords
    tets: Integer[t.LongTensor, "tet 4"] = tet_mesh.tets
    tet_vert_coords: Float[t.Tensor, "tet 4 3"] = vert_coords[tets]

    n_verts = tet_mesh.n_verts
    n_tris = tet_mesh.n_tris

    dtype = vert_coords.dtype
    device = vert_coords.device

    # For each tet, denote the inner product between the 2-form basis functions
    # associated with triangle faces i and j as int_ij; recall that
    #
    #               int_ij = sum_k,l[C_kl*<ik,jl>]/(180V)
    #
    # Where C_kl = 1 + delta_kl (delta is the Kronecker delta function). Here,
    # the summation represents the inner products between all edge vectors emanating
    # from vertices i and j. Then, one can show that the Jacobian of int_ij wrt
    # the coordinates of vertex p, grad_p[int_ij], is given by
    #
    #     grad_p[int_ij] = (
    #         sum_k,l[C_kl*(delta_pk - delta_pi)*jl]/(180*V) +
    #         sum_k,l[C_kl*(delta_pl - delta_pj)*ik]/(180*V) -
    #         int_ij*grad_p[V]/V
    #     )
    #
    # The first two terms here can be further simplified to give
    #
    #     grad_p[int_ij] = (
    #         (p + 4*c)/(90*V) -
    #         (i + j)/(36*V) -
    #         (delta_pi*(c - j) + delta_pj*(c - i))/(9*V) -
    #         int_ij*grad_p[V]/V
    #     )
    #
    # where "c" is the centroid of the tet.

    # First, collect all the constituent terms required to compute the Jacobian.
    tet_signed_vols: Float[t.Tensor, "tet 1 1 1 1"] = _tet_signed_vols(
        vert_coords, tets
    ).view(-1, 1, 1, 1, 1)

    tet_vols = t.abs(tet_signed_vols)

    d_signed_vols_d_vert_coords: Float[t.Tensor, "tet 1 1 4 3"] = (
        _d_tet_signed_vols_d_vert_coords(vert_coords, tets)
    ).view(-1, 1, 1, 4, 3)

    identity = t.eye(4, dtype=dtype, device=device)

    sign_corrections, whitney_inner_prods = _whitney_2_form_inner_prods(tet_mesh)
    sign_corrections_shaped: Float[t.Tensor, "tet 4 4 1 1"] = (
        sign_corrections.view(-1, 1, 4) * sign_corrections.view(-1, 4, 1)
    ).view(-1, 4, 4, 1, 1)
    whitney_inner_prods_shaped = whitney_inner_prods.view(-1, 4, 4, 1, 1)

    centroids: Float[t.Tensor, "tet 1 3"] = t.mean(tet_vert_coords, dim=1, keepdim=True)

    # Prepare all terms in the sum into the form (tet, i, j, p, coords).
    # Note that all but the last term require a correction for the triangle and
    # tet orientations. The last term does not require this correction since
    # the function _whitney_2_form_inner_prods() already applies this correction
    # to the inner products.

    whitney_inner_prod_grad: Float[t.Tensor, "tet i=4 j=4 p=4 3"] = (
        tet_vert_coords + 4.0 * centroids
    ).view(-1, 1, 1, 4, 3) / (90.0 * tet_vols)

    whitney_inner_prod_grad.subtract_(
        (tet_vert_coords.view(-1, 4, 1, 1, 3) + tet_vert_coords.view(-1, 1, 4, 1, 3))
        / (36.0 * tet_vols)
    )

    sum_delta = t.einsum("pi,tjc->tijpc", identity, centroids - tet_vert_coords) / (
        9.0 * tet_vols
    )
    whitney_inner_prod_grad.subtract_(sum_delta + sum_delta.transpose(1, 2))

    whitney_inner_prod_grad.multiply_(sign_corrections_shaped)

    whitney_inner_prod_grad.subtract_(
        whitney_inner_prods_shaped * d_signed_vols_d_vert_coords / tet_signed_vols
    )

    all_canon_tris_idx: Integer[t.LongTensor, "tet 4"] = tet_mesh.tet_tri_idx

    # Assemble the mass matrix Jacobian.
    dMdV_idx_i = all_canon_tris_idx.view(-1, 4, 1, 1).expand(-1, 4, 4, 4).flatten()
    dMdV_idx_j = all_canon_tris_idx.view(-1, 1, 4, 1).expand(-1, 4, 4, 4).flatten()
    dMdV_idx_p = tet_mesh.tets.view(-1, 1, 1, 4).expand(-1, 4, 4, 4).flatten()

    dMdV_idx = t.vstack(
        (
            dMdV_idx_i,
            dMdV_idx_j,
            dMdV_idx_p,
        )
    )

    dMdV_val = whitney_inner_prod_grad.flatten(end_dim=-2)

    dMdV = t.sparse_coo_tensor(
        dMdV_idx, dMdV_val, (n_tris, n_tris, n_verts, 3)
    ).coalesce()

    return dMdV


def mass_3(tet_mesh: SimplicialComplex) -> Float[t.Tensor, "tet"]:
    """
    Compute the diagonal of the tet/3-form mass matrix, which is equivalent to
    the inverse of 3-star.
    """
    return t.abs(_tet_signed_vols(tet_mesh.vert_coords, tet_mesh.tets))

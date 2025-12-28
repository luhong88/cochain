import torch as t
from jaxtyping import Float, Integer

from ..complex import SimplicialComplex
from ..geometry.tet.tet_geometry import (
    bary_coord_grad_inner_prods,
    d_tet_signed_vols_d_vert_coords,
    get_tet_signed_vols,
    tet_face_vector_areas,
    whitney_2_form_inner_prods,
)


def _d2_tet_signed_vols_d2_vert_coords(
    vert_coords: Float[t.Tensor, "vert 3"],
    tets: Integer[t.LongTensor, "tet 4"],
    vec: Float[t.Tensor, "tet 4 3"],
) -> Float[t.Tensor, "tet vert vert vert 3"]:
    """
    For each tet, given the gradient of the signed volumes grad_x[V]
    (shape: (tet, x=4, 3)) and a vector field v_y associated with each vertex
    (shape: (tet, y=4, 3)), compute the pairwise "vector-Hessian product" (VHP)
    as VHP_xyp = hess_xp[V]@v_y. This is useful for computing the gradient vectors
    for inner products of the form I_xy = <grad_x[V], v_y>.
    """
    i, j, k, l = 0, 1, 2, 3

    # Since the gradient of signed tet volumes is given by a triangle face
    # area vector, the Hessian of the signed tet volumes can be represented by
    # skew-symmetric matrices of tet edges associated with cross product with
    # the edges:
    #
    # vert  base tri  area normal  grad_i  grad_j  grad_k  grad_l
    # -----------------------------------------------------------
    # i     jkl       jl x jk      [ii]    [lk]    [jl]    [kj]
    # j     ikl       ik x il      [kl]    [jj]    [li]    [ik]
    # k     ijl       il x ij      [lj]    [il]    [kk]    [ji]
    # l     ijk       ij x ik      [jk]    [ik]    [ji]    [ll]

    tet_vert_coords: Float[t.Tensor, "tet 4 3"] = vert_coords[tets]

    area_vec_grad: Float[t.Tensor, "tet 4 4 3"] = (
        tet_vert_coords[
            :,
            [[i, k, l, j], [l, j, i, k], [j, l, k, i], [k, k, i, l]],
        ]
        - tet_vert_coords[
            :,
            [[i, l, j, k], [k, j, l, i], [l, i, k, j], [j, i, j, l]],
        ]
    )

    # (t,x,p,3) x (t,y,3) -> (t,x,y,p,3)
    signed_vols_grad_grad = (
        t.cross(area_vec_grad.view(-1, 4, 1, 4, 3), vec.view(-1, 1, 4, 1, 3), dim=-1)
        / 6.0
    )

    return signed_vols_grad_grad


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
    tet_signed_vols: Float[t.Tensor, " tet"] = get_tet_signed_vols(vert_coords, tets)
    tet_signs = tet_signed_vols.sign()
    d_signed_vols_d_vert_coords: Float[t.Tensor, "tet 4 3"] = (
        d_tet_signed_vols_d_vert_coords(vert_coords, tets)
    )

    tet_vol_vhp: Float[t.Tensor, "tet x=4 y=4 p=4 3"] = (
        _d2_tet_signed_vols_d2_vert_coords(
            vert_coords, tets, d_signed_vols_d_vert_coords
        )
    )

    bary_coords_grad_dot: Float[t.Tensor, "tet 4 4"] = bary_coord_grad_inner_prods(
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
    tet_signed_vols: Float[t.Tensor, "tet 1 1 1 1"] = get_tet_signed_vols(
        vert_coords, tets
    ).view(-1, 1, 1, 1, 1)

    tet_vols = t.abs(tet_signed_vols)

    d_signed_vols_d_vert_coords: Float[t.Tensor, "tet 1 1 4 3"] = (
        d_tet_signed_vols_d_vert_coords(vert_coords, tets)
    ).view(-1, 1, 1, 4, 3)

    identity = t.eye(4, dtype=dtype, device=device)

    sign_corrections, whitney_inner_prods = whitney_2_form_inner_prods(
        tet_mesh.vert_coords, tet_mesh.tets, tet_mesh.tet_tri_orientations
    )
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


def _d_cotan_weights_d_vert_coords(
    vert_coords: Float[t.Tensor, "vert 3"],
    tets: Integer[t.LongTensor, "tet 4"],
    n_verts: int,
) -> Float[t.Tensor, "vert vert vert 3"]:
    i, j, k, l = 0, 1, 2, 3
    # For each tet ijkl and each edge s, find the gradient of w_o (i.e., its
    # contribution to the cotan weights), which is (<th x o, hh x o> / 36 * vol_ijkl),
    # wrt each vertex i, j, k, and l. For vertex p, this gradient is given by
    #
    # grad_p(w_o) = (grad_p(<th x o, hh x o>)/36 - w_o*grad_p(vol_ijkl))/vol_ijkl
    tet_vert_coords: Float[t.Tensor, "tet 4 3"] = vert_coords[tets]

    norm_tri_to, norm_tri_ho, weight_o = tet_face_vector_areas(vert_coords, tets)

    norm_tri_to_shaped: Float[t.Tensor, "tet 6 1 3"] = norm_tri_to.view(-1, 6, 1, 3)
    norm_tri_ho_shaped: Float[t.Tensor, "tet 6 1 3"] = norm_tri_ho.view(-1, 6, 1, 3)
    weight_o_shaped: Float[t.Tensor, "tet 6 1 1"] = weight_o.view(-1, 6, 1, 1)

    tet_signed_vols: Float[t.Tensor, " tet"] = get_tet_signed_vols(vert_coords, tets)
    tet_signs = tet_signed_vols.sign()

    vols_shaped: Float[t.Tensor, "tet 1 1 1"] = t.abs(tet_signed_vols).view(-1, 1, 1, 1)

    # Multiply the gradient of the signed volumes with the signs of the volumes
    # to get the gradient of the unsigned/absolute volumes.
    vol_grad: Float[t.Tensor, "tet 1 4 3"] = (
        d_tet_signed_vols_d_vert_coords(vert_coords, tets) * tet_signs.view(-1, 1, 1)
    ).view(-1, 1, 4, 3)

    # Compute the "Jacobian" of the th x o normal vector wrt each vertex in the
    # tet. Note that, technically, the Jacobian should have the shape (tet, 6, 4,
    # 3, 3) (e.g., grad_i(il x kl) = [kl], a 3x3 skew-symmetric matrix).
    norm_tri_to_grad: Float[t.Tensor, "tet 6 4 3"] = (
        tet_vert_coords[
            :,
            [
                [k, l, l, i],
                [l, i, l, j],
                [l, i, l, j],
                [j, k, i, k],
                [j, k, i, j],
                [k, l, j, k],
            ],
        ]
        - tet_vert_coords[
            :,
            [
                [l, l, i, k],
                [j, l, l, i],
                [j, l, l, i],
                [k, i, j, k],
                [k, i, j, j],
                [k, k, l, j],
            ],
        ]
    )

    # Compute the "Jacobian" of the th x o normal vector wrt each vertex in the tet.
    norm_tri_ho_grad: Float[t.Tensor, "tet 6 4 3"] = (
        tet_vert_coords[
            :,
            [
                [l, l, j, k],
                [l, l, j, k],
                [k, l, l, i],
                [k, k, l, i],
                [l, i, j, j],
                [j, k, i, k],
            ],
        ]
        - tet_vert_coords[
            :,
            [
                [l, k, l, j],
                [l, k, l, j],
                [l, l, i, k],
                [l, k, i, k],
                [j, l, j, i],
                [k, i, j, k],
            ],
        ]
    )

    # Compute grad_p(<th x o, hh x o>) using the dot product chain rule. Here,
    # we use the special property of the Jacobian to reduce this into the sum of
    # two cross products. For example, for s = ij and p = k, the gradient is
    #
    # grad_k(<il x kl, jl x kl>) = [li].T@(jl x kl) + [lj]@(il x kl)
    #                            = (jl x kl) x li + (il x kl) x lj
    #
    # where the second equality follows from the skew-symmetric property of [].
    area_normal_dot_grad: Float[t.Tensor, "tet 6 4 3"] = t.cross(
        norm_tri_ho_shaped, norm_tri_to_grad, dim=-1
    ) + t.cross(norm_tri_to_shaped, norm_tri_ho_grad, dim=-1)

    # Compute the dense gradient of w_o wrt each vertex in a tet.
    weight_o_grad: Float[t.Tensor, "tet 6 4 3"] = (
        area_normal_dot_grad / 36.0 - weight_o_shaped * vol_grad
    ) / vols_shaped

    # Assemble the final, sparse Jacobian
    # fmt:off
    dWdV_idx= (
        tets[
            :, 
            [
                i, i, i, i, i, i, i, i, j, j, j, j, j, j, j, j, k, k, k, k, i, i, i, i,
                j, j, j, j, k, k, k, k, k, k, k, k, l, l, l, l, l, l, l, l, l, l, l, l,
                i, j, k, l, i, j, k, l, i, j, k, l, i, j, k, l, i, j, k, l, i, j, k, l,
            ]
        ]
        .T
        .flatten()
        .reshape(3, -1)
    )
    # fmt:on

    # Use permute to reshape to (edge=6, vert=4, tet, 3)
    dWdV_val = weight_o_grad.permute(1, 2, 0, 3).flatten(end_dim=-2)

    asym_dWdV = t.sparse_coo_tensor(
        dWdV_idx, dWdV_val, (n_verts, n_verts, n_verts, 3)
    ).coalesce()

    # Symmetrize so that dW_ijk = dW_jki
    sym_dWdV = (asym_dWdV + asym_dWdV.transpose(0, 1)).coalesce()

    return sym_dWdV


def d_stiffness_d_vert_coords(
    tri_mesh: SimplicialComplex,
) -> Float[t.Tensor, "vert vert vert 3"]:
    """
    Compute the jacobian of the stiffness matrix/cotan Laplacian with respect to
    the vertex coordinates.
    """
    vert_coords: Float[t.Tensor, "vert 3"] = tri_mesh.vert_coords
    tets: Integer[t.LongTensor, "tet 4"] = tri_mesh.tets
    n_verts = tri_mesh.n_verts

    # dWdV gives dSdV except for the diagonal elements.
    sym_dSdV = _d_cotan_weights_d_vert_coords(vert_coords, tets, n_verts)

    # Compute the "diagonal" elements dS_iik
    dSdV_diag: Float[t.Tensor, "vert vert 3"] = t.sparse.sum(sym_dSdV, dim=1)
    # Note that the last dim is dense and does not show up in indices()
    diag_idx_i, diag_idx_k = dSdV_diag.indices()
    diag_idx = t.vstack((diag_idx_i, diag_idx_i, diag_idx_k))

    # Generate the final, complete dSdV gradients.
    dSdV = t.sparse_coo_tensor(
        t.hstack((sym_dSdV.indices(), diag_idx)),
        t.concatenate((sym_dSdV.values(), -dSdV_diag.values())),
        (n_verts, n_verts, n_verts, 3),
    ).coalesce()

    return dSdV

import torch as t
from jaxtyping import Float, Integer

from ...utils.constants import EPS


def _tet_signed_vols(
    vert_coords: Float[t.Tensor, "vert 3"], tets: Integer[t.LongTensor, "tet 4"]
) -> Float[t.Tensor, " tet"]:
    """
    Compute the signed volume of each tetrahedron in a 3D mesh. A tet is assigned
    a positive volume if it satisfies the right-hand rule (or, equivalently, its
    vertex indices can be reordered into ascending order with an even permutation).
    """
    tet_vert_coords: Float[t.Tensor, "tet 4 3"] = vert_coords[tets]

    # For each tet ijkl, compute the edge vectors ij, ik, and il. The volume of
    # the tet is given by the absolute value of the scalar triple product of these
    # three vectors, divided by 6.
    tet_edges = tet_vert_coords[:, [1, 2, 3], :] - tet_vert_coords[:, [0, 0, 0], :]

    tet_signed_vols = (
        t.sum(
            t.cross(tet_edges[:, 0], tet_edges[:, 1], dim=-1) * tet_edges[:, 2],
            dim=-1,
        )
        / 6.0
    )

    return tet_signed_vols


def _d_tet_signed_vols_d_vert_coords(
    vert_coords: Float[t.Tensor, "vert 3"], tets: Integer[t.LongTensor, "tet 4"]
) -> Float[t.Tensor, "tet 4 3"]:
    """
    Compute the gradient of the signed volume of each tetrahedron wrt the coordinates
    of its four vertices.
    """
    i, j, k, l = 0, 1, 2, 3

    tet_vert_coords: Float[t.Tensor, "tet 4 3"] = vert_coords[tets]

    # For each tet ijkl and for each vertex, find the (inward) area normal of the
    # base triangle, which is proportional to the gradient of the volume wrt the
    # vertex position.
    #
    # vert   base tri   area normal
    # ----------------------------------
    # i      jkl        jl x jk
    # j      ikl        ik x il
    # k      ijl        il x ij
    # l      ijk        ij x ik
    #
    # Note that, if a tet has a negative orientation, the resulting gradient will
    # also carry a negative sign (i.e., it points in the direction that minimizes
    # the unsigned/absolute volume of the tet).
    base_tri_edge_1 = (
        tet_vert_coords[:, [l, k, l, j]] - tet_vert_coords[:, [j, i, i, i]]
    )
    base_tri_edge_2 = (
        tet_vert_coords[:, [k, l, j, k]] - tet_vert_coords[:, [j, i, i, i]]
    )

    dVdV = t.cross(base_tri_edge_1, base_tri_edge_2, dim=-1) / 6.0

    return dVdV


def _tet_face_vector_areas(
    vert_coords: Float[t.Tensor, "vert 3"], tets: Integer[t.LongTensor, "tet 4"]
) -> tuple[
    Float[t.Tensor, "tet 6 3"], Float[t.Tensor, "tet 6 3"], Float[t.Tensor, "tet 6"]
]:
    """
    Compute the outward pointing vector areas for triangles in a tet mesh and their
    dihedral angles.

    Specifically, for each tet and each edge s, this function computes
    * the double area vectors for the two triangles sharing the opposite edge o,
    * the dihedral angle between these two triangles, weighted by the length of o;
      or, more precisely, the cotan weight -|o|cot(theta_o)/6
    """
    i, j, k, l = 0, 1, 2, 3

    tet_vols = t.abs(_tet_signed_vols(vert_coords, tets))

    tet_vert_coords: Float[t.Tensor, "tet 4 3"] = vert_coords[tets]

    # For each tet ijkl and each edge s, computes the (outward) normal on the two
    # triangles with o as the shared edge (i.e., th x o and hh x o).
    area_vec_to: Float[t.Tensor, "tet 6 3"] = t.cross(
        tet_vert_coords[:, [k, i, i, j, i, l]] - tet_vert_coords[:, [l, l, l, k, j, k]],
        tet_vert_coords[:, [i, j, j, i, k, j]] - tet_vert_coords[:, [l, l, l, k, j, k]],
        dim=-1,
    )
    area_vec_ho: Float[t.Tensor, "tet 6 3"] = t.cross(
        tet_vert_coords[:, [j, j, k, i, l, j]] - tet_vert_coords[:, [l, l, l, k, j, k]],
        tet_vert_coords[:, [k, k, i, l, i, i]] - tet_vert_coords[:, [l, l, l, k, j, k]],
        dim=-1,
    )

    # For each tet ijkl and each edge s, computes the contribution of s to the
    # cotan Laplacian (restricted to ijkl), which is given by -|o|cot(theta_o)/6,
    # where |o| is the length of the opposite edge, and theta_o is the dihedral
    # angle formed by the two triangles with o as the shared edge. This contribution
    # can also be written as <th x o, hh x o> / 36 * vol_ijkl; here, vol_ijkl is
    # the unsigned/absolute volume of the tet ijkl.
    weight_o: Float[t.Tensor, "tet 6"] = (
        t.sum(area_vec_to * area_vec_ho, dim=-1) / (36.0 * tet_vols + EPS)[:, None]
    )

    return area_vec_to, area_vec_ho, weight_o


def _bary_coord_grad_inner_prods(
    tet_signed_vols: Float[t.Tensor, " tet"],
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


def _whitney_2_form_inner_prods(
    vert_coords: Float[t.Tensor, "vert 3"], tets: Integer[t.LongTensor, "tet 4"]
) -> tuple[Float[t.Tensor, "tet 1"], Float[t.Tensor, "tet 4 4"]]:
    """
    For each tet, compute the pairwise inner product of the Whitney 2-form basis
    functions associated with the faces of the tet, and correct for the face and
    tet orientation.
    """
    i, j, k, l = 0, 1, 2, 3

    tet_vert_coords: Float[t.Tensor, "tet 4 3"] = vert_coords[tets]

    tet_signed_vols: Float[t.Tensor, " tet"] = _tet_signed_vols(vert_coords, tets)
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


def _cotan_weights(
    vert_coords: Float[t.Tensor, "vert 3"],
    tets: Integer[t.LongTensor, "tri 3"],
    n_verts: int,
) -> Float[t.Tensor, "vert vert"]:
    i, j, k, l = 0, 1, 2, 3

    _, _, weight_o = _tet_face_vector_areas(vert_coords, tets)

    # For each tet ijkl, each edge s contributes one term w_o to the weight matrix,
    # thus each tet contributes six terms (in COO format):
    #
    # [
    #   (i, j, w_kl = w_0),
    #   (i, k, w_jl = w_1),
    #   (j, k, w_il = w_2),
    #   (j, l, w_ik = w_3),
    #   (k, l, w_ij = w_4),
    #   (l, i, w_jk = w_5),
    # ]

    weights_idx = (
        tets[:, [i, i, j, j, k, l, j, k, k, l, l, i]].T.flatten().reshape(2, -1)
    )
    weights_val = weight_o.T.flatten()

    asym_weights = t.sparse_coo_tensor(weights_idx, weights_val, (n_verts, n_verts))
    sym_weights = (asym_weights + asym_weights.T).coalesce()

    return sym_weights

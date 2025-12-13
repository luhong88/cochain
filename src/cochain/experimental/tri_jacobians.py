import torch as t
from jaxtyping import Float, Integer

from ..complex import SimplicialComplex
from ..geometry.tri.tri_geometry import (
    _bary_coord_grad_inner_prods,
    _d_tri_areas_d_vert_coords,
    _tri_areas,
)
from ..geometry.tri.tri_hodge_stars import _star_1_circumcentric, star_0, star_2
from ..utils.constants import EPS


def _d2_tri_areas_d2_vert_coords(
    vert_coords: Float[t.Tensor, "vert 3"],
    tris: Integer[t.LongTensor, "tri 3"],
    vec: Float[t.Tensor, "tri 3 3"],
) -> Float[t.Tensor, "tri vert vert vert 3"]:
    """
    For each tri, given the gradient of the area grad_x[A] (shape: (tri, x=3, 3))
    and a vector field v_y associated with each vertex (shape: (tri, y=3, 3)),
    compute the pairwise "vector-Hessian product" (VHP) as VHP_xyp = hess_xp[A]@v_y.
    This is useful for computing the gradient vectors for inner products of the
    form I_xy = <grad_x[A], v_y>.
    """
    i, j, k = 0, 1, 2

    # vert  triple cross    grad_i                           grad_j                           grad_k
    # ---------------------------------------------------------------------------------------------------------
    # i     (ji x ki) x jk  <jk, jk>I - jk@jk.T              <ki, jk>I + jk@ki.T - 2ki@jk.T   -<ji, jk> - jk@ji.T + 2ji@jk.T
    # j     (kj x ij) x ki  -<kj, ki>I - ki@kj.T + 2kj@ki.T  <ki, ki>I - ki@ki.T              <kj, ki>I + ki@ij.T - 2ij@ki.T
    # k     (ik x jk) x ij  <ik, ij>I + ij@jk.T - 2jk@ij.T   -<ik, ij>I - ij@ik.T + 2ik@ij.T  <ij, ij>I - ij@ij.T

    tri_vert_coords: Float[t.Tensor, "tri 3 3"] = vert_coords[tris]

    vec1: Float[t.Tensor, "tri 3 3 3"] = (
        tri_vert_coords[:, [[k, i, i], [j, i, j], [k, k, j]]]
        - tri_vert_coords[:, [[j, k, j], [k, k, k], [i, i, i]]]
    )

    vec2: Float[t.Tensor, "tri 3 3 3"] = (
        tri_vert_coords[:, [[k, k, k], [i, i, i], [j, j, j]]]
        - tri_vert_coords[:, [[j, j, j], [k, k, k], [i, i, i]]]
    )

    sign_mask = t.Tensor(
        [[1, 1, -1], [-1, 1, 1], [1, -1, 1]],
        dtype=vert_coords.dtype,
        device=vert_coords.device,
    ).view(1, 1, 1, 3, 3)
    inner_prod: Float[t.Tensor, "tri 3 3 1 1"] = t.sum(vec1 * vec2, dim=-1).view(
        -1, 3, 3, 1, 1
    ) * t.eye(
        3,
        dtype=vert_coords.dtype,
        device=vert_coords.device,
    ).view(1, 1, 1, 3, 3)
    outer_prod: Float[t.Tensor, "tri 3 3 3 3"] = vec1.view(-1, 3, 3, 3, 1) * vec2.view(
        -1, 3, 3, 1, 3
    )
    outer_diff = -2.0 * outer_prod + outer_prod.transpose(-1, -2)

    triple_cross_grad: Float[t.Tensor, "tri 3 3 3 3"] = (
        inner_prod + outer_diff
    ) * sign_mask

    tri_area_grad: Float[t.Tensor, "tri 3 3"] = _d_tri_areas_d_vert_coords(
        vert_coords, tris
    )

    tri_areas: Float[t.Tensor, "tri 1 1 1 1"] = _tri_areas(vert_coords, tris).view(
        -1, 1, 1, 1, 1
    )

    hess: Float[t.Tensor, "tri 3 3 3 3"] = -tri_area_grad.view(
        -1, 3, 1, 1, 3
    ) * tri_area_grad.view(-1, 1, 3, 1, 3) / tri_areas + triple_cross_grad / (
        4.0 * tri_areas
    )

    hvp = t.einsum("txpcd,tyd->txypc", hess, vec)

    return hvp


def d_inv_star_2_d_vert_coords(
    tri_mesh: SimplicialComplex,
) -> Float[t.Tensor, "tri vert 3"]:
    """
    Compute the Jacobian of the inverse Hodge 2-star matrix (diagonal elements)
    with respect to vertex coordinates.
    """
    vert_coords: Float[t.Tensor, "vert 3"] = tri_mesh.vert_coords
    tris: Integer[t.LongTensor, "tri 3"] = tri_mesh.tris

    n_verts = tri_mesh.n_verts
    n_tris = tri_mesh.n_tris

    dAdV = _d_tri_areas_d_vert_coords(vert_coords, tris)

    dSdV_idx = t.vstack(
        (t.repeat_interleave(t.arange(n_tris, device=tris.device), 3), tris.flatten())
    )
    dSdV_val = dAdV.flatten(end_dim=1)
    dSdV = t.sparse_coo_tensor(dSdV_idx, dSdV_val, (n_tris, n_verts, 3)).coalesce()

    return dSdV


def d_star_2_d_vert_coords(
    tri_mesh: SimplicialComplex,
) -> Float[t.Tensor, "tri vert 3"]:
    """
    Compute the Jacobian of the Hodge 2-star matrix (diagonal elements) with respect
    to vertex coordinates.
    """
    d_inv_S_dV = d_inv_star_2_d_vert_coords(tri_mesh)

    s2 = star_2(tri_mesh)[d_inv_S_dV.indices()[0]]
    inv_scale = -s2.square()[:, None]

    dSdV = t.sparse_coo_tensor(
        d_inv_S_dV.indices(), d_inv_S_dV.values() * inv_scale, d_inv_S_dV.shape
    ).coalesce()

    return dSdV


def d_star_1_circumcentric_d_vert_coords(
    tri_mesh: SimplicialComplex,
) -> Float[t.Tensor, "edge vert 3"]:
    """
    Compute the Jacobian of the Hodge 1-star matrix (diagonal elements) with respect
    to vertex coordinates.
    """
    vert_coords: Float[t.Tensor, "vert 3"] = tri_mesh.vert_coords
    tris: Integer[t.LongTensor, "tri 3"] = tri_mesh.tris

    n_verts = tri_mesh.n_verts
    n_edges = tri_mesh.n_edges

    # Similar to how the 1-star can be computed by extracting the correct elements
    # from the cotan weights matrix, its jacobian can be computed by extracting
    # the correct elements from the jacobian of the cotan weights matrix. More
    # specifically, We will extract gradient vectors -dWdV_ijk for all canonical
    # edges e = ij, and store them in a new tensor dSdV_ek.
    dWdV: Float[t.Tensor, "vert vert vert 3"] = _d_cotan_weights_d_vert_coords(
        vert_coords, tris, n_verts
    )

    # Compute a flat index for all edges ij represented in dWdV_ijk.
    all_edge_idx = dWdV.indices()
    all_edge_idx_flat = all_edge_idx[0] * n_verts + all_edge_idx[1]

    # Similarly, compute a flat index for all canonical edges.
    canon_edges: Integer[t.Tensor, "edge 2"] = tri_mesh.edges
    canon_edge_idx_flat = canon_edges[:, 0] * n_verts + canon_edges[:, 1]

    # Find the "insertion location" of each edge into the list of canonical edges,
    # in a way that preserves the flat index ordering.
    all_edge_insert_loc = t.searchsorted(canon_edge_idx_flat, all_edge_idx_flat)
    # An edge is canonical iff its flat index matches the flat index of the canonical
    # edge at its insertion location. To perform this check, we need to prevent
    # out-of-bound errors by capping insertion locations to the number of canonical
    # edges.
    edge_insert_loc_clipped = t.clip(all_edge_insert_loc, 0, n_edges - 1)
    canon_edge_mask = canon_edge_idx_flat[edge_insert_loc_clipped] == all_edge_idx_flat

    # Final assembly.
    dSdV_e_idx = all_edge_insert_loc[canon_edge_mask]
    dSdV_k_idx = dWdV.indices()[2, canon_edge_mask]
    dSdV_val = -dWdV.values()[canon_edge_mask]

    dSdV = t.sparse_coo_tensor(
        t.vstack((dSdV_e_idx, dSdV_k_idx)),
        dSdV_val,
        (n_edges, n_verts, 3),
    ).coalesce()

    return dSdV


def d_inv_star_1_circumcentric_d_vert_coords(
    tri_mesh: SimplicialComplex,
) -> Float[t.Tensor, "edge vert 3"]:
    """
    Compute the Jacobian of the inverse Hodge 1-star matrix (diagonal elements)
    with respect to vertex coordinates.
    """
    dSdV = d_star_1_circumcentric_d_vert_coords(tri_mesh)

    s1 = _star_1_circumcentric(tri_mesh)[dSdV.indices()[0]]
    inv_scale = -1.0 / (s1.square()[:, None] + EPS)

    d_inv_S_dV = t.sparse_coo_tensor(
        dSdV.indices(), dSdV.values() * inv_scale, dSdV.shape
    ).coalesce()

    return d_inv_S_dV


def d_star_0_d_vert_coords(
    tri_mesh: SimplicialComplex,
) -> Float[t.Tensor, "vert vert 3"]:
    """
    Compute the Jacobian of the Hodge 0-star matrix (diagonal elements) with respect
    to vertex coordinates.
    """
    vert_coords: Float[t.Tensor, "vert 3"] = tri_mesh.vert_coords
    tris: Integer[t.LongTensor, "tri 3"] = tri_mesh.tris
    n_verts = tri_mesh.n_verts

    dAdV: Float[t.Tensor, "tri 3 3"] = _d_tri_areas_d_vert_coords(vert_coords, tris)

    # For each triangle ijk and each vertex s, dAdV_ijk_s contributes to the gradient
    # star0_ll wrt s whenever l = s or js is an edge in the mesh. Therefore, each
    # triangle ijk contributes 9 gradient terms, in COO format:
    # [
    #   (i, i, dAdV_ijk_i/3),
    #   (i, j, dAdV_ijk_j/3),
    #   (i, k, dAdV_ijk_k/3),
    #
    #   (j, i, dAdV_ijk_i/3),
    #   (j, j, dAdV_ijk_j/3),
    #   (j, k, dAdV_ijk_k/3),
    #
    #   (k, i, dAdV_ijk_i/3),
    #   (k, j, dAdV_ijk_j/3),
    #   (k, k, dAdV_ijk_k/3),
    # ]

    # Translate the ijk notation to actual indices to access tensor elements.
    i, j, k = 0, 1, 2

    dSdV_idx = (
        tris[
            :,
            [
                [i, i, i, j, j, j, k, k, k],  # first column/index
                [i, j, k, i, j, k, i, j, k],  # second column/index
            ],
        ]
        .transpose(0, 1)
        .flatten(start_dim=1)
    )

    dSdV_val = t.repeat_interleave(dAdV, repeats=3, dim=0).flatten(end_dim=1) / 3.0
    dSdV = t.sparse_coo_tensor(dSdV_idx, dSdV_val, (n_verts, n_verts, 3)).coalesce()

    return dSdV


def d_inv_star_0_d_vert_coords(
    tri_mesh: SimplicialComplex,
) -> Float[t.Tensor, "vert vert 3"]:
    """
    Compute the Jacobian of the inverse Hodge 0-star matrix (diagonal elements)
    with respect to vertex coordinates.
    """
    dSdV = d_star_0_d_vert_coords(tri_mesh)

    s0 = star_0(tri_mesh)[dSdV.indices()[0]]
    inv_scale = -1.0 / (s0.square()[:, None] + EPS)

    d_inv_S_dV = t.sparse_coo_tensor(
        dSdV.indices(), dSdV.values() * inv_scale, dSdV.shape
    ).coalesce()

    return d_inv_S_dV


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


def _d_cotan_weights_d_vert_coords(
    vert_coords: Float[t.Tensor, "vert 3"],
    tris: Integer[t.LongTensor, "tri 3"],
    n_verts: int,
) -> Float[t.Tensor, "vert vert vert 3"]:
    # For each triangle snp, and each vertex s, find the edge vectors sn and sp,
    # and a vector normal to the triangle at s (sn x sp), and the sine (squared)
    # of the angle at s.
    vert_s_coord: Float[t.Tensor, "tri 3 3"] = vert_coords[tris]

    edge_ns = vert_s_coord[:, [1, 2, 0], :] - vert_s_coord
    edge_ps = vert_s_coord[:, [2, 0, 1], :] - vert_s_coord

    edge_ns_len = t.linalg.norm(edge_ns, dim=-1, keepdim=True) + EPS
    edge_ps_len = t.linalg.norm(edge_ps, dim=-1, keepdim=True) + EPS

    uedge_ns = edge_ns / edge_ns_len
    uedge_ps = edge_ps / edge_ps_len

    norm_s: Float[t.Tensor, "tri 3 3"] = t.cross(uedge_ns, uedge_ps, dim=-1)
    sin_squared_s: Float[t.Tensor, "tri 3 1"] = (
        t.sum(norm_s.square(), dim=-1, keepdim=True) + EPS
    )
    unorm_s = norm_s / t.sqrt(sin_squared_s)

    # For each triangle snp, and for each of its vertex s, compute
    #   * cot_grad_sn: the gradient of cotan at s wrt n with
    #     * length 1/(|sn|*sin_s**2), where |sn| is the length of edge sn,
    #     * along the direction (sn x sp) x sn;
    #   * cot_grad_sp: the gradient of cotan at s wrt p with
    #     * length 1/(|sp|*sin_s**2),
    #     * along the direction (sn x sp) x ps (note the sign flip)
    #   * cot_grad_ss: the gradient of cotan at s wrt s itself; this is given
    #     by -(cot_grad_sn + cot_grad_sp), due to translational symmetry.
    cot_grad_sn = t.cross(unorm_s, uedge_ns, dim=-1) / (edge_ns_len * sin_squared_s)
    cot_grad_sp = t.cross(unorm_s, -uedge_ps, dim=-1) / (edge_ps_len * sin_squared_s)
    cot_grad_ss = -(cot_grad_sn + cot_grad_sp)

    # note that the neighbor dimension is ordered by local relation (snp), while
    # the vert dimension is ordered by global orientation (ijk)
    cot_grad: Float[t.Tensor, "tri vert=3 neighbor=3 coord=3"] = t.stack(
        (cot_grad_ss, cot_grad_sn, cot_grad_sp), dim=2
    )

    # First, we build the asymmetric, "off-diagonal" version of dW_ijk.
    #
    # For a given vertex s in triangle snp, because cot_s contributes to
    # L_np and cot_s is a function of all three vertices s, n, and p,
    # this vertex contributes three gradient terms:
    #
    #   * cot_grad_ss contributes to dW_nps,
    #   * cot_grad_sn contributes to dW_npn,
    #   * cot_grad_sp contributes to dW_npp,
    #
    # We can therefore workout all 9 contributions of each triangle ijk to the
    # asymmetric dW_ijk, in the COO format, by setting s to i, j, k and using
    # the local snp -> global ijk index mapping:
    #
    # [
    #   (j, k, i, -0.5*cot_grad_is),
    #   (j, k, j, -0.5*cot_grad_in),
    #   (j, k, k, -0.5*cot_grad_ip),
    #
    #   (k, i, j, -0.5*cot_grad_js),
    #   (k, i, k, -0.5*cot_grad_jn),
    #   (k, i, i, -0.5*cot_grad_jp),
    #
    #   (i, j, k, -0.5*cot_grad_ks),
    #   (i, j, i, -0.5*cot_grad_kn),
    #   (i, j, j, -0.5*cot_grad_kp),
    # ]
    #
    # Note that, since the neighbor dimension of cot_grad is ordered by snp,
    # it is unaffected by how i, j, or k relates to s.

    # Translate the ijk and snp notation to actual indices to access tensor elements.
    i, j, k = 0, 1, 2
    s, n, p = 0, 1, 2

    # fmt: off
    dWdV_idx = (
        tris[
            :,
            [
                j, j, j, k, k, k, i, i, i, # first column/index
                k, k, k, i, i, i, j, j, j, # second column/index
                i, j, k, j, k, i, k, i, j, # third column/index
            ],
        ]
        .T
        .flatten()
        .reshape(3, -1)
    )
    # fmt: on
    dWdV_val = -0.5 * cot_grad[
        :,
        [i, i, i, j, j, j, k, k, k],
        [s, n, p, s, n, p, s, n, p],
    ].transpose(0, 1).flatten(end_dim=-2)
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
    tris: Integer[t.LongTensor, "tri 3"] = tri_mesh.tris
    n_verts = tri_mesh.n_verts

    # dWdV gives dSdV except for the diagonal elements.
    sym_dSdV = _d_cotan_weights_d_vert_coords(vert_coords, tris, n_verts)

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

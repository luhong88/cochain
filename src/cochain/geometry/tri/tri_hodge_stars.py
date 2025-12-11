import torch as t
from jaxtyping import Float, Integer

from ...complex import SimplicialComplex
from ...utils.constants import EPS
from .tri_stiffness import _cotan_weights, _d_cotan_weights_d_vert_coords


def _tri_areas(
    vert_coords: Float[t.Tensor, "vert 3"], tris: Integer[t.LongTensor, "tri 3"]
) -> Float[t.Tensor, "tri"]:
    """
    Compute the area of all triangles in a 2D mesh.
    """
    vert_s_coord: Float[t.Tensor, "tri 3 3"] = vert_coords[tris]

    edge_ij = vert_s_coord[:, 1] - vert_s_coord[:, 0]
    edge_ik = vert_s_coord[:, 2] - vert_s_coord[:, 0]

    area = 0.5 * t.linalg.norm(t.cross(edge_ij, edge_ik, dim=-1), dim=-1) + EPS

    return area


def _d_tri_areas_d_vert_coords(
    vert_coords: Float[t.Tensor, "vert 3"], tris: Integer[t.LongTensor, "tri 3"]
) -> Float[t.Tensor, "tri 3 3"]:
    """
    Compute the gradient of the triangle areas with respect to vertex coordinates.
    """
    # For each triangle snp, and each vertex s, find the edge vectors sn, sp, and
    # np, and a vector normal to the triangle at s (sn x sp).
    vert_s_coord: Float[t.Tensor, "tri 3 3"] = vert_coords[tris]

    edge_ns = vert_s_coord[:, [1, 2, 0], :] - vert_s_coord
    edge_ps = vert_s_coord[:, [2, 0, 1], :] - vert_s_coord
    edge_np = vert_s_coord[:, [2, 0, 1], :] - vert_s_coord[:, [1, 2, 0], :]

    norm_s: Float[t.Tensor, "tri 3 3"] = t.cross(edge_ns, edge_ps, dim=-1)
    norm_s_len = t.linalg.norm(norm_s, dim=-1, keepdim=True) + EPS

    unorm_s = norm_s / norm_s_len

    # For each triangle snp, the gradient of its area with respect to each vertex
    # s is given by (unorm_s x edge_np)/2
    dAdV = t.cross(unorm_s, edge_np, dim=-1) / 2.0

    return dAdV


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


def star_2(tri_mesh: SimplicialComplex) -> Float[t.Tensor, "tri"]:
    """
    The Hodge 2-star operator maps the 2-simplices (triangles) in a mesh to their
    dual 0-cells. This function computes the ratio of the "volume" of the dual 0-cells
    (which is 1 by convention) to the area of the primal triangles. The returned tensor
    forms the diagonal of the 2-star tensor.
    """
    return 1.0 / _tri_areas(tri_mesh.vert_coords, tri_mesh.tris)


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


def star_1(tri_mesh: SimplicialComplex) -> Float[t.Tensor, "edge"]:
    """
    The Hodge 1-star operator maps the 1-simplices (edges) in a mesh to the
    circumcentric dual 1-cells. This function computes the length ratio of the dual
    1-cells to the primal edges, which is given by the cotan formula. The returned
    tensor forms the diagonal of the 1-star tensor.
    """
    vert_coords: Float[t.Tensor, "vert 3"] = tri_mesh.vert_coords
    tris: Integer[t.LongTensor, "tri 3"] = tri_mesh.tris

    n_verts = tri_mesh.n_verts

    # The cotan weights matrix (i.e., off-diagonal elements of the stiffness matrix)
    # already contains the desired values; i.e., S_ij = -0.5*sum_k[cot_k] for all
    # k that forms a triangle with i and j.
    weights: Float[t.Tensor, "vert vert"] = _cotan_weights(vert_coords, tris, n_verts)

    # For the weights matrix W_ij in COO format, the first row of its index tensor
    # is for the i dimension, and the second row is for the j dimension. We flatten
    # the two dimensions into a 1D index with (i, j) -> i*n_verts + j.
    all_idx = weights.indices()
    all_idx_flat = all_idx[0] * n_verts + all_idx[1]

    # Similarly, we convert the canonical edges represented by a vertex-indexing tuple
    # (i, j) into a flat index with the same formulae.
    edges: Integer[t.Tensor, "edge 2"] = tri_mesh.edges
    edge_idx_flat = edges[:, 0] * n_verts + edges[:, 1]

    # Identify the location of the canonical edge ij in the sparse W_ij indices,
    # using the flattened indices, and use the location to extract the cotan values.
    subset_idx = t.searchsorted(all_idx_flat, edge_idx_flat)
    subset_vals = weights.values()[subset_idx]

    return -subset_vals  # note the negative sign to get dual edge lengths


def d_star_1_d_vert_coords(
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


def d_inv_star_1_d_vert_coords(
    tri_mesh: SimplicialComplex,
) -> Float[t.Tensor, "edge vert 3"]:
    """
    Compute the Jacobian of the inverse Hodge 1-star matrix (diagonal elements)
    with respect to vertex coordinates.
    """
    dSdV = d_star_1_d_vert_coords(tri_mesh)

    s1 = star_1(tri_mesh)[dSdV.indices()[0]]
    inv_scale = -1.0 / (s1.square()[:, None] + EPS)

    d_inv_S_dV = t.sparse_coo_tensor(
        dSdV.indices(), dSdV.values() * inv_scale, dSdV.shape
    ).coalesce()

    return d_inv_S_dV


def _tri_edge_faces(
    tri_mesh: SimplicialComplex,
) -> tuple[Float[t.Tensor, "tri 3"], Integer[t.LongTensor, "tri 3"]]:
    """
    Enumerate all edges for each tri and find their orientations and indices on
    the tri_mesh.edges list.
    """
    device = tri_mesh.vert_coords.device

    n_verts = tri_mesh.n_verts

    # Enumerate all unique edges via their vertex position in the tris.
    i, j, k = 0, 1, 2
    unique_edges = t.tensor([[i, j], [i, k], [j, k]], dtype=t.long, device=device)

    # For each tri and each unique edge pair, find the orientations of the edges
    # and their indices on the list of unique, canonical edges (tri_mesh.edges).
    whitney_edges: Float[t.Tensor, "tri*3 2"] = tri_mesh.tris[:, unique_edges].flatten(
        end_dim=-2
    )

    # Same method as used in the construction of coboundary operators to use
    # sort() to identify edge orientations.
    whitney_canon_edges, whitney_edge_orientations = whitney_edges.sort(dim=-1)
    whitney_edge_signs: Float[t.Tensor, "tri 3"] = t.where(
        whitney_edge_orientations[:, 1] > 0, whitney_edge_orientations[:, 1], -1
    ).view(-1, 3)

    # This assumes that the edge indices in tri_mesh.edges are already in canonical
    # orders.
    unique_canon_edges_packed = tri_mesh.edges[:, 0] * n_verts + tri_mesh.edges[:, 1]
    canon_edges_packed_sorted, canon_edges_idx = t.sort(unique_canon_edges_packed)

    whitney_edges_packed = (
        whitney_canon_edges[:, 0] * n_verts + whitney_canon_edges[:, 1]
    )
    whitney_edges_idx: Float[t.Tensor, "tri 3"] = canon_edges_idx[
        t.searchsorted(canon_edges_packed_sorted, whitney_edges_packed)
    ].view(-1, 3)

    return whitney_edge_signs, whitney_edges_idx


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
    whitney_edge_signs, whitney_edges_idx = _tri_edge_faces(tri_mesh)

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
    whitney_edge_signs, whitney_edges_idx = _tri_edge_faces(tri_mesh)

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


def star_0(tri_mesh: SimplicialComplex) -> Float[t.Tensor, "vert"]:
    """
    The Hodge 0-star operator maps the 0-simplices (vertices) in a mesh to their
    barycentric dual 2-cells. This function computes the ratio of the area of the
    dual 2-cells to the "size" of the vertices (which is 1 by convention). The
    returned tensor forms the diagonal of the 0-star tensor.

    The barycentric dual area for each vertex is the sum of 1/3 of the areas of
    all triangles that share the vertex as a face.
    """
    n_verts = tri_mesh.n_verts

    tri_area = _tri_areas(tri_mesh.vert_coords, tri_mesh.tris)

    diag = t.zeros(n_verts, device=tri_mesh.vert_coords.device)
    diag.scatter_add_(
        dim=0,
        index=tri_mesh.tris.flatten(),
        src=t.repeat_interleave(tri_area / 3.0, 3),
    )

    return diag


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

import torch as t
from jaxtyping import Float, Integer

from ...complex import SimplicialComplex
from .tet_geometry import (
    _d_tet_signed_vols_d_vert_coords,
    _tet_face_vector_areas,
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
    n_verts = tet_mesh.n_verts

    tet_signed_vols = _tet_signed_vols(vert_coords, tets).view(-1, 1, 1)
    d_signed_vols_d_vert_coords = _d_tet_signed_vols_d_vert_coords(vert_coords, tets)

    # For a tet ijkl, let p be a position vector inside ijkl and let lambda_i(p)
    # be the barycentric coordinate of p wrt vertex i. The gradient of lambda_i(p)
    # wrt p is given by grad_i(vol_ijkl)/vol_ijkl, a constant wrt p.
    bary_coords_grad: Float[t.Tensor, "tet 4 3"] = (
        d_signed_vols_d_vert_coords / tet_signed_vols
    )
    # For each tet ijkl, compute all pairwise inner products of the barycentric
    # coordinate gradients wrt each pair of vertices.
    barry_coords_grad_dot: Float[t.Tensor, "tet 4 4"] = t.einsum(
        "ijk,ilk->ijl", bary_coords_grad, bary_coords_grad
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
    # gradient inner product (bary_coords_grad). Note that, since this expression
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
        bary_coords_int[:, x_idx, p_idx] * barry_coords_grad_dot[:, y_idx, q_idx]
        - bary_coords_int[:, x_idx, q_idx] * barry_coords_grad_dot[:, y_idx, p_idx]
        - bary_coords_int[:, y_idx, p_idx] * barry_coords_grad_dot[:, x_idx, q_idx]
        + bary_coords_int[:, y_idx, q_idx] * barry_coords_grad_dot[:, x_idx, p_idx]
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


def mass_2(tet_mesh: SimplicialComplex) -> Float[t.Tensor, "tri tri"]:
    """
    Compute the Galerkin triangle/2-form mass matrix.

    For each tet, each (canonical) triangle pairs xyz and rst and their associated
    Whitney 2-form basis functions W_xyz and W_rst contribute the inner product
    term int[W_t1*W_t2*dV] to the mass matrix element M[xyz, rst], where W_xyz =
    e_o[yz]/3V. Here, e_o[yz] is the length of the edge opposite to the edge yz
    (e.g., for the triangle ijk, this edge is il).
    """
    i, j, k, l = 0, 1, 2, 3

    vert_coords: Float[t.Tensor, "vert 3"] = tet_mesh.vert_coords
    tets: Integer[t.LongTensor, "tet 4"] = tet_mesh.tets

    n_verts = tet_mesh.n_verts
    n_tris = tet_mesh.n_tris
    n_tets = tet_mesh.n_tets

    # For each tet ijkl and each edge s, get the cotan weight
    # <th x o, hh x o> / 36 * vol_ijkl
    # Note that these weights only give the off-diagonal elements of the mass matrix.
    norm_tri_to, _, weight_o = _tet_face_vector_areas(vert_coords, tets)

    # For each tet and each cotan weight associated with edge s, find the two
    # triangles in the tet sharing the edge s, find their orientations, and
    # convert them to the canonical orientations.
    canon_pos_orientation = t.tensor([0, 1, 2], dtype=t.long, device=tets.device)

    weight_o_tri_to: Integer[t.LongTensor, "tet edge=6 vert=3"] = tets[
        :, [[i, k, l], [i, j, l], [i, j, l], [i, j, k], [i, j, k], [j, k, l]]
    ]
    weight_o_tri_to_canon, weight_o_tri_to_orientations = weight_o_tri_to.sort(dim=-1)
    # Same method as used in the construction of coboundary operators to use
    # sort() to identify triangle orientations.
    weight_o_tri_to_signs: Float[t.Tensor, "tet 6"] = t.where(
        condition=t.sum(weight_o_tri_to_orientations == canon_pos_orientation, dim=-1)
        == 1,
        self=-1.0,
        other=1.0,
    ).to(dtype=vert_coords.dtype)

    weight_o_tri_ho: Integer[t.LongTensor, "tet edge=6 vert=3"] = tets[
        :, [[j, k, l], [j, k, l], [i, k, l], [i, k, l], [i, j, l], [i, j, k]]
    ]
    weight_o_tri_ho_canon, weight_o_tri_ho_orientations = weight_o_tri_ho.sort(dim=-1)
    weight_o_tri_ho_signs: Float[t.Tensor, "tet 6"] = t.where(
        condition=t.sum(weight_o_tri_ho_orientations == canon_pos_orientation, dim=-1)
        == 1,
        self=-1.0,
        other=1.0,
    ).to(dtype=vert_coords.dtype)

    # Use the triangle orientation signs to correct weight_o to get the contribution
    # from canonical triangles. In addition, multiply the weight by 4 since the
    # contribution from each triangle pair is <th x o, hh x o> / 9 * vol_ijkl.
    weight_o_signed = 4.0 * weight_o * weight_o_tri_to_signs * weight_o_tri_ho_signs

    # Find the indices of the triangles on the list of unique, canonical triangles
    # (tet_mesh.tris) by radix encoding and searchsorted(). Because each triangle
    # ijk is encoded as i*n_verts^2 + j*n_verts + k and the max value of t.int64
    # is ~ 2^63, the max number of vertices this method can accommodate is
    # ~ n_verts < 2^21. Note that this method assumes that the triangle indices
    # in tet_mesh.tris are already in canonical orders.
    unique_canon_tris_packed = (
        tet_mesh.tris[:, 0] * n_verts**2
        + tet_mesh.tris[:, 1] * n_verts
        + tet_mesh.tris[:, 2]
    )
    canon_tris_packed_sorted, canon_tris_idx = t.sort(unique_canon_tris_packed)

    weight_o_tri_to_canon_flat: Integer[t.LongTensor, "tet*6 3"] = (
        weight_o_tri_to_canon.flatten(end_dim=-2)
    )
    tri_to_packed = (
        weight_o_tri_to_canon_flat[:, 0] * n_verts**2
        + weight_o_tri_to_canon_flat[:, 1] * n_verts
        + weight_o_tri_to_canon_flat[:, 2]
    )
    tri_to_idx = canon_tris_idx[t.searchsorted(canon_tris_packed_sorted, tri_to_packed)]

    weight_o_tri_ho_canon_flat: Integer[t.LongTensor, "tet*6 3"] = (
        weight_o_tri_ho_canon.flatten(end_dim=-2)
    )
    tri_ho_packed = (
        weight_o_tri_ho_canon_flat[:, 0] * n_verts**2
        + weight_o_tri_ho_canon_flat[:, 1] * n_verts
        + weight_o_tri_ho_canon_flat[:, 2]
    )
    tri_ho_idx = canon_tris_idx[t.searchsorted(canon_tris_packed_sorted, tri_ho_packed)]

    # First build the symmetric, off-diagonal version of the mass matrix.
    mass_asym = t.sparse_coo_tensor(
        t.vstack((tri_to_idx, tri_ho_idx)), weight_o_signed.flatten(), (n_tris, n_tris)
    )
    mass_off_diag = (mass_asym + mass_asym.T).coalesce()

    # Then, compute the diagonal elements.
    #
    # The norm_tri_to tensor contains triangle normal vectors for each tet and is
    # of shape (tet, edge=6, coord=3); since each tet has only four triangles,
    # the second "edge" dimension encodes duplicate triangles. In particular, by
    # its construction, the 0th, 1st, 3rd, and 5th elements along the "edge"
    # dimension encodes the four unique triangles (ikl, ijl, ijk, jkl).
    #
    # Note that one could have used the norm_tri_ho tensor, since both contains
    # the same duplicate info.
    unique_tri_to_idx = [0, 1, 3, 5]
    norm_unique_tri_to = norm_tri_to[:, unique_tri_to_idx, :].flatten(end_dim=-2)

    # Compute the diagonal mass matrix elements as <th x o, th x o> / 9 * vol_ijkl
    # Note that the orientation of the area vectors here is irrelevant, since the
    # orientation sign cancels in the dot product.
    norm_unique_tri_to_dot = t.sum(norm_unique_tri_to * norm_unique_tri_to, dim=-1)
    tet_vols_expanded = t.repeat_interleave(
        t.abs(_tet_signed_vols(vert_coords, tets)), 4
    )

    mass_diag_val = norm_unique_tri_to_dot / (9.0 * tet_vols_expanded)
    mass_diag_idx = tri_to_idx.view(n_tets, 6)[:, unique_tri_to_idx].flatten()

    mass_diag = t.sparse_coo_tensor(
        t.vstack((mass_diag_idx, mass_diag_idx)), mass_diag_val, (n_tris, n_tris)
    )

    mass = (mass_diag + mass_off_diag).coalesce()

    return mass


def mass_3(tet_mesh: SimplicialComplex) -> Float[t.Tensor, "tet"]:
    """
    Compute the diagonal of the tet/3-form mass matrix, which is equivalent to
    the inverse of 3-star.
    """
    return t.abs(_tet_signed_vols(tet_mesh.vert_coords, tet_mesh.tets))

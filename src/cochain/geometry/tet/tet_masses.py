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
    term int[W_xyz*W_rst*dV] to the mass matrix element M[xyz, rst], where

    W_xyz(p) = sign[xyz]*(p - v[-xyz])/3V

    Here, v[-xyz] is the coordinate vector of the vertex opposite to xyz. sign[xyz]
    is +1 whenever the triangle xyz satisfies the right-hand rule (i.e., the normal
    vector formed by the right hand points out of the tet), and -1 if not.
    """
    i, j, k, l = 0, 1, 2, 3

    vert_coords: Float[t.Tensor, "vert 3"] = tet_mesh.vert_coords
    tets: Integer[t.LongTensor, "tet 4"] = tet_mesh.tets
    tet_vert_coords: Float[t.Tensor, "tet 4 3"] = vert_coords[tets]

    dtype = vert_coords.dtype
    device = vert_coords.device

    n_verts = tet_mesh.n_verts
    n_tris = tet_mesh.n_tris

    tet_signed_vols: Float[t.Tensor, "tet"] = _tet_signed_vols(vert_coords, tets)
    tet_vols = t.abs(tet_signed_vols)
    tet_signs = t.sign(tet_signed_vols)

    # For each tet, associate the 1-form basis function with the opposite vertex.
    # Then, the inner product between the basis functions is given by
    #
    #               int[W_i*W_j*dV] = sum_k,l[C_kl*ik*lj]/(180V)
    #
    # Where C_kl = 1 + delta_kl (delta is the Kronecker delta function). Here,
    # the summation represents the inner products between all edge vectors emanating
    # from vertices i and j.

    all_edges: Float[t.Tensor, "tet 4 4 3"] = tet_vert_coords.view(
        -1, 1, 4, 3
    ) - tet_vert_coords.view(-1, 4, 1, 3)

    int_weights: Float[t.Tensor, "4 4"] = t.ones(
        (4, 4), dtype=dtype, device=device
    ) + t.eye(4, dtype=dtype, device=device)

    whitney_inner_prod: Float[t.Tensor, "tet 4 4"] = t.einsum(
        "bijc,bklc,jl->bik", all_edges, all_edges, int_weights
    ) / (180.0 * tet_vols.view(-1, 1, 1))

    # For each tet and each vertex, find the outward-facing triangle opposite
    # to the vertex (note that the way the triangles are indexed here satisfies
    # the right-hand rule for positively oriented tets).
    all_tris: Integer[t.LongTensor, "tet 4 3"] = tets[
        :, [[j, k, l], [i, l, k], [i, j, l], [i, k, j]]
    ]

    canon_pos_orientation = t.tensor([0, 1, 2], dtype=t.long, device=tets.device)

    all_canon_tris, all_tris_orientations = all_tris.sort(dim=-1)
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
    unique_canon_tris_packed_sorted, unique_canon_tris_idx = t.sort(
        unique_canon_tris_packed
    )

    all_canon_tris_flat: Integer[t.LongTensor, "tet*4 3"] = all_canon_tris.flatten(
        end_dim=-2
    )
    all_canon_tris_packed = (
        all_canon_tris_flat[:, 0] * n_verts**2
        + all_canon_tris_flat[:, 1] * n_verts
        + all_canon_tris_flat[:, 2]
    )
    all_canon_tris_idx: Integer[t.LongTensor, "tet 4"] = unique_canon_tris_idx[
        t.searchsorted(unique_canon_tris_packed_sorted, all_canon_tris_packed)
    ].view(-1, 4)

    # Assemble the mass matrix
    mass_idx = t.vstack(
        (
            all_canon_tris_idx.view(-1, 4, 1).expand(-1, 4, 4).flatten(),
            all_canon_tris_idx.view(-1, 1, 4).expand(-1, 4, 4).flatten(),
        )
    )
    mass_val = whitney_inner_prod_signed.flatten()
    mass = t.sparse_coo_tensor(mass_idx, mass_val, (n_tris, n_tris)).coalesce()

    return mass


def mass_3(tet_mesh: SimplicialComplex) -> Float[t.Tensor, "tet"]:
    """
    Compute the diagonal of the tet/3-form mass matrix, which is equivalent to
    the inverse of 3-star.
    """
    return t.abs(_tet_signed_vols(tet_mesh.vert_coords, tet_mesh.tets))

import torch as t
from jaxtyping import Float, Integer

from ...complex import SimplicialComplex
from .tet_geometry import (
    _d_tet_signed_vols_d_vert_coords,
    _tet_signed_vols,
)


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

    dtype = vert_coords.dtype
    device = vert_coords.device

    tet_signed_vols: Float[t.Tensor, "tet"] = _tet_signed_vols(vert_coords, tets)
    tet_vols = t.abs(tet_signed_vols)
    tet_signs = t.sign(tet_signed_vols)

    # For each tet, associate the 2-form basis function with the opposite vertex.
    # Then, the inner product between the basis functions is given by
    #
    #               int[W_i*W_j*dV] = sum_k,l[C_kl*<ik,jl>]/(180V)
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


def _tet_tri_face_idx(
    tets: Integer[t.LongTensor, "tet 4"],
    tris: Integer[t.LongTensor, "tri 3"],
    n_verts: int,
) -> Integer[t.LongTensor, "tet 4"]:
    """
    For each tet and each of its vertices, find the triangle face opposite to the
    vertex and its index in the tet_mesh.tris list.
    """
    i, j, k, l = 0, 1, 2, 3

    # For each tet and each vertex, triangle opposite to the vertex.
    all_tris: Integer[t.LongTensor, "tet 4 3"] = tets[
        :, [[j, k, l], [i, l, k], [i, j, l], [i, k, j]]
    ]

    all_canon_tris = all_tris.sort(dim=-1).values

    # Find the indices of the triangles on the list of unique, canonical triangles
    # (tet_mesh.tris) by radix encoding and searchsorted(). Because each triangle
    # ijk is encoded as i*n_verts^2 + j*n_verts + k and the max value of t.int64
    # is ~ 2^63, the max number of vertices this method can accommodate is
    # ~ n_verts < 2^21. Note that this method assumes that the triangle indices
    # in tet_mesh.tris are already in canonical orders.
    unique_canon_tris_packed = (
        tris[:, 0] * n_verts**2 + tris[:, 1] * n_verts + tris[:, 2]
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

    return all_canon_tris_idx


def _d_mass_2_d_vert_coords(
    vert_coords: Float[t.Tensor, "vert 3"],
    tets: Integer[t.LongTensor, "tet 4"],
    tris: Integer[t.LongTensor, "tri 3"],
    n_verts: int,
) -> tuple[
    Float[t.Tensor, "tet*64 3"],
    Integer[t.Tensor, "tet*64"],
    Integer[t.Tensor, "tet*64"],
    Integer[t.Tensor, "tet*64"],
]:
    tet_vert_coords: Float[t.Tensor, "tet 4 3"] = vert_coords[tets]

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

    # First, collect all the constituent terms required to compute the Jacobian.
    tet_signed_vols: Float[t.Tensor, "tet 1 1 1 1"] = _tet_signed_vols(
        vert_coords, tets
    ).view(-1, 1, 1, 1, 1)

    tet_vols = t.abs(tet_signed_vols)

    d_signed_vols_d_vert_coords: Float[t.Tensor, "tet 1 1 4 3"] = (
        _d_tet_signed_vols_d_vert_coords(vert_coords, tets)
    ).view(-1, 1, 1, 4, 3)

    all_edges: Float[t.Tensor, "tet 4 4 3"] = tet_vert_coords.view(
        -1, 1, 4, 3
    ) - tet_vert_coords.view(-1, 4, 1, 3)

    identity = t.eye(4, dtype=dtype, device=device)

    int_weights: Float[t.Tensor, "4 4"] = (
        t.ones((4, 4), dtype=dtype, device=device) + identity
    )

    sign_corrections, whitney_inner_prods = _whitney_2_form_inner_prods(
        vert_coords, tets
    )
    sign_corrections_shaped: Float[t.Tensor, "tet 4 4"] = sign_corrections.view(
        -1, 1, 4
    ) * sign_corrections.view(-1, 4, 1)
    whitney_inner_prods_shaped = whitney_inner_prods.view(-1, 4, 4, 1, 1)

    # Compute the delta_pk - delta_pi term of shape (i, p, k); this is equivalent
    # to the delta_pl - delta_pj term of shape (j, p, l).
    delta = identity.view(4, 4, 1) - identity.view(4, 1, 4)

    # Prepare all three terms in the sum into the form (tet, i, j, p, coords).
    # Note that the first two terms require a correction for the triangle and
    # tet orientations. The third term does not require this correction since
    # the function _whitney_2_form_inner_prods() already applies this correction
    # to the inner products.
    sum_1 = t.einsum("kl,pki,tjlc->tijpc", int_weights, delta, all_edges) / (
        180 * tet_vols
    )
    sum_2 = t.einsum("kl,plj,tikc->tijpc", int_weights, delta, all_edges) / (
        180 * tet_vols
    )
    sum_3 = whitney_inner_prods_shaped * d_signed_vols_d_vert_coords / tet_signed_vols

    whitney_inner_prod_grad: Float[t.Tensor, "tet i=4 j=4 p=4 3"] = (
        t.einsum("tij,tijpc->tijpc", sign_corrections_shaped, sum_1 + sum_2) - sum_3
    )

    all_canon_tris_idx: Integer[t.LongTensor, "tet 4"] = _tet_tri_face_idx(
        tets, tris, n_verts
    )

    dMdV_idx_i = all_canon_tris_idx.view(-1, 4, 1, 1).expand(-1, 4, 4, 4).flatten()
    dMdV_idx_j = all_canon_tris_idx.view(-1, 1, 4, 1).expand(-1, 4, 4, 4).flatten()
    dMdV_idx_p = tets.view(-1, 1, 1, 4).expand(-1, 4, 4, 4).flatten()

    dMdV_val = whitney_inner_prod_grad.flatten(end_dim=-2)

    return dMdV_val, dMdV_idx_i, dMdV_idx_j, dMdV_idx_p


class _Mass2(t.autograd.Function):
    @staticmethod
    def forward(
        vert_coords: Float[t.Tensor, "vert 3"],
        tets: Integer[t.LongTensor, "tet 4"],
        tris: Integer[t.LongTensor, "tri 3"],
        n_tris: int,
        n_verts: int,
    ) -> Float[t.Tensor, "tri tri"]:
        # First, compute the inner products of the Whitney 2-form basis functions.
        _, whitney_inner_prod_signed = _whitney_2_form_inner_prods(vert_coords, tets)

        # Then, find the indices of the tet triangle faces associated with the basis
        # functions.
        all_canon_tris_idx = _tet_tri_face_idx(tets, tris, n_verts)

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

    @staticmethod
    def setup_context(ctx, inputs, output):
        vert_coords, tets, tris, n_tris, n_verts = inputs

        ctx.save_for_backward(vert_coords, tets, tris)
        ctx.n_tris = n_tris
        ctx.n_verts = n_verts

    @staticmethod
    def backward(
        ctx, dLdM: Float[t.Tensor, "tri tri"]
    ) -> tuple[Float[t.Tensor, "vert 3"] | None, None, None, None, None]:
        if ctx.needs_input_grad[0] is None:
            return (None, None, None, None, None)

        vert_coords, tets, tris = ctx.saved_tensors

        dMdV_val, dMdV_idx_i, dMdV_idx_j, dMdV_idx_p = _d_mass_2_d_vert_coords(
            vert_coords, tets, tris, ctx.n_verts
        )

        # Compute the vector-Jacobian product between the "vector" dLdM of shape
        # (tri, tri) and the Jacobian dMdV of shape (tri, tri, vert, 3) as
        # t.einsum("ij,ijpc->pc", dLdM, dMdV). Note that, since tet_mesh.vert_coords
        # is always a dense tensor, the VJP also outputs a dense tensor.

        if dLdM.layout == t.strided:
            # If dLdM is a dense tensor, simply extract the elements from dLdM
            # that correspond to the nonzero elements of dMdV (along its first
            # two tri dimensions).
            dLdM_flat = dLdM[dMdV_idx_i, dMdV_idx_j].view(-1, 1)

            # Perform the einsum as a simple product between the nonzero elements
            # of triangle pairs, and scatter according to the vert dimension index
            # to generate the final VJP.
            dLdV = t.zeros_like(vert_coords)
            dLdV.scatter_add_(
                dim=0,
                index=dMdV_idx_p.view(-1, 1).expand(-1, 3),
                src=dLdM_flat * dMdV_val,
            )

        else:
            # TODO: the indexing logic can be cached for repeated backward passes

            # If dLdM is a sparse tensor, we need to find the "common nonzeros"
            # (cnz) of triangle pairs in both dLdM and dMdV, then performs the
            # product/scatter as before.

            # The dLdM needs to be a coalesced COO sparse tensor.
            dLdM_coo = dLdM.to_sparse_coo().coalesce()
            dLdM_idx = dLdM_coo.indices()

            # Perform radix packing to index the nonzero triangle pair elements
            # in both dLdM and dMdV.
            dLdM_idx_flat = dLdM_idx[0] * ctx.n_tris + dLdM_idx[1]
            dLdM_nnz = dLdM_idx_flat.size(0)

            dMdV_idx_flat = dMdV_idx_i * ctx.n_tris + dMdV_idx_j

            # For each nonzero triangle pair element in dMdV, determine if it is
            # a "common nonzero", and, if it is, how to map the dMdV element to
            # the corresponding dLdM element.

            # First, use searchsorted to find the "insertion locaiton" of each
            # dMdV nonzero element into dLdM nonzeros.
            dMdV_idx_insert_loc = t.searchsorted(dLdM_idx_flat, dMdV_idx_flat)
            dMdV_insert_loc_clipped = t.clip(dMdV_idx_insert_loc, 0, dLdM_nnz - 1)

            # If a dMdV nonzero element has a corresponding dLdM nonzero element,
            # then its packed index matches the packed index of the dLdM nonzero
            # element at its insertion location. Checking this gives a dMdV "common
            # nonzero" index mask.
            dMdV_cnz_idx_mask = dLdM_idx_flat[dMdV_insert_loc_clipped] == dMdV_idx_flat
            # The "insertion location" for the dMdV "common nonzero" elements give
            # the "common nonzero" indices of dLdM
            dLdM_cnz_idx = dMdV_idx_insert_loc[dMdV_cnz_idx_mask]

            # Multiply and scatter the "common nonzeros" to dLdV.
            dLdM_cnz_val = dLdM.values()[dLdM_cnz_idx].view(-1, 1)

            dMdV_cnz_val = dMdV_val[dMdV_cnz_idx_mask]
            dMdV_cnz_idx_p = dMdV_idx_p[dMdV_cnz_idx_mask].view(-1, 1).expand(-1, 3)

            dLdV = t.zeros_like(vert_coords)
            dLdV.scatter_add_(
                dim=0,
                index=dMdV_cnz_idx_p,
                src=dLdM_cnz_val * dMdV_cnz_val,
            )

        return dLdV, None, None, None, None

    def jvp(
        ctx,
        tangent_vert_coords: Float[t.Tensor, "vert 3"] | None,
        tangent_tets: None,
        tangent_tris: None,
        tangent_n_tris: None,
        tangent_n_verts: None,
    ) -> Float[t.Tensor, "tri tri"]:
        vert_coords, tets, tris = ctx.saved_tensors

        if tangent_vert_coords is None:
            dMdt = t.sparse_coo_tensor(
                indices=t.empty((2, 0), dtype=t.long, device=vert_coords.device),
                values=t.empty(
                    (0,), dtype=vert_coords.dtype, device=vert_coords.device
                ),
                size=(ctx.n_tris, ctx.n_tris),
            )

        else:
            # Compute the Jacobian-vector product between the "Jacobian" dMdV
            # of shape (tri, tri, vert, 3) and the tangent vector dVdt of shape
            # (vert, 3) as t.einsum("ijpc,pc->ij", dMdV, dVdt). Note that, since
            # the 2-form mass matrix is always a sparse tensor, the JVP also
            # outputs a sparse tensor.
            dMdV_val, dMdV_idx_i, dMdV_idx_j, dMdV_idx_p = _d_mass_2_d_vert_coords(
                vert_coords, tets, tris, ctx.n_verts
            )
            dVdt_flat = tangent_vert_coords[dMdV_idx_p]

            dMdt_val = (dMdV_val * dVdt_flat).sum(dim=-1)
            dMdt_idx = t.vstack((dMdV_idx_i, dMdV_idx_j))
            dMdt = t.sparse_coo_tensor(
                dMdt_idx, dMdt_val, (ctx.n_tris, ctx.n_tris)
            ).coalesce()

        return dMdt


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
    return _Mass2.apply(
        tet_mesh.vert_coords,
        tet_mesh.tets,
        tet_mesh.tris,
        tet_mesh.n_tris,
        tet_mesh.n_verts,
    )


def d_mass_2_d_vert_coords(
    tet_mesh: SimplicialComplex,
) -> Float[t.Tensor, "tri tri vert 3"]:
    """
    Compute the Jacobian of the 2-form mass matrix wrt the vertex coordinates.
    """
    dMdV_val, dMdV_idx_i, dMdV_idx_j, dMdV_idx_p = _d_mass_2_d_vert_coords(
        tet_mesh.vert_coords, tet_mesh.tets, tet_mesh.tris, tet_mesh.n_verts
    )

    dMdV_idx = t.vstack(
        (
            dMdV_idx_i,
            dMdV_idx_j,
            dMdV_idx_p,
        )
    )

    dMdV = t.sparse_coo_tensor(
        dMdV_idx, dMdV_val, (tet_mesh.n_tris, tet_mesh.n_tris, tet_mesh.n_verts, 3)
    ).coalesce()

    return dMdV

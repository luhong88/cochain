import torch as t
from jaxtyping import Float, Integer

from ...complex import SimplicialComplex
from ...utils.constants import EPS
from .tet_geometry import _d_tet_signed_vols_d_vert_coords, _tet_signed_vols


def _cotan_weights(
    vert_coords: Float[t.Tensor, "vert 3"],
    tets: Integer[t.LongTensor, "tri 3"],
    n_verts: int,
) -> tuple[
    Float[t.Tensor, "tet 6 3"],
    Float[t.Tensor, "tet 6 3"],
    Float[t.Tensor, "tet 6"],
    Float[t.Tensor, "vert vert"],
]:
    i, j, k, l = 0, 1, 2, 3

    tet_vert_coords: Float[t.Tensor, "tet 4 3"] = vert_coords[tets]
    tet_vols = t.abs(_tet_signed_vols(vert_coords, tets))

    # For each tet ijkl and each edge s, computes the (outward) normal on the two
    # triangles with o as the shared edge (i.e., th x o and hh x o).
    norm_tri_to: Float[t.Tensor, "tet 6 3"] = t.cross(
        tet_vert_coords[:, [k, i, i, j, i, l]] - tet_vert_coords[:, [l, l, l, k, j, k]],
        tet_vert_coords[:, [i, j, j, i, k, j]] - tet_vert_coords[:, [l, l, l, k, j, k]],
        dim=-1,
    )
    norm_tri_ho: Float[t.Tensor, "tet 6 3"] = t.cross(
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
        t.sum(norm_tri_to * norm_tri_ho, dim=-1) / (36.0 * tet_vols + EPS)[:, None]
    )

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

    return norm_tri_to, norm_tri_ho, weight_o, sym_weights


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
    # grad_p(w_o) = -(w_o*grad_p(vol_ijkl) + grad_p(<th x o, hh x o>)/36)/vol_ijkl
    tet_vert_coords: Float[t.Tensor, "tet 4 3"] = vert_coords[tets]

    norm_tri_to, norm_tri_ho, weight_o, _ = _cotan_weights(vert_coords, tets, n_verts)
    norm_tri_to_shaped: Float[t.Tensor, "tet 6 1 3"] = norm_tri_to.view(-1, 6, 1, 3)
    norm_tri_ho_shaped: Float[t.Tensor, "tet 6 1 3"] = norm_tri_ho.view(-1, 6, 1, 3)
    weight_o_shaped: Float[t.Tensor, "tet 6 1 1"] = weight_o.view(-1, 6, 1, 1)

    tet_signed_vols: Float[t.Tensor, "tet"] = _tet_signed_vols(vert_coords, tets)
    tet_signs = tet_signed_vols.sign()

    vols_shaped: Float[t.Tensor, "tet 1 1 1"] = t.abs(tet_signed_vols).view(-1, 1, 1, 1)

    # Multiply the gradient of the signed volumes with the signs of the volumes
    # to get the gradient of the unsigned/absolute volumes.
    vol_grad: Float[t.Tensor, "tet 1 4 3"] = (
        _d_tet_signed_vols_d_vert_coords(vert_coords, tets) * tet_signs.view(-1, 1, 1)
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


def stiffness_matrix(
    tet_mesh: SimplicialComplex,
) -> Float[t.Tensor, "vert vert"]:
    """
    Computes the stiffness matrix for a 3D mesh, sometimes also known as the "cotan
    Laplacian".
    """
    # The cotan weight matrix W gives the stiffness matrix except for the diagonal
    # elements.
    _, _, _, sym_stiffness = _cotan_weights(
        tet_mesh.vert_coords, tet_mesh.tets, tet_mesh.n_verts
    )

    # Compute the diagonal elements of the stiffness matrix.
    stiffness_diag = t.sparse.sum(sym_stiffness, dim=-1)
    # laplacian_diag.indices() has shape (1, nnz_diag)
    diag_idx = t.concatenate([stiffness_diag.indices(), stiffness_diag.indices()])

    # Generate the final, complete stiffness matrix.
    stiffness = t.sparse_coo_tensor(
        t.hstack((sym_stiffness.indices(), diag_idx)),
        t.concatenate((sym_stiffness.values(), -stiffness_diag.values())),
    ).coalesce()

    return stiffness


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

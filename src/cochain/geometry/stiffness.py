import torch as t
from jaxtyping import Float, Integer

from ..complex import Simplicial2Complex


def _cotan_weights(
    vert_coords: Float[t.Tensor, "vert 3"],
    tris: Integer[t.LongTensor, "tri 3"],
    n_verts: int,
) -> Float[t.Tensor, "vert vert"]:
    # For each triangle snp, and each vertex s, find the edge vectors sn and sp,
    # and use them to compute the cotan of the angle at s.
    vert_s_coord: Float[t.Tensor, "tri 3 3"] = vert_coords[tris]

    edge_ns = vert_s_coord[:, [1, 2, 0], :] - vert_s_coord
    edge_ps = vert_s_coord[:, [2, 0, 1], :] - vert_s_coord

    edge_ns_ps_dot = t.sum(edge_ns * edge_ps, dim=-1)
    edge_ns_ps_cross = t.linalg.norm(t.cross(edge_ns, edge_ps, dim=-1), dim=-1)
    cot_s: Float[t.Tensor, "tri 3"] = edge_ns_ps_dot / (1e-9 + edge_ns_ps_cross)

    # For each triangle snp, and each vertex s, scatter cot_s to edge np in the
    # weight matrix (W_np); i.e., each triangle ijk contributes the following
    # values to the asym_laplacian (in COO format):
    #
    # [
    #   (j, k, -0.5*cot_i),
    #   (i, k, -0.5*cot_j),
    #   (i, j, -0.5*cot_k),
    # ]

    # Translate the ijk notation to actual indices to access tensor elements.
    i, j, k = 0, 1, 2

    weights_idx = tris[:, [j, i, i, k, k, j]].T.flatten().reshape(2, -1)
    weights_val = -0.5 * cot_s[:, [i, j, k]].T.flatten()
    asym_weights = t.sparse_coo_tensor(weights_idx, weights_val, (n_verts, n_verts))

    # Symmetrize so that the cotan at i is scattered to both jk and kj.
    sym_weights = (asym_weights + asym_weights.T).coalesce()

    return sym_weights


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

    edge_ns_len = t.linalg.norm(edge_ns, dim=-1, keepdim=True) + 1e-9
    edge_ps_len = t.linalg.norm(edge_ps, dim=-1, keepdim=True) + 1e-9

    uedge_ns = edge_ns / edge_ns_len
    uedge_ps = edge_ps / edge_ps_len

    norm_s: Float[t.Tensor, "tri 3 3"] = t.cross(uedge_ns, uedge_ps, dim=-1)
    sin_squared_s: Float[t.Tensor, "tri 3 1"] = (
        t.sum(norm_s.square(), dim=-1, keepdim=True) + 1e-9
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


def stiffness_matrix(
    simplicial_mesh: Simplicial2Complex,
) -> Float[t.Tensor, "vert vert"]:
    """
    Computes the stiffness matrix for a 2D mesh, sometimes also known as the "cotan
    Laplacian".
    """
    # The cotan weight matrix W gives the stiffness matrix except for the diagonal
    # elements.
    sym_stiffness = _cotan_weights(
        simplicial_mesh.vert_coords, simplicial_mesh.tris, simplicial_mesh.n_verts
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
    simplicial_mesh: Simplicial2Complex,
) -> Float[t.Tensor, "vert vert vert 3"]:
    """
    Compute the jacobian of the stiffness matrix/cotan Laplacian with respect to
    the vertex coordinates.
    """
    vert_coords: Float[t.Tensor, "vert 3"] = simplicial_mesh.vert_coords
    tris: Integer[t.LongTensor, "tri 3"] = simplicial_mesh.tris
    n_verts = simplicial_mesh.n_verts

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

import torch as t
from jaxtyping import Float, Integer

from ..complex import Simplicial2Complex


def cotan_laplacian(
    simplicial_mesh: Simplicial2Complex,
) -> Float[t.Tensor, "vert vert"]:
    """
    Computes the cotan Laplacian (L0) for a 2D mesh.

    The input vert_coords and tris need to be on the same device
    """
    vert_coords: Float[t.Tensor, "vert 3"] = simplicial_mesh.vert_coords
    tris: Integer[t.LongTensor, "tri 3"] = simplicial_mesh.tris
    n_verts = simplicial_mesh.n_verts

    # For each triangle snp, and each vertex s, find the edge vectors sn and sp,
    # and use them to compute the cotan of the angle at s.
    vert_s_coord: Float[t.Tensor, "tri 3 3"] = vert_coords[tris]

    edge_ns = vert_s_coord[:, [1, 2, 0], :] - vert_s_coord
    edge_ps = vert_s_coord[:, [2, 0, 1], :] - vert_s_coord

    edge_ns_ps_dot = t.sum(edge_ns * edge_ps, dim=-1)
    edge_ns_ps_cross = t.linalg.norm(t.cross(edge_ns, edge_ps, dim=-1), dim=-1)
    cot_s: Float[t.Tensor, "tri 3"] = edge_ns_ps_dot / (1e-9 + edge_ns_ps_cross)

    # For each triangle snp, and each vertex s, scatter cot_s to edge np in the
    # laplacian (L_np); i.e., each triangle ijk contributes the following values
    # to the asym_laplacian (in COO format):
    #
    # [
    #   (j, k, -0.5*cot_i),
    #   (i, k, -0.5*cot_j),
    #   (i, j, -0.5*cot_k),
    # ]

    # Translate the ijk notation to actual indices to access tensor elements.
    i, j, k = 0, 1, 2

    laplacian_idx = tris[:, [j, i, i, k, k, j]].T.flatten().reshape(2, -1)
    laplacian_val = -0.5 * cot_s[:, [i, j, k]].T.flatten()
    asym_laplacian = t.sparse_coo_tensor(
        laplacian_idx, laplacian_val, (n_verts, n_verts)
    )

    # Symmetrize so that the cotan at i is scattered to both jk and kj.
    sym_laplacian = (asym_laplacian + asym_laplacian.T).coalesce()

    # Compute the diagonal elements of the laplacian.
    laplacian_diag = t.sparse.sum(sym_laplacian, dim=-1)
    # laplacian_diag.indices() has shape (1, nnz_diag)
    diag_idx = t.concatenate([laplacian_diag.indices(), laplacian_diag.indices()])

    # Generate the final, complete Laplacian operator.
    laplacian = (
        t.sparse_coo_tensor(
            t.hstack((sym_laplacian.indices(), diag_idx)),
            t.concatenate((sym_laplacian.values(), -laplacian_diag.values())),
        )
        .coalesce()
        .to_sparse_csr()
    )

    return laplacian


def d_cotan_laplacian_d_vert_coords(
    simplicial_mesh: Simplicial2Complex,
) -> Float[t.Tensor, "vert vert vert 3"]:
    """
    Compute the jacobian of the cotan Laplacian with respect to the vertex coordinates.
    """
    vert_coords: Float[t.Tensor, "vert 3"] = simplicial_mesh.vert_coords
    tris: Integer[t.LongTensor, "tri 3"] = simplicial_mesh.tris
    n_verts = simplicial_mesh.n_verts

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
        t.sum(norm_s**2, dim=-1, keepdim=True) + 1e-9
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

    # First, we build the asymmetric, "off-diagonal" version of dL_ijk.
    #
    # For a given vertex s in triangle snp, because cot_s contributes to
    # L_np and cot_s is a function of all three vertices s, n, and p,
    # this vertex contributes three gradient terms:
    #
    #   * cot_grad_ss contributes to dL_nps,
    #   * cot_grad_sn contributes to dL_npn,
    #   * cot_grad_sp contributes to dL_npp,
    #
    # We can therefore workout all 9 contributions of each triangle ijk to the
    # asymmetric dL_ijk, in the COO format, by setting s to i, j, k and using
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
    dLdV_idx = (
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
    dLdV_val = -0.5 * cot_grad[
        :,
        [i, i, i, j, j, j, k, k, k],
        [s, n, p, s, n, p, s, n, p],
    ].transpose(dim0=0, dim1=1).flatten(end_dim=-2)
    asym_dLdV = t.sparse_coo_tensor(
        dLdV_idx, dLdV_val, (n_verts, n_verts, n_verts, 3)
    ).coalesce()

    # Symmetrize so that dL_ijk = dL_jki
    sym_dLdV = (asym_dLdV + asym_dLdV.transpose(dim0=0, dim1=1)).coalesce()

    # Compute the "diagonal" elements dL_iik
    dLdV_diag: Float[t.Tensor, "vert vert 3"] = t.sparse.sum(sym_dLdV, dim=1)
    # Note that the last dim is dense and does not show up in indices()
    diag_idx_i, diag_idx_k = dLdV_diag.indices()
    diag_idx = t.vstack((diag_idx_i, diag_idx_i, diag_idx_k))

    # Generate the final, complete dLdV gradients.
    dLdV = (
        t.sparse_coo_tensor(
            t.hstack((sym_dLdV.indices(), diag_idx)),
            t.concatenate((sym_dLdV.values(), -dLdV_diag.values())),
            (n_verts, n_verts, n_verts, 3),
        )
        .coalesce()
        .to_sparse_csr()
    )

    return dLdV

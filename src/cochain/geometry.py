import torch as t
from jaxtyping import Float, Integer

from .complex import Simplicial2Complex

# We adopt the following convention for describing the relation between vertices
# in a triangle. For a given triangle represented by three vertex indices, we refer
# the first, second, and third vertex with index i, j, and k. This effectively
# assigns an orientation to the triangle, and allows us to distinguish the neighbors
# for each vertex ("self", or s) as either the "next" (n) and "previous" (p) vertex.
# For a triangle ijk, the "self"/"next"/"prev" relation is defined as follows:
#
# -------
# s  n  p
# -------
# i  j  k
# j  k  i
# k  i  j
# -------
#
# We will refer to a triangle as ijk or snp, depending on the context.


def _cotan_laplacian(
    vert_coords: Float[t.Tensor, "vert 3"], tris: Integer[t.LongTensor, "tri 3"]
) -> Float[t.Tensor, "vert vert"]:
    """
    Computes the cotan Laplacian (L0) for a 2D mesh.

    The input vert_coords and tris need to be on the same device
    """
    n_verts = vert_coords.shape[0]

    # For each triangle, compute the cotan of the angle at each vertex.
    tri_vert_coord: Float[t.Tensor, "tri 3 3"] = vert_coords[tris]

    tri_vert_vec1 = tri_vert_coord[:, [1, 2, 0], :] - tri_vert_coord
    tri_vert_vec2 = tri_vert_coord[:, [2, 0, 1], :] - tri_vert_coord

    tri_vert_ang_dot = t.sum(tri_vert_vec1 * tri_vert_vec2, dim=-1)
    tri_vert_ang_cross = t.linalg.norm(
        t.cross(tri_vert_vec1, tri_vert_vec2, dim=-1), dim=-1
    )
    tri_vert_ang_cotan: Float[t.Tensor, "tri 3"] = tri_vert_ang_dot / (
        1e-9 + tri_vert_ang_cross
    )

    # For each triangle ijk, and each of its vertex i, scatter the cotan at i to
    # edge jk in the laplacian (L_jk); i.e., each triangle ijk contributes the
    # following values to the asym_laplacian (in COO format):
    # [
    #   (j, k, -0.5*cot_i),
    #   (i, k, -0.5*cot_j),
    #   (i, j, -0.5*cot_k),
    # ]
    laplacian_idx = tris[:, [1, 0, 0, 2, 2, 1]].T.flatten().reshape(2, -1)
    laplacian_val = -0.5 * tri_vert_ang_cotan[:, [0, 1, 2]].T.flatten()
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
    laplacian = t.sparse_coo_tensor(
        t.hstack((sym_laplacian.indices(), diag_idx)),
        t.concatenate((sym_laplacian.values(), -laplacian_diag.values())),
    ).coalesce()

    return laplacian


def _d_cotan_laplacian_d_vert_coords(
    vert_coords: Float[t.Tensor, "vert 3"],
    tris: Integer[t.LongTensor, "tri 3"],
) -> Float[t.Tensor, "vert vert vert 3"]:
    n_verts = vert_coords.shape[0]

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
    # asymmetric dL_ijk, in the COO format, by setting self to i, j, k and using
    # the local -> global index mapping:
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
    dLdV = t.sparse_coo_tensor(
        t.hstack((sym_dLdV.indices(), diag_idx)),
        t.concatenate((sym_dLdV.values(), -dLdV_diag.values())),
        (n_verts, n_verts, n_verts, 3),
    ).coalesce()

    return dLdV


class _DifferentiableCotanLaplacian(t.autograd.Function):
    def forward(
        ctx,
        vert_coords: Float[t.Tensor, "vert 3"],
        tris: Integer[t.LongTensor, "tri 3"],
    ) -> Float[t.Tensor, "vert vert"]:
        ctx.save_for_backward(vert_coords, tris)

        return _cotan_laplacian(vert_coords, tris).to_sparse_csr()

    def backward(
        ctx,
        grad_outputs: Float[t.Tensor, "vert vert"],
    ):
        vert_coords, tris = ctx.saved_tensors

        dLdV: Float[t.Tensor, "vert vert vert 3"] = _d_cotan_laplacian_d_vert_coords(
            vert_coords, tris
        )

        # Force dense grad_outputs
        grad = (
            grad_outputs.to_dense()
            if grad_outputs.layout != t.strided
            else grad_outputs
        )
        # The final gradient of loss w.r.t. vertex coordinates, which we denote
        # as dV_kl, can be computed via chain rule as dV_kl= sum_ij[grad_ij*dLdV_ijkl];
        # note that l is a dense dimension. In addition, since vert_coords is dense,
        # dV will also need to be a dense tensor.
        dLdV_values: Float[t.Tensor, "nz 3"] = dLdV.values()
        dLdV_idx_i, dLdV_idx_j, dLdV_idx_k = dLdV.indices()

        dV_values: Float[t.Tensor, "nz 3"] = (
            grad[dLdV_idx_i, dLdV_idx_j].unsqueeze(-1) * dLdV_values
        )

        dV = t.zeros_like(vert_coords)
        # TODO: use torch_scatter to improve performance
        dV.index_add_(0, dLdV_idx_k, dV_values)

        # Cannot compute gradient w.r.t. topology (yet)
        dT = None

        return (dV, dT)


def cotan_laplacian(
    simplicial_mesh: Simplicial2Complex,
) -> Float[t.Tensor, "vert vert"]:
    return _DifferentiableCotanLaplacian().apply(
        simplicial_mesh.vert_coords, simplicial_mesh.tris
    )

import torch as t
from jaxtyping import Float, Integer

from .complex import Simplicial2Complex


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

    # For each triangle ijk, and for each of its vertex i, define (according to
    # the given triangle orientation), the "next" neighbor edge ij (vec1) and the
    # "previous" neighbor edge ik (vec2). Compute the ij, ik edge lengths, the
    # vector normal to ijk at i (ij x ik), and the sine (squared) of the angle at i.
    vert_self_coord: Float[t.Tensor, "tri 3 3"] = vert_coords[tris]

    edge_next_self = vert_self_coord[:, [1, 2, 0], :] - vert_self_coord
    edge_prev_selv = vert_self_coord[:, [2, 0, 1], :] - vert_self_coord

    edge_next_self_len = t.linalg.norm(edge_next_self, dim=-1, keepdim=True) + 1e-9
    edge_prev_self_len = t.linalg.norm(edge_prev_selv, dim=-1, keepdim=True) + 1e-9

    uedge_next_self = edge_next_self / edge_next_self_len
    uedge_prev_self = edge_prev_selv / edge_prev_self_len

    norm_self: Float[t.Tensor, "tri 3 3"] = t.cross(
        uedge_next_self, uedge_prev_self, dim=-1
    )
    sin_sq_self: Float[t.Tensor, "tri 3 1"] = (
        t.sum(norm_self**2, dim=-1, keepdim=True) + 1e-9
    )
    unorm_self = norm_self / t.sqrt(sin_sq_self)

    # For each triangle ijk, and for each of its vertex i, compute
    #   * grad_j: the gradient of cotan at i w.r.t. CW neighbor vertex j with
    #     * length 1/(|ij|*sin_i**2), where |ij| is the length of edge ij,
    #     * along the direction (ij x ik) x ij;
    #   * grad_k: the gradient of cotan at i w.r.t. CCW neighbor vertex k with
    #     * length 1/(|ik|*sin_i**2),
    #     * along the direction (ij x ik) x ki (note the sign flip)
    #   * grad_i: the gradient of cotan at i w.r.t. vertex i itself; this is given
    #     by -(grad1 + grad2), due to translational symmetry.
    cot_grad_self_wrt_next = t.cross(unorm_self, uedge_next_self, dim=-1) / (
        edge_next_self_len * sin_sq_self
    )
    cot_grad_self_wrt_prev = t.cross(unorm_self, -uedge_prev_self, dim=-1) / (
        edge_prev_self_len * sin_sq_self
    )
    cot_grad_self_wrt_self = -(cot_grad_self_wrt_next + cot_grad_self_wrt_prev)

    cot_grad: Float[t.Tensor, "tri vert=3 neighbor=3 coord=3"] = t.stack(
        (cot_grad_self_wrt_self, cot_grad_self_wrt_next, cot_grad_self_wrt_prev), dim=2
    )

    # First, we build the asymmetric, "off-diagonal" version of dL_ijk.
    #
    # For a triangle ijk, the "self"/"next"/"prev" relation is defined as follows:
    #
    # --------------
    # self next prev
    # --------------
    # i    j    k
    # j    k    i
    # k    i    j
    # --------------
    #
    # For a given vertex "self" in triangle ijk, because cot_self contributes to
    # L_next/prev and cot_self is a function of all three vertices ("self", "next",
    # and "prev"), this vertex contributes three gradient terms:
    #   * cot_grad_self_wrt_self contributes to dL_next/prev/self,
    #   * cot_grad_self_wrt_next contributes to dL_next/prev/next,
    #   * cot_grad_self_wrt_prev contributes to dL_next/prev/prev,
    #
    # We can therefore workout all 9 contributions of each triangle ijk to the
    # asymmetric dL_ijk, in the COO format, by setting self to i, j, k and using
    # the local -> global index mapping:
    #
    # [
    #   (j, k, i, -0.5*cot_grad_ii),
    #   (j, k, j, -0.5*cot_grad_ij),
    #   (j, k, k, -0.5*cot_grad_ik),
    #
    #   (k, i, j, -0.5*cot_grad_ji),
    #   (k, i, k, -0.5*cot_grad_jj),
    #   (k, i, i, -0.5*cot_grad_jk),
    #
    #   (i, j, k, -0.5*cot_grad_ki),
    #   (i, j, i, -0.5*cot_grad_kj),
    #   (i, j, j, -0.5*cot_grad_kk),
    # ]

    # Translate the i,j,k notation to actual indices to access tensor elements.
    i, j, k = 0, 1, 2

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
        [i, j, k, i, j, k, i, j, k],
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

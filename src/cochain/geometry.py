import torch as t
from jaxtyping import Float, Integer

from .complex import Simplicial2Complex


def cotan_laplacian(
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
    vert_coords: Float[t.Tensor, "vert 3"],
    tris: Integer[t.LongTensor, "tri 3"],
) -> Float[t.Tensor, "vert vert vert 3"]:
    n_verts = vert_coords.shape[0]

    # For each triangle ijk, and for each of its vertex i, define (according to
    # the given triangle orientation), the CW neighbor edge ij (vec1) and the CCW
    # neighbor edge ik (vec2). Compute the ij, ik edge lengths, the vector normal
    # to ijk at i (ij x ik), and the sine of the angle at i.
    tri_vert_coord: Float[t.Tensor, "tri 3 3"] = vert_coords[tris]

    tri_vert_vec1 = tri_vert_coord[:, [1, 2, 0], :] - tri_vert_coord
    tri_vert_vec2 = tri_vert_coord[:, [2, 0, 1], :] - tri_vert_coord

    tri_vert_vec1_len = t.linalg.norm(tri_vert_vec1, dim=-1, keepdim=True) + 1e-9
    tri_vert_vec2_len = t.linalg.norm(tri_vert_vec2, dim=-1, keepdim=True) + 1e-9

    tri_vert_uvec1 = tri_vert_vec1 / tri_vert_vec1_len
    tri_vert_uvec2 = tri_vert_vec2 / tri_vert_vec2_len

    tri_vert_norm: Float[t.Tensor, "tri 3 3"] = t.cross(
        tri_vert_uvec1, tri_vert_uvec2, dim=-1
    )
    tri_vert_ang_sin_sq: Float[t.Tensor, "tri 3 1"] = (
        t.sum(tri_vert_norm**2, dim=-1, keepdim=True) + 1e-9
    )
    tri_vert_unorm = tri_vert_norm / t.sqrt(tri_vert_ang_sin_sq)

    # For each triangle ijk, and for each of its vertex i, compute
    #   * grad_j: the gradient of cotan at i w.r.t. CW neighbor vertex j with
    #     * length 1/(|ij|*sin_i**2), where |ij| is the length of edge ij,
    #     * along the direction (ij x ik) x ij;
    #   * grad_k: the gradient of cotan at i w.r.t. CCW neighbor vertex k with
    #     * length 1/(|ik|*sin_i**2),
    #     * along the direction (ij x ik) x ki (note the sign flip)
    #   * grad_i: the gradient of cotan at i w.r.t. vertex i itself; this is given
    #     by -(grad1 + grad2), due to translational symmetry.
    tri_vert_grad_j = t.cross(tri_vert_unorm, tri_vert_uvec1, dim=-1) / (
        tri_vert_vec1_len * tri_vert_ang_sin_sq
    )
    tri_vert_grad_k = t.cross(tri_vert_unorm, -tri_vert_uvec2, dim=-1) / (
        tri_vert_vec2_len * tri_vert_ang_sin_sq
    )
    tri_vert_grad_i = -(tri_vert_grad_j + tri_vert_grad_k)

    tri_vert_grad: Float[t.Tensor, "tri vert=3 grad=3 coord=3"] = t.stack(
        (tri_vert_grad_i, tri_vert_grad_j, tri_vert_grad_k), dim=2
    )

    # First, we build the asymmetric, "off-diagonal" version of dL_ijk.
    #
    # For a given triangle ijk, because cot_i contributes to L_jk and cot_i is a
    # function of all three vertices i, j, and k, vertex i contributes three gradient
    # terms:
    #   * grad_ii (grad_i of cot_i) contributes to dL_jki,
    #   * grad_ij (grad_j of cot_i) contributes to dL_jkj,
    #   * grad_ik (grad_k of cot_i) contributes to dL_jkk,
    #
    # Using symmetry, we can work out all 9 contributions to the asymmetric dL_ijk;
    # writing this in COO format:
    # [
    #   (j, k, i, -0.5*grad_ii),
    #   (j, k, j, -0.5*grad_ij),
    #   (j, k, k, -0.5*grad_ik),
    #
    #   (i, k, i, -0.5*grad_ji),
    #   (i, k, j, -0.5*grad_jj),
    #   (i, k, k, -0.5*grad_jk),
    #
    #   (i, j, i, -0.5*grad_ki),
    #   (i, j, j, -0.5*grad_kj),
    #   (i, j, k, -0.5*grad_kk),
    # ]
    dLdV_idx = (
        tris[
            :,
            [
                1,
                1,
                1,
                0,
                0,
                0,
                0,
                0,
                0,
                2,
                2,
                2,
                2,
                2,
                2,
                1,
                1,
                1,
                0,
                1,
                2,
                0,
                1,
                2,
                0,
                1,
                2,
            ],
        ]
        .T.flatten()
        .reshape(3, -1)
    )
    dLdV_val = -0.5 * tri_vert_grad[
        :,
        [0, 0, 0, 1, 1, 1, 2, 2, 2],
        [0, 1, 2, 0, 1, 2, 0, 1, 2],
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


class DifferentiableCotanLaplacian(t.autograd.Function):
    def forward(
        ctx,
        vert_coords: Float[t.Tensor, "vert 3"],
        tris: Integer[t.LongTensor, "tri 3"],
    ) -> Float[t.Tensor, "vert vert"]:
        ctx.save_for_backward(vert_coords, tris)

        return cotan_laplacian(vert_coords, tris)

    def backward(
        ctx,
        grad_outputs,
    ):
        vert_coords, tris = ctx.saved_tensors

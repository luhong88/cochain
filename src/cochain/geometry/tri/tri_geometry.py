import torch as t
from jaxtyping import Float, Integer

from ...utils.constants import EPS


def compute_tri_areas(
    vert_coords: Float[t.Tensor, "vert 3"], tris: Integer[t.LongTensor, "tri 3"]
) -> Float[t.Tensor, " tri"]:
    """
    Compute the area of all triangles in a 2D mesh.
    """
    vert_s_coord: Float[t.Tensor, "tri 3 3"] = vert_coords[tris]

    edge_ij = vert_s_coord[:, 1] - vert_s_coord[:, 0]
    edge_ik = vert_s_coord[:, 2] - vert_s_coord[:, 0]

    area = 0.5 * t.linalg.norm(t.cross(edge_ij, edge_ik, dim=-1), dim=-1) + EPS

    return area


def compute_d_tri_areas_d_vert_coords(
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


def bary_coord_grad_inner_prods(
    tri_areas: Float[t.Tensor, " tri"],
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


def cotan_weights(
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
    cot_s: Float[t.Tensor, "tri 3"] = edge_ns_ps_dot / (EPS + edge_ns_ps_cross)

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

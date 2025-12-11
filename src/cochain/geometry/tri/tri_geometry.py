import torch as t
from jaxtyping import Float, Integer

from ...utils.constants import EPS


def _tri_areas(
    vert_coords: Float[t.Tensor, "vert 3"], tris: Integer[t.LongTensor, "tri 3"]
) -> Float[t.Tensor, "tri"]:
    """
    Compute the area of all triangles in a 2D mesh.
    """
    vert_s_coord: Float[t.Tensor, "tri 3 3"] = vert_coords[tris]

    edge_ij = vert_s_coord[:, 1] - vert_s_coord[:, 0]
    edge_ik = vert_s_coord[:, 2] - vert_s_coord[:, 0]

    area = 0.5 * t.linalg.norm(t.cross(edge_ij, edge_ik, dim=-1), dim=-1) + EPS

    return area


def _d_tri_areas_d_vert_coords(
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


def _d2_tri_areas_d2_vert_coords(
    vert_coords: Float[t.Tensor, "vert 3"],
    tris: Integer[t.LongTensor, "tri 3"],
    vec: Float[t.Tensor, "tri 3 3"],
) -> Float[t.Tensor, "tri vert vert vert 3"]:
    """
    For each tri, given the gradient of the area grad_x[A] (shape: (tri, x=3, 3))
    and a vector field v_y associated with each vertex (shape: (tri, y=3, 3)),
    compute the pairwise "vector-Hessian product" (VHP) as VHP_xyp = hess_xp[A]@v_y.
    This is useful for computing the gradient vectors for inner products of the
    form I_xy = <grad_x[A], v_y>.
    """
    i, j, k = 0, 1, 2

    # vert  triple cross    grad_i                           grad_j                           grad_k
    # ---------------------------------------------------------------------------------------------------------
    # i     (ji x ki) x jk  <jk, jk>I - jk@jk.T              <ki, jk>I + jk@ki.T - 2ki@jk.T   -<ji, jk> - jk@ji.T + 2ji@jk.T
    # j     (kj x ij) x ki  -<kj, ki>I - ki@kj.T + 2kj@ki.T  <ki, ki>I - ki@ki.T              <kj, ki>I + ki@ij.T - 2ij@ki.T
    # k     (ik x jk) x ij  <ik, ij>I + ij@jk.T - 2jk@ij.T   -<ik, ij>I - ij@ik.T + 2ik@ij.T  <ij, ij>I - ij@ij.T

    tri_vert_coords: Float[t.Tensor, "tri 3 3"] = vert_coords[tris]

    vec1: Float[t.Tensor, "tri 3 3 3"] = (
        tri_vert_coords[:, [[k, i, i], [j, i, j], [k, k, j]]]
        - tri_vert_coords[:, [[j, k, j], [k, k, k], [i, i, i]]]
    )

    vec2: Float[t.Tensor, "tri 3 3 3"] = (
        tri_vert_coords[:, [[k, k, k], [i, i, i], [j, j, j]]]
        - tri_vert_coords[:, [[j, j, j], [k, k, k], [i, i, i]]]
    )

    sign_mask = t.Tensor(
        [[1, 1, -1], [-1, 1, 1], [1, -1, 1]],
        dtype=vert_coords.dtype,
        device=vert_coords.device,
    ).view(1, 1, 1, 3, 3)
    inner_prod: Float[t.Tensor, "tri 3 3 1 1"] = t.sum(vec1 * vec2, dim=-1).view(
        -1, 3, 3, 1, 1
    ) * t.eye(
        3,
        dtype=vert_coords.dtype,
        device=vert_coords.device,
    ).view(1, 1, 1, 3, 3)
    outer_prod: Float[t.Tensor, "tri 3 3 3 3"] = vec1.view(-1, 3, 3, 3, 1) * vec2.view(
        -1, 3, 3, 1, 3
    )
    outer_diff = -2.0 * outer_prod + outer_prod.transpose(-1, -2)

    triple_cross_grad: Float[t.Tensor, "tri 3 3 3 3"] = (
        inner_prod + outer_diff
    ) * sign_mask

    tri_area_grad: Float[t.Tensor, "tri 3 3"] = _d_tri_areas_d_vert_coords(
        vert_coords, tris
    )

    tri_areas: Float[t.Tensor, "tri 1 1 1 1"] = _tri_areas(vert_coords, tris).view(
        -1, 1, 1, 1, 1
    )

    hess: Float[t.Tensor, "tri 3 3 3 3"] = -tri_area_grad.view(
        -1, 3, 1, 1, 3
    ) * tri_area_grad.view(-1, 1, 3, 1, 3) / tri_areas + triple_cross_grad / (
        4.0 * tri_areas
    )

    hvp = t.einsum("txpcd,tyd->txypc", hess, vec)

    return hvp

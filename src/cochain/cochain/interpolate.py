import torch as t
from jaxtyping import Float, Integer

from ..complex import SimplicialComplex
from ..geometry.tet.tet_geometry import (
    d_tet_signed_vols_d_vert_coords,
    get_tet_signed_vols,
)
from ..geometry.tri.tri_geometry import (
    compute_d_tri_areas_d_vert_coords,
    compute_tri_areas,
)


def _bary_whitney_tri_cochain_0(
    cochain_0: Float[t.Tensor, " vert *ch"],
    tris: Integer[t.LongTensor, "tri 3"],
    bary_coords: Float[t.Tensor, "point 3"],
) -> Float[t.Tensor, "tri point *ch coord=1"]:
    # W_i = λ_i for i = 0, 1, 2.
    basis: Float[t.Tensor, "point vert=3"] = bary_coords

    cochain_0_at_vert_faces: Float[t.Tensor, "tri vert=3 *ch"] = cochain_0[tris]

    form_0: Float[t.Tensor, "tri point *ch"] = t.einsum(
        "pv,tv...->tp...", basis, cochain_0_at_vert_faces
    )

    return form_0.unsqueeze(-1)


def _bary_whitney_tri_cochain_1(
    cochain_1: Float[t.Tensor, " edge *ch"],
    tri_edge_idx: Integer[t.LongTensor, "tri 3"],
    tri_edge_orientations: Float[t.Tensor, "tri 3"],
    bary_coords: Float[t.Tensor, "point 3"],
    bary_coords_grad: Float[t.Tensor, "tri vert=3 coord=3"],
) -> Float[t.Tensor, "tri point *ch coord=3"]:
    bary_coords_grad_shaped: Float[t.Tensor, "tri 1 vert=3 coord=3"] = (
        bary_coords_grad.view(-1, 1, 3, 3)
    )

    bary_coords_shaped: Float[t.Tensor, "1 point vert=3 1"] = bary_coords.view(
        1, -1, 3, 1
    )

    # W_ij = λ_i∇λ_j - λ_j∇λ_i for (i, j) = (0, 1), (0, 2), (1, 2)
    # Note that i, j switch positions for the second term.
    basis: Float[t.Tensor, "tri point edge=3 coord=3"] = (
        bary_coords_shaped[:, :, [0, 0, 1]]
        * bary_coords_grad_shaped[:, :, [1, 2, 2], :]
        - bary_coords_shaped[:, :, [1, 2, 2]]
        * bary_coords_grad_shaped[:, :, [0, 0, 1], :]
    )

    # tri_edge_face_idx contains the index of edges 01, 02, and 12 in the list of
    # canonical edges of the triangular mesh.
    cochain_1_at_edge_faces: Float[t.Tensor, "tri edge=3 *ch"] = cochain_1[tri_edge_idx]

    # If the edges 01, 02, and 12 are not in their canonical orientation, then
    # the corresponding basis form needs a sign correction given by tri_edge_orientations.
    form_1 = t.einsum(
        "tpec,te,te...->tp...c", basis, tri_edge_orientations, cochain_1_at_edge_faces
    )

    return form_1


def _bary_whitney_tri_cochain_2(
    cochain_2: Float[t.Tensor, " tri *ch"],
    tri_orientations: Float[t.Tensor, " tri"],
    bary_coords_grad: Float[t.Tensor, "tri vert=3 coord=3"],
) -> Float[t.Tensor, "tri point=1 *ch coord=3"]:
    # There is only one basis form W_012 = 2(∇λ_1 x ∇λ_2); note that this basis
    # function is a constant of barycentric coordinates, which means that the
    # interpolated 2-forms will be constant on each triangle.
    basis: Float[t.Tensor, "tri coord=3"] = 2.0 * t.cross(
        bary_coords_grad[:, 1, :], bary_coords_grad[:, 2, :], dim=-1
    )

    # If the triangle is not in a canonical orientation, then the basis form
    # needs a sign correction given by tri_orientations.
    form_2: Float[t.Tensor, "tri *ch coord=3"] = t.einsum(
        "tc,t,t...->t...c", basis, tri_orientations, cochain_2
    ).view(-1, 1, 3)

    return form_2.unsqueeze(1)


def _bary_whitney_tet_cochain_0(
    cochain_0: Float[t.Tensor, " vert *ch"],
    tets: Integer[t.LongTensor, "tet 4"],
    bary_coords: Float[t.Tensor, "point 4"],
) -> Float[t.Tensor, "tet point *ch coord=1"]:
    # W_i = λ_i for i = 0, 1, 2, 3.
    basis: Float[t.Tensor, "point vert=4"] = bary_coords

    cochain_0_at_vert_faces: Float[t.Tensor, "tet vert=4 *ch"] = cochain_0[tets]

    form_0: Float[t.Tensor, "tet *ch point"] = t.einsum(
        "pv,tv...->tp...", basis, cochain_0_at_vert_faces
    )

    return form_0.unsqueeze(-1)


def _bary_whitney_tet_cochain_1(
    cochain_1: Float[t.Tensor, " edge *ch"],
    tet_edge_idx: Integer[t.LongTensor, "tet 6"],
    tet_edge_orientations: Float[t.Tensor, "tet 6"],
    bary_coords: Float[t.Tensor, "point 4"],
    bary_coords_grad: Float[t.Tensor, "tet vert=4 coord=3"],
) -> Float[t.Tensor, "tet point *ch coord=3"]:
    bary_coords_grad_shaped: Float[t.Tensor, "tet 1 vert=4 coord=3"] = (
        bary_coords_grad.view(-1, 1, 4, 3)
    )

    bary_coords_shaped: Float[t.Tensor, "1 point vert=4 1"] = bary_coords.view(
        1, -1, 4, 1
    )

    # W_ij = λ_i∇λ_j - λ_j∇λ_i for ij = 01, 02, 12, 13, 23, 03
    # Note that i, j switch positions for the second term.
    basis: Float[t.Tensor, "tet point edge=6 coord=3"] = (
        bary_coords_shaped[:, :, [0, 0, 1, 1, 2, 0]]
        * bary_coords_grad_shaped[:, :, [1, 2, 2, 3, 3, 3]]
        - bary_coords_shaped[:, :, [1, 2, 2, 3, 3, 3]]
        * bary_coords_grad_shaped[:, :, [0, 0, 1, 1, 2, 0]]
    )

    # tet_edge_face_idx contains the index of edges 01, 02, 12, 13, 23, and 03 in
    # the list of canonical edges of the tet mesh.
    cochain_1_at_edge_faces: Float[t.Tensor, "tet edge=6 *ch"] = cochain_1[tet_edge_idx]

    # If the edges are not in their canonical orientation, then the corresponding
    # basis form needs a sign correction given by tet_edge_orientations.
    form_1 = t.einsum(
        "tpec,te,te...->tp...c", basis, tet_edge_orientations, cochain_1_at_edge_faces
    )

    return form_1


def _bary_whitney_tet_cochain_2(
    cochain_2: Float[t.Tensor, " tri *ch"],
    tet_tri_idx: Integer[t.LongTensor, "tet 4"],
    tet_tri_orientations: Float[t.Tensor, "tet 4"],
    bary_coords: Float[t.Tensor, "point 4"],
    bary_coords_grad: Float[t.Tensor, "tet vert=4 coord=3"],
) -> Float[t.Tensor, "tet point *ch coord=3"]:
    bary_coords_grad_shaped: Float[t.Tensor, "tet 1 vert=4 coord=3"] = (
        bary_coords_grad.view(-1, 1, 4, 3)
    )

    bary_coords_shaped: Float[t.Tensor, "1 point vert=4 1"] = bary_coords.view(
        1, -1, 4, 1
    )

    # W_ijk = 2(λ_i ∇λ_jx∇λ_k + λ_j ∇λ_kx∇λ_i + λ_k ∇λ_ix∇ λ_j)
    # for ijk in 123, 032, 013, 021
    perm_i = [1, 0, 0, 0]
    perm_j = [2, 3, 1, 2]
    perm_k = [3, 2, 3, 1]
    basis: Float[t.Tensor, "tet point tri=4 coord=3"] = 2.0 * (
        bary_coords_shaped[:, :, perm_i]
        * t.cross(
            bary_coords_grad_shaped[:, :, perm_j],
            bary_coords_grad_shaped[:, :, perm_k],
            dim=-1,
        )
        + bary_coords_shaped[:, :, perm_j]
        * t.cross(
            bary_coords_grad_shaped[:, :, perm_k],
            bary_coords_grad_shaped[:, :, perm_i],
            dim=-1,
        )
        + bary_coords_shaped[:, :, perm_k]
        * t.cross(
            bary_coords_grad_shaped[:, :, perm_i],
            bary_coords_grad_shaped[:, :, perm_j],
            dim=-1,
        )
    )

    # tet_tri_face_idx contains the index of triangles 123, 032, 013, 021 in
    # the list of canonical triangles of the tet mesh.
    cochain_2_at_tri_faces: Float[t.Tensor, "tet tri=4"] = cochain_2[tet_tri_idx]

    # If the triangles are not in their canonical orientation, then the corresponding
    # basis form needs a sign correction given by tet_edge_orientations.
    form_2 = t.einsum(
        "tpec,te,te...->tp...c", basis, tet_tri_orientations, cochain_2_at_tri_faces
    )

    return form_2


def _bary_whitney_tet_cochain_3(
    cochain_3: Float[t.Tensor, " tet *ch"],
    tet_signed_vols: Float[t.Tensor, " tet"],
    tet_orientations: Float[t.Tensor, " tet"],
) -> Float[t.Tensor, "tet point=1 *ch coord=1"]:
    # There is only one basis form W_0123 = 1/vol; note that this basis
    # function is a constant of barycentric coordinates, which means that the
    # interpolated 3-forms will be constant on each tet.
    basis = 1.0 / tet_signed_vols

    # If the tet is not in a canonical orientation, then the basis form
    # needs a sign correction given by tet_orientations.
    form_3: Float[t.Tensor, " tet *ch"] = t.einsum(
        "t,t,t...->t...", basis, tet_orientations, cochain_3
    )

    return form_3.unsqueeze(1).unsqueeze(-1)


def _bary_whitney_tri(
    k: int,
    k_cochain: Float[t.Tensor, " simp *ch"],
    bary_coords: Float[t.Tensor, "point bary"],
    mesh: SimplicialComplex,
) -> Float[t.Tensor, "tri point *ch coord"]:
    if k in [1, 2]:
        tri_areas = compute_tri_areas(mesh.vert_coords, mesh.tris).view(-1, 1, 1)
        d_tri_areas_d_vert_coords = compute_d_tri_areas_d_vert_coords(
            mesh.vert_coords, mesh.tris
        )
        bary_coords_grad: Float[t.Tensor, "tri 3 3"] = (
            d_tri_areas_d_vert_coords / tri_areas
        )

    match k:
        case 0:
            return _bary_whitney_tri_cochain_0(
                cochain_0=k_cochain, tris=mesh.tris, bary_coords=bary_coords
            )
        case 1:
            return _bary_whitney_tri_cochain_1(
                cochain_1=k_cochain,
                tri_edge_idx=mesh.tri_edge_idx,
                tri_edge_orientations=mesh.tri_edge_orientations,
                bary_coords=bary_coords,
                bary_coords_grad=bary_coords_grad,
            )
        case 2:
            return _bary_whitney_tri_cochain_2(
                cochain_2=k_cochain,
                tri_orientations=mesh.tri_orientations,
                bary_coords_grad=bary_coords_grad,
            )
        case _:
            raise ValueError()


def _bary_whitney_tet(
    k: int,
    k_cochain: Float[t.Tensor, " simp *ch"],
    bary_coords: Float[t.Tensor, "point bary"],
    mesh: SimplicialComplex,
) -> Float[t.Tensor, "tet point *ch coord"]:
    if k in [1, 2, 3]:
        tet_signed_vols = get_tet_signed_vols(mesh.vert_coords, mesh.tets).view(
            -1, 1, 1
        )

    if k in [1, 2]:
        d_signed_vols_d_vert_coords = d_tet_signed_vols_d_vert_coords(
            mesh.vert_coords, mesh.tets
        )
        bary_coords_grad: Float[t.Tensor, "tet 4 3"] = (
            d_signed_vols_d_vert_coords / tet_signed_vols
        )

    match k:
        case 0:
            return _bary_whitney_tet_cochain_0(
                cochain_0=k_cochain, tets=mesh.tets, bary_coords=bary_coords
            )
        case 1:
            return _bary_whitney_tet_cochain_1(
                cochain_1=k_cochain,
                tet_edge_idx=mesh.tet_edge_idx,
                tet_edge_orientations=mesh.tet_edge_orientations,
                bary_coords=bary_coords,
                bary_coords_grad=bary_coords_grad,
            )
        case 2:
            return _bary_whitney_tet_cochain_2(
                cochain_2=k_cochain,
                tet_tri_idx=mesh.tet_tri_idx,
                tet_tri_orientations=mesh.tet_tri_orientations,
                bary_coords=bary_coords,
                bary_coords_grad=bary_coords_grad,
            )
        case 3:
            return _bary_whitney_tet_cochain_3(
                cochain_3=k_cochain,
                tet_signed_vols=tet_signed_vols,
                tet_orientations=mesh.tet_orientations,
            )
        case _:
            raise ValueError()


def barycentric_whitney_map(
    k: int,
    k_cochain: Float[t.Tensor, " simp *ch"],
    bary_coords: Float[t.Tensor, "point bary"],
    mesh: SimplicialComplex,
) -> Float[t.Tensor, "top_sim point *ch coord"]:
    match mesh.dim:
        case 2:
            return _bary_whitney_tri(k, k_cochain, bary_coords, mesh)
        case 3:
            return _bary_whitney_tet(k, k_cochain, bary_coords, mesh)
        case _:
            raise ValueError()

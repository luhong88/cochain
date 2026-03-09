import torch as t
from jaxtyping import Float, Integer


def bary_whitney_tri_cochain_0(
    tris: Integer[t.LongTensor, "tri 3"],
    cochain_0: Float[t.Tensor, " vert"],
    bary_coords: Float[t.Tensor, "point 3"],
) -> Float[t.Tensor, "tri point coord=1"]:
    # W_i = λ_i for i = 0, 1, 2.
    basis: Float[t.Tensor, "point vert=3"] = bary_coords

    cochain_0_at_vert_faces: Float[t.Tensor, "tri vert=3"] = cochain_0[tris]

    form_0: Float[t.Tensor, "tri point"] = t.einsum(
        "pv,tv->tp", basis, cochain_0_at_vert_faces
    ).unsqueeze(-1)

    return form_0


def bary_whitney_tri_cochain_1(
    bary_coords_grad: Float[t.Tensor, "tri vert=3 coord=3"],
    tri_edge_idx: Integer[t.LongTensor, "tri 3"],
    tri_edge_orientations: Float[t.Tensor, "tri 3"],
    cochain_1: Float[t.Tensor, " edge"],
    bary_coords: Float[t.Tensor, "point 3"],
) -> Float[t.Tensor, "tri point 3"]:
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
    cochain_1_at_edge_faces: Float[t.Tensor, "tri edge=3"] = cochain_1[tri_edge_idx]

    # If the edges 01, 02, and 12 are not in their canonical orientation, then
    # the corresponding basis form needs a sign correction given by tri_edge_orientations.
    form_1: Float[t.Tensor, "tri point coord=3"] = t.einsum(
        "tpec,te,te->tpc", basis, tri_edge_orientations, cochain_1_at_edge_faces
    )

    return form_1


def bary_whitney_tri_cochain_2(
    bary_coords_grad: Float[t.Tensor, "tri vert=3 coord=3"],
    tri_orientations: Float[t.Tensor, " tri"],
    cochain_2: Float[t.Tensor, " tri"],
) -> Float[t.Tensor, "tri point=1 3"]:
    # There is only one basis form W_012 = 2(∇λ_1 x ∇λ_2); note that this basis
    # function is a constant of barycentric coordinates, which means that the
    # interpolated 2-forms will be constant on each triangle.
    basis: Float[t.Tensor, "tri coord=3"] = t.cross(
        bary_coords_grad[:, 1, :], bary_coords_grad[:, 2, :], dim=-1
    )

    # If the triangle is not in a canonical orientation, then the basis form
    # needs a sign correction given by tri_orientations.
    form_2: Float[t.Tensor, "tri point=1 coord=3"] = t.einsum(
        "tc,t,t->tc", basis, tri_orientations, cochain_2
    ).view(-1, 1, 3)

    return form_2

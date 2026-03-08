import torch as t
from jaxtyping import Float, Integer

from ..geometry.tet.tet_geometry import d_tet_signed_vols_d_vert_coords
from ..geometry.tri.tri_geometry import compute_d_tri_areas_d_vert_coords


def _bary_whitney_tri_cochain_1(
    tri_edge_idx: Integer[t.LongTensor, "tri 3"],
    tri_edge_orientations: Float[t.Tensor, "tri 3"],
    tri_areas: Float[t.Tensor, " tri 1 1"],
    d_tri_areas_d_vert_coords: Float[t.Tensor, "tri 3 3"],
    cochain_1: Float[t.Tensor, " edge"],
    bary_coords: Float[t.Tensor, "point 3"],
):
    # The gradient of lambda_i(p) wrt p is given by grad_i(area_ijk)/area_ijk, a
    # constant wrt p.
    bary_coords_grad: Float[t.Tensor, "tri 1 vert=3 coord=3"] = (
        d_tri_areas_d_vert_coords / tri_areas
    ).view(-1, 1, 3, 3)

    bary_coords_shaped: Float[t.Tensor, "1 point vert=3 1"] = bary_coords.view(
        1, -1, 3, 1
    )

    # W_ij = λ_i∇λ_j - λ_j∇λ_i for (i, j) = (0, 1), (0, 2), (1, 2)
    # Note that i, j switch positions for the second term.
    basis: Float[t.Tensor, "tri point edge=3 coord=3"] = (
        bary_coords_shaped[:, :, [0, 0, 1]] * bary_coords_grad[:, :, [1, 2, 2], :]
        - bary_coords_shaped[:, :, [1, 2, 2]] * bary_coords_grad[:, :, [0, 0, 1], :]
    )

    # tri_edge_face_idx contains the index of edges 01, 02, and 12 in the list of
    # canonical edges of the triangular mesh.
    cochain_1_at_edge_faces: Float[t.Tensor, "tri edge=3"] = cochain_1[tri_edge_idx]

    # If the edges 01, 02, and 12 are not in their canonical orientation, then
    # the corresponding basis function needs a sign correction given by tri_edge_orientations.
    form_1: Float[t.Tensor, "tri point coord=3"] = t.einsum(
        "tpec,te,te->tpc", basis, tri_edge_orientations, cochain_1_at_edge_faces
    )

    return form_1

__all__ = ["barycentric_whitney_map"]

from typing import Literal

import torch
from einops import einsum, rearrange, repeat
from jaxtyping import Float, Integer
from torch import Tensor

from ...complex import SimplicialMesh
from ...metric.tet import _tet_geometry
from ...metric.tri import _tri_geometry
from ...utils.faces import enumerate_local_faces


def _bary_whitney_tri_cochain_0(
    cochain_0: Float[Tensor, " vert *ch"],
    tris: Integer[Tensor, "tri vert=3"],
    bary_coords: Float[Tensor, "tri pt vert=3"],
) -> Float[Tensor, "tri pt *ch coord=1"]:
    # W_i = λ_i for i = 0, 1, 2.
    basis = bary_coords

    cochain_0_at_vert_faces: Float[Tensor, "tri vert=3 *ch"] = cochain_0[tris]

    form_0 = einsum(
        basis, cochain_0_at_vert_faces, "tri pt vert, tri vert ... -> tri pt ..."
    )

    return rearrange(form_0, "tri pt ... -> tri pt ... 1")


def _bary_whitney_tri_cochain_1(
    cochain_1: Float[Tensor, " edge *ch"],
    tri_edge_idx: Integer[Tensor, "tri edge=3"],
    tri_edge_orientations: Float[Tensor, "tri edge=3"],
    bary_coords: Float[Tensor, "tri pt vert=3"],
    bary_coords_grad: Float[Tensor, "tri vert=3 coord=3"],
) -> Float[Tensor, "tri pt *ch coord=3"]:
    bary_coords_shaped = rearrange(bary_coords, "tri pt vert -> tri pt vert 1")
    bary_coords_grad_shaped = rearrange(
        bary_coords_grad, "tri vert coord -> tri 1 vert coord"
    )

    # W_ij = λ_i∇λ_j - λ_j∇λ_i for (i, j) = (0, 1), (0, 2), (1, 2)
    # Note that i, j switch positions for the second term.
    local_edge_idx = enumerate_local_faces(
        splx_dim=2, face_dim=1, device=bary_coords.device
    )
    basis: Float[Tensor, "tri pt edge=3 coord=3"] = (
        bary_coords_shaped[:, :, local_edge_idx[:, 0]]
        * bary_coords_grad_shaped[:, :, local_edge_idx[:, 1], :]
        - bary_coords_shaped[:, :, local_edge_idx[:, 1]]
        * bary_coords_grad_shaped[:, :, local_edge_idx[:, 0], :]
    )

    # tri_edge_face_idx contains the index of edges 01, 02, and 12 in the list of
    # canonical edges of the triangular mesh.
    cochain_1_at_edge_faces: Float[Tensor, "tri edge=3 *ch"] = cochain_1[tri_edge_idx]

    # If the edges 01, 02, and 12 are not in their canonical orientation, then
    # the corresponding basis form needs a sign correction given by tri_edge_orientations.
    form_1 = einsum(
        basis,
        tri_edge_orientations,
        cochain_1_at_edge_faces,
        "tri pt edge coord, tri edge, tri edge ... -> tri pt ... coord",
    )

    return form_1


def _bary_whitney_tri_cochain_2(
    cochain_2: Float[Tensor, " tri *ch"],
    bary_coords: Float[Tensor, "tri pt vert=3"],
    bary_coords_grad: Float[Tensor, "tri vert=3 coord=3"],
) -> Float[Tensor, "tri pt *ch coord=3"]:
    # There is only one basis form W_012 = 2(∇λ_1 x ∇λ_2); note that this basis
    # function is a constant of barycentric coordinates, which means that the
    # interpolated 2-forms will be constant on each triangle.
    basis: Float[Tensor, "tri coord=3"] = 2.0 * torch.cross(
        bary_coords_grad[:, 1, :], bary_coords_grad[:, 2, :], dim=-1
    )

    # Note that no orientation sign correction is needed here since the top-level
    # simplices are stored as is rather than lex-sorted.
    form_2 = einsum(basis, cochain_2, "tri coord, tri ... -> tri ... coord")

    # Note that the bary_coords argument is only used to determine the number
    # of sampled points.
    form_2_shaped = repeat(
        form_2, "tri ... coord -> tri pt ... coord", pt=bary_coords.size(-2)
    )

    return form_2_shaped


def bary_whitney_tri(
    k: int,
    k_cochain: Float[Tensor, " splx *ch"],
    bary_coords: Float[Tensor, "tri pt vert"],
    mesh: SimplicialMesh,
) -> Float[Tensor, "tri pt *ch coord"]:
    if k in [1, 2]:
        _, bary_coords_grad = _tri_geometry.compute_bc_grads(
            vert_coords=mesh.vert_coords, tris=mesh.tris
        )

    match k:
        case 0:
            return _bary_whitney_tri_cochain_0(
                cochain_0=k_cochain, tris=mesh.tris, bary_coords=bary_coords
            )
        case 1:
            return _bary_whitney_tri_cochain_1(
                cochain_1=k_cochain,
                tri_edge_idx=mesh.edge_faces.idx,
                tri_edge_orientations=mesh.edge_faces.parity,
                bary_coords=bary_coords,
                bary_coords_grad=bary_coords_grad,
            )
        case 2:
            return _bary_whitney_tri_cochain_2(
                cochain_2=k_cochain,
                bary_coords=bary_coords,
                bary_coords_grad=bary_coords_grad,
            )
        case _:
            raise ValueError(
                "'k' must be a nonnegative integer less than or equal to the "
                "dimension of the mesh."
            )

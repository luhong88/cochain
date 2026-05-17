import torch
from einops import einsum, rearrange, repeat
from jaxtyping import Float, Integer
from torch import Tensor

from ...complex import SimplicialMesh
from ...metric.tet import _tet_geometry
from ...utils.faces import enumerate_local_faces


def _bary_whitney_tet_cochain_0(
    cochain_0: Float[Tensor, " global_vert *ch"],
    tets: Integer[Tensor, "tet local_vert=4"],
    bary_coords: Float[Tensor, "tet pt local_vert=4"],
) -> Float[Tensor, "tet pt *ch coord=1"]:
    # W_i = λ_i for i = 0, 1, 2, 3.
    basis = bary_coords

    cochain_0_at_vert_faces: Float[Tensor, "tet vert=4 *ch"] = cochain_0[tets]

    form_0 = einsum(
        basis, cochain_0_at_vert_faces, "tet pt vert, tet vert ... -> tet pt ..."
    )

    return rearrange(form_0, "tet pt ... -> tet pt ... 1")


def _bary_whitney_tet_cochain_1(
    cochain_1: Float[Tensor, " global_edge *ch"],
    tet_edge_idx: Integer[Tensor, "tet local_edge=6"],
    tet_edge_orientations: Float[Tensor, "tet local_edge=6"],
    bary_coords: Float[Tensor, "tet pt local_vert=4"],
    bary_coords_grad: Float[Tensor, "tet local_vert=4 coord=3"],
) -> Float[Tensor, "tet pt *ch coord=3"]:
    bary_coords_shaped = rearrange(bary_coords, "tet pt vert -> tet pt vert 1")
    bary_coords_grad_shaped = rearrange(
        bary_coords_grad, "tet vert coord -> tet 1 vert coord"
    )

    # W_ij = λ_i∇λ_j - λ_j∇λ_i
    # Note that i, j switch positions for the second term.
    local_edge_idx = enumerate_local_faces(
        splx_dim=3, face_dim=1, device=bary_coords.device
    )
    basis: Float[Tensor, "tet pt edge=6 coord=3"] = (
        bary_coords_shaped[:, :, local_edge_idx[:, 0]]
        * bary_coords_grad_shaped[:, :, local_edge_idx[:, 1]]
        - bary_coords_shaped[:, :, local_edge_idx[:, 1]]
        * bary_coords_grad_shaped[:, :, local_edge_idx[:, 0]]
    )

    # tet_edge_face_idx contains the index of 1-faces in the list of canonical
    # edges of the tet mesh.
    cochain_1_at_edge_faces: Float[Tensor, "tet edge=6 *ch"] = cochain_1[tet_edge_idx]

    # If the edges are not in their canonical orientation, then the corresponding
    # basis form needs a sign correction given by tet_edge_orientations.
    form_1 = einsum(
        basis,
        tet_edge_orientations,
        cochain_1_at_edge_faces,
        "tet pt edge coord, tet edge, tet edge ... -> tet pt ... coord",
    )

    return form_1


def _bary_whitney_tet_cochain_2(
    cochain_2: Float[Tensor, " global_tri *ch"],
    tet_tri_idx: Integer[Tensor, "tet local_tri=4"],
    tet_tri_orientations: Float[Tensor, "tet local_tri=4"],
    bary_coords: Float[Tensor, "tet pt local_vert=4"],
    bary_coords_grad: Float[Tensor, "tet local_vert=4 coord=3"],
) -> Float[Tensor, "tet pt *ch coord=3"]:
    bary_coords_shaped = rearrange(bary_coords, "tet pt vert -> tet pt vert 1")
    bary_coords_grad_shaped = rearrange(
        bary_coords_grad, "tet vert coord -> tet 1 vert coord"
    )

    # W_ijk = 2(λ_i ∇λ_jx∇λ_k + λ_j ∇λ_kx∇λ_i + λ_k ∇λ_ix∇ λ_j)
    local_tri_idx = enumerate_local_faces(
        splx_dim=3, face_dim=2, device=bary_coords.device
    )
    perm_i = local_tri_idx[:, 0]
    perm_j = local_tri_idx[:, 1]
    perm_k = local_tri_idx[:, 2]
    basis: Float[Tensor, "tet pt tri=4 coord=3"] = 2.0 * (
        bary_coords_shaped[:, :, perm_i]
        * torch.cross(
            bary_coords_grad_shaped[:, :, perm_j],
            bary_coords_grad_shaped[:, :, perm_k],
            dim=-1,
        )
        + bary_coords_shaped[:, :, perm_j]
        * torch.cross(
            bary_coords_grad_shaped[:, :, perm_k],
            bary_coords_grad_shaped[:, :, perm_i],
            dim=-1,
        )
        + bary_coords_shaped[:, :, perm_k]
        * torch.cross(
            bary_coords_grad_shaped[:, :, perm_i],
            bary_coords_grad_shaped[:, :, perm_j],
            dim=-1,
        )
    )

    # tet_tri_face_idx contains the index of 2-faces in the list of canonical
    # triangles of the tet mesh.
    cochain_2_at_tri_faces: Float[Tensor, "tet tri=4"] = cochain_2[tet_tri_idx]

    # If the triangles are not in their canonical orientation, then the corresponding
    # basis form needs a sign correction given by tet_edge_orientations.
    form_2 = einsum(
        basis,
        tet_tri_orientations,
        cochain_2_at_tri_faces,
        "tet pt tri coord, tet tri, tet tri ... -> tet pt ... coord",
    )

    return form_2


def _bary_whitney_tet_cochain_3(
    cochain_3: Float[Tensor, " tet *ch"],
    tet_signed_vols: Float[Tensor, " tet"],
    bary_coords: Float[Tensor, "tet pt vert=4"],
) -> Float[Tensor, "tet pt *ch coord=1"]:
    # There is only one basis form W_0123 = 1/vol; note that this basis
    # function is a constant of barycentric coordinates, which means that the
    # interpolated 3-forms will be constant on each tet.
    basis = 1.0 / tet_signed_vols

    # Note that no orientation sign correction is needed here since the top-level
    # simplices are stored as is rather than lex-sorted.
    form_3 = einsum(basis, cochain_3, "tet, tet ... -> tet ...")

    # Note that the bary_coords argument is only used here to determine the
    # number of sampled points.
    form_3_shaped = repeat(
        form_3, "tet ... -> tet pt ... coord", pt=bary_coords.size(-2), coord=1
    )

    return form_3_shaped


def bary_whitney_tet(
    k: int,
    k_cochain: Float[Tensor, " splx *ch"],
    bary_coords: Float[Tensor, "tet pt vert"],
    mesh: SimplicialMesh,
) -> Float[Tensor, "tet pt *ch coord"]:
    if k in [1, 2]:
        tet_signed_vols, bary_coords_grad = _tet_geometry.compute_bc_grads(
            mesh.vert_coords, mesh.tets
        )
    elif k == 3:
        tet_signed_vols = _tet_geometry.compute_tet_signed_vols(
            mesh.vert_coords, mesh.tets
        )

    match k:
        case 0:
            return _bary_whitney_tet_cochain_0(
                cochain_0=k_cochain, tets=mesh.tets, bary_coords=bary_coords
            )
        case 1:
            return _bary_whitney_tet_cochain_1(
                cochain_1=k_cochain,
                tet_edge_idx=mesh.edge_faces.idx,
                tet_edge_orientations=mesh.edge_faces.parity,
                bary_coords=bary_coords,
                bary_coords_grad=bary_coords_grad,
            )
        case 2:
            return _bary_whitney_tet_cochain_2(
                cochain_2=k_cochain,
                tet_tri_idx=mesh.tri_faces.idx,
                tet_tri_orientations=mesh.tri_faces.parity,
                bary_coords=bary_coords,
                bary_coords_grad=bary_coords_grad,
            )
        case 3:
            return _bary_whitney_tet_cochain_3(
                cochain_3=k_cochain,
                tet_signed_vols=tet_signed_vols,
                bary_coords=bary_coords,
            )
        case _:
            raise ValueError(
                "'k' must be a nonnegative integer less than or equal to the "
                "dimension of the mesh."
            )

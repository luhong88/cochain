from typing import Any

import torch
from einops import einsum, rearrange, repeat
from jaxtyping import Float
from torch import Tensor

from ...complex import SimplicialMesh
from ...sparse.decoupled_tensor import (
    BaseDecoupledTensor,
    DiagDecoupledTensor,
    SparseDecoupledTensor,
)
from ...sparse.linalg.solvers._sparse_solver import InvSparseOperator
from ...utils.faces import enumerate_local_faces


def element_based_tri_mixed_mass_matrix(
    mesh: SimplicialMesh,
    tri_areas: Float[Tensor, " tri"],
    bary_coords_grad: Float[Tensor, "tri vert=3 coord=3"],
) -> Float[SparseDecoupledTensor, "tri*coord global_edge"]:
    n_edges = mesh.n_edges

    tri_edge_faces = mesh.edge_faces
    tri_edge_idx = tri_edge_faces.idx
    tri_edge_orientations = tri_edge_faces.parity

    local_edge_idx = enumerate_local_faces(
        splx_dim=2, face_dim=1, device=bary_coords_grad.device
    )

    # The integral of W_ij over a triangle is given by
    #
    # int[W_ij*dA] = A*(∇λ_j - ∇λ_i)/(d+1), d = 2
    #
    # Then, the inner product P_k(ij) is given by
    #
    # int[g(e_k, W_ij)*dA] = <e_k, int[W_ij*dA]>
    #
    # which is simply the k-th component of the vector int[W_ij*dA].
    basis_int = (
        einsum(
            tri_areas,
            tri_edge_orientations,
            bary_coords_grad[:, local_edge_idx[:, 1], :]
            - bary_coords_grad[:, local_edge_idx[:, 0], :],
            "tri, tri edge, tri edge coord -> tri coord edge",
        )
        / 3.0
    )

    n_tris = tri_areas.size(0)
    n_coords = 3
    n_edges_per_tri = 3

    # Scatter add the basis_int tensor by the edge dimension to convert the
    # (global triangle, local edge) relation to the (global triangle, global edge)
    # representation.
    coo_idx = torch.stack(
        (
            repeat(
                torch.arange(
                    n_tris * n_coords,
                    device=bary_coords_grad.device,
                    dtype=tri_edge_idx.dtype,
                ),
                "tri_by_coord -> (tri_by_coord edge)",
                edge=n_edges_per_tri,
            ),
            repeat(tri_edge_idx, "tri edge -> (tri coord edge)", coord=n_coords),
        )
    )

    cross_mass = mesh._sparse_coalesced_matrix(
        operator="element_based_tri_mixed_mass_matrix",
        indices=coo_idx,
        values=basis_int.flatten(),
        size=(n_tris * n_coords, n_edges),
    )

    return cross_mass


def element_based_tet_mixed_mass_matrix(
    mesh: SimplicialMesh,
    tet_unsigned_vols: Float[Tensor, " tet"],
    bary_coords_grad: Float[Tensor, "tet vert=4 coord=3"],
) -> Float[SparseDecoupledTensor, "tri*coord global_edge"]:
    n_edges = mesh.n_edges

    tet_edge_faces = mesh.edge_faces
    tet_edge_idx = tet_edge_faces.idx
    tet_edge_orientations = tet_edge_faces.parity

    local_edge_idx = enumerate_local_faces(
        splx_dim=3, face_dim=1, device=bary_coords_grad.device
    )

    # int[W_ij*dA] = A*(∇λ_j - ∇λ_i)/(d+1), d = 3
    basis_int = (
        einsum(
            tet_unsigned_vols,
            tet_edge_orientations,
            bary_coords_grad[:, local_edge_idx[:, 1], :]
            - bary_coords_grad[:, local_edge_idx[:, 0], :],
            "tet, tet edge, tet edge coord -> tet coord edge",
        )
        / 4.0
    )

    n_tets = tet_unsigned_vols.size(0)
    n_coords = 3
    n_edges_per_tet = 6

    # Scatter add the basis_int tensor by the edge dimension to convert the
    # (global tet, local edge) relation to the (global tet, global edge)
    # representation.
    coo_idx = torch.stack(
        (
            repeat(
                torch.arange(
                    n_tets * n_coords,
                    device=bary_coords_grad.device,
                    dtype=tet_edge_idx.dtype,
                ),
                "tet_by_coord -> (tet_by_coord edge)",
                edge=n_edges_per_tet,
            ),
            repeat(tet_edge_idx, "tet edge -> (tet coord edge)", coord=n_coords),
        )
    )

    cross_mass = mesh._sparse_coalesced_matrix(
        operator="element_based_tet_mixed_mass_matrix",
        indices=coo_idx,
        values=basis_int.flatten(),
        size=(n_tets * n_coords, n_edges),
    )

    return cross_mass


def element_based_tri_vector_mass_matrix(
    tri_areas: Float[Tensor, " tri"],
) -> Float[DiagDecoupledTensor, "tri*coord tri*coord"]:
    val = repeat(tri_areas, "tri -> (tri coord)", coord=3)
    return DiagDecoupledTensor(val)


def element_based_tet_vector_mass_matrix(
    tet_unsigned_vols: Float[Tensor, " tet"],
) -> Float[DiagDecoupledTensor, "tri*coord tri*coord"]:
    val = repeat(tet_unsigned_vols, "tet -> (tet coord)", coord=3)
    return DiagDecoupledTensor(val)


def element_based_galerkin_flat(
    vec_field: Float[Tensor, "top_splx coord"],
    mass_1: Float[BaseDecoupledTensor, "edge edge"]
    | Float[InvSparseOperator, "edge edge"],
    mass_mixed: Float[SparseDecoupledTensor, "top_splx*coord edge"],
    solver_kwargs: dict[str, Any],
) -> Float[Tensor, " edge"]:
    # Formula: M_1@η = P.T@v, where M_1 is the edge mass matrix, η is the 1-cochain,
    # P is the mixed mass matrix, and v is the vector field.
    rhs = mass_mixed.T @ vec_field.flatten()

    match mass_1:
        case InvSparseOperator():
            return mass_1(rhs, **solver_kwargs)
        case DiagDecoupledTensor():
            return mass_1.inv @ rhs
        case _:
            return torch.linalg.solve(mass_1.to_dense(), rhs)


def element_based_galerkin_sharp(
    cochain_1: Float[Tensor, " edge"],
    mass_vec: Float[DiagDecoupledTensor, "top_splx*coord top_splx*coord"],
    mass_mixed: Float[SparseDecoupledTensor, "top_splx*coord edge"],
) -> Float[Tensor, "top_splx coord=3"]:
    # Formula: M_V@v = P@η, where M_V is the vector mass matrix, v is the vector field,
    # P is the mixed mass matrix, and η is the 1-cochain.
    return rearrange(
        mass_vec.inv @ mass_mixed @ cochain_1,
        "(top_splx coord) -> top_splx coord",
        coord=3,
    )

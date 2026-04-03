from typing import Literal

import torch as t
from jaxtyping import Float

from ...complex import SimplicialMesh
from ...geometry.tet import tet_hodge_stars, tet_masses
from ...geometry.tet.tet_geometry import (
    compute_tet_signed_vols,
    dompute_d_tet_signed_vols_d_vert_coords,
)
from ...geometry.tri import tri_hodge_stars, tri_masses
from ...geometry.tri.tri_geometry import (
    compute_d_tri_areas_d_vert_coords,
    compute_tri_areas,
)
from ...sparse.decoupled_tensor import (
    BaseDecoupledTensor,
    SparseDecoupledTensor,
)
from . import _galerkin_element, _galerkin_vertex


def mixed_mass(
    mesh: SimplicialMesh, mode: Literal["element", "vertex"]
) -> Float[SparseDecoupledTensor, "splx*coord edge"]:
    match mesh.dim:
        case 2:
            tri_areas = compute_tri_areas(mesh.vert_coords, mesh.tris)

            d_tri_areas_d_vert_coords = compute_d_tri_areas_d_vert_coords(
                mesh.vert_coords, mesh.tris
            )
            bary_coords_grad: Float[t.Tensor, "tri vert=3 coord=3"] = (
                d_tri_areas_d_vert_coords / tri_areas.view(-1, 1, 1)
            )

        case 3:
            tet_signed_vols = compute_tet_signed_vols(mesh.vert_coords, mesh.tets)
            tet_unsigned_vols = t.abs(tet_signed_vols)

            d_signed_vols_d_vert_coords = dompute_d_tet_signed_vols_d_vert_coords(
                mesh.vert_coords, mesh.tets
            )
            bary_coords_grad: Float[t.Tensor, "tet vert=4 coord=3"] = (
                d_signed_vols_d_vert_coords / tet_signed_vols.view(-1, 1, 1)
            )

    match (mode, mesh.dim):
        case ("element", 2):
            return _galerkin_element.element_based_tri_mixed_mass_matrix(
                n_edges=mesh.n_edges,
                tri_edge_idx=mesh.edge_faces.idx,
                tri_edge_orientations=mesh.edge_faces.parity,
                tri_areas=tri_areas,
                bary_coords_grad=bary_coords_grad,
            )

        case ("element", 3):
            return _galerkin_element.element_based_tet_mixed_mass_matrix(
                n_edges=mesh.n_edges,
                tet_edge_idx=mesh.edge_faces.idx,
                tet_edge_orientations=mesh.edge_faces.parity,
                tet_unsigned_vols=tet_unsigned_vols,
                bary_coords_grad=bary_coords_grad,
            )

        case ("vertex", 2):
            return _galerkin_vertex.vertex_based_tri_mixed_mass_matrix(
                n_verts=mesh.n_verts,
                n_edges=mesh.n_edges,
                tris=mesh.tris,
                tri_edge_idx=mesh.edge_faces.idx,
                tri_edge_orientations=mesh.edge_faces.parity,
                tri_areas=tri_areas,
                bary_coords_grad=bary_coords_grad,
            )

        case ("vertex", 3):
            return _galerkin_vertex.vertex_based_tet_mixed_mass_matrix(
                n_verts=mesh.n_verts,
                n_edges=mesh.n_edges,
                tets=mesh.tets,
                tet_edge_idx=mesh.edge_faces.idx,
                tet_edge_orientations=mesh.edge_faces.parity,
                tet_unsigned_vols=tet_unsigned_vols,
                bary_coords_grad=bary_coords_grad,
            )

        case _:
            raise ValueError()


def vector_mass(
    mesh: SimplicialMesh, mode: Literal["element", "vertex"], diagonal: bool = False
) -> Float[BaseDecoupledTensor, "splx*coord splx*coord"]:
    """
    Note that the `diagonal` argument is only relevant for the vertex-based method,
    and the element-based method will always produce diagonal vector mass matrices.
    """
    match (mode, mesh.dim):
        case ("element", 2):
            tri_areas = compute_tri_areas(mesh.vert_coords, mesh.tris)
            return _galerkin_element.element_based_tri_vector_mass_matrix(tri_areas)

        case ("element", 3):
            tet_signed_vols = compute_tet_signed_vols(mesh.vert_coords, mesh.tets)
            tet_unsigned_vols = t.abs(tet_signed_vols)
            return _galerkin_element.element_based_tet_vector_mass_matrix(
                tet_unsigned_vols
            )

        case ("vertex", 2):
            if diagonal:
                star_0 = tri_hodge_stars.star_0(mesh)
                return _galerkin_vertex.vertex_based_diag_vector_mass_matrix(star_0)
            else:
                mass_0 = tri_masses.mass_0(mesh)
                return _galerkin_vertex.vertex_based_consistent_vector_mass_matrix(
                    mass_0
                )

        case ("vertex", 3):
            if diagonal:
                star_0 = tet_hodge_stars.star_0(mesh)
                return _galerkin_vertex.vertex_based_diag_vector_mass_matrix(star_0)
            else:
                mass_0 = tet_masses.mass_0(mesh)
                return _galerkin_vertex.vertex_based_consistent_vector_mass_matrix(
                    mass_0
                )

        case _:
            raise ValueError()


def galerkin_flat(
    vec_field: Float[t.Tensor, "splx coord"],
    mass_1: Float[BaseDecoupledTensor, "edge edge"],
    mass_mixed: Float[SparseDecoupledTensor, "splx*coord edge"],
    mode: Literal["element", "vertex"],
    method: Literal["dense", "solver", "inv_star"],
) -> Float[t.Tensor, " edge"]:
    match mode:
        case "element":
            return _galerkin_element.element_based_galerkin_flat(
                vec_field, mass_1, mass_mixed, method
            )

        case "vertex":
            return _galerkin_vertex.vertex_based_galerkin_flat(
                vec_field, mass_1, mass_mixed, method
            )

        case _:
            raise ValueError()


def galerkin_sharp(
    cochain_1: Float[t.Tensor, " edge"],
    mass_vec: Float[BaseDecoupledTensor, "splx*coord splx*coord"],
    mass_mixed: Float[SparseDecoupledTensor, "splx*coord edge"],
    mode: Literal["element", "vertex"],
    method: Literal["dense", "solver", "inv_star"] | None = None,
) -> Float[t.Tensor, "splx coord=3"]:
    """
    Note that the `method` argument is only relevant for the vertex-based method.
    """
    match mode:
        case "element":
            return _galerkin_element.element_based_galerkin_sharp(
                cochain_1, mass_vec, mass_mixed
            )

        case "vertex":
            if method is None:
                raise ValueError()

            return _galerkin_vertex.vertex_based_galerkin_sharp(
                cochain_1, mass_vec, mass_mixed, method
            )

        case _:
            raise ValueError()

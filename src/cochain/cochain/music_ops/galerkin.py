from typing import Literal

import torch as t
from einops import einsum, rearrange, repeat
from jaxtyping import Float, Integer

from ...complex import SimplicialMesh
from ...geometry.tet import tet_hodge_stars, tet_masses
from ...geometry.tet.tet_geometry import (
    d_tet_signed_vols_d_vert_coords,
    get_tet_signed_vols,
)
from ...geometry.tri import tri_hodge_stars, tri_masses
from ...geometry.tri.tri_geometry import (
    compute_d_tri_areas_d_vert_coords,
    compute_tri_areas,
)
from . import _galerkin_element, _galerkin_vertex


def mixed_mass(mesh: SimplicialMesh, mode: Literal["element", "vertex"]):
    if mesh.dim == 2:
        tri_areas = rearrange(
            compute_tri_areas(mesh.vert_coords, mesh.tris), "tri -> tri 1 1"
        )

        d_tri_areas_d_vert_coords = compute_d_tri_areas_d_vert_coords(
            mesh.vert_coords, mesh.tris
        )
        bary_coords_grad: Float[t.Tensor, "tri vert=3 coord=3"] = (
            d_tri_areas_d_vert_coords / tri_areas
        )

        if mode == "element":
            return _galerkin_element.element_based_tri_mixed_mass_matrix(
                n_edges=mesh.n_edges,
                tri_edge_idx=mesh.edge_faces.idx,
                tri_edge_orientations=mesh.edge_faces.parity,
                tri_areas=tri_areas,
                bary_coords_grad=bary_coords_grad,
            )

        elif mode == "vertex":
            return _galerkin_vertex.vertex_based_tri_mixed_mass_matrix(
                n_verts=mesh.n_verts,
                n_edges=mesh.n_edges,
                tris=mesh.tris,
                tri_edge_idx=mesh.edge_faces.idx,
                tri_edge_orientations=mesh.edge_faces.parity,
                tri_areas=tri_areas,
                bary_coords_grad=bary_coords_grad,
            )

        else:
            raise ValueError()

    elif mesh.dim == 3:
        tet_signed_vols = get_tet_signed_vols(mesh.vert_coords, mesh.tets)
        tet_unsigned_vols = t.abs(tet_signed_vols)

        d_signed_vols_d_vert_coords = d_tet_signed_vols_d_vert_coords(
            mesh.vert_coords, mesh.tets
        )
        bary_coords_grad: Float[t.Tensor, "tet vert=4 coord=3"] = (
            d_signed_vols_d_vert_coords / tet_signed_vols.view(-1, 1, 1)
        )

        if mode == "element":
            return _galerkin_element.element_based_tet_mixed_mass_matrix(
                n_edges=mesh.n_edges,
                tet_edge_idx=mesh.edge_faces.idx,
                tet_edge_orientations=mesh.edge_faces.parity,
                tet_unsigned_vols=tet_unsigned_vols,
                bary_coords_grad=bary_coords_grad,
            )

        elif mode == "vertex":
            return _galerkin_vertex.vertex_based_tet_mixed_mass_matrix(
                n_verts=mesh.n_verts,
                n_edges=mesh.n_edges,
                tets=mesh.tets,
                tet_edge_idx=mesh.edge_faces.idx,
                tet_edge_orientations=mesh.edge_faces.parity,
                tet_unsigned_vols=tet_unsigned_vols,
                bary_coords_grad=bary_coords_grad,
            )

    else:
        raise ValueError()


def vector_mass(
    mesh: SimplicialMesh, mode: Literal["element", "vertex"], diagonal: bool
):
    if mode == "element":
        if mesh.dim == 2:
            tri_areas = compute_tri_areas(mesh.vert_coords, mesh.tris)
            return _galerkin_element.element_based_tri_vector_mass_matrix(tri_areas)

        elif mesh.dim == 3:
            tet_signed_vols = get_tet_signed_vols(mesh.vert_coords, mesh.tets)
            tet_unsigned_vols = t.abs(tet_signed_vols)
            return _galerkin_element.element_based_tet_vector_mass_matrix(
                tet_unsigned_vols
            )

    elif mode == "vertex":
        if mesh.dim == 2:
            if diagonal:
                star_0 = tri_hodge_stars.star_0(mesh)
                return _galerkin_vertex.vertex_based_diag_vector_mass_matrix(star_0)
            else:
                mass_0 = tri_masses.mass_0(mesh)
                return _galerkin_vertex.vertex_based_consistent_vector_mass_matrix(
                    mass_0
                )

        elif mesh.dim == 3:
            if diagonal:
                star_0 = tet_hodge_stars.star_0(mesh)
                return _galerkin_vertex.vertex_based_diag_vector_mass_matrix(star_0)
            else:
                mass_0 = tet_masses.mass_0(mesh)
                return _galerkin_vertex.vertex_based_consistent_vector_mass_matrix(
                    mass_0
                )

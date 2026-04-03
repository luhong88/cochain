from typing import Literal

import torch as t
from jaxtyping import Float

from ...complex import SimplicialMesh
from ...geometry.tet import tet_hodge_stars
from ...geometry.tet.tet_geometry import (
    compute_tet_signed_vols,
    dompute_d_tet_signed_vols_d_vert_coords,
)
from ...geometry.tri import tri_hodge_stars
from ...geometry.tri.tri_geometry import (
    compute_d_tri_areas_d_vert_coords,
    compute_tri_areas,
)
from . import _geometric_element, _geometric_vertex


def geometric_flat(
    vec_field: Float[t.Tensor, "splx coord"],
    mesh: SimplicialMesh,
    mode: Literal["element", "vertex"],
):
    match (mode, mesh.dim):
        case ("element", 2):
            return _geometric_element.element_based_tri_geometric_flat(
                vec_field=vec_field,
                vert_coords=mesh.vert_coords,
                tri_edge_idx=mesh.edge_faces.idx,
                edges=mesh.edges,
            )

        case ("element", 3):
            return _geometric_element.element_based_tet_geometric_flat(
                vec_field=vec_field,
                vert_coords=mesh.vert_coords,
                tet_edge_idx=mesh.edge_faces.idx,
                edges=mesh.edges,
            )

        case ("vertex", dim) if dim in [2, 3]:
            return _geometric_vertex.vertex_based_geometric_flat(
                vec_field=vec_field, vert_coords=mesh.vert_coords, edges=mesh.edges
            )

        case _:
            raise ValueError()


def geometric_sharp(
    cochain_1: Float[t.Tensor, " edge"],
    mesh: SimplicialMesh,
    mode: Literal["element", "vertex"],
    location: Literal["barycenter", "circumcenter"] = "barycenter",
):
    """
    "location" is only relevant for tri meshes and element-based sharp.
    """
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

            d_signed_vols_d_vert_coords = dompute_d_tet_signed_vols_d_vert_coords(
                mesh.vert_coords, mesh.tets
            )
            bary_coords_grad: Float[t.Tensor, "tet vert=4 coord=3"] = (
                d_signed_vols_d_vert_coords / tet_signed_vols.view(-1, 1, 1)
            )

        case _:
            raise ValueError()

    match (mode, mesh.dim):
        case ("element", 2):
            return _geometric_element.element_based_tri_geometric_sharp(
                cochain_1=cochain_1,
                tris=mesh.tris,
                tri_edge_idx=mesh.edge_faces.idx,
                tri_edge_orientations=mesh.edge_faces.parity,
                vert_coords=mesh.vert_coords,
                bary_coords_grad=bary_coords_grad,
                location=location,
            )

        case ("element", 3):
            return _geometric_element.element_based_tet_geometric_sharp(
                cochain_1=cochain_1,
                tet_edge_idx=mesh.edge_faces.idx,
                tet_edge_orientations=mesh.edge_faces.parity,
                bary_coords_grad=bary_coords_grad,
            )

        case ("vertex", 2):
            star_0 = tri_hodge_stars.star_0(mesh)

            return _geometric_vertex.vertex_based_tri_geometric_sharp(
                cochain_1=cochain_1,
                star_0=star_0,
                n_verts=mesh.n_verts,
                tris=mesh.tris,
                tri_edge_idx=mesh.edge_faces.idx,
                tri_edge_orientations=mesh.edge_faces.parity,
                tri_areas=tri_areas,
                vert_coords=mesh.vert_coords,
                bary_coords_grad=bary_coords_grad,
            )

        case ("vertex", 3):
            star_0 = tet_hodge_stars.star_0(mesh)
            tet_unsigned_vols = t.abs(tet_signed_vols)

            return _geometric_vertex.vertex_based_tet_geometric_sharp(
                cochain_1=cochain_1,
                star_0=star_0,
                n_verts=mesh.n_verts,
                tets=mesh.tets,
                tet_edge_idx=mesh.edge_faces.idx,
                tet_edge_orientations=mesh.edge_faces.parity,
                tet_unsigned_vols=tet_unsigned_vols,
                bary_coords_grad=bary_coords_grad,
            )

        case _:
            raise ValueError()

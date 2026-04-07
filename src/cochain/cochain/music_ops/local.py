from typing import Literal

import torch
from jaxtyping import Float
from torch import Tensor

from ...complex import SimplicialMesh
from ...geometry.tet import tet_hodge_stars
from ...geometry.tet.tet_geometry import (
    compute_tet_signed_vols,
    dompute_d_tet_signed_vols_d_vert_coords,
)
from ...geometry.tri import tri_hodge_stars
from ...geometry.tri.tri_geometry import compute_bc_grads
from . import _local_element, _local_vertex


def local_flat(
    vec_field: Float[Tensor, "splx coord"],
    mesh: SimplicialMesh,
    mode: Literal["element", "vertex"],
) -> Float[Tensor, " edge"]:
    """
    Compute the flat of a vector field using a local, interpolation based method.

    If `mode` is "element", the input vector field is assumed to be piecewise
    constant and associated with the top-level simplices of the mesh, and the flat
    is computed at each edge by taking the dot product between the edge vector
    and the mean of the field across all cofaces of the edge. If `mode` is "vertex",
    the input vector field is assumed to be associated with the vertices of the
    mesh and the flat is computed at each edge as the dot product between the edge
    vector and the mean of the field at the two end vertices of the edge.

    Note that this local, interpolation based method is in general faster than
    the Galerkin approach, but the flat and sharp operators do not satisfy the
    exact adjoint relation.
    """
    match (mode, mesh.dim):
        case ("element", 2):
            return _local_element.element_based_tri_local_flat(
                vec_field=vec_field,
                vert_coords=mesh.vert_coords,
                tri_edge_idx=mesh.edge_faces.idx,
                edges=mesh.edges,
            )

        case ("element", 3):
            return _local_element.element_based_tet_local_flat(
                vec_field=vec_field,
                vert_coords=mesh.vert_coords,
                tet_edge_idx=mesh.edge_faces.idx,
                edges=mesh.edges,
            )

        case ("vertex", dim) if dim in [2, 3]:
            return _local_vertex.vertex_based_local_flat(
                vec_field=vec_field, vert_coords=mesh.vert_coords, edges=mesh.edges
            )

        case _:
            raise ValueError()


def local_sharp(
    cochain_1: Float[Tensor, " edge"],
    mesh: SimplicialMesh,
    mode: Literal["element", "vertex"],
    location: Literal["barycenter", "circumcenter"] = "barycenter",
):
    """
    Compute the sharp of a 1-cochain using a local, interpolation based method.

    If `mode` is "element", the sharp is computed by interpolating the 1-cochain
    at either the barycenter (supported for both tri and tet meshes) or the
    circumcenter (supported for only tri meshes) of the top-level simplices. If
    `mode` is "vertex", the function first call the element-based method to interpolate
    the 1-cochain at the barycenters and then compute a per-vertex value by taking
    an area/volume-weighted average of the interpolated 1-form over all top-level
    simplices shsaring the vertex as a face. Therefore, the `location` argument
    is only relevant for tri meshes and element-based sharp.

    Note that this local, interpolation based method is in general faster than
    the Galerkin approach, but the flat and sharp operators do not satisfy the
    exact adjoint relation.
    """
    match mesh.dim:
        case 2:
            tri_areas, bary_coords_grad = compute_bc_grads(
                vert_coords=mesh.vert_coords, tris=mesh.tris
            )

        case 3:
            tet_signed_vols = compute_tet_signed_vols(mesh.vert_coords, mesh.tets)

            d_signed_vols_d_vert_coords = dompute_d_tet_signed_vols_d_vert_coords(
                mesh.vert_coords, mesh.tets
            )
            bary_coords_grad: Float[Tensor, "tet vert=4 coord=3"] = (
                d_signed_vols_d_vert_coords / tet_signed_vols.view(-1, 1, 1)
            )

        case _:
            raise ValueError()

    match (mode, mesh.dim):
        case ("element", 2):
            return _local_element.element_based_tri_local_sharp(
                cochain_1=cochain_1,
                tris=mesh.tris,
                tri_edge_idx=mesh.edge_faces.idx,
                tri_edge_orientations=mesh.edge_faces.parity,
                vert_coords=mesh.vert_coords,
                bary_coords_grad=bary_coords_grad,
                location=location,
            )

        case ("element", 3):
            return _local_element.element_based_tet_local_sharp(
                cochain_1=cochain_1,
                tet_edge_idx=mesh.edge_faces.idx,
                tet_edge_orientations=mesh.edge_faces.parity,
                bary_coords_grad=bary_coords_grad,
            )

        case ("vertex", 2):
            star_0 = tri_hodge_stars.star_0(mesh)

            return _local_vertex.vertex_based_tri_local_sharp(
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
            tet_unsigned_vols = torch.abs(tet_signed_vols)

            return _local_vertex.vertex_based_tet_local_sharp(
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

__all__ = ["local_flat", "local_sharp"]

from typing import Literal

import torch
from jaxtyping import Float
from torch import Tensor

from ...complex import SimplicialMesh
from ...metric.tet import _tet_geometry, tet_hodge_stars
from ...metric.tri import _tri_geometry, tri_hodge_stars
from . import _local_element, _local_vertex


def local_flat(
    vec_field: Float[Tensor, "splx coord=3"],
    mesh: SimplicialMesh,
    mode: Literal["element", "vertex"],
) -> Float[Tensor, " edge"]:
    """
    Compute the flat of a vector field using a local, interpolation based method.

    Parameters
    ----------
    vec_field : [splx, coord]
        The input vector field. The interpretation of the first `splx` dimension
        depends on the `mode` argument. If `mode` is "element", the input vector
        field is assumed to be discrete/piecewise constant and associated with
        the top-level simplices of the mesh. If `mode` is "vertex", the input vector
        field is assumed to be piecewise linear associated with the vertices of
        the mesh.
    mesh
        A simplicial mesh.
    mode
        How the flat is computed. If `mode` is "element`", the flat is computed
        at each edge by taking the dot product between the edge vector and the
        mean of the field across all cofaces of the edge; if `mode` is "vertex",
        the flat is computed at each edge as the dot product between the edge
        vector and the mean of the field at the two end vertices of the edge.

    Returns
    -------
    [edge,]
        A 1-cochain representing the flat of the input vector field.

    Notes
    -----
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
            raise ValueError(
                f"Unknown mode ('{mode}') and mesh dim ({mesh.dim}) combination."
            )


def local_sharp(
    cochain_1: Float[Tensor, " edge"],
    mesh: SimplicialMesh,
    mode: Literal["element", "vertex"],
    location: Literal["barycenter", "circumcenter"] = "barycenter",
) -> Float[Tensor, "splx coord=3"]:
    """
    Compute the sharp of a 1-cochain using a local, interpolation based method.

    Parameters
    ----------
    cochain_1 : [edge,]
        The input 1-cochain.
    mesh
        A simplicial mesh.
    mode
        How the sharp is computed. If `mode` is "element", the sharp is computed
        by interpolating the 1-cochain at either the barycenter or the circumcenter
        of the top-level simplices. If `mode` is "vertex", the function first call
        the element-based method to interpolate the 1-cochain at the barycenters
        and then compute a per-vertex value by taking an area/volume-weighted
        average of the interpolated 1-form over all top-level simplices sharing
        the vertex as a face.
    location
        Whether to perform interpolation at the barycenter (supported for both
        tri and tet meshes) or circumcenter (supported for only tri meshes); this
        is only relevant for tri meshes and element-based sharp.

    Returns
    -------
    [splx, coord]
        The sharp of the input 1-cochain. The first `splx` dimension can refer
        to either the top-level simplices or the vertices of the mesh, depending
        on the `mode` argument.

    Notes
    -----
    Note that this local, interpolation based method is in general faster than
    the Galerkin approach, but the flat and sharp operators do not satisfy the
    exact adjoint relation.
    """
    match mesh.dim:
        case 2:
            tri_areas, bary_coords_grad = _tri_geometry.compute_bc_grads(
                vert_coords=mesh.vert_coords, tris=mesh.tris
            )

        case 3:
            tet_signed_vols, bary_coords_grad = _tet_geometry.compute_bc_grads(
                vert_coords=mesh.vert_coords, tets=mesh.tets
            )

        case _:
            raise ValueError(f"Unsupported mesh dimension {mesh.dim}.")

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
                n_global_verts=mesh.n_verts,
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
                n_global_verts=mesh.n_verts,
                tets=mesh.tets,
                tet_edge_idx=mesh.edge_faces.idx,
                tet_edge_orientations=mesh.edge_faces.parity,
                tet_unsigned_vols=tet_unsigned_vols,
                bary_coords_grad=bary_coords_grad,
            )

        case _:
            raise ValueError(
                f"Unknown mode ('{mode}') and mesh dim ({mesh.dim}) combination."
            )

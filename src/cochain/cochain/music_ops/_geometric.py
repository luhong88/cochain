from typing import Literal

import torch as t
from einops import einsum, rearrange, repeat
from jaxtyping import Float, Integer

from ...sparse.decoupled_tensor import DiagDecoupledTensor
from ...utils.bary_coords import get_k_splx_barycenters, get_tri_circumcenters
from ..interpolate import _bary_whitney_tet_cochain_1, _bary_whitney_tri_cochain_1


def vertex_based_geometric_flat(
    vec_field: Float[t.Tensor, "global_vert coord=3"],
    vert_coords: Float[t.Tensor, "global_vert coord=3"],
    edges: Integer[t.LongTensor, "global_edge local_vert=2"],
) -> Float[t.Tensor, " global_edge"]:
    """
    Compute the flat of a vector field associated with the mesh vertices by taking
    the dot product between the mean of the field at the two vertices of each
    edge with the edge vector. Note that this function works identically for both
    tri and tet meshes.
    """
    edge_verts = vert_coords[edges]
    edge_vecs = edge_verts[:, 1] - edge_verts[:, 0]

    vec_field_mean = vec_field[edges].mean(dim=1)

    dot_prod = einsum(edge_vecs, vec_field_mean, "edge coord, edge coord -> edge")

    return dot_prod


def element_based_tri_geometric_flat(
    vec_field: Float[t.Tensor, "tri coord=3"],
    vert_coords: Float[t.Tensor, "global_vert coord=3"],
    tri_edge_idx: Integer[t.LongTensor, "tri local_edge=3"],
    edges: Integer[t.LongTensor, "global_edge local_vert=2"],
) -> Float[t.Tensor, " global_edge"]:
    """
    Compute the flat of a piecewise constant vector field associated with the tris
    of the mesh by taking the dot product between the mean of the field across all
    cofaces of each edge with the edge vector.
    """
    n_coords = 3
    n_edges_per_tri = 3
    n_edges = edges.size(0)

    # Reshape vec_field and tri_edge_idx in preparation for scatter reduce.
    vec_field_shaped = repeat(
        vec_field, "tri coord -> (tri edge) coord", edge=n_edges_per_tri
    )
    tri_edge_idx_shaped = repeat(
        tri_edge_idx, "tri edge -> (tri edge) coord", coord=n_coords
    )

    vec_field_mean = t.zeros(
        (n_edges, n_coords), dtype=vec_field.dtype, device=vec_field.device
    )

    # self[idx[tri_by_edge][coord]][coord] += src[tri_by_edge][coord]
    vec_field_mean.scatter_reduce_(
        dim=0,
        index=tri_edge_idx_shaped,
        src=vec_field_shaped,
        reduce="mean",
        include_self=False,
    )

    edge_verts = vert_coords[edges]
    edge_vecs = edge_verts[:, 1] - edge_verts[:, 0]

    dot_prod = einsum(edge_vecs, vec_field_mean, "edge coord, edge coord -> edge")

    return dot_prod


def element_based_tet_geometric_flat(
    vec_field: Float[t.Tensor, "tet coord=4"],
    vert_coords: Float[t.Tensor, "global_vert coord=4"],
    tet_edge_idx: Integer[t.LongTensor, "tet local_edge=6"],
    edges: Integer[t.LongTensor, "global_edge local_vert=2"],
) -> Float[t.Tensor, " global_edge"]:
    """
    Compute the flat of a piecewise constant vector field associated with the tets
    of the mesh by taking the dot product between the mean of the field across all
    cofaces of each edge with the edge vector.
    """
    n_coords = 3
    n_edges_per_tet = 6
    n_edges = edges.size(0)

    # Reshape vec_field and tet_edge_idx in preparation for scatter reduce.
    vec_field_shaped = repeat(
        vec_field, "tet coord -> (tet edge) coord", edge=n_edges_per_tet
    )
    tet_edge_idx_shaped = repeat(
        tet_edge_idx, "tet edge -> (tet edge) coord", coord=n_coords
    )

    vec_field_mean = t.zeros(
        (n_edges, n_coords), dtype=vec_field.dtype, device=vec_field.device
    )

    # self[idx[tet_by_edge][coord]][coord] += src[tet_by_edge][coord]
    vec_field_mean.scatter_reduce_(
        dim=0,
        index=tet_edge_idx_shaped,
        src=vec_field_shaped,
        reduce="mean",
        include_self=False,
    )

    edge_verts = vert_coords[edges]
    edge_vecs = edge_verts[:, 1] - edge_verts[:, 0]

    dot_prod = einsum(edge_vecs, vec_field_mean, "edge coord, edge coord -> edge")

    return dot_prod


def element_based_tri_geometric_sharp(
    cochain_1: Float[t.Tensor, " edge"],
    tris: Integer[t.LongTensor, "tri vert=3"],
    tri_edge_idx: Integer[t.LongTensor, "tri edge=3"],
    tri_edge_orientations: Float[t.Tensor, "tri edge=3"],
    vert_coords: Float[t.Tensor, "vert coord=3"],
    bary_coords_grad: Float[t.Tensor, "tri vert=3 coord=3"],
    location: Literal["barycenter", "circumcenter"] = "barycenter",
) -> Float[t.Tensor, "tri coord=3"]:
    """
    Compute the sharp of a 1-cochain by interpolating it at either the barycenter
    or the circumcenter of the triangles.

    Note that, for tri meshes with obtuse triangles, the circumcenters are not
    guaranteed to be inside such triangles, and taking the sharp this way may
    result in extrapolation or numerical instability.
    """
    match location:
        case "barycenter":
            bary_coords = get_k_splx_barycenters(
                k=2, dtype=cochain_1.dtype, device=cochain_1.device
            )
        case "circumcenter":
            bary_coords = get_tri_circumcenters(tris=tris, vert_coords=vert_coords)

    form_1 = _bary_whitney_tri_cochain_1(
        cochain_1=cochain_1,
        tri_edge_idx=tri_edge_idx,
        tri_edge_orientations=tri_edge_orientations,
        bary_coords=bary_coords,
        bary_coords_grad=bary_coords_grad,
    ).squeeze(dim=1)

    return form_1


def element_based_tet_geometric_sharp(
    cochain_1: Float[t.Tensor, " edge"],
    tet_edge_idx: Integer[t.LongTensor, "tet edge=6"],
    tet_edge_orientations: Float[t.Tensor, "tet edge=6"],
    bary_coords_grad: Float[t.Tensor, "tet vert=4 coord=3"],
) -> Float[t.Tensor, "tet coord=3"]:
    """
    Compute the sharp of a 1-cochain by interpolating it at the barycenter of the
    tets.
    """
    bary_coords = get_k_splx_barycenters(
        k=3, dtype=cochain_1.dtype, device=cochain_1.device
    )

    form_1 = _bary_whitney_tet_cochain_1(
        cochain_1=cochain_1,
        tet_edge_idx=tet_edge_idx,
        tet_edge_orientations=tet_edge_orientations,
        bary_coords=bary_coords,
        bary_coords_grad=bary_coords_grad,
    ).squeeze()

    return form_1


def vertex_based_tri_geometric_sharp(
    cochain_1: Float[t.Tensor, " edge"],
    star_0: Float[DiagDecoupledTensor, "vert vert"],
    n_verts: int,
    tris: Integer[t.LongTensor, "tri vert=3"],
    tri_edge_idx: Integer[t.LongTensor, "tri edge=3"],
    tri_edge_orientations: Float[t.Tensor, "tri edge=3"],
    tri_areas: Float[t.Tensor, " tri"],
    vert_coords: Float[t.Tensor, "vert coord=3"],
    bary_coords_grad: Float[t.Tensor, "tri vert=3 coord=3"],
) -> Float[t.Tensor, "vert coord=3"]:
    """
    Compute the vertex-based sharp of a 1-cochain by first using the element-based
    approach and then taking an area-weighted average of the 1-form over all triangles
    sharing a given vertex.

    Note that, since the Whitney 1-form basis functions are discontinuous across
    the tris, the value of the interpolated 1-form from the element-based approach
    is not well-defined at the vertices. Therefore, this area-weighted approach
    does not satisfy the adjoint relation with the vertex-based geometric flat
    operator and can introduce numerical artifacts.
    """
    n_coords = 3

    # Interpolate the 1-cochain at the barycenter, which is equivalent to computing
    # the average of the interpolated 1-form over the tris.
    form_1_on_tris = element_based_tri_geometric_sharp(
        cochain_1=cochain_1,
        tris=tris,
        tri_edge_idx=tri_edge_idx,
        tri_edge_orientations=tri_edge_orientations,
        vert_coords=vert_coords,
        bary_coords_grad=bary_coords_grad,
        location="barycenter",
    )
    area_weighted_form_1 = tri_areas.unsqueeze(-1) * form_1_on_tris

    # Reshape tris and area_weighted_form_1 in preparation for scatter add.
    tris_shaped = repeat(tris, "tri vert -> (tri vert) coord", coord=n_coords)
    area_weighted_form_1_shaped = repeat(
        area_weighted_form_1, "tri coord -> (tri vert) coord", vert=n_verts
    )

    area_weighted_form_1_on_verts = t.zeros(
        (n_verts, n_coords), dtype=cochain_1.dtype, device=cochain_1.device
    )

    # self[idx[tri_by_vert][coord]][coord] += src[tri_by_vert][coord]
    area_weighted_form_1_on_verts.scatter_add_(
        dim=0,
        index=tris_shaped,
        src=area_weighted_form_1_shaped,
    )

    # The barycentric star-0 gives 1/3 of the total area of all triangles sharing
    # each vert as a face.
    area_weights = star_0.val.unsqueeze(-1) * 3.0

    # Use the area-weighted average to assign the 1-form to vertices.
    sharp = area_weighted_form_1_on_verts / area_weights

    return sharp


def vertex_based_tet_geometric_sharp(
    cochain_1: Float[t.Tensor, " edge"],
    star_0: Float[DiagDecoupledTensor, "vert vert"],
    n_verts: int,
    tets: Integer[t.LongTensor, "tet vert=4"],
    tet_edge_idx: Integer[t.LongTensor, "tet local_edge=6"],
    tet_edge_orientations: Float[t.Tensor, "tet local_edge=6"],
    tet_unsigned_vols: Float[t.Tensor, " tet"],
    bary_coords_grad: Float[t.Tensor, "tet vert=4 coord=3"],
) -> Float[t.Tensor, "vert coord=3"]:
    """
    Compute the vertex-based sharp of a 1-cochain by first using the element-based
    approach and then taking an area-weighted average of the 1-form over all tets
    sharing a given vertex.
    """
    n_coords = 3

    # Interpolate the 1-cochain at the barycenter, which is equivalent to computing
    # the average of the interpolated 1-form over the tets.
    form_1_on_tets = element_based_tet_geometric_sharp(
        cochain_1=cochain_1,
        tet_edge_idx=tet_edge_idx,
        tet_edge_orientations=tet_edge_orientations,
        bary_coords_grad=bary_coords_grad,
    )
    vol_weighted_form_1 = tet_unsigned_vols.unsqueeze(-1) * form_1_on_tets

    # Reshape tets and vol_weighted_form_1 in preparation for scatter add.
    tets_shaped = repeat(tets, "tet vert -> (tet vert) coord", coord=n_coords)
    vol_weighted_form_1_shaped = repeat(
        vol_weighted_form_1, "tet coord -> (tet vert) coord", vert=n_verts
    )

    vol_weighted_form_1_on_verts = t.zeros(
        (n_verts, n_coords), dtype=cochain_1.dtype, device=cochain_1.device
    )

    # self[idx[tet_by_vert][coord]][coord] += src[tet_by_vert][coord]
    vol_weighted_form_1_on_verts.scatter_add_(
        dim=0,
        index=tets_shaped,
        src=vol_weighted_form_1_shaped,
    )

    # The barycentric star-0 gives 1/4 of the total volume of all tets sharing
    # each vert as a face.
    vol_weights = star_0.val.unsqueeze(-1) * 4.0

    # Use the volume-weighted average to assign the 1-form to vertices.
    sharp = vol_weighted_form_1_on_verts / vol_weights

    return sharp

import torch
from einops import einsum, repeat
from jaxtyping import Float, Integer
from torch import Tensor

from ...sparse.decoupled_tensor import DiagDecoupledTensor
from ._local_element import (
    element_based_tet_local_sharp,
    element_based_tri_local_sharp,
)


def vertex_based_local_flat(
    vec_field: Float[Tensor, "global_vert coord=3"],
    vert_coords: Float[Tensor, "global_vert coord=3"],
    edges: Integer[Tensor, "global_edge local_vert=2"],
) -> Float[Tensor, " global_edge"]:
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


def vertex_based_tri_local_sharp(
    cochain_1: Float[Tensor, " edge"],
    star_0: Float[DiagDecoupledTensor, "vert vert"],
    n_verts: int,
    tris: Integer[Tensor, "tri vert=3"],
    tri_edge_idx: Integer[Tensor, "tri edge=3"],
    tri_edge_orientations: Float[Tensor, "tri edge=3"],
    tri_areas: Float[Tensor, " tri"],
    vert_coords: Float[Tensor, "vert coord=3"],
    bary_coords_grad: Float[Tensor, "tri vert=3 coord=3"],
) -> Float[Tensor, "vert coord=3"]:
    """
    Compute the vertex-based sharp of a 1-cochain by first using the element-based
    approach and then taking an area-weighted average of the 1-form over all triangles
    sharing a given vertex.

    Note that, since the Whitney 1-form basis functions are discontinuous across
    the tris, the value of the interpolated 1-form from the element-based approach
    is not well-defined at the vertices. Therefore, this area-weighted approach
    does not satisfy the adjoint relation with the vertex-based local flat
    operator and can introduce numerical artifacts.
    """
    n_coords = 3

    # Interpolate the 1-cochain at the barycenter, which is equivalent to computing
    # the average of the interpolated 1-form over the tris.
    form_1_on_tris = element_based_tri_local_sharp(
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

    area_weighted_form_1_on_verts = torch.zeros(
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
    area_weights = star_0.values.unsqueeze(-1) * 3.0

    # Use the area-weighted average to assign the 1-form to vertices.
    sharp = area_weighted_form_1_on_verts / area_weights

    return sharp


def vertex_based_tet_local_sharp(
    cochain_1: Float[Tensor, " edge"],
    star_0: Float[DiagDecoupledTensor, "vert vert"],
    n_verts: int,
    tets: Integer[Tensor, "tet vert=4"],
    tet_edge_idx: Integer[Tensor, "tet local_edge=6"],
    tet_edge_orientations: Float[Tensor, "tet local_edge=6"],
    tet_unsigned_vols: Float[Tensor, " tet"],
    bary_coords_grad: Float[Tensor, "tet vert=4 coord=3"],
) -> Float[Tensor, "vert coord=3"]:
    """
    Compute the vertex-based sharp of a 1-cochain by first using the element-based
    approach and then taking an area-weighted average of the 1-form over all tets
    sharing a given vertex.
    """
    n_coords = 3

    # Interpolate the 1-cochain at the barycenter, which is equivalent to computing
    # the average of the interpolated 1-form over the tets.
    form_1_on_tets = element_based_tet_local_sharp(
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

    vol_weighted_form_1_on_verts = torch.zeros(
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
    vol_weights = star_0.values.unsqueeze(-1) * 4.0

    # Use the volume-weighted average to assign the 1-form to vertices.
    sharp = vol_weighted_form_1_on_verts / vol_weights

    return sharp

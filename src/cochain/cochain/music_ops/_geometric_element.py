from typing import Literal

import torch as t
from einops import einsum, repeat
from jaxtyping import Float, Integer

from ...utils.bary_coords import get_k_splx_barycenters, get_tri_circumcenters
from ..interpolate import _bary_whitney_tet_cochain_1, _bary_whitney_tri_cochain_1


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

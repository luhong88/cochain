from typing import Literal

import torch
from einops import einsum, repeat
from jaxtyping import Float, Integer
from torch import Tensor

from ..interpolate._bary_whitney_tet import _bary_whitney_tet_cochain_1
from ..interpolate._bary_whitney_tri import _bary_whitney_tri_cochain_1


def _get_k_splx_barycenters(
    k: int, dtype: torch.dtype, device: torch.device
) -> Float[Tensor, "k_splx=1 pt=1 vert=k+1"]:
    """Determine the barycentric coordinates of the barycenters of k-simplices."""
    return torch.tensor([[[1.0 / (k + 1.0)] * (k + 1)]], dtype=dtype, device=device)


def _get_tri_circumcenters(
    tris: Integer[Tensor, "tri local_vert=3"],
    vert_coords: Float[Tensor, "global_vert coord=3"],
) -> Float[Tensor, "tri pt=1 local_vert=3"]:
    """Determine the barycentric coordinates of the circumcenters of tris."""
    tri_coords = vert_coords[tris]

    # For each triangle ijk, find the length squared of the edges opposite to
    # the vertices i, j, and k.
    a2 = torch.sum((tri_coords[:, 2] - tri_coords[:, 1]) ** 2, dim=-1)
    b2 = torch.sum((tri_coords[:, 2] - tri_coords[:, 0]) ** 2, dim=-1)
    c2 = torch.sum((tri_coords[:, 1] - tri_coords[:, 0]) ** 2, dim=-1)

    # Use the edge lengths to compute the unnormalized barycentric coordinates
    # for the triangle circumcenters.
    bary_coords_unorm: Float[Tensor, "tri vert=3"] = torch.stack(
        [
            a2 * (b2 + c2 - a2),
            b2 * (c2 + a2 - b2),
            c2 * (a2 + b2 - c2),
        ],
        dim=-1,
    )

    # Normalize by enforcing the λ_0 + λ_1 + λ_2 = 1 condition per tri.
    bary_coords_norm = bary_coords_unorm / bary_coords_unorm.sum(dim=-1, keepdim=True)

    return bary_coords_norm.unsqueeze(1)


def element_based_tri_local_flat(
    vec_field: Float[Tensor, "tri coord=3"],
    vert_coords: Float[Tensor, "global_vert coord=3"],
    tri_edge_idx: Integer[Tensor, "tri local_edge=3"],
    edges: Integer[Tensor, "global_edge local_vert=2"],
) -> Float[Tensor, " global_edge"]:
    """
    Compute the flat of a discrete vector field on a tri mesh.

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

    vec_field_mean = torch.zeros(
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


def element_based_tet_local_flat(
    vec_field: Float[Tensor, "tet coord=4"],
    vert_coords: Float[Tensor, "global_vert coord=4"],
    tet_edge_idx: Integer[Tensor, "tet local_edge=6"],
    edges: Integer[Tensor, "global_edge local_vert=2"],
) -> Float[Tensor, " global_edge"]:
    """
    Compute the flat of a discrete vector field on a tet mesh.

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

    vec_field_mean = torch.zeros(
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


def element_based_tri_local_sharp(
    cochain_1: Float[Tensor, " global_edge"],
    tris: Integer[Tensor, "tri local_vert=3"],
    tri_edge_idx: Integer[Tensor, "tri local_edge=3"],
    tri_edge_orientations: Float[Tensor, "tri local_edge=3"],
    vert_coords: Float[Tensor, "vert local_coord=3"],
    bary_coords_grad: Float[Tensor, "tri local_vert=3 coord=3"],
    location: Literal["barycenter", "circumcenter"],
) -> Float[Tensor, "tri coord=3"]:
    """
    Compute the sharp of a 1-cochain on a tri mesh.

    Compute the sharp of a 1-cochain by interpolating it at either the barycenter
    or the circumcenter of the triangles. Note that, for tri meshes with obtuse
    triangles, the circumcenters are not guaranteed to be inside such triangles,
    and taking the sharp this way may result in extrapolation or numerical instability.
    """
    match location:
        case "barycenter":
            bary_coords = _get_k_splx_barycenters(
                k=2, dtype=cochain_1.dtype, device=cochain_1.device
            )

        case "circumcenter":
            bary_coords = _get_tri_circumcenters(tris=tris, vert_coords=vert_coords)

        case _:
            raise ValueError(f"Unknown interpolation location '{location}'.")

    form_1 = _bary_whitney_tri_cochain_1(
        cochain_1=cochain_1,
        tri_edge_idx=tri_edge_idx,
        tri_edge_orientations=tri_edge_orientations,
        bary_coords=bary_coords,
        bary_coords_grad=bary_coords_grad,
    ).squeeze(dim=1)

    return form_1


def element_based_tet_local_sharp(
    cochain_1: Float[Tensor, " global_edge"],
    tet_edge_idx: Integer[Tensor, "tet local_edge=6"],
    tet_edge_orientations: Float[Tensor, "tet local_edge=6"],
    bary_coords_grad: Float[Tensor, "tet local_vert=4 coord=3"],
) -> Float[Tensor, "tet coord=3"]:
    """
    Compute the sharp of a 1-cochain on a tet mesh.

    Compute the sharp of a 1-cochain by interpolating it at the barycenter of the
    tets. Note that, unlike the tet mesh, interpolation at the circumcenter is
    not supported.
    """
    bary_coords = _get_k_splx_barycenters(
        k=3, dtype=cochain_1.dtype, device=cochain_1.device
    )

    form_1 = _bary_whitney_tet_cochain_1(
        cochain_1=cochain_1,
        tet_edge_idx=tet_edge_idx,
        tet_edge_orientations=tet_edge_orientations,
        bary_coords=bary_coords,
        bary_coords_grad=bary_coords_grad,
    ).squeeze(dim=1)

    return form_1

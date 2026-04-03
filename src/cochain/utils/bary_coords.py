import torch as t
from jaxtyping import Float, Integer


def get_k_splx_barycenters(
    k: int, dtype: t.dtype, device: t.device
) -> Float[t.Tensor, "k_splx=1 pt=1 vert=k+1"]:
    return t.tensor([[[1.0 / (k + 1.0)] * (k + 1)]], dtype=dtype, device=device)


# TODO: consider backward numerical safety
def get_tri_circumcenters(
    tris: Integer[t.LongTensor, "tri vert=3"],
    vert_coords: Float[t.Tensor, "vert coord=3"],
) -> Float[t.Tensor, "tri pt=1 vert=3"]:
    tri_coords = vert_coords[tris]

    # For each triangle ijk, find the length squared of the edges opposite to
    # the vertices i, j, and k.
    a2 = t.sum((tri_coords[:, 2] - tri_coords[:, 1]) ** 2, dim=-1)
    b2 = t.sum((tri_coords[:, 2] - tri_coords[:, 0]) ** 2, dim=-1)
    c2 = t.sum((tri_coords[:, 1] - tri_coords[:, 0]) ** 2, dim=-1)

    # Use the edge lengths to compute the unnormalized barycentric coordinates
    # for the triangle circumcenters.
    bary_coords_unorm: Float[t.Tensor, "tri vert=3"] = t.stack(
        [a2 * (b2 + c2 - a2), b2 * (c2 + a2 - b2), c2 * (a2 + b2 - c2)], dim=-1
    )

    # Normalize by enforcing the λ_0 + λ_1 + λ_2 = 1 condition per tri.
    bary_coords_norm = bary_coords_unorm / bary_coords_unorm.sum(dim=-1, keepdim=True)

    return bary_coords_norm.unsqueeze(1)

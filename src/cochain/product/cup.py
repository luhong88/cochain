from typing import Literal

import torch as t
from jaxtyping import Float

from ..complex import SimplicialComplex


def cup_product(
    k_cochain: Float[t.Tensor, " k-simp *ch"],
    l_cochain: Float[t.Tensor, " l-simp *ch"],
    k: int,
    l: int,
    mesh: SimplicialComplex,
    pairing: Literal["scalar", "dot", "cross", "outer"],
) -> Float[t.Tensor, " (k+l)-simp *ch"]:
    dtype = mesh.edges.dtype
    device = mesh.edges.device

    verts = t.arange(mesh.n_verts, dtype=dtype, device=device)

    simp_dim_map: dict[int, t.Tensor] = {
        dim: simp for dim, simp in enumerate([verts, mesh.edges, mesh.tris, mesh.tets])
    }
    n_simp_map: dict[int, int] = {
        dim: n_simp
        for dim, n_simp in enumerate(
            [mesh.n_verts, mesh.n_edges, mesh.n_tris, mesh.n_tets]
        )
    }

    if k_cochain.size(0) != n_simp_map[k]:
        raise ValueError()
    if l_cochain.size(0) != n_simp_map[l]:
        raise ValueError()

    # Sort the (k+l)-simplices by vert indices; for an simplicial n-complex, this
    # is unnecessary if k + l < n, because all but the top-level simplices are
    # sorted by vert indices by default.
    kl_simp: Float[t.Tensor, " (k+l)-simp *ch"] = (
        simp_dim_map[k + l].sort(dim=-1).values
    )

    n_k_simp = n_simp_map[k]
    k_simp = simp_dim_map[k]
    k_front_face = kl_simp[:, : k + 1]

    n_l_simp = n_simp_map[l]
    l_simp = simp_dim_map[l]
    k_back_face = kl_simp[:, k:]

    # Find the index of the k-front faces in the list of k-simplices and the index
    # of the k-back faces in the list of l-simplices.
    k_simp_packed = t.zeros(n_k_simp, dtype=dtype, device=device)
    for idx in range(k + 1):
        k_simp_packed.add_(k_simp[:, idx] * (mesh.n_verts ** (k - idx)))

    k_front_face_packed = t.zeros(k_front_face.size(0), dtype=dtype, device=device)
    for idx in range(l + 1):
        k_front_face_packed.add_(k_front_face[:, idx] * (mesh.n_verts ** (k - idx)))

    k_front_face_idx = t.searchsorted(k_simp_packed, k_front_face_packed)
    k_cochain_at_front_face = k_cochain[k_front_face_idx]

    l_simp_packed = t.zeros(n_l_simp, dtype=dtype, device=device)
    for idx in range(k + 1):
        l_simp_packed.add_(l_simp[:, idx] * (mesh.n_verts ** (l - idx)))

    k_back_face_packed = t.zeros(k_back_face.size(0), dtype=dtype, device=device)
    for idx in range(l + 1):
        k_back_face_packed.add_(k_back_face[:, idx] * (mesh.n_verts ** (k - idx)))

    k_back_face_idx = t.searchsorted(l_simp_packed, k_back_face_packed)
    l_cochain_at_back_face = l_cochain[k_back_face_idx]

    match pairing:
        case "scalar":
            prod = k_cochain_at_front_face * l_cochain_at_back_face
        case "dot":
            prod = t.sum(k_cochain_at_front_face * l_cochain_at_back_face, dim=-1)
        case "cross":
            prod = t.cross(k_cochain_at_front_face, l_cochain_at_back_face, dim=-1)
        case "outer":
            prod = t.einsum(
                "nk,nl->nkl", k_cochain_at_front_face, l_cochain_at_back_face
            )
        case _:
            raise ValueError()

    return prod

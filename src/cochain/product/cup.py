from typing import Literal

import torch as t
from jaxtyping import Float, Integer

from ..complex import SimplicialComplex
from ._perm_lut import perm_idx_lut


# TODO: cache front/back face index mapping
def cup_product(
    k_cochain: Float[t.Tensor, " k_simp ch"],
    l_cochain: Float[t.Tensor, " l_simp ch"],
    k: int,
    l: int,
    mesh: SimplicialComplex,
    pairing: Literal["scalar", "dot", "cross", "outer"],
) -> Float[t.Tensor, " (k+l)_simp ch"]:
    dtype = mesh.edges.dtype
    pack_dtype = t.int64
    device = mesh.edges.device

    verts = t.arange(mesh.n_verts, dtype=dtype, device=device).view(-1, 1)

    simp_map: dict[int, t.Tensor] = {
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
    kl_simp: Integer[t.Tensor, " (k+l)_simp k+l+1"] = (
        simp_map[k + l].sort(dim=-1).values
    )

    n_k_simp = n_simp_map[k]
    k_simp = simp_map[k]
    k_front_face = kl_simp[:, : k + 1]

    n_l_simp = n_simp_map[l]
    l_simp = simp_map[l]
    k_back_face = kl_simp[:, k:]

    # Find the index of the k-front faces in the list of k-simplices and the index
    # of the k-back faces in the list of l-simplices.
    k_simp_packed = t.zeros(n_k_simp, dtype=pack_dtype, device=device)
    for idx in range(k + 1):
        k_simp_packed.add_(k_simp[:, idx] * (mesh.n_verts ** (k - idx)))

    k_front_face_packed = t.zeros(k_front_face.size(0), dtype=pack_dtype, device=device)
    for idx in range(k + 1):
        k_front_face_packed.add_(k_front_face[:, idx] * (mesh.n_verts ** (k - idx)))

    k_front_face_idx = t.searchsorted(k_simp_packed, k_front_face_packed)
    k_cochain_at_front_face = k_cochain[k_front_face_idx]

    l_simp_packed = t.zeros(n_l_simp, dtype=pack_dtype, device=device)
    for idx in range(l + 1):
        l_simp_packed.add_(l_simp[:, idx] * (mesh.n_verts ** (l - idx)))

    k_back_face_packed = t.zeros(k_back_face.size(0), dtype=pack_dtype, device=device)
    for idx in range(l + 1):
        k_back_face_packed.add_(k_back_face[:, idx] * (mesh.n_verts ** (l - idx)))

    k_back_face_idx = t.searchsorted(l_simp_packed, k_back_face_packed)
    l_cochain_at_back_face = l_cochain[k_back_face_idx]

    match pairing:
        case "scalar":
            prod = k_cochain_at_front_face * l_cochain_at_back_face
        case "dot":
            prod = t.sum(
                k_cochain_at_front_face * l_cochain_at_back_face, dim=-1, keepdim=True
            )
        case "cross":
            prod = t.cross(k_cochain_at_front_face, l_cochain_at_back_face, dim=-1)
        case "outer":
            prod = t.einsum(
                "nk,nl->nkl", k_cochain_at_front_face, l_cochain_at_back_face
            )
        case _:
            raise ValueError()

    return prod


def antisymmetric_cup_product(
    k_cochain: Float[t.Tensor, " k_simp ch"],
    l_cochain: Float[t.Tensor, " l_simp ch"],
    k: int,
    l: int,
    mesh: SimplicialComplex,
    pairing: Literal["scalar", "dot", "cross", "outer"],
):
    dtype = mesh.edges.dtype
    pack_dtype = t.int64
    device = mesh.edges.device

    verts = t.arange(mesh.n_verts, dtype=dtype, device=device).view(-1, 1)

    simp_map: dict[int, t.Tensor] = {
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
    n_kl_simp = n_simp_map[k + l]
    kl_simp: Integer[t.Tensor, " (k+l)_simp k+l+1"] = (
        simp_map[k + l].sort(dim=-1).values
    )

    perm = perm_idx_lut[(k, l)].to(device)

    n_k_simp = n_simp_map[k]
    k_simp = simp_map[k]
    uf_face: Integer[t.Tensor, " (k+l)_simp uf_face k+1"] = kl_simp[
        :, perm.unique_front
    ]
    uf_face_flat = uf_face.view(-1, k + 1)

    n_l_simp = n_simp_map[l]
    l_simp = simp_map[l]
    ub_face: Integer[t.Tensor, " (k+l)_simp ub_face l+1"] = kl_simp[:, perm.unique_back]
    ub_face_flat = ub_face.view(-1, l + 1)

    # Find the index of the k-front faces in the list of k-simplices and the index
    # of the k-back faces in the list of l-simplices.
    k_simp_packed = t.zeros(n_k_simp, dtype=pack_dtype, device=device)
    for idx in range(k + 1):
        k_simp_packed.add_(k_simp[:, idx] * (mesh.n_verts ** (k - idx)))

    uf_face_flat_packed = t.zeros(uf_face_flat.size(0), dtype=pack_dtype, device=device)
    for idx in range(k + 1):
        uf_face_flat_packed.add_(uf_face_flat[:, idx] * (mesh.n_verts ** (k - idx)))

    uf_face_idx: Integer[t.Tensor, " (k+l)_simp uf_face"] = t.searchsorted(
        k_simp_packed, uf_face_flat_packed
    ).view(n_kl_simp, -1)
    f_face_idx = uf_face_idx[:, perm.front_idx]

    k_cochain_at_f_face: Float[t.Tensor, "(k+l)_simp f_face ch"] = k_cochain[f_face_idx]

    l_simp_packed = t.zeros(n_l_simp, dtype=pack_dtype, device=device)
    for idx in range(l + 1):
        l_simp_packed.add_(l_simp[:, idx] * (mesh.n_verts ** (l - idx)))

    ub_face_flat_packed = t.zeros(ub_face_flat.size(0), dtype=pack_dtype, device=device)
    for idx in range(l + 1):
        ub_face_flat_packed.add_(ub_face_flat[:, idx] * (mesh.n_verts ** (l - idx)))

    ub_face_idx: Integer[t.Tensor, " (k+l)_simp ub_face"] = t.searchsorted(
        l_simp_packed, ub_face_flat_packed
    ).view(n_kl_simp, -1)
    b_face_idx = ub_face_idx[:, perm.back_idx]

    l_cochain_at_b_face: Float[t.Tensor, "(k+l)_simp b_face ch"] = l_cochain[b_face_idx]

    match pairing:
        case "scalar":
            prod = t.mean(perm.sign * k_cochain_at_f_face * l_cochain_at_b_face, dim=1)
        case "dot":
            prod = t.mean(
                t.sum(
                    perm.sign * k_cochain_at_f_face * l_cochain_at_b_face,
                    dim=-1,
                    keepdim=True,
                ),
                dim=1,
            )
        case "cross":
            prod = t.mean(
                perm.sign * t.cross(k_cochain_at_f_face, l_cochain_at_b_face, dim=-1),
                dim=1,
            )
        case "outer":
            prod = t.einsum(
                "nf,nfk,nfl->nkl",
                perm.sign.flatten(start_dim=1),
                k_cochain_at_f_face,
                l_cochain_at_b_face,
            ) / perm.sign.size(1)
        case _:
            raise ValueError()

    return prod

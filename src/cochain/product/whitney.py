import itertools
import math
from typing import Literal

import torch as t
from jaxtyping import Float, Integer

from ..complex import SimplicialComplex
from ..geometry.tet import tet_geometry
from ..geometry.tri import tri_geometry
from ..sparse.operators import SparseOperator
from ..utils.perm_parity import compute_lex_rel_orient


def enumerate_faces(simp_dim: int, face_dim: int) -> Integer[t.Tensor, "face vert"]:
    if face_dim > simp_dim:
        raise ValueError()

    return t.tensor(
        list(itertools.combinations(list(range(simp_dim + 1)), face_dim + 1))
    )


def compute_face_sign(
    mesh: SimplicialComplex, simp_dim: int, face_dim: int
) -> Float[t.Tensor, "simp face 1"]:
    simp_map = {
        dim: simp
        for dim, simp in enumerate([mesh.verts, mesh.edges, mesh.tris, mesh.tets])
    }

    face_idx = enumerate_faces(simp_dim, face_dim)
    all_faces = simp_map[simp_dim][:, face_idx]

    signs = compute_lex_rel_orient(all_faces.flatten(end_dim=-2))
    signs_shaped = signs.view(simp_map[simp_dim].size(0), face_idx.size(0), 1)

    return signs_shaped


def compute_whitney_router(
    simp_dim: int, form_deg: int, device: t.device, dtype: t.dtype = t.float
) -> Float[t.Tensor, "face lambda *d_lambda"]:
    """
    Compute the coefficients required to construct the Whitney forms from the
    λ's and the dλ's.
    """
    faces = enumerate_faces(simp_dim, form_deg)

    router_shape = (faces.size(0),) + (simp_dim + 1,) * (form_deg + 1)
    router = t.zeros(router_shape, dtype=dtype, device=device)

    for simp_idx, simp in enumerate(faces):
        perms = t.tensor(list(itertools.permutations(simp)))
        signs = compute_lex_rel_orient(perms).to(dtype=dtype, device=device)
        router[simp_idx][perms.T.unbind(0)] = signs

    return router


def compute_moments(
    order: int, simp_dim: int, device: t.device, dtype: t.dtype = t.float
) -> t.Tensor:
    """
    For an n-simplex with unit area/volume and n + 1 barycentric coordinate functions
    λ_i, use the magic formula

    int[prod_i[λ_i^m_i]dV] = (n! * prod_i[m_i!]) / (n + sum_i[m_i])!

    to compute the moment tensors.
    """
    verts = list(range(simp_dim + 1))
    moments = t.zeros((len(verts),) * order)

    for lambdas in itertools.product(verts, repeat=order):
        exponents = [lambdas.count(i) for i in verts]
        numerator = math.factorial(simp_dim) * math.prod(
            [math.factorial(i) for i in exponents]
        )
        denominator = math.factorial(simp_dim + sum(exponents))
        moments[lambdas] = numerator / denominator

    return moments.to(device=device, dtype=dtype)


def inv_gram_det(g: Float[t.Tensor, "simp vert vert"], form_deg=int):
    match form_deg:
        case 0:
            grad_wedge_dot = t.ones(g.shape[0], device=g.device)

        case 1:
            grad_wedge_dot = g

        case 2:
            n_simp = g.size(0)
            n_vert = g.size(-1)
            grad_wedge_dot = t.zeros(
                n_simp,
                n_vert,
                n_vert,
                n_vert,
                n_vert,
                dtype=g.dtype,
                device=g.device,
            )

            # det_ijkl = g_ik*g_jl - g_il*g_jk

            grad_wedge_dot.add_(t.einsum("tik,tjl->tijkl", g, g))
            grad_wedge_dot.sub_(t.einsum("til,tjk->tijkl", g, g))

        # TODO: memory optimization
        case 3:
            n_simp = g.size(0)
            n_vert = g.size(-1)
            grad_wedge_dot = t.zeros(
                n_simp,
                n_vert,
                n_vert,
                n_vert,
                n_vert,
                n_vert,
                n_vert,
                dtype=g.dtype,
                device=g.device,
            )

            # Calculate all permutations of the product g_ia*g_jb*g_kc
            # Det = (123) - (132) - (213) + (231) + (312) - (321)
            # | g_ir g_is g_it |
            # | g_jr g_js g_jt |
            # | g_kr g_ks g_kt |

            grad_wedge_dot.add_(t.einsum("tia, tjb, tkc -> tijkabc", g, g, g))
            grad_wedge_dot.sub_(t.einsum("tia, tjc, tkb -> tijkabc", g, g, g))
            grad_wedge_dot.sub_(t.einsum("tib, tja, tkc -> tijkabc", g, g, g))
            grad_wedge_dot.add_(t.einsum("tib, tjc, tka -> tijkabc", g, g, g))
            grad_wedge_dot.add_(t.einsum("tic, tja, tkb -> tijkabc", g, g, g))
            grad_wedge_dot.sub_(t.einsum("tic, tjb, tka -> tijkabc", g, g, g))

    return grad_wedge_dot


def _einsum_str(k: int, l: int) -> str:
    match (k, l):
        # 0-simplex
        case (0, 0):
            # 0-form_router: (u=vert, a=λ)
            # 0-form_router: (v=vert, b=λ)
            # 0-form_router: (w=vert, c=λ)
            # size: (t=simp,)
            # moments: (a=λ, b=λ, c=λ)
            # wedge_dot: (t=simp,)
            # output: (t=simp, u=λ, v=λ, w=dλ)
            einsum = "ua, vb, wc, t, abc, t -> tuvw"

        # 1-simplex
        case (0, 1):
            # 0-form_router: (u=vert, a=λ)
            # 1-form_router: (v=edge, b=λ, y=dλ)
            # 1-form_router: (w=edge, c=λ, p=dλ)
            # size: (t=simp,)
            # moments: (a=λ, b=λ, c=λ)
            # wedge_dot: (t=simp, y=dλ, p=dλ)
            # output: (t=simp, u=λ, v=λ, w=dλ)
            einsum = "ua, vby, wcp, t, abc, typ -> tuvw"

        case (1, 0):
            # 0-form_router: (u=vert, a=λ, x=dλ)
            # 1-form_router: (v=edge, b=λ)
            # 1-form_router: (w=edge, c=λ, p=dλ)
            # size: (t=simp,)
            # moments: (a=λ, b=λ, c=λ)
            # wedge_dot: (t=simp, x=dλ, p=dλ)
            # output: (t=simp, u=λ, v=λ, w=dλ)
            einsum = "uax, vb, wcp, t, abc, txp -> tuvw"

        # 2-simplex
        case (0, 2):
            # 0-form_router: (u=vert, a=λ)
            # 2-form_router: (v=tri, b=λ, x=dλ, y= dλ)
            # 2-form_router: (w=tri, c=λ, p=dλ, q=dλ)
            # size: (t=simp,)
            # moments: (a=λ, b=λ, c=λ)
            # wedge_dot: (t=simp, x=dλ, y=dλ, p=dλ, q=dλ)
            # output: (t=simp, u=λ, v=λ, w=dλ)
            einsum = "ua, vbxy, wcpq, t, abc, txypq -> tuvw"

        case (1, 1):
            # 1-form_router: (u=edge, a=λ, x=dλ)
            # 1-form_router: (v=edge, b=λ, y=dλ)
            # 2-form_router: (w=tri, c=λ, p=dλ, q=dλ)
            # size: (t=simp,)
            # moments: (a=λ, b=λ, c=λ)
            # wedge_dot: (t=simp, x=dλ, y=dλ, p=dλ, q=dλ)
            # output: (t=simp, u=λ, v=λ, w=dλ)
            einsum = "uax, vby, wcpq, t, abc, txypq -> tuvw"

        case (2, 0):
            # 2-form_router: (u=tri, a=λ, x=dλ, y=dλ)
            # 0-form_router: (v=vert, b=λ)
            # 2-form_router: (w=tri, c=λ, p=dλ, q=dλ)
            # size: (t=simp,)
            # moments: (a=λ, b=λ, c=λ)
            # wedge_dot: (t=simp, x=dλ, y=dλ, p=dλ, q=dλ)
            # output: (t=simp, u=λ, v=λ, w=dλ)
            einsum = "uaxy, vb, wcpq, t, abc, txypq -> tuvw"

        # 3-simplex
        case (0, 3):
            # 0-form_router: (u=vert, a=λ)
            # 3-form_router: (v=tet, b=λ, x=dλ, y=dλ, z= dλ)
            # 3-form_router: (w=tet, c=λ, p=dλ, q=dλ, r=dλ)
            # size: (t=simp,)
            # moments: (a=λ, b=λ, c=λ)
            # wedge_dot: (t=simp, x=dλ, y=dλ, z=dλ, p=dλ, q=dλ, r=dλ)
            # output: (t=simp, u=λ, v=λ, w=dλ)
            einsum = "ua, vbxyz, wcpqr, t, abc, txyzpqr -> tuvw"

        case (1, 2):
            # 1-form_router: (u=edge, a=λ, x=dλ)
            # 2-form_router: (v=tri, b=λ, y=dλ, z= dλ)
            # 3-form_router: (w=tet, c=λ, p=dλ, q=dλ, r=dλ)
            # size: (t=simp,)
            # moments: (a=λ, b=λ, c=λ)
            # wedge_dot: (t=simp, x=dλ, y=dλ, z=dλ, p=dλ, q=dλ, r=dλ)
            # output: (t=simp, u=λ, v=λ, w=dλ)
            einsum = "uax, vbyz, wcpqr, t, abc, txyzpqr -> tuvw"

        case (2, 1):
            # 2-form_router: (u=tri, a=λ, x=dλ, y=dλ)
            # 1-form_router: (v=edge, b=λ, z= dλ)
            # 3-form_router: (w=tet, c=λ, p=dλ, q=dλ, r=dλ)
            # size: (t=simp,)
            # moments: (a=λ, b=λ, c=λ)
            # wedge_dot: (t=simp, x=dλ, y=dλ, z=dλ, p=dλ, q=dλ, r=dλ)
            # output: (t=simp, u=λ, v=λ, w=dλ)
            einsum = "uaxy, vbz, wcpqr, t, abc, txyzpqr -> tuvw"

        case (3, 0):
            # 3-form_router: (u=tet, a=λ, x=dλ, y=dλ, z= dλ)
            # 0-form_router: (v=vert, b=λ)
            # 3-form_router: (w=tet, c=λ, p=dλ, q=dλ, r=dλ)
            # size: (t=simp,)
            # moments: (a=λ, b=λ, c=λ)
            # wedge_dot: (t=simp, x=dλ, y=dλ, z=dλ, p=dλ, q=dλ, r=dλ)
            # output: (t=simp, u=λ, v=λ, w=dλ)
            einsum = "uaxyz, vb, wcpqr, t, abc, txyzpqr -> tuvw"

        case _:
            raise NotImplementedError()

    return einsum


def triple_scalar_prod(
    k: int,
    l: int,
    mesh: SimplicialComplex,
):
    sc_dim = mesh.dim

    device = mesh.vert_coords.device
    float_dtype = mesh.vert_coords.dtype
    int_dtype = mesh.edges.dtype

    k_form_router = compute_whitney_router(sc_dim, k, device, float_dtype)
    l_form_router = compute_whitney_router(sc_dim, l, device, float_dtype)
    kl_form_router = compute_whitney_router(sc_dim, k + l, device, float_dtype)

    moments = compute_moments(3, sc_dim, device, float_dtype)

    match sc_dim:
        case 2:
            abs_simp_size = tri_geometry.compute_tri_areas(mesh.vert_coords, mesh.tris)
            abs_simp_size_grad = tri_geometry.compute_d_tri_areas_d_vert_coords(
                mesh.vert_coords, mesh.tris
            )
            bary_cord_grad = tri_geometry.bary_coord_grad_inner_prods(
                abs_simp_size, abs_simp_size_grad
            )

        case 3:
            simp_size = tet_geometry.get_tet_signed_vols(mesh.vert_coords, mesh.tets)
            abs_simp_size = t.abs(simp_size)
            simp_size_grad = tet_geometry.d_tet_signed_vols_d_vert_coords(
                mesh.vert_coords, mesh.tets
            )
            wedge_dot = tet_geometry.bary_coord_grad_inner_prods(
                simp_size, simp_size_grad
            )
        case _:
            raise NotImplementedError()

    wedge_dot = inv_gram_det(bary_cord_grad, k + l)

    einsum_str = _einsum_str(k, l)

    return t.einsum(
        einsum_str,
        k_form_router,
        l_form_router,
        kl_form_router,
        abs_simp_size,
        moments,
        wedge_dot,
    )


# TODO: further optimization when the face_dim = mesh_dim
def _find_face_global_idx(
    face_dim: int,
    mesh_dim: int,
    n_verts: int,
    simp_map: dict[int, t.Tensor],
    n_simp_map: dict[int, int],
    pack_dtype: t.dtype,
    device=t.device,
):
    k = face_dim

    n_k_simp = n_simp_map[k]
    # this is not necessary unless k is the top dimension in the complex.
    k_simp = simp_map[k].sort(dim=-1).values
    all_k_faces = enumerate_faces(mesh_dim, k)
    k_face = simp_map[mesh_dim][:, all_k_faces]
    k_face_flat = k_face.view(-1, k + 1)
    k_face_sorted = k_face_flat.sort(dim=-1).values

    k_simp_packed = t.zeros(n_k_simp, dtype=pack_dtype, device=device)
    for idx in range(k + 1):
        k_simp_packed.add_(k_simp[:, idx] * (n_verts ** (k - idx)))

    k_face_packed = t.zeros(k_face_sorted.size(0), dtype=pack_dtype, device=device)
    for idx in range(k + 1):
        k_face_packed.add_(k_face_sorted[:, idx] * (n_verts ** (k - idx)))

    k_face_idx = t.searchsorted(k_simp_packed, k_face_packed)

    n_k_face_per_top_simp = len(all_k_faces)

    return k_face_idx, n_k_face_per_top_simp


def whitney_wedge_product(
    k_cochain: Float[t.Tensor, " k_simp ch"],
    l_cochain: Float[t.Tensor, " l_simp ch"],
    k: int,
    l: int,
    mass: Float[SparseOperator, "(k+l)_simp (k+l)_simp"],
    mesh: SimplicialComplex,
    pairing: Literal["scalar", "dot", "cross", "outer"],
) -> Float[t.Tensor, " (k+l)_simp ch"]:
    dtype = mesh.edges.dtype
    pack_dtype = t.int64
    device = mesh.edges.device

    m = k + l

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

    # Do all k-faces
    k_face_idx, n_k_face = _find_face_global_idx(
        k, mesh.dim, mesh.n_verts, simp_map, n_simp_map, pack_dtype, device
    )
    signed_k_cochain_at_k_face = compute_face_sign(mesh, mesh.dim, k) * k_cochain[
        k_face_idx
    ].view(n_simp_map[mesh.dim], n_k_face, -1)

    # Do all l-faces
    l_face_idx, n_l_face = _find_face_global_idx(
        l, mesh.dim, mesh.n_verts, simp_map, n_simp_map, pack_dtype, device
    )
    signed_l_cochain_at_l_face = compute_face_sign(mesh, mesh.dim, l) * l_cochain[
        l_face_idx
    ].view(n_simp_map[mesh.dim], n_l_face, -1)

    # Do all m-faces
    m_face_idx, n_m_face = _find_face_global_idx(
        m, mesh.dim, mesh.n_verts, simp_map, n_simp_map, pack_dtype, device
    )
    m_sign = compute_face_sign(mesh, mesh.dim, m)

    load = triple_scalar_prod(k, l, mesh)

    match pairing:
        case "scalar":
            signed_m_cochain_at_m_face = m_sign * t.einsum(
                "tuvw,tuc,tvc->twc",
                load,
                signed_k_cochain_at_k_face,
                signed_l_cochain_at_l_face,
            )
        case "dot":
            signed_m_cochain_at_m_face = m_sign * t.einsum(
                "tuvw,tuc,tvc->tw",
                load,
                signed_k_cochain_at_k_face,
                signed_l_cochain_at_l_face,
            ).unsqueeze(-1)
        case "cross":
            epsilon = t.tensor(
                [
                    [[0, 0, 0], [0, 0, 1], [0, -1, 0]],
                    [[0, 0, -1], [0, 0, 0], [1, 0, 0]],
                    [[0, 1, 0], [-1, 0, 0], [0, 0, 0]],
                ],
                device=device,
                dtype=dtype,
            )
            signed_m_cochain_at_m_face = m_sign * t.einsum(
                "tuvw,tuc,tvd,cde->twe",
                load,
                signed_k_cochain_at_k_face,
                signed_l_cochain_at_l_face,
                epsilon,
            )
        case _:
            raise NotImplementedError()

    n_out_ch = signed_m_cochain_at_m_face.shape[-1]
    m_cochain = t.zeros((n_simp_map[m], n_out_ch), device=device, dtype=dtype)
    m_cochain.index_add_(
        0, m_face_idx.flatten(), signed_m_cochain_at_m_face.reshape(-1, n_out_ch)
    )

    # TODO: use implemented sparse solver wrapper
    x = t.linalg.solve(mass.to_dense(), m_cochain)

    return x

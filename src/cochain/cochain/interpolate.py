from typing import Literal

import torch as t
from einops import einsum, rearrange, repeat
from jaxtyping import Float, Integer

from ..complex import SimplicialComplex
from ..geometry.tet.tet_geometry import (
    d_tet_signed_vols_d_vert_coords,
    get_tet_signed_vols,
)
from ..geometry.tri.tri_geometry import (
    compute_d_tri_areas_d_vert_coords,
    compute_tri_areas,
)
from ..utils.faces import enumerate_unique_faces
from ..utils.perm_parity import compute_lex_rel_orient
from ..utils.search import simplex_search


def _bary_whitney_tri_cochain_0(
    cochain_0: Float[t.Tensor, " vert *ch"],
    tris: Integer[t.LongTensor, "tri vert=3"],
    bary_coords: Float[t.Tensor, "tri pt vert=3"],
) -> Float[t.Tensor, "tri pt *ch coord=1"]:
    # W_i = λ_i for i = 0, 1, 2.
    basis = bary_coords

    cochain_0_at_vert_faces: Float[t.Tensor, "tri vert=3 *ch"] = cochain_0[tris]

    form_0 = einsum(
        basis, cochain_0_at_vert_faces, "tri pt vert, tri vert ... -> tri pt ..."
    )

    return rearrange(form_0, "tri pt ... -> tri pt ... 1")


def _bary_whitney_tri_cochain_1(
    cochain_1: Float[t.Tensor, " edge *ch"],
    tri_edge_idx: Integer[t.LongTensor, "tri vert=3"],
    tri_edge_orientations: Float[t.Tensor, "tri vert=3"],
    bary_coords: Float[t.Tensor, "tri pt vert=3"],
    bary_coords_grad: Float[t.Tensor, "tri vert=3 coord=3"],
) -> Float[t.Tensor, "tri pt *ch coord=3"]:
    bary_coords_shaped = rearrange(bary_coords, "tri pt vert -> tri pt vert 1")
    bary_coords_grad_shaped = rearrange(
        bary_coords_grad, "tri vert coord -> tri 1 vert coord"
    )

    # W_ij = λ_i∇λ_j - λ_j∇λ_i for (i, j) = (0, 1), (0, 2), (1, 2)
    # Note that i, j switch positions for the second term.
    local_edge_idx = enumerate_unique_faces(
        simp_dim=2, face_dim=1, device=bary_coords.device
    )
    basis: Float[t.Tensor, "tri pt edge=3 coord=3"] = (
        bary_coords_shaped[:, :, local_edge_idx[:, 0]]
        * bary_coords_grad_shaped[:, :, local_edge_idx[:, 1], :]
        - bary_coords_shaped[:, :, local_edge_idx[:, 1]]
        * bary_coords_grad_shaped[:, :, local_edge_idx[:, 0], :]
    )

    # tri_edge_face_idx contains the index of edges 01, 02, and 12 in the list of
    # canonical edges of the triangular mesh.
    cochain_1_at_edge_faces: Float[t.Tensor, "tri edge=3 *ch"] = cochain_1[tri_edge_idx]

    # If the edges 01, 02, and 12 are not in their canonical orientation, then
    # the corresponding basis form needs a sign correction given by tri_edge_orientations.
    form_1 = einsum(
        basis,
        tri_edge_orientations,
        cochain_1_at_edge_faces,
        "tri pt edge coord, tri edge, tri edge ... -> tri pt ... coord",
    )

    return form_1


def _bary_whitney_tri_cochain_2(
    cochain_2: Float[t.Tensor, " tri *ch"],
    tri_orientations: Float[t.Tensor, " tri"],
    bary_coords: Float[t.Tensor, "tri pt vert=3"],
    bary_coords_grad: Float[t.Tensor, "tri vert=3 coord=3"],
) -> Float[t.Tensor, "tri pt *ch coord=3"]:
    # There is only one basis form W_012 = 2(∇λ_1 x ∇λ_2); note that this basis
    # function is a constant of barycentric coordinates, which means that the
    # interpolated 2-forms will be constant on each triangle.
    basis: Float[t.Tensor, "tri coord=3"] = 2.0 * t.cross(
        bary_coords_grad[:, 1, :], bary_coords_grad[:, 2, :], dim=-1
    )

    # If the triangle is not in a canonical orientation, then the basis form
    # needs a sign correction given by tri_orientations.
    form_2 = einsum(
        basis, tri_orientations, cochain_2, "tri coord, tri, tri ... -> tri ... coord"
    )

    # Note that the bary_coords argument is only used to determine the number
    # of sampled points.
    form_2_shaped = repeat(
        form_2, "tri ... coord -> tri pt ... coord", pt=bary_coords.size(-2)
    )

    return form_2_shaped


def _bary_whitney_tet_cochain_0(
    cochain_0: Float[t.Tensor, " vert *ch"],
    tets: Integer[t.LongTensor, "tet 4"],
    bary_coords: Float[t.Tensor, "tet pt 4"],
) -> Float[t.Tensor, "tet pt *ch coord=1"]:
    # W_i = λ_i for i = 0, 1, 2, 3.
    basis = bary_coords

    cochain_0_at_vert_faces: Float[t.Tensor, "tet vert=4 *ch"] = cochain_0[tets]

    form_0 = einsum(
        basis, cochain_0_at_vert_faces, "tet pt vert, tet vert ... -> tet pt ..."
    )

    return rearrange(form_0, "tet pt ... -> tet pt ... 1")


def _bary_whitney_tet_cochain_1(
    cochain_1: Float[t.Tensor, " edge *ch"],
    tet_edge_idx: Integer[t.LongTensor, "tet edge=6"],
    tet_edge_orientations: Float[t.Tensor, "tet edge=6"],
    bary_coords: Float[t.Tensor, "tet pt vert=4"],
    bary_coords_grad: Float[t.Tensor, "tet vert=4 coord=3"],
) -> Float[t.Tensor, "tet pt *ch coord=3"]:
    bary_coords_shaped = rearrange(bary_coords, "tet pt vert -> tet pt vert p 1")
    bary_coords_grad_shaped = rearrange(
        bary_coords_grad, "tet vert coord -> tet 1 vert coord"
    )

    # W_ij = λ_i∇λ_j - λ_j∇λ_i for ij = 01, 02, 12, 13, 23, 03
    # Note that i, j switch positions for the second term.
    local_edge_idx = enumerate_unique_faces(
        simp_dim=3, face_dim=1, device=bary_coords.device
    )
    basis: Float[t.Tensor, "tet pt edge=6 coord=3"] = (
        bary_coords_shaped[:, :, local_edge_idx[:, 0]]
        * bary_coords_grad_shaped[:, :, local_edge_idx[:, 1]]
        - bary_coords_shaped[:, :, local_edge_idx[:, 1]]
        * bary_coords_grad_shaped[:, :, local_edge_idx[:, 0]]
    )

    # tet_edge_face_idx contains the index of edges 01, 02, 12, 13, 23, and 03 in
    # the list of canonical edges of the tet mesh.
    cochain_1_at_edge_faces: Float[t.Tensor, "tet edge=6 *ch"] = cochain_1[tet_edge_idx]

    # If the edges are not in their canonical orientation, then the corresponding
    # basis form needs a sign correction given by tet_edge_orientations.
    form_1 = einsum(
        basis,
        tet_edge_orientations,
        cochain_1_at_edge_faces,
        "tet pt edge coord, tet edge, tet edge ... -> tet pt ... coord",
    )

    return form_1


def _bary_whitney_tet_cochain_2(
    cochain_2: Float[t.Tensor, " tri *ch"],
    tet_tri_idx: Integer[t.LongTensor, "tet tri=4"],
    tet_tri_orientations: Float[t.Tensor, "tet tri=4"],
    bary_coords: Float[t.Tensor, "tet pt vert=4"],
    bary_coords_grad: Float[t.Tensor, "tet vert=4 coord=3"],
) -> Float[t.Tensor, "tet pt *ch coord=3"]:
    bary_coords_shaped = rearrange(bary_coords, "tet pt vert -> tet pt vert 1")
    bary_coords_grad_shaped = rearrange(
        bary_coords_grad, "tet vert coord -> tet 1 vert coord"
    )

    # W_ijk = 2(λ_i ∇λ_jx∇λ_k + λ_j ∇λ_kx∇λ_i + λ_k ∇λ_ix∇ λ_j)
    # for ijk in 123, 032, 013, 021
    local_tri_idx = enumerate_unique_faces(
        simp_dim=3, face_dim=2, device=bary_coords.device
    )
    perm_i = local_tri_idx[:, 0]
    perm_j = local_tri_idx[:, 1]
    perm_k = local_tri_idx[:, 2]
    basis: Float[t.Tensor, "tet pt tri=4 coord=3"] = 2.0 * (
        bary_coords_shaped[:, :, perm_i]
        * t.cross(
            bary_coords_grad_shaped[:, :, perm_j],
            bary_coords_grad_shaped[:, :, perm_k],
            dim=-1,
        )
        + bary_coords_shaped[:, :, perm_j]
        * t.cross(
            bary_coords_grad_shaped[:, :, perm_k],
            bary_coords_grad_shaped[:, :, perm_i],
            dim=-1,
        )
        + bary_coords_shaped[:, :, perm_k]
        * t.cross(
            bary_coords_grad_shaped[:, :, perm_i],
            bary_coords_grad_shaped[:, :, perm_j],
            dim=-1,
        )
    )

    # tet_tri_face_idx contains the index of triangles 123, 032, 013, 021 in
    # the list of canonical triangles of the tet mesh.
    cochain_2_at_tri_faces: Float[t.Tensor, "tet tri=4"] = cochain_2[tet_tri_idx]

    # If the triangles are not in their canonical orientation, then the corresponding
    # basis form needs a sign correction given by tet_edge_orientations.
    form_2 = einsum(
        basis,
        tet_tri_orientations,
        cochain_2_at_tri_faces,
        "tet pt tri coord, tet tri, tet tri... -> tet pt ... coord",
    )

    return form_2


def _bary_whitney_tet_cochain_3(
    cochain_3: Float[t.Tensor, " tet *ch"],
    tet_signed_vols: Float[t.Tensor, " tet"],
    tet_orientations: Float[t.Tensor, " tet"],
    bary_coords: Float[t.Tensor, "tet pt vert=4"],
) -> Float[t.Tensor, "tet pt *ch coord=1"]:
    # There is only one basis form W_0123 = 1/vol; note that this basis
    # function is a constant of barycentric coordinates, which means that the
    # interpolated 3-forms will be constant on each tet.
    basis = 1.0 / tet_signed_vols

    # If the tet is not in a canonical orientation, then the basis form
    # needs a sign correction given by tet_orientations.
    form_3 = einsum(basis, tet_orientations, cochain_3, "tet, tet, tet ... -> tet ...")

    # Note that the bary_coords argument is only used here to determine the
    # number of sampled points.
    form_3_shaped = repeat(
        form_3, "tet ... -> tet pt ... coord", pt=bary_coords.size(-2), coord=1
    )

    return form_3_shaped


def _bary_whitney_tri(
    k: int,
    k_cochain: Float[t.Tensor, " simp *ch"],
    bary_coords: Float[t.Tensor, "tri pt vert"],
    mesh: SimplicialComplex,
) -> Float[t.Tensor, "tri pt *ch coord"]:
    if k in [1, 2]:
        tri_areas = rearrange(
            compute_tri_areas(mesh.vert_coords, mesh.tris), "tri -> tri 1 1"
        )
        d_tri_areas_d_vert_coords = compute_d_tri_areas_d_vert_coords(
            mesh.vert_coords, mesh.tris
        )
        bary_coords_grad: Float[t.Tensor, "tri vert=3 coord=3"] = (
            d_tri_areas_d_vert_coords / tri_areas
        )

    match k:
        case 0:
            return _bary_whitney_tri_cochain_0(
                cochain_0=k_cochain, tris=mesh.tris, bary_coords=bary_coords
            )
        case 1:
            return _bary_whitney_tri_cochain_1(
                cochain_1=k_cochain,
                tri_edge_idx=mesh.tri_edge_idx,
                tri_edge_orientations=mesh.tri_edge_orientations,
                bary_coords=bary_coords,
                bary_coords_grad=bary_coords_grad,
            )
        case 2:
            return _bary_whitney_tri_cochain_2(
                cochain_2=k_cochain,
                tri_orientations=mesh.tri_orientations,
                bary_coords=bary_coords,
                bary_coords_grad=bary_coords_grad,
            )
        case _:
            raise ValueError()


def _bary_whitney_tet(
    k: int,
    k_cochain: Float[t.Tensor, " simp *ch"],
    bary_coords: Float[t.Tensor, "tet pt vert"],
    mesh: SimplicialComplex,
) -> Float[t.Tensor, "tet pt *ch coord"]:
    if k in [1, 2, 3]:
        tet_signed_vols = rearrange(
            get_tet_signed_vols(mesh.vert_coords, mesh.tets), "tet -> tet 1 1"
        )

    if k in [1, 2]:
        d_signed_vols_d_vert_coords = d_tet_signed_vols_d_vert_coords(
            mesh.vert_coords, mesh.tets
        )
        bary_coords_grad: Float[t.Tensor, "tet vert=4 coord=3"] = (
            d_signed_vols_d_vert_coords / tet_signed_vols
        )

    match k:
        case 0:
            return _bary_whitney_tet_cochain_0(
                cochain_0=k_cochain, tets=mesh.tets, bary_coords=bary_coords
            )
        case 1:
            return _bary_whitney_tet_cochain_1(
                cochain_1=k_cochain,
                tet_edge_idx=mesh.tet_edge_idx,
                tet_edge_orientations=mesh.tet_edge_orientations,
                bary_coords=bary_coords,
                bary_coords_grad=bary_coords_grad,
            )
        case 2:
            return _bary_whitney_tet_cochain_2(
                cochain_2=k_cochain,
                tet_tri_idx=mesh.tet_tri_idx,
                tet_tri_orientations=mesh.tet_tri_orientations,
                bary_coords=bary_coords,
                bary_coords_grad=bary_coords_grad,
            )
        case 3:
            return _bary_whitney_tet_cochain_3(
                cochain_3=k_cochain,
                tet_signed_vols=tet_signed_vols,
                bary_coords=bary_coords,
                tet_orientations=mesh.tet_orientations,
            )
        case _:
            raise ValueError()


def _bary_embed(
    m_simps: Integer[t.LongTensor, "m_simp m_vert"],
    k_simps_bary_coords: Float[t.Tensor, "k_simp pt k_vert"],
    k_faces_local_vert_idx: Integer[t.LongTensor, "k_face k_vert"],
    k_faces_global_idx: Integer[t.LongTensor, "m_simp k_face"],
    n_k_simps: int,
) -> Float[t.Tensor, "m_simp k_face pt m_vert"]:
    """
    Embed barycentric coordinates for a k-simplex onto the barycentric coordinates
    of the k-dimensional faces of a higher m-simplex. Note that the `k_simp`
    dimension of `k_simps_bary_coords` is allowed to be trivial.

    Example:

    Given a two-point quadrature rule on a 1-simplex
    ```
    k_simps_bary_coords = [
        [0.2, 0.8],
        [0.8, 0.2]
    ]
    ```
    and a 2-simplex with edges (represented by local vertex index)
    ```
    k_simps_local_vert_idx = [[0, 1], [0, 2], [1, 2]]
    ```
    To embed the 1-simplex barycentric coordinates onto the 1-faces 01, 02, and 12
    of a 2-simplex (m = 2), the function returns
    ```
    [[[0.2, 0.8, 0.0],
      [0.8, 0.2, 0.0]],

     [[0.2, 0.0, 0.8],
      [0.8, 0.0, 0.2]],

     [[0.0, 0.2, 0.8],
      [0.0, 0.8, 0.2]]]
    ```
    Note that this function also handles orientation/permutation mismatch between
    the k-faces and the canonical k-simplices (not shown in this example).
    """
    # Collect shape/size information.
    n_m_simps, n_m_verts = m_simps.shape
    n_pts = k_simps_bary_coords.size(1)
    n_k_faces_per_m_simp = k_faces_local_vert_idx.size(0)

    # In case the first dimension of k_simps_bary_coords is trivial, inflate to the
    # correct shape.
    k_simps_bary_coords_shaped = k_simps_bary_coords.expand(
        n_k_simps, *k_simps_bary_coords.shape[1:]
    )

    # Scatter the barycentric coordinates defined on the canonical k-simplices
    # to the k-faces of the m-simplices.
    k_simps_bary_coords_scattered: Float[t.Tensor, "m_simp k_face pt k_vert"] = (
        k_simps_bary_coords_shaped[k_faces_global_idx]
    )

    # For each k-face of an m-simplex, identify the permutation required to reorder
    # the corresponding canonical k-simplex to match the k-face. For example,
    # For a 2-face [30, 2, 40], the permutation [1, 0, 2] is required to permute
    # the canonical 2-simplex [2, 30, 40] to [30, 2, 40].
    all_k_faces: Integer[t.LongTensor, "m_simp k_face k_vert"] = m_simps[
        :, k_faces_local_vert_idx
    ]

    # Note that the two argsort() here is required; the first argsort() computes
    # the permutation required to reorder the k-face to match the canonical
    # k-simplex, and the second argsort() computes the inverse of this mapping.
    k_face_perm_map = repeat(
        t.argsort(
            t.argsort(all_k_faces, dim=-1, descending=False),
            dim=-1,
            descending=False,
        ),
        "m_simp k_face k_vert -> m_simp k_face pt k_vert",
        pt=n_pts,
    )

    # Use the permutation map to reorder the last dimension of k_simps_bary_coords_scattered.
    # This is required because the local k-face definition in k_faces_local_vert_idx
    # ignores the orientation/permutation difference between a k-face and the
    # canonical k-simplex. For example, if a canonical 1-simplex [28, 29] is the
    # face of two 2-simplices [27, 28, 29] and [30, 29, 28], then the barycentric
    # coordinates evaluated at edge 12 in the two 2-simplices are flipped. This
    # becomes a problem later on in _barycentric_whitney_map_boundary() when the
    # k-forms interpolated on the k-faces are used to produce an averaged k-form
    # per canonical k-simplex, where the k-form evaluated at the wrong points may
    # be averaged together. Note that this permutation correction is necessary
    # (1) on top of the k-form sign corrections, (2) even if the set of points
    # is invariant to vertex permutation, and (3) even if the Whitney bases used
    # are of the lowest order.
    k_simps_bary_coords_permuted = t.gather(
        input=k_simps_bary_coords_scattered, dim=-1, index=k_face_perm_map
    )

    # Broadcast and embed the k-simplex barycentric coordinates into the m-simplices.
    k_faces_local_vert_idx_shaped = repeat(
        k_faces_local_vert_idx,
        "k_face k_vert -> m_simp k_face pt k_vert",
        m_simp=n_m_simps,
        pt=n_pts,
    )

    bary_coords_embedded = t.zeros(
        n_m_simps,
        n_k_faces_per_m_simp,
        n_pts,
        n_m_verts,
        dtype=k_simps_bary_coords.dtype,
        device=k_simps_bary_coords.device,
    )

    bary_coords_embedded.scatter_(
        dim=-1, index=k_faces_local_vert_idx_shaped, src=k_simps_bary_coords_permuted
    )

    return bary_coords_embedded


def _barycentric_whitney_map_interior(
    k: int,
    k_cochain: Float[t.Tensor, " simp *ch"],
    bary_coords: Float[t.Tensor, "top_simp pt vert"],
    mesh: SimplicialComplex,
) -> Float[t.Tensor, "top_simp pt *ch coord"]:
    match mesh.dim:
        case 2:
            return _bary_whitney_tri(k, k_cochain, bary_coords, mesh)
        case 3:
            return _bary_whitney_tet(k, k_cochain, bary_coords, mesh)
        case _:
            raise ValueError()


def _barycentric_whitney_map_boundary(
    k: int,
    k_cochain: Float[t.Tensor, " k_simp *ch"],
    bary_coords: Float[t.Tensor, "k_simp pt vert"],
    mesh: SimplicialComplex,
    reduction: Literal["mean", "none"] = "none",
) -> Float[t.Tensor, "*m_simp k_simp_or_face pt *ch coord"]:
    m = mesh.dim
    n_k_simps = mesh.simplices[k].size(0)
    n_pts = bary_coords.size(-2)

    local_face_idx: Integer[t.LongTensor, "k_face k_vert"] = enumerate_unique_faces(
        simp_dim=m, face_dim=k, device=mesh.vert_coords.device
    )

    # Find the global indices of all k-faces.
    # TODO: this is not ideal since we are repeating topo calculations
    all_faces = mesh.simplices[m][:, local_face_idx]

    global_face_idx: Integer[t.LongTensor, "m_simp k_face"] = simplex_search(
        key_simps=mesh.simplices[k],
        query_simps=all_faces,
        sort_key_simp=False,
        sort_key_vert=False,
        sort_query_vert=True,
        method="lex_sort",
    )

    # Perform barycentric coordinate embedding and then compute the whitney
    # interpolation at the top simplex level.
    bary_coords_embedded: Float[t.Tensor, "m_simp k_face pt m_vert"] = _bary_embed(
        m_simps=mesh.simplices[m],
        k_simps_bary_coords=bary_coords,
        k_faces_local_vert_idx=local_face_idx,
        k_faces_global_idx=global_face_idx,
        n_k_simps=n_k_simps,
    )

    k_forms = _barycentric_whitney_map_interior(
        k,
        k_cochain,
        rearrange(
            bary_coords_embedded,
            "m_simp k_face pt m_vert -> m_simp (k_face pt) m_vert",
        ),
        mesh,
    )

    match reduction:
        case "none":
            k_forms_shaped = rearrange(
                k_forms,
                "m_simp (k_face pt) ... coord -> m_simp k_face pt ... coord",
                pt=n_pts,
            )
            return k_forms_shaped

        # For each unique/canonical k-simplex in the mesh, compute the average
        # k-forms evaluated at the barycentric coordinates per top-level simplex.
        case "mean":
            k_forms_shaped = rearrange(
                k_forms,
                "m_simp (k_face pt) ... coord -> (m_simp k_face) pt ... coord",
                pt=n_pts,
            )

            global_face_idx_shaped: Integer[
                t.LongTensor, "m_simp*k_face pt *ch coord"
            ] = (
                global_face_idx.flatten()
                .view(-1, *[1] * (k_forms_shaped.ndim - 1))
                .expand_as(k_forms_shaped)
            )

            canon_k_forms: Float[t.Tensor, "k_simp pt *ch coord"] = t.zeros(
                (n_k_simps, *k_forms_shaped.shape[1:]),
                dtype=k_cochain.dtype,
                device=k_cochain.device,
            )
            canon_k_forms.scatter_reduce_(
                dim=0,
                index=global_face_idx_shaped,
                src=k_forms_shaped,
                reduce="mean",
                include_self=False,
            )

            return canon_k_forms


def barycentric_whitney_map(
    k: int,
    k_cochain: Float[t.Tensor, " k_simp *ch"],
    bary_coords: Float[t.Tensor, "simp pt vert"],
    mesh: SimplicialComplex,
    mode: Literal["interior", "boundary"],
    boundary_reduction: Literal["mean", "none"] = "none",
) -> Float[t.Tensor, "*simp pt *ch coord"]:
    """
    This function implements an "element-local" version of the Whitney map for
    interpolating discrete k-cochains using Whitney basis functions of the lowest
    order.

    In the `interior` mode, the function maps the k-cochains to k-forms interpolated
    at local barycentric coordinates across all m-simplices, where m is the dimension
    of the mesh; in the `boundary` mode, the function maps the k-cochains to k-forms
    interpolated at local barycentric coordinates across the k-simplices. If
    k = m, the `interior` mode is used regardless of the `mode` argument. Currently,
    interpolation of k-cochains on l-simplices in an m-dimensional mesh, where
    k < l < m, is not supported.

    The `boundary_reduction` argument modifies the output of the `boundary` mode,
    if set to `'none'`, then the returned tensor is of shape (`m_simp`, `k_face`,
    `pt`, `*ch`, `coord`); if set to `'mean'`, then the returned tensor is of shape
    (`k_simp`, `pt`, `*ch`, `coord`), where the interpolated k-forms are averaged
    over all k-faces of m-simplices corresponding to the same canonical k-simplex.
    Note that k-forms interpolated using Whitney bases have discontinuous, multi-
    valued, unconstrained components (e.g., for 1-forms, the tangential components
    are continuous but the normal components jump; for 2-forms, the normal components
    are continuous but the tangential components jump between higher-order simplices);
    as such, the `'mean'` reduction creates distortions and should only be used
    for downstream applications when the unconstrained components are irrelevant
    (e.g., for de Rham map).

    For the `boundary` mode, the `pt` dimension is always ordered relative to the
    canonical simplices. For example, consider a two point quadrature on 1-simplices
    consisting of barycentric coordinates (0.2, 0.8) and (0.8, 0.2); if a canonical
    1-simplex [28, 29] is the face of two 2-simplices [27, 28, 29] and [30, 29, 28],
    then the `pt` dimension for both faces will refer to the point closer to vertex
    29 first, and the point closer to vertex 28 second, regardless of the local
    vertex ordering of the 1-face in the 2-simplices. Note that, for the `interior`
    mode, the values across the `pt` dimension will be constant since the interpolated
    m-forms (i.e., volume forms) on the m-simplices are always piecewise constant.

    Note that this function does not perform global spatial interpolation (i.e.,
    it cannot directly evaluate the k-form at arbitrary cartesian coordinates on
    the mesh.)

    The input `k_cochain` is allowed to have an arbitrary number of trailing
    channel/batch dimensions. The `bary_coords` argument is allowed to have a trivial
    first `simp` dimension, in which case the k-cochain is interpolated at the
    same fixed local barycentric coordinates over all target simplices.
    """
    if k == mesh.dim:
        mode = "interior"

    match mode:
        case "interior":
            return _barycentric_whitney_map_interior(k, k_cochain, bary_coords, mesh)
        case "boundary":
            return _barycentric_whitney_map_boundary(
                k, k_cochain, bary_coords, mesh, boundary_reduction
            )
        case _:
            raise ValueError()

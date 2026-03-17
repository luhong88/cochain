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
from ..utils.search import simplex_search


def _bary_whitney_tri_cochain_0(
    cochain_0: Float[t.Tensor, " vert *ch"],
    tris: Integer[t.LongTensor, "tri vert=3"],
    bary_coords: Float[t.Tensor, "pt vert=3"],
) -> Float[t.Tensor, "tri pt *ch coord=1"]:
    # W_i = λ_i for i = 0, 1, 2.
    basis: Float[t.Tensor, "pt vert=3"] = bary_coords

    cochain_0_at_vert_faces: Float[t.Tensor, "tri vert=3 *ch"] = cochain_0[tris]

    form_0 = einsum(
        basis, cochain_0_at_vert_faces, "pt vert, tri vert ... -> tri pt ..."
    )

    return rearrange(form_0, "tri pt ... -> tri pt ... 1")


def _bary_whitney_tri_cochain_1(
    cochain_1: Float[t.Tensor, " edge *ch"],
    tri_edge_idx: Integer[t.LongTensor, "tri vert=3"],
    tri_edge_orientations: Float[t.Tensor, "tri vert=3"],
    bary_coords: Float[t.Tensor, "pt vert=3"],
    bary_coords_grad: Float[t.Tensor, "tri vert=3 coord=3"],
) -> Float[t.Tensor, "tri pt *ch coord=3"]:
    bary_coords_shaped = rearrange(bary_coords, "pt coord -> 1 pt vert 1")
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
        "tri pt edge coord, tri edge, tri eedge ... -> tri pt ... coord",
    )

    return form_1


def _bary_whitney_tri_cochain_2(
    cochain_2: Float[t.Tensor, " tri *ch"],
    tri_orientations: Float[t.Tensor, " tri"],
    bary_coords_grad: Float[t.Tensor, "tri vert=3 coord=3"],
) -> Float[t.Tensor, "tri pt=1 *ch coord=3"]:
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

    return rearrange(
        form_2,
        "tri ... coord -> tri 1 ... coord",
    )


def _bary_whitney_tet_cochain_0(
    cochain_0: Float[t.Tensor, " vert *ch"],
    tets: Integer[t.LongTensor, "tet 4"],
    bary_coords: Float[t.Tensor, "pt 4"],
) -> Float[t.Tensor, "tet pt *ch coord=1"]:
    # W_i = λ_i for i = 0, 1, 2, 3.
    basis: Float[t.Tensor, "pt vert=4"] = bary_coords

    cochain_0_at_vert_faces: Float[t.Tensor, "tet vert=4 *ch"] = cochain_0[tets]

    form_0 = einsum(
        basis, cochain_0_at_vert_faces, "pt vert, tet vert ... -> tet pt ..."
    )

    return rearrange(form_0, "tet pt ... -> tet pt ... 1")


def _bary_whitney_tet_cochain_1(
    cochain_1: Float[t.Tensor, " edge *ch"],
    tet_edge_idx: Integer[t.LongTensor, "tet edge=6"],
    tet_edge_orientations: Float[t.Tensor, "tet edge=6"],
    bary_coords: Float[t.Tensor, "pt vert=4"],
    bary_coords_grad: Float[t.Tensor, "tet vert=4 coord=3"],
) -> Float[t.Tensor, "tet pt *ch coord=3"]:
    bary_coords_shaped = rearrange(bary_coords, "pt vert -> 1 pt vert p1")
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
    bary_coords: Float[t.Tensor, "pt vert=4"],
    bary_coords_grad: Float[t.Tensor, "tet vert=4 coord=3"],
) -> Float[t.Tensor, "tet pt *ch coord=3"]:
    bary_coords_shaped = rearrange(bary_coords, "pt vert -> 1 pt vert 1")
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
) -> Float[t.Tensor, "tet pt=1 *ch coord=1"]:
    # There is only one basis form W_0123 = 1/vol; note that this basis
    # function is a constant of barycentric coordinates, which means that the
    # interpolated 3-forms will be constant on each tet.
    basis = 1.0 / tet_signed_vols

    # If the tet is not in a canonical orientation, then the basis form
    # needs a sign correction given by tet_orientations.
    form_3 = einsum(basis, tet_orientations, cochain_3, "tet, tet, tet ... -> tet ...")

    return rearrange(form_3, "tet 1 ... 1")


def _bary_whitney_tri(
    k: int,
    k_cochain: Float[t.Tensor, " simp *ch"],
    bary_coords: Float[t.Tensor, "pt vert"],
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
                bary_coords_grad=bary_coords_grad,
            )
        case _:
            raise ValueError()


def _bary_whitney_tet(
    k: int,
    k_cochain: Float[t.Tensor, " simp *ch"],
    bary_coords: Float[t.Tensor, "pt vert"],
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
                tet_orientations=mesh.tet_orientations,
            )
        case _:
            raise ValueError()


def _bary_embed(
    m: int,
    k_simps_bary_coords: Float[t.Tensor, "pt k_vert"],
    k_simps_local_vert_idx: Integer[t.LongTensor, "k_simp k_vert"],
) -> Float[t.Tensor, "k_simp pt m_vert"]:
    """
    Embed barycentric coordinates for a k-simplex onto the barycentric coordinates
    of the k-dimensional faces of a higher m-simplex.

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
    """
    device = k_simps_bary_coords.device
    dtype = k_simps_bary_coords.dtype

    n_pts = k_simps_bary_coords.size(0)
    n_k_faces = k_simps_local_vert_idx.size(0)
    n_coords_embedded = m + 1

    # Prepare for scatter by casting all tensors to the target shape
    # (n_k_simps, n_pts, n_coords_embedded)
    bary_coords_embedded = t.zeros(
        n_k_faces,
        n_pts,
        n_coords_embedded,
        dtype=dtype,
        device=device,
    )
    src = repeat(k_simps_bary_coords, "pt coord -> k_face pt coord", k_face=n_k_faces)
    idx = repeat(k_simps_local_vert_idx, "k_simp vert -> k_simp pt vert", pt=n_pts)

    bary_coords_embedded.scatter_(dim=2, index=idx, src=src)

    return bary_coords_embedded


def _barycentric_whitney_map_interior(
    k: int,
    k_cochain: Float[t.Tensor, " simp *ch"],
    bary_coords: Float[t.Tensor, "pt vert"],
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
    bary_coords: Float[t.Tensor, "pt vert"],
    mesh: SimplicialComplex,
) -> Float[t.Tensor, "k_simp pt *ch coord"]:
    m = mesh.dim

    local_face_idx: Integer[t.LongTensor, "k_face k_vert"] = enumerate_unique_faces(
        simp_dim=m, face_dim=k, device=mesh.vert_coords.device
    )

    # Perform barycentric coordinate embedding and then compute the whitney
    # interpolation at the top simplex level.
    bary_coords_embedded: Float[t.Tensor, "k_face pt m_vert"] = _bary_embed(
        m,
        bary_coords,
        local_face_idx,
    )
    n_k_faces, n_pts, _ = bary_coords_embedded.shape

    k_forms = _barycentric_whitney_map_interior(
        k,
        k_cochain,
        rearrange(bary_coords_embedded, "k_face pt m_vert -> (k_face pt) m_vert"),
        mesh,
    )
    k_forms_shaped = rearrange(
        k_forms,
        "m_simp (k_face pt) ... coord -> (m_simp k_face) pt ... coord",
        k_face=n_k_faces,
        pt=n_pts,
    )

    # Find the global indices of all k-faces.
    # TODO: this is not ideal since we are repating topo calculations
    all_faces = rearrange(
        mesh.simplices[m][:, local_face_idx],
        "m_simp k_face k_vert -> (m_simp k_face) k_vert",
    )

    global_face_idx: Integer[t.LongTensor, "m_simp*k_face pt *ch coord"] = (
        simplex_search(
            key_simps=mesh.simplices[m],
            query_simps=all_faces,
            sort_key_simp=False,
            sort_key_vert=False,
            sort_query_vert=True,
            method="lex_sort",
        )
        .view(-1, *[1] * (k_forms_shaped.ndim - 1))
        .expand_as(k_forms_shaped)
    )

    # For each unique/canonical k-simplex in the mesh, compute the average
    # k-forms evaluated at the barycentric coordinates per top-level simplex.
    # Another way to achieve the same result would be to find, for each canonical
    # k-simplex, a representative top-level simplex that contains the k-simplex
    # as a face, and use the k-forms evaluated on the representative. Topologically,
    # the two approaches achieve the same results, but they differ geometrically,
    # in that 1) the normal components of the interpolated k-forms are not constrained
    # by the whitney map and thus the two approaches will give k-forms with different
    # normal components on the k-simplices, and 2) the second approach is not
    # safe for autograd since it doesn't correctly link each k-simplex to all of
    # its top-level cofaces.
    canon_k_forms: Float[t.Tensor, "k_simp pt *ch coord"] = t.zeros(
        (mesh.simplices[k].size(0), *k_forms_shaped.shape[1:]),
        dtype=k_cochain.dtype,
        device=k_cochain.device,
    )
    canon_k_forms.scatter_reduce_(
        dim=0,
        index=global_face_idx,
        src=k_forms_shaped,
        reduce="mean",
        include_self=False,
    )

    return canon_k_forms


def barycentric_whitney_map(
    k: int,
    k_cochain: Float[t.Tensor, " k_simp *ch"],
    bary_coords: Float[t.Tensor, "pt vert"],
    mesh: SimplicialComplex,
    mode: Literal["interior", "boundary"],
) -> Float[t.Tensor, "simp pt *ch coord"]:
    """
    This function implements an "element-local" version of the Whitney map for
    interpolating discrete k-cochains, which is useful for numerical quadrature.

    In the `interior` mode, the function maps the k-cochains to k-forms interpolated
    at a fixed set of local barycentric coordinates across all top-level simplices;
    in the `boundary` mode, the function maps the k-cochains to k-forms interpolated
    at a fixed set of local barycentric coordinates across the k-simplices.

    Note that this function does not perform global spatial interpolation (i.e.,
    it cannot directly evaluate the k-form at arbitrary cartesian coordinates on
    the mesh.)

    The input k-cochain is allowed to have an arbitrary number of trailing
    channel/batch dimensions.
    """
    match mode:
        case "interior":
            return _barycentric_whitney_map_interior(k, k_cochain, bary_coords, mesh)
        case "boundary":
            return _barycentric_whitney_map_boundary(k, k_cochain, bary_coords, mesh)
        case _:
            raise ValueError()

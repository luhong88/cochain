from typing import Literal

import torch
from einops import einsum, rearrange, repeat
from jaxtyping import Float, Integer
from torch import LongTensor, Tensor

from ..complex import SimplicialMesh
from ..metric.tet import _tet_geometry
from ..metric.tri import _tri_geometry
from ..utils.faces import enumerate_local_faces
from ..utils.search import splx_search


def _bary_whitney_tri_cochain_0(
    cochain_0: Float[Tensor, " vert *ch"],
    tris: Integer[LongTensor, "tri vert=3"],
    bary_coords: Float[Tensor, "tri pt vert=3"],
) -> Float[Tensor, "tri pt *ch coord=1"]:
    # W_i = λ_i for i = 0, 1, 2.
    basis = bary_coords

    cochain_0_at_vert_faces: Float[Tensor, "tri vert=3 *ch"] = cochain_0[tris]

    form_0 = einsum(
        basis, cochain_0_at_vert_faces, "tri pt vert, tri vert ... -> tri pt ..."
    )

    return rearrange(form_0, "tri pt ... -> tri pt ... 1")


def _bary_whitney_tri_cochain_1(
    cochain_1: Float[Tensor, " edge *ch"],
    tri_edge_idx: Integer[LongTensor, "tri edge=3"],
    tri_edge_orientations: Float[Tensor, "tri edge=3"],
    bary_coords: Float[Tensor, "tri pt vert=3"],
    bary_coords_grad: Float[Tensor, "tri vert=3 coord=3"],
) -> Float[Tensor, "tri pt *ch coord=3"]:
    bary_coords_shaped = rearrange(bary_coords, "tri pt vert -> tri pt vert 1")
    bary_coords_grad_shaped = rearrange(
        bary_coords_grad, "tri vert coord -> tri 1 vert coord"
    )

    # W_ij = λ_i∇λ_j - λ_j∇λ_i for (i, j) = (0, 1), (0, 2), (1, 2)
    # Note that i, j switch positions for the second term.
    local_edge_idx = enumerate_local_faces(
        splx_dim=2, face_dim=1, device=bary_coords.device
    )
    basis: Float[Tensor, "tri pt edge=3 coord=3"] = (
        bary_coords_shaped[:, :, local_edge_idx[:, 0]]
        * bary_coords_grad_shaped[:, :, local_edge_idx[:, 1], :]
        - bary_coords_shaped[:, :, local_edge_idx[:, 1]]
        * bary_coords_grad_shaped[:, :, local_edge_idx[:, 0], :]
    )

    # tri_edge_face_idx contains the index of edges 01, 02, and 12 in the list of
    # canonical edges of the triangular mesh.
    cochain_1_at_edge_faces: Float[Tensor, "tri edge=3 *ch"] = cochain_1[tri_edge_idx]

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
    cochain_2: Float[Tensor, " tri *ch"],
    bary_coords: Float[Tensor, "tri pt vert=3"],
    bary_coords_grad: Float[Tensor, "tri vert=3 coord=3"],
) -> Float[Tensor, "tri pt *ch coord=3"]:
    # There is only one basis form W_012 = 2(∇λ_1 x ∇λ_2); note that this basis
    # function is a constant of barycentric coordinates, which means that the
    # interpolated 2-forms will be constant on each triangle.
    basis: Float[Tensor, "tri coord=3"] = 2.0 * torch.cross(
        bary_coords_grad[:, 1, :], bary_coords_grad[:, 2, :], dim=-1
    )

    # Note that no orientation sign correction is needed here since the top-level
    # simplices are stored as is rather than lex-sorted.
    form_2 = einsum(basis, cochain_2, "tri coord, tri ... -> tri ... coord")

    # Note that the bary_coords argument is only used to determine the number
    # of sampled points.
    form_2_shaped = repeat(
        form_2, "tri ... coord -> tri pt ... coord", pt=bary_coords.size(-2)
    )

    return form_2_shaped


def _bary_whitney_tet_cochain_0(
    cochain_0: Float[Tensor, " vert *ch"],
    tets: Integer[LongTensor, "tet vert=4"],
    bary_coords: Float[Tensor, "tet pt vert=4"],
) -> Float[Tensor, "tet pt *ch coord=1"]:
    # W_i = λ_i for i = 0, 1, 2, 3.
    basis = bary_coords

    cochain_0_at_vert_faces: Float[Tensor, "tet vert=4 *ch"] = cochain_0[tets]

    form_0 = einsum(
        basis, cochain_0_at_vert_faces, "tet pt vert, tet vert ... -> tet pt ..."
    )

    return rearrange(form_0, "tet pt ... -> tet pt ... 1")


def _bary_whitney_tet_cochain_1(
    cochain_1: Float[Tensor, " edge *ch"],
    tet_edge_idx: Integer[LongTensor, "tet edge=6"],
    tet_edge_orientations: Float[Tensor, "tet edge=6"],
    bary_coords: Float[Tensor, "tet pt vert=4"],
    bary_coords_grad: Float[Tensor, "tet vert=4 coord=3"],
) -> Float[Tensor, "tet pt *ch coord=3"]:
    bary_coords_shaped = rearrange(bary_coords, "tet pt vert -> tet pt vert 1")
    bary_coords_grad_shaped = rearrange(
        bary_coords_grad, "tet vert coord -> tet 1 vert coord"
    )

    # W_ij = λ_i∇λ_j - λ_j∇λ_i
    # Note that i, j switch positions for the second term.
    local_edge_idx = enumerate_local_faces(
        splx_dim=3, face_dim=1, device=bary_coords.device
    )
    basis: Float[Tensor, "tet pt edge=6 coord=3"] = (
        bary_coords_shaped[:, :, local_edge_idx[:, 0]]
        * bary_coords_grad_shaped[:, :, local_edge_idx[:, 1]]
        - bary_coords_shaped[:, :, local_edge_idx[:, 1]]
        * bary_coords_grad_shaped[:, :, local_edge_idx[:, 0]]
    )

    # tet_edge_face_idx contains the index of 1-faces in the list of canonical
    # edges of the tet mesh.
    cochain_1_at_edge_faces: Float[Tensor, "tet edge=6 *ch"] = cochain_1[tet_edge_idx]

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
    cochain_2: Float[Tensor, " tri *ch"],
    tet_tri_idx: Integer[LongTensor, "tet tri=4"],
    tet_tri_orientations: Float[Tensor, "tet tri=4"],
    bary_coords: Float[Tensor, "tet pt vert=4"],
    bary_coords_grad: Float[Tensor, "tet vert=4 coord=3"],
) -> Float[Tensor, "tet pt *ch coord=3"]:
    bary_coords_shaped = rearrange(bary_coords, "tet pt vert -> tet pt vert 1")
    bary_coords_grad_shaped = rearrange(
        bary_coords_grad, "tet vert coord -> tet 1 vert coord"
    )

    # W_ijk = 2(λ_i ∇λ_jx∇λ_k + λ_j ∇λ_kx∇λ_i + λ_k ∇λ_ix∇ λ_j)
    local_tri_idx = enumerate_local_faces(
        splx_dim=3, face_dim=2, device=bary_coords.device
    )
    perm_i = local_tri_idx[:, 0]
    perm_j = local_tri_idx[:, 1]
    perm_k = local_tri_idx[:, 2]
    basis: Float[Tensor, "tet pt tri=4 coord=3"] = 2.0 * (
        bary_coords_shaped[:, :, perm_i]
        * torch.cross(
            bary_coords_grad_shaped[:, :, perm_j],
            bary_coords_grad_shaped[:, :, perm_k],
            dim=-1,
        )
        + bary_coords_shaped[:, :, perm_j]
        * torch.cross(
            bary_coords_grad_shaped[:, :, perm_k],
            bary_coords_grad_shaped[:, :, perm_i],
            dim=-1,
        )
        + bary_coords_shaped[:, :, perm_k]
        * torch.cross(
            bary_coords_grad_shaped[:, :, perm_i],
            bary_coords_grad_shaped[:, :, perm_j],
            dim=-1,
        )
    )

    # tet_tri_face_idx contains the index of 2-faces in the list of canonical
    # triangles of the tet mesh.
    cochain_2_at_tri_faces: Float[Tensor, "tet tri=4"] = cochain_2[tet_tri_idx]

    # If the triangles are not in their canonical orientation, then the corresponding
    # basis form needs a sign correction given by tet_edge_orientations.
    form_2 = einsum(
        basis,
        tet_tri_orientations,
        cochain_2_at_tri_faces,
        "tet pt tri coord, tet tri, tet tri ... -> tet pt ... coord",
    )

    return form_2


def _bary_whitney_tet_cochain_3(
    cochain_3: Float[Tensor, " tet *ch"],
    tet_signed_vols: Float[Tensor, " tet"],
    bary_coords: Float[Tensor, "tet pt vert=4"],
) -> Float[Tensor, "tet pt *ch coord=1"]:
    # There is only one basis form W_0123 = 1/vol; note that this basis
    # function is a constant of barycentric coordinates, which means that the
    # interpolated 3-forms will be constant on each tet.
    basis = 1.0 / tet_signed_vols

    # Note that no orientation sign correction is needed here since the top-level
    # simplices are stored as is rather than lex-sorted.
    form_3 = einsum(basis, cochain_3, "tet, tet ... -> tet ...")

    # Note that the bary_coords argument is only used here to determine the
    # number of sampled points.
    form_3_shaped = repeat(
        form_3, "tet ... -> tet pt ... coord", pt=bary_coords.size(-2), coord=1
    )

    return form_3_shaped


def _bary_whitney_tri(
    k: int,
    k_cochain: Float[Tensor, " splx *ch"],
    bary_coords: Float[Tensor, "tri pt vert"],
    mesh: SimplicialMesh,
) -> Float[Tensor, "tri pt *ch coord"]:
    if k in [1, 2]:
        _, bary_coords_grad = _tri_geometry.compute_bc_grads(
            vert_coords=mesh.vert_coords, tris=mesh.tris
        )

    match k:
        case 0:
            return _bary_whitney_tri_cochain_0(
                cochain_0=k_cochain, tris=mesh.tris, bary_coords=bary_coords
            )
        case 1:
            return _bary_whitney_tri_cochain_1(
                cochain_1=k_cochain,
                tri_edge_idx=mesh.edge_faces.idx,
                tri_edge_orientations=mesh.edge_faces.parity,
                bary_coords=bary_coords,
                bary_coords_grad=bary_coords_grad,
            )
        case 2:
            return _bary_whitney_tri_cochain_2(
                cochain_2=k_cochain,
                bary_coords=bary_coords,
                bary_coords_grad=bary_coords_grad,
            )
        case _:
            raise ValueError()


def _bary_whitney_tet(
    k: int,
    k_cochain: Float[Tensor, " splx *ch"],
    bary_coords: Float[Tensor, "tet pt vert"],
    mesh: SimplicialMesh,
) -> Float[Tensor, "tet pt *ch coord"]:
    if k in [1, 2]:
        tet_signed_vols, bary_coords_grad = _tet_geometry.compute_bc_grads(
            mesh.vert_coords, mesh.tets
        )
    elif k == 3:
        tet_signed_vols = _tet_geometry.compute_tet_signed_vols(
            mesh.vert_coords, mesh.tets
        )

    match k:
        case 0:
            return _bary_whitney_tet_cochain_0(
                cochain_0=k_cochain, tets=mesh.tets, bary_coords=bary_coords
            )
        case 1:
            return _bary_whitney_tet_cochain_1(
                cochain_1=k_cochain,
                tet_edge_idx=mesh.edge_faces.idx,
                tet_edge_orientations=mesh.edge_faces.parity,
                bary_coords=bary_coords,
                bary_coords_grad=bary_coords_grad,
            )
        case 2:
            return _bary_whitney_tet_cochain_2(
                cochain_2=k_cochain,
                tet_tri_idx=mesh.tri_faces.idx,
                tet_tri_orientations=mesh.tri_faces.parity,
                bary_coords=bary_coords,
                bary_coords_grad=bary_coords_grad,
            )
        case 3:
            return _bary_whitney_tet_cochain_3(
                cochain_3=k_cochain,
                tet_signed_vols=tet_signed_vols,
                bary_coords=bary_coords,
            )
        case _:
            raise ValueError()


def _bary_embed(
    m_splx: Integer[LongTensor, "m_splx m_vert"],
    k_splx_bary_coords: Float[Tensor, "k_splx pt k_vert"],
    k_faces_local_vert_idx: Integer[LongTensor, "k_face k_vert"],
    k_faces_global_idx: Integer[LongTensor, "m_splx k_face"],
    n_k_splx: int,
) -> Float[Tensor, "m_splx k_face pt m_vert"]:
    """
    Embed barycentric coordinates for a k-simplex onto the barycentric coordinates
    of the k-dimensional faces of a higher m-simplex. Note that the `k_splx`
    dimension of `k_splx_bary_coords` is allowed to be trivial.

    Example:

    Given a two-point quadrature rule on a 1-simplex
    ```
    k_splx_bary_coords = [
        [0.2, 0.8],
        [0.8, 0.2]
    ]
    ```
    and a 2-simplex with edges (represented by local vertex index)
    ```
    k_splx_local_vert_idx = [[0, 1], [0, 2], [1, 2]]
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
    n_m_splx, n_m_verts = m_splx.shape
    n_pts = k_splx_bary_coords.size(1)
    n_k_faces_per_m_splx = k_faces_local_vert_idx.size(0)

    # In case the first dimension of k_splx_bary_coords is trivial, inflate to the
    # correct shape.
    k_splx_bary_coords_shaped = k_splx_bary_coords.expand(
        n_k_splx, *k_splx_bary_coords.shape[1:]
    )

    # Scatter the barycentric coordinates defined on the canonical k-simplices
    # to the k-faces of the m-simplices.
    k_splx_bary_coords_scattered: Float[Tensor, "m_splx k_face pt k_vert"] = (
        k_splx_bary_coords_shaped[k_faces_global_idx]
    )

    # For each k-face of an m-simplex, identify the permutation required to reorder
    # the corresponding canonical k-simplex to match the k-face. For example,
    # For a 2-face [30, 2, 40], the permutation [1, 0, 2] is required to permute
    # the canonical 2-simplex [2, 30, 40] to [30, 2, 40].
    all_k_faces: Integer[LongTensor, "m_splx k_face k_vert"] = m_splx[
        :, k_faces_local_vert_idx
    ]

    # Note that the two argsort() here is required; the first argsort() computes
    # the permutation required to reorder the k-face to match the canonical
    # k-simplex, and the second argsort() computes the inverse of this mapping.
    k_face_perm_map = repeat(
        torch.argsort(
            torch.argsort(all_k_faces, dim=-1, descending=False),
            dim=-1,
            descending=False,
        ),
        "m_splx k_face k_vert -> m_splx k_face pt k_vert",
        pt=n_pts,
    )

    # Use the permutation map to reorder the last dimension of k_splx_bary_coords_scattered.
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
    k_splx_bary_coords_permuted = torch.gather(
        input=k_splx_bary_coords_scattered, dim=-1, index=k_face_perm_map
    )

    # Broadcast and embed the k-simplex barycentric coordinates into the m-simplices.
    k_faces_local_vert_idx_shaped = repeat(
        k_faces_local_vert_idx,
        "k_face k_vert -> m_splx k_face pt k_vert",
        m_splx=n_m_splx,
        pt=n_pts,
    )

    bary_coords_embedded = torch.zeros(
        n_m_splx,
        n_k_faces_per_m_splx,
        n_pts,
        n_m_verts,
        dtype=k_splx_bary_coords.dtype,
        device=k_splx_bary_coords.device,
    )

    bary_coords_embedded.scatter_(
        dim=-1, index=k_faces_local_vert_idx_shaped, src=k_splx_bary_coords_permuted
    )

    return bary_coords_embedded


def _barycentric_whitney_map_interior(
    k: int,
    k_cochain: Float[Tensor, " splx *ch"],
    bary_coords: Float[Tensor, "top_splx pt vert"],
    mesh: SimplicialMesh,
) -> Float[Tensor, "top_splx pt *ch coord"]:
    match mesh.dim:
        case 2:
            return _bary_whitney_tri(k, k_cochain, bary_coords, mesh)
        case 3:
            return _bary_whitney_tet(k, k_cochain, bary_coords, mesh)
        case _:
            raise ValueError()


def _barycentric_whitney_map_boundary(
    k: int,
    k_cochain: Float[Tensor, " k_splx *ch"],
    bary_coords: Float[Tensor, "k_splx pt vert"],
    mesh: SimplicialMesh,
    reduction: Literal["mean", "none"] = "none",
) -> Float[Tensor, "m_splx k_face pt *ch coord"] | Float[Tensor, "k_splx pt *ch coord"]:
    m = mesh.dim
    n_k_splx = mesh.splx[k].size(0)
    n_pts = bary_coords.size(-2)

    local_face_idx: Integer[LongTensor, "k_face k_vert"] = enumerate_local_faces(
        splx_dim=m, face_dim=k, device=mesh.device
    )

    # Find the global indices of all k-faces.
    # TODO: this is not ideal since we are repeating topo calculations
    all_faces = mesh.splx[m][:, local_face_idx]

    global_face_idx: Integer[LongTensor, "m_splx k_face"] = splx_search(
        key_splx=mesh.splx[k],
        query_splx=all_faces,
        sort_key_splx=False,
        sort_key_vert=False,
        sort_query_vert=True,
        method="lex_sort",
    )

    # Perform barycentric coordinate embedding and then compute the whitney
    # interpolation at the top simplex level.
    bary_coords_embedded: Float[Tensor, "m_splx k_face pt m_vert"] = _bary_embed(
        m_splx=mesh.splx[m],
        k_splx_bary_coords=bary_coords,
        k_faces_local_vert_idx=local_face_idx,
        k_faces_global_idx=global_face_idx,
        n_k_splx=n_k_splx,
    )

    k_forms = _barycentric_whitney_map_interior(
        k,
        k_cochain,
        rearrange(
            bary_coords_embedded,
            "m_splx k_face pt m_vert -> m_splx (k_face pt) m_vert",
        ),
        mesh,
    )

    match reduction:
        case "none":
            k_forms_shaped = rearrange(
                k_forms,
                "m_splx (k_face pt) ... coord -> m_splx k_face pt ... coord",
                pt=n_pts,
            )
            return k_forms_shaped

        # For each unique/canonical k-simplex in the mesh, compute the average
        # k-forms evaluated at the barycentric coordinates per top-level simplex.
        case "mean":
            k_forms_shaped = rearrange(
                k_forms,
                "m_splx (k_face pt) ... coord -> (m_splx k_face) pt ... coord",
                pt=n_pts,
            )

            global_face_idx_shaped: Integer[
                LongTensor, "m_splx*k_face pt *ch coord"
            ] = (
                global_face_idx.flatten()
                .view(-1, *[1] * (k_forms_shaped.ndim - 1))
                .expand_as(k_forms_shaped)
            )

            canon_k_forms: Float[Tensor, "k_splx pt *ch coord"] = torch.zeros(
                (n_k_splx, *k_forms_shaped.shape[1:]),
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
    k_cochain: Float[Tensor, " k_splx *ch"],
    bary_coords: Float[Tensor, "splx pt vert"],
    mesh: SimplicialMesh,
    mode: Literal["interior", "boundary"],
    boundary_reduction: Literal["mean", "none"] = "none",
) -> Tensor:
    """
    This function implements an "element-local" version of the Whitney map for
    interpolating discrete k-cochains using Whitney basis functions of the lowest
    order.

    In the `interior` mode, the function maps the k-cochains to k-forms interpolated
    at local barycentric coordinates across all m-simplices, where m is the dimension
    of the mesh, and the returned tensor is of shape (`m_splx`, `pt`, `*ch`, `coord`);
    in the `boundary` mode, the function maps the k-cochains to k-forms interpolated
    at local barycentric coordinates across the k-simplices. If k = m, the `interior`
    mode is used regardless of the `mode` argument. Currently, interpolation of
    k-cochains on l-simplices in an m-dimensional mesh, where k < l < m, is not
    supported.

    The `boundary_reduction` argument modifies the output of the `boundary` mode,
    if set to `'none'`, then the returned tensor is of shape (`m_splx`, `k_face`,
    `pt`, `*ch`, `coord`); if set to `'mean'`, then the returned tensor is of shape
    (`k_splx`, `pt`, `*ch`, `coord`), where the interpolated k-forms are averaged
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
    first `splx` dimension, in which case the k-cochain is interpolated at the
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

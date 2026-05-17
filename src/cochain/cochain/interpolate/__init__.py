__all__ = ["barycentric_whitney_map"]

from typing import Literal

import torch
from einops import rearrange, repeat
from jaxtyping import Float, Integer
from torch import Tensor

from ...complex import SimplicialMesh
from ...utils.faces import enumerate_local_faces
from ._bary_whitney_tet import bary_whitney_tet
from ._bary_whitney_tri import bary_whitney_tri


def _bary_embed(
    m_splx: Integer[Tensor, "m_splx m_vert"],
    k_splx_bary_coords: Float[Tensor, "k_splx pt k_vert"],
    k_faces_local_vert_idx: Integer[Tensor, "k_face k_vert"],
    k_faces_global_idx: Integer[Tensor, "m_splx k_face"],
    n_k_splx: int,
) -> Float[Tensor, "m_splx k_face pt m_vert"]:
    """
    Embed the barycentric coordinates onto the faces of higher-order simplices.

    Parameters
    ----------
    m_splx : [m_splx, m_vert]
        A list of m-simplices whose k-faces are the embedding targets.
    k_splx_bary_coords : [k_splx, pt, k_vert]
        A list of barycentric coordinates defined on the canonical k-simplices.
        If the `k_splx` dimension is trivial, then it is assumed that the same
        barycentric coordinates are defined on all canonical k-simplices.
    k_faces_local_vert_idx : [k_face, k_vert]
        The k-face definitions of the m-simplices in terms of local vertex indices.
        This defines the set of k-faces per m-simplex onto which the barycentric
        coordinates are embedded. The local vertex indices for each k-face should
        be sorted in ascending order.
    k_faces_global_idx : [m_splx, k_face]
        The indices of the k-faces on the list of canonical k-simplices.
    n_k_splx
        The number of canonical k-simplices.

    Returns
    -------
    [m_splx, k_face, pt, m_vert]
        The embedded barycentric coordinates.

    Notes
    -----
    There are some subtleties with regard to how this function handles the
    vertex ordering along the `m_vert` dimension in the returned tensor.

    Consider a simple example where a canonical 1-simplex `[28, 29]` is the face
    of two 2-simplices `[27, 28, 29]` and `[30, 29, 28]` in a tri mesh. If one were
    to embed the barycentric coordinate `[0.2, 0.8]` to the local 1-face/edge `12`,
    then the embedded barycentric coordinate would be `[0.0, 0.2, 0.8]` for both
    2-simplices; however, the embedded coordinates (in the real space) are actually
    `0.2*x28 + 0.8*x29` for the first 2-simplex but `0.8*x28 + 0.2*x29` for the
    second 2-simplex. While this is the correct embedding locally within each
    2-simplex, it becomes a problem later in `_barycentric_whitney_map_boundary()`
    when the 1-forms interpolated on the 1-faces are used to produce an averaged
    1-form per canonical 1-simplex, where this mismatch causes the 1-form evaluated
    at the wrong points to be averaged together.

    To prevent this problem, this function reorders the k-face vertices to conform
    to the vertex ordering in the corresponding canonical k-simplex prior to
    performing the barycentric coordinate embedding. Continuing with the example
    above, this would mean that the embdeed barycentric coordinates for edge `12`
    is `[0.0, 0.2, 0.8]` for the first 2-simplex, but `[0.0, 0.8, 0.2]` for the second
    2-simplex.

    Note that this permutation correction is necessary (1) on top of the k-form
    sign corrections, (2) even if the set of points is invariant to vertex permutation,
    and (3) even if the Whitney bases used are of the lowest order.

    Examples
    --------
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
    """
    # Collect shape/size information.
    n_m_splx, n_m_verts = m_splx.shape
    _, n_pts, n_k_verts = k_splx_bary_coords.shape
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

    # Compute the permutation required to reorder the k-faces to match the
    # canonical k-simplices.
    all_k_faces: Integer[Tensor, "m_splx k_face k_vert"] = m_splx[
        :, k_faces_local_vert_idx
    ]
    face_to_splx_map = torch.argsort(all_k_faces, dim=-1, descending=False)

    # Compute the inverse of this permutation using scatter(); this is equivalent
    # to splx_to_face_map = torch.argsort(face_to_splx_map, dim=-1).
    splx_to_face_map = torch.empty_like(face_to_splx_map)
    k_verts = repeat(
        torch.arange(
            n_k_verts,
            dtype=face_to_splx_map.dtype,
            device=face_to_splx_map.device,
        ),
        "k_vert -> m_splx k_face k_vert",
        m_splx=n_m_splx,
        k_face=n_k_faces_per_m_splx,
    )
    # k_splx_to_face_map[m][k][k_face_to_splx_map[m][k][v]]=k_verts[m][k][v]
    splx_to_face_map.scatter_(
        dim=-1,
        index=face_to_splx_map,
        src=k_verts,
    )
    # Expand in preparation for gather().
    k_face_perm_map = repeat(
        splx_to_face_map,
        "m_splx k_face k_vert -> m_splx k_face pt k_vert",
        pt=n_pts,
    )

    # Use k_face_perm_map to reorder the last dimension of k_splx_bary_coords_scattered
    # so that the vertex ordering in the k-faces match that of the corresponding
    # canonical k-simplices.
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

    # bary_coords_embedded[m][k][p][k_faces_local_vert_idx_shaped[m][k][p][v]] =
    # k_splx_bary_coords_permuted[m][k][p][v]
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
            return bary_whitney_tri(k, k_cochain, bary_coords, mesh)
        case 3:
            return bary_whitney_tet(k, k_cochain, bary_coords, mesh)
        case _:
            raise ValueError("Only tri and tet meshes are supported.")


def _barycentric_whitney_map_boundary(
    k: int,
    k_cochain: Float[Tensor, " k_splx *ch"],
    bary_coords: Float[Tensor, "k_splx pt vert"],
    mesh: SimplicialMesh,
    reduction: Literal["mean", "none"] = "none",
) -> Float[Tensor, "m_splx k_face pt *ch coord"] | Float[Tensor, "k_splx pt *ch coord"]:
    m = mesh.dim
    n_k_splx = mesh.n_splx[k]
    n_pts = bary_coords.size(-2)

    local_face_idx: Integer[Tensor, "k_face k_vert"] = enumerate_local_faces(
        splx_dim=m, face_dim=k, device=mesh.device
    )
    global_face_idx = mesh.faces[k].idx

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
        k=k,
        k_cochain=k_cochain,
        bary_coords=rearrange(
            bary_coords_embedded,
            "m_splx k_face pt m_vert -> m_splx (k_face pt) m_vert",
        ),
        mesh=mesh,
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

            global_face_idx_shaped: Integer[Tensor, "m_splx*k_face pt *ch coord"] = (
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
    Interpolate discrete k-cochains using Whitney basis functions of the lowest order.

    Parameters
    ----------
    k
        The order of the input `k_cochain`.
    k_cochain : [k_splx, *ch]
        The k-cochain to interpolate. The input is allowed to have an arbitrary
        number of trailing channel/batch dimensions.
    bary_coords : [splx, pt, vert]
        The barycentric coordinates at which to interpolate the k-cochain. The
        meaning of the first `splx` dimension depends on the `mode` argument. In
        the `interior` mode, the barycentric coordinates should be defined over
        the top-level simplices; in the `boundary` mode, the barycentric coordinates
        should be defined over the canonical k-simplices. The `splx` dimension
        can be trivial, in which case the k-cochain is interpolated at the same
        fixed local barycentric coordinates over all target simplices.
    mesh
        A simplicial mesh.
    mode
        The interpolation mode. In the `interior` mode, the function maps the
        k-cochains to k-forms interpolated at the local barycentric coordinates
        defined over the top-level simplices; in the `boundary` mode, the function
        maps the k-cochains to k-forms interpolated at the local barycentric
        coordinates defined over the the canonical k-simplices. Currently,
        interpolation of k-cochains on l-simplices in an m-dimensional mesh, where
        k < l < m, is not supported. If k is equal to the dimension of the mesh,
        then the `interior` mode is used regardless of the `mode` argument.
    boundary_reduction
        Whether to average the interpolated k-forms over all k-faces of the
        top-level simplices corresponding to the same canonical k-simplices.
        Only relevant in the `boundary` mode.

    Returns
    -------
    The interpolated k-form. The shape of this tensor depends on the `mode`
    and `boundary_reduction` arguments. If `m` is the dimension of the mesh, then

    * In the `interior` mode, the function returns a tensor of shape
      `[m_splx, pt, *ch, coord]`.
    * In the `boundary` mode, if `boundary_reduction` is set to `'none'`, then
      the function returns a tensor of shape `[m_splx, k_face, pt, *ch, coord]`;
      if `boundary_reduction` is set to `'mean'`, then the function returns a
      tensor of shape `[k_splx, pt, *ch, coord]`.

    Notes
    -----
    The Whitney map inplemented in this function can be described as "element-local",
    which is to be contrasted with global spatial interpolation (i.e., evaluation
    of the k-form at arbitrary cartesian coordinates on the mesh.)

    For the `boundary` mode, the `pt` dimension is always ordered relative to the
    canonical simplices. For example, consider a two-point quadrature on 1-simplices
    consisting of barycentric coordinates `(0.2, 0.8)` and `(0.8, 0.2)`; if a canonical
    1-simplex `[28, 29]` is the face of two 2-simplices `[27, 28, 29]` and `[30, 29, 28]`,
    then the `pt` dimension for both faces will refer to the point closer to vertex
    `29` first, and the point closer to vertex `28` second, regardless of the local
    vertex ordering of the 1-face in the 2-simplices. Note that, for the `interior`
    mode, the values across the `pt` dimension will be constant since the interpolated
    m-forms (i.e., volume forms) on the m-simplices are always piecewise constant.

    The `boundary_reduction` argument should be used with care in the `boundary`
    mode. Note that k-forms interpolated using Whitney bases have discontinuous,
    multi-valued, unconstrained components (e.g., for 1-forms, the tangential
    components are continuous but the normal components jump; for 2-forms, the
    normal components are continuous but the tangential components jump between
    higher-order simplices); as such, the `'mean'` reduction creates distortions
    and should only be used for downstream applications when the unconstrained
    components are irrelevant (e.g., for de Rham map).
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
            raise ValueError(f"Unknown mode argument: {mode}.")

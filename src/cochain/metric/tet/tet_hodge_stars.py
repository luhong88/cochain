__all__ = ["star_0", "star_1", "star_2", "star_3"]

import torch
from einops import reduce, repeat
from jaxtyping import Float, Integer
from torch import LongTensor, Tensor

from ...complex import SimplicialMesh
from ...sparse.decoupled_tensor import DiagDecoupledTensor
from ...utils.faces import enumerate_local_faces
from ..tri._tri_geometry import compute_tri_areas
from ._tet_geometry import compute_tet_signed_vols
from .tet_masses import mass_3


def star_3(tet_mesh: SimplicialMesh) -> Float[DiagDecoupledTensor, "tri tri"]:
    """
    Compute the discrete Hodge star operator on 3-forms for a tet mesh.

    The barycentric Hodge 3-star operator maps the 3-simplices (tets) on a mesh
    to their barycentric dual 0-cells (vertices). This function computes the ratio
    of the volume of the dual 0-cells (which is 1 by convention) to the volume of
    the primal tets, and this ratio tensor forms the diagonal of the 3-star tensor.
    This matrix is also known as the diagonal 3-form mass matrix.

    Note that this function is equivalent to `mass_3()`.

    Parameters
    ----------
    tet_mesh
        A tet mesh.

    Returns
    -------
    [tet, tet]
        The Hodge 3-star matrix.
    """
    return mass_3(tet_mesh)


def star_2(tet_mesh: SimplicialMesh) -> Float[DiagDecoupledTensor, "tri tri"]:
    """
    Compute the discrete, barycentric Hodge star operator on 2-forms for a tet mesh.

    The barycentric Hodge 2-star operator maps the 2-simplices (triangles) on a
    mesh to their barycentric dual 1-cells. This function computes the ratio of
    the length of the dual 1-cells to the area of the primal triangles, and this
    ratio tensor forms the diagonal of the 2-star tensor. This matrix is also known
    as the diagonal 2-form mass matrix.

    Parameters
    ----------
    tet_mesh
        A tet mesh.

    Returns
    -------
    [tri, tri]
        The Hodge 2-star matrix.
    """
    tet_vert_coords: Float[Tensor, "tet 4 3"] = tet_mesh.vert_coords[tet_mesh.tets]

    # For each tet, find all of its tri faces, and for each such tet-tri pair,
    # find the dual edge that connects the barycenters of the tet and the tri.
    tet_bcs: Float[Tensor, "tet 1 3"] = reduce(
        tet_vert_coords, "tri vert coord -> tri 1 coord", "mean"
    )

    all_tri_faces = enumerate_local_faces(
        splx_dim=3, face_dim=2, device=tet_mesh.device
    )
    tet_tri_face_bcs: Float[Tensor, "tet 4 3"] = torch.mean(
        tet_vert_coords[:, all_tri_faces], dim=-2
    )

    dual_edges = tet_bcs - tet_tri_face_bcs
    dual_edge_lens: Float[Tensor, "tet 4"] = torch.linalg.norm(dual_edges, dim=-1)

    # For each tri, find all tet containing the tri as a face, and sum together
    # the tet-tri pair dual edge lengths.
    all_canon_tris_idx: Integer[LongTensor, "tet 4"] = tet_mesh.tri_faces.idx

    dual_edge_lens_agg = torch.zeros(
        tet_mesh.n_tris,
        dtype=tet_mesh.dtype,
        device=tet_mesh.device,
    )
    dual_edge_lens_agg.scatter_add_(
        dim=0,
        index=all_canon_tris_idx.flatten(),
        src=dual_edge_lens.flatten(),
    )

    # Divide the dual edge length sum by the tri area to get the Hodge 2-star.
    tri_areas = compute_tri_areas(tet_mesh.vert_coords, tet_mesh.tris)
    diag_vals = dual_edge_lens_agg / tri_areas

    return DiagDecoupledTensor.from_tensor(diag_vals)


def star_1(tet_mesh: SimplicialMesh) -> Float[DiagDecoupledTensor, "edge edge"]:
    """
    Compute the discrete, barycentric Hodge star operator on 1-forms for a tet mesh.

    The Hodge 1-star operator maps the 1-simplices (edges) in a mesh to the
    dual 2-cells. This function computes the ratio of the dual 2-cells to the
    primal edges, and this ratio tensor forms the diagonal of the 1-star tensor.

    Parameters
    ----------
    tet_mesh
        A tet mesh.

    Returns
    -------
    [edge, edge]
        The Hodge 1-star matrix.
    """
    tet_vert_coords: Float[Tensor, "tet 4 3"] = tet_mesh.vert_coords[tet_mesh.tets]

    # For each tet, find its barycenter and the barycenters of its tri faces and
    # edge faces.
    tet_bcs: Float[Tensor, "tet 1 3"] = reduce(
        tet_vert_coords, "tet vert coord -> tet 1 coord", "mean"
    )

    all_tri_faces = enumerate_local_faces(
        splx_dim=3, face_dim=2, device=tet_mesh.device
    )
    tet_tri_face_bcs = reduce(
        tet_vert_coords[:, all_tri_faces],
        "tet tri vert coord -> tet tri coord",
        "mean",
    )

    all_edge_faces = enumerate_local_faces(
        splx_dim=3, face_dim=1, device=tet_mesh.device
    )
    tet_edge_face_bcs = reduce(
        tet_vert_coords[:, all_edge_faces],
        "tet edge vert coord -> tet edge coord",
        "mean",
    )

    # For each tet and each of its edge, find the sum of the areas of the two
    # triangles formed by the edge barycenter, the tet barycenter, and the tri
    # barycenter of one of the two tris sharing the edge as a face.
    #
    # edge tri_1 tri_1_idx tri_2 tri_2_idx
    # ------------------------------------
    # ij   ijk   0         ijl   1
    # ik   ijk   0         ikl   2
    # il   ijl   1         ikl   2
    # jk   ijk   0         jkl   3
    # jl   ijl   1         jkl   3
    # kl   ikl   2         jkl   3
    tet_tri_vec = tet_bcs - tet_tri_face_bcs
    tet_edge_vec = tet_bcs - tet_edge_face_bcs

    edge_coface_1 = [0, 0, 1, 0, 1, 2]
    edge_coface_2 = [1, 2, 2, 3, 3, 3]

    bc_tris_areas = torch.zeros(
        (tet_mesh.n_tets, 6), dtype=tet_mesh.dtype, device=tet_mesh.device
    )
    bc_tris_areas.add_(
        torch.linalg.norm(
            torch.cross(tet_edge_vec, tet_tri_vec[:, edge_coface_1], dim=-1),
            dim=-1,
        )
        / 2.0
    )
    bc_tris_areas.add_(
        torch.linalg.norm(
            torch.cross(tet_edge_vec, tet_tri_vec[:, edge_coface_2], dim=-1),
            dim=-1,
        )
        / 2.0
    )

    # For each edge, find all tet containing the edge as a face, and sum together
    # the subareas of the two barycentric triangles.
    all_canon_edges_idx = tet_mesh.edge_faces.idx

    dual_area_agg = torch.zeros(
        tet_mesh.n_edges,
        dtype=tet_mesh.dtype,
        device=tet_mesh.device,
    )
    dual_area_agg.scatter_add_(
        dim=0,
        index=all_canon_edges_idx.flatten(),
        src=bc_tris_areas.flatten(),
    )

    # Divide the dual area sums by the primal edge lengths to get the Hodge 1-star.
    edge_lens = torch.linalg.norm(
        tet_mesh.vert_coords[tet_mesh.edges[:, 1]]
        - tet_mesh.vert_coords[tet_mesh.edges[:, 0]],
        dim=-1,
    )

    diag_vals = dual_area_agg / edge_lens

    return DiagDecoupledTensor.from_tensor(diag_vals)


def star_0(tet_mesh: SimplicialMesh) -> Float[DiagDecoupledTensor, "vert vert"]:
    """
    Compute the discrete, barycentric Hodge star operator on 0-forms for a tet mesh.

    The Hodge 0-star operator maps the 0-simplices (vertices) in a mesh to their
    barycentric dual 3-cells. This function computes the ratio of the volume of the
    dual 3-cells to the "volume" of the vertices (which is 1 by convention), and
    this volume ratio tensor forms the diagonal of the 0-star tensor. This matrix
    is also known as the diagonal 0-form mass matrix.

    The barycentric dual volume for each vertex is the sum of 1/4 of the volumes
    of all tets that share the vertex as a face.

    Parameters
    ----------
    tet_mesh
        A tet mesh.

    Returns
    -------
    [vert, vert]
        The Hodge 0-star matrix.
    """
    tet_vol = torch.abs(compute_tet_signed_vols(tet_mesh.vert_coords, tet_mesh.tets))

    diag = torch.zeros(tet_mesh.n_verts, dtype=tet_mesh.dtype, device=tet_mesh.device)
    diag.scatter_add_(
        dim=0,
        index=tet_mesh.tets.flatten(),
        src=repeat(tet_vol / 4.0, "tet -> (tet vert)", vert=4),
    )

    return DiagDecoupledTensor.from_tensor(diag)

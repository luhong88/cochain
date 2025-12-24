import torch as t
from jaxtyping import Float, Integer

from ...complex import SimplicialComplex
from ...sparse.operators import DiagOperator
from ..tri.tri_geometry import _tri_areas
from .tet_masses import mass_0, mass_3


def star_3(tet_mesh: SimplicialComplex) -> Float[DiagOperator, "tet tet"]:
    """
    Compute the Hodge 3-star, which is the inverse of the mass-3 matrix.
    """
    return mass_3(tet_mesh).inv


def star_2(tet_mesh: SimplicialComplex) -> Float[DiagOperator, "tri tri"]:
    """
    Compute the barycentric Hodge 2-star operator.
    """
    vert_coords: Float[t.Tensor, "vert 3"] = tet_mesh.vert_coords
    tets: Integer[t.LongTensor, "tet 4"] = tet_mesh.tets
    tris: Integer[t.LongTensor, "tri 3"] = tet_mesh.tris
    tet_vert_coords: Float[t.Tensor, "tet 4 3"] = vert_coords[tets]

    i, j, k, l = 0, 1, 2, 3

    # For each tet, find all of its tri faces, and for each such tet-tri pair,
    # find the dual edge that connects the barycenters of the tet and the tri.
    tet_barys: Float[t.Tensor, "tet 1 3"] = t.mean(
        tet_vert_coords, dim=-2, keepdim=True
    )
    tet_tri_face_barys: Float[t.Tensor, "tet 4 3"] = t.mean(
        tet_vert_coords[:, [[j, k, l], [i, l, k], [i, j, l], [i, k, j]]], dim=-2
    )

    dual_edges = tet_barys - tet_tri_face_barys
    dual_edge_lens: Float[t.Tensor, "tet 4"] = t.linalg.norm(dual_edges, dim=-1)

    # For each tri, find all tet containing the tri as a face, and sum together
    # the tet-tri pair dual edge lengths.
    all_canon_tris_idx: Integer[t.LongTensor, "tet 4"] = tet_mesh.tet_tri_idx

    diag = t.zeros(
        tet_mesh.n_tris,
        dtype=vert_coords.dtype,
        device=vert_coords.device,
    )
    diag.scatter_add_(
        dim=0,
        index=all_canon_tris_idx.flatten(),
        src=dual_edge_lens.flatten(),
    )

    # Divide the dual edge length sum by the tri area to get the Hodge 2-star.
    tri_areas = _tri_areas(vert_coords, tris)
    diag.divide_(tri_areas)

    return DiagOperator.from_tensor(diag)


def star_1(tet_mesh: SimplicialComplex) -> Float[DiagOperator, "edge edge"]:
    """
    Compute the barycentric Hodge 1-star operator.
    """
    vert_coords: Float[t.Tensor, "vert 3"] = tet_mesh.vert_coords
    tets: Integer[t.LongTensor, "tet 4"] = tet_mesh.tets
    edges: Integer[t.LongTensor, "edge 2"] = tet_mesh.edges
    tet_vert_coords: Float[t.Tensor, "tet 4 3"] = vert_coords[tets]

    i, j, k, l = 0, 1, 2, 3

    # For each tet, find its barycenter and the barycenters of its tri faces and
    # edge faces.
    tet_barys: Float[t.Tensor, "tet 1 3"] = t.mean(
        tet_vert_coords, dim=-2, keepdim=True
    )
    tet_tri_face_barys: Float[t.Tensor, "tet 4 3"] = t.mean(
        tet_vert_coords[:, [[j, k, l], [i, l, k], [i, j, l], [i, k, j]]], dim=-2
    )
    tet_edge_face_barys: Float[t.Tensor, "tet 6 3"] = t.mean(
        tet_vert_coords[:, [[i, j], [i, k], [j, k], [j, l], [k, l], [i, l]]],
        dim=-2,
    )

    # For each tet and each of its edge, find the sum of the areas of the two
    # triangles formed by the edge barycenter, the tet barycenter, and the tri
    # barycenter of one of the two tris containing the edge.
    tet_tri_vec = tet_barys - tet_tri_face_barys
    tet_edge_vec = tet_barys - tet_edge_face_barys

    subareas: Float[t.Tensor, "tet 6"] = (
        t.linalg.norm(
            t.cross(tet_edge_vec, tet_tri_vec[:, [2, 1, 0, 0, 0, 1]], dim=-1), dim=-1
        )
        + t.linalg.norm(
            t.cross(tet_edge_vec, tet_tri_vec[:, [3, 3, 3, 2, 1, 2]], dim=-1), dim=-1
        )
    ) / 2.0

    # For each edge, find all tet containing the edge as a face, and sum together
    # the subareas of the two barycentric triangles.
    all_canon_edges_idx = tet_mesh.tet_edge_idx

    diag = t.zeros(
        tet_mesh.n_edges,
        dtype=vert_coords.dtype,
        device=vert_coords.device,
    )
    diag.scatter_add_(
        dim=0,
        index=all_canon_edges_idx.flatten(),
        src=subareas.flatten(),
    )

    # Divide the dual area sums by the edge lengths to get the Hodge 1-star.
    edge_lens = t.linalg.norm(
        vert_coords[edges[:, 1]] - vert_coords[edges[:, 0]],
        dim=-1,
    )
    diag.divide_(edge_lens)

    return DiagOperator.from_tensor(diag)


star_0 = mass_0

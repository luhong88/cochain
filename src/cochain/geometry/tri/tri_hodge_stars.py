from typing import Literal

import torch as t
from jaxtyping import Float, Integer

from ...complex import SimplicialComplex
from .tri_geometry import _tri_areas
from .tri_stiffness import _cotan_weights


def star_2(tri_mesh: SimplicialComplex) -> Float[t.Tensor, " tri"]:
    """
    The Hodge 2-star operator maps the 2-simplices (triangles) in a mesh to their
    dual 0-cells. This function computes the ratio of the "volume" of the dual 0-cells
    (which is 1 by convention) to the area of the primal triangles. The returned tensor
    forms the diagonal of the 2-star tensor.
    """
    return 1.0 / _tri_areas(tri_mesh.vert_coords, tri_mesh.tris)


def _star_1_circumcentric(tri_mesh: SimplicialComplex) -> Float[t.Tensor, " edge"]:
    """
    The Hodge 1-star operator maps the 1-simplices (edges) in a mesh to the
    circumcentric dual 1-cells. This function computes the length ratio of the dual
    1-cells to the primal edges, which is given by the cotan formula. The returned
    tensor forms the diagonal of the 1-star tensor.
    """
    vert_coords: Float[t.Tensor, "vert 3"] = tri_mesh.vert_coords
    tris: Integer[t.LongTensor, "tri 3"] = tri_mesh.tris

    n_verts = tri_mesh.n_verts

    # The cotan weights matrix (i.e., off-diagonal elements of the stiffness matrix)
    # already contains the desired values; i.e., S_ij = -0.5*sum_k[cot_k] for all
    # k that forms a triangle with i and j.
    weights: Float[t.Tensor, "vert vert"] = _cotan_weights(vert_coords, tris, n_verts)

    # For the weights matrix W_ij in COO format, the first row of its index tensor
    # is for the i dimension, and the second row is for the j dimension. We flatten
    # the two dimensions into a 1D index with (i, j) -> i*n_verts + j.
    all_idx = weights.indices()
    all_idx_flat = all_idx[0] * n_verts + all_idx[1]

    # Similarly, we convert the canonical edges represented by a vertex-indexing tuple
    # (i, j) into a flat index with the same formulae.
    edges: Integer[t.Tensor, "edge 2"] = tri_mesh.edges
    edge_idx_flat = edges[:, 0] * n_verts + edges[:, 1]

    # Identify the location of the canonical edge ij in the sparse W_ij indices,
    # using the flattened indices, and use the location to extract the cotan values.
    subset_idx = t.searchsorted(all_idx_flat, edge_idx_flat)
    subset_vals = weights.values()[subset_idx]

    return -subset_vals  # note the negative sign to get dual edge lengths


def _star_1_barycentric(tri_mesh: SimplicialComplex) -> Float[t.Tensor, " edge"]:
    """
    Compute the barycentric Hodge 1-star operator.
    """
    vert_coords: Float[t.Tensor, "vert 3"] = tri_mesh.vert_coords
    tris: Integer[t.LongTensor, "tri 3"] = tri_mesh.tris
    edges: Integer[t.LongTensor, "edge 2"] = tri_mesh.edges
    tri_vert_coords: Float[t.Tensor, "tet 3 3"] = vert_coords[tris]

    i, j, k = 0, 1, 2

    # For each tri, find its barycenter and the barycenters of its edge faces,
    # as well as the dual edges that connect the barycenters
    tri_barycenters: Float[t.Tensor, "tri 1 3"] = t.mean(
        tri_vert_coords, dim=-2, keepdim=True
    )
    tri_edge_face_barycenters: Float[t.Tensor, "tet 3 3"] = t.mean(
        tri_vert_coords[:, [[i, j], [i, k], [j, k]]],
        dim=-2,
    )

    dual_edges = tri_barycenters - tri_edge_face_barycenters
    dual_edge_lens: Float[t.Tensor, "tet 4"] = t.linalg.norm(dual_edges, dim=-1)

    # For each edge, find all tri containing the edge as a face, and sum together
    # the tri-edge pair dual edge lengths.
    all_canon_edges_idx = tri_mesh.tri_edge_idx

    diag = t.zeros(
        tri_mesh.n_edges,
        dtype=vert_coords.dtype,
        device=vert_coords.device,
    )
    diag.scatter_add_(
        dim=0,
        index=all_canon_edges_idx.flatten(),
        src=dual_edge_lens.flatten(),
    )

    # Divide the dual edge length sum by the primal edge length to get the Hodge 1-star.
    edge_lens = t.linalg.norm(
        vert_coords[edges[:, 1]] - vert_coords[edges[:, 0]],
        dim=-1,
    )
    diag.divide_(edge_lens)

    return diag


def star_1(
    tri_mesh: SimplicialComplex,
    dual_complex: Literal["circumcentric", "barycentric"] = "barycentric",
):
    match dual_complex:
        case "circumcentric":
            return _star_1_circumcentric(tri_mesh)
        case "barycentric":
            return _star_1_barycentric(tri_mesh)
        case _:
            raise ValueError()


def star_0(tri_mesh: SimplicialComplex) -> Float[t.Tensor, " vert"]:
    """
    The Hodge 0-star operator maps the 0-simplices (vertices) in a mesh to their
    barycentric dual 2-cells. This function computes the ratio of the area of the
    dual 2-cells to the "size" of the vertices (which is 1 by convention). The
    returned tensor forms the diagonal of the 0-star tensor.

    The barycentric dual area for each vertex is the sum of 1/3 of the areas of
    all triangles that share the vertex as a face.
    """
    tri_area = _tri_areas(tri_mesh.vert_coords, tri_mesh.tris)

    diag = t.zeros(
        tri_mesh.n_verts,
        dtype=tri_mesh.vert_coords.dtype,
        device=tri_mesh.vert_coords.device,
    )
    diag.scatter_add_(
        dim=0,
        index=tri_mesh.tris.flatten(),
        src=t.repeat_interleave(tri_area / 3.0, 3),
    )

    return diag

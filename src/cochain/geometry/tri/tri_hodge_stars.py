from typing import Literal

import torch
from jaxtyping import Float, Integer
from torch import LongTensor, Tensor

from ...complex import SimplicialMesh
from ...sparse.decoupled_tensor import DiagDecoupledTensor
from ...utils.faces import enumerate_local_faces
from ...utils.search import splx_search
from ._tri_geometry import compute_tri_areas
from .tri_stiffness import compute_cotan_weights

__all__ = ["star_0", "star_1", "star_2"]


def star_2(tri_mesh: SimplicialMesh) -> Float[DiagDecoupledTensor, "tri tri"]:
    """
    Compute the discrete Hodge star operator on 2-forms for a tri mesh.

    The Hodge 2-star operator maps the 2-simplices (triangles) on a mesh to their
    dual 0-cells (points). This function computes the ratio of the "area" of the
    dual 0-cells (which is 1 by convention) to the area of the primal triangles,
    and this area ratio tensor forms the diagonal of the 2-star tensor. This matrix
    is also known as the diagonal 2-form mass matrix.

    Parameters
    ----------
    tri_mesh
        A tri mesh.

    Returns
    -------
    (tri, tri)
        The Hodge 2-star matrix.
    """
    return DiagDecoupledTensor(
        1.0 / compute_tri_areas(tri_mesh.vert_coords, tri_mesh.tris)
    )


def _star_1_circumcentric(
    tri_mesh: SimplicialMesh,
) -> Float[DiagDecoupledTensor, "edge edge"]:
    vert_coords: Float[Tensor, "global_vert coord=3"] = tri_mesh.vert_coords
    tris: Integer[LongTensor, "tri local_vert=3"] = tri_mesh.tris

    # The cotan weights matrix (i.e., off-diagonal elements of the stiffness matrix)
    # already contains the desired values; i.e., S_ij = -0.5*sum_k[cot_k] for all
    # vertices k such that ijk forms a triangle.
    weights: Float[Tensor, "global_vert global_vert"] = compute_cotan_weights(
        vert_coords, tris
    )

    # Identify the location of the canonical edge ij in the sparse W_ij indices,
    # and use the location to extract the cotan values.
    subset_idx = splx_search(
        key_splx=weights.indices().T,
        query_splx=tri_mesh.edges,
        sort_key_splx=False,
        sort_key_vert=False,
        sort_query_vert=False,
    )
    subset_vals = weights.values()[subset_idx]

    return DiagDecoupledTensor(
        -subset_vals
    )  # note the negative sign to get dual edge lengths


def _star_1_barycentric(
    tri_mesh: SimplicialMesh,
) -> Float[DiagDecoupledTensor, "edge edge"]:
    vert_coords: Float[Tensor, "vert 3"] = tri_mesh.vert_coords
    tris: Integer[LongTensor, "tri 3"] = tri_mesh.tris
    edges: Integer[LongTensor, "edge 2"] = tri_mesh.edges
    tri_vert_coords: Float[Tensor, "tri 3 3"] = vert_coords[tris]

    # For each tri, find its barycenter and the barycenters of its edge faces,
    # as well as the dual edges that connect the barycenters
    tri_barys: Float[Tensor, "tri 1 3"] = torch.mean(
        tri_vert_coords, dim=-2, keepdim=True
    )

    all_edge_faces = enumerate_local_faces(
        splx_dim=2, face_dim=1, device=tri_mesh.device
    )
    tri_edge_face_barys: Float[Tensor, "tri 3 3"] = torch.mean(
        tri_vert_coords[:, all_edge_faces],
        dim=-2,
    )

    dual_edges = tri_barys - tri_edge_face_barys
    dual_edge_lens: Float[Tensor, "tri 4"] = torch.linalg.norm(dual_edges, dim=-1)

    # For each edge, find all tri containing the edge as a face, and sum together
    # the tri-edge pair dual edge lengths.
    all_canon_edges_idx = tri_mesh.edge_faces.idx

    diag = torch.zeros(
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
    edge_lens = torch.linalg.norm(
        vert_coords[edges[:, 1]] - vert_coords[edges[:, 0]],
        dim=-1,
    )
    diag.divide_(edge_lens)

    return DiagDecoupledTensor(diag)


def star_1(
    tri_mesh: SimplicialMesh,
    dual_complex: Literal["circumcentric", "barycentric"] = "barycentric",
) -> Float[DiagDecoupledTensor, "edge edge"]:
    """
    Compute the discrete Hodge star operator on 1-forms for a tri mesh.

    The Hodge 1-star operator maps the 1-simplices (edges) in a mesh to the
    dual 1-cells (edges). This function computes the length ratio of the dual
    1-cells to the primal edges, and this length ratio tensor forms the diagonal
    of the 1-star tensor.

    Parameters
    ----------
    tri_mesh
        A tri mesh.
    dual_complex
        The type of dual mesh used to compute the operator. See the Note section
        for more details.

    Returns
    -------
    (edge, edge)
        The Hodge 1-star matrix.

    Note
    ----
    When the `dual_complex` is "circumcentric", the dual complex is (implicitly)
    constructed by placing the 0-cells at the circumcenters of the primal triangles;
    when the `dual_complex` is "barycentric", the 0-cells are placed at the barycenters
    of the primal triangles. In general, the circumcentric dual is more accurate
    in that the dual edges are guaranteed to be orthogonal to the corresponding
    primal edges. However, for obtuse triangles, the circumcenter of the triangle
    is outside of the triangle, which could lead to negative dual edge lengths,
    which destroys the positive defifnite property of the Hodge star operator.
    The barycenter, on the other hand, is always guaranteed to be inside each
    triangle.
    """
    match dual_complex:
        case "circumcentric":
            return _star_1_circumcentric(tri_mesh)
        case "barycentric":
            return _star_1_barycentric(tri_mesh)
        case _:
            raise ValueError("Unknown 'dual_complex' argument.")


def star_0(tri_mesh: SimplicialMesh) -> Float[DiagDecoupledTensor, "vert vert"]:
    """
    Compute the discrete Hodge star operator on 0-forms for a tri mesh.

    The Hodge 0-star operator maps the 0-simplices (vertices) in a mesh to their
    barycentric dual 2-cells. This function computes the ratio of the area of the
    dual 2-cells to the "area" of the vertices (which is 1 by convention), and this
    area ratio tensor forms the diagonal of the 0-star tensor. This matrix is also
    known as the diagonal 0-form mass matrix.

    The barycentric dual area for each vertex is the sum of 1/3 of the areas of
    all triangles that share the vertex as a face.

    Parameters
    ----------
    tri_mesh
        A tri mesh.

    Returns
    -------
    (vert, vert)
        The Hodge 0-star matrix.
    """
    tri_area = compute_tri_areas(tri_mesh.vert_coords, tri_mesh.tris)

    diag = torch.zeros(
        tri_mesh.n_verts,
        dtype=tri_mesh.dtype,
        device=tri_mesh.device,
    )
    diag.scatter_add_(
        dim=0,
        index=tri_mesh.tris.flatten(),
        src=torch.repeat_interleave(tri_area / 3.0, 3),
    )

    return DiagDecoupledTensor(diag)

import torch as t
from jaxtyping import Float, Integer

from ..complex import Simplicial2Complex
from .stiffness import _compute_cotan_weights_matrix


def _tri_area(
    vert_coords: Float[t.Tensor, "vert 3"], tris: Integer[t.LongTensor, "tri 3"]
) -> Float[t.Tensor, "tri"]:
    """
    Compute the area of all triangles in a 2D mesh.
    """
    vert_s_coord: Float[t.Tensor, "tri 3 3"] = vert_coords[tris]

    edge_ij = vert_s_coord[:, 1] - vert_s_coord[:, 0]
    edge_ik = vert_s_coord[:, 2] - vert_s_coord[:, 0]

    area = 0.5 * t.linalg.norm(t.cross(edge_ij, edge_ik, dim=-1), dim=-1) + 1e-9

    return area


def _d_tri_area_d_vert_coords(
    vert_coords: Float[t.Tensor, "vert 3"], tris: Integer[t.LongTensor, "tri 3"]
) -> Float[t.Tensor, "tri vert 3"]:
    """
    Compute the gradient of the triangle areas with respect to vertex coordinates.
    """
    # For each triangle snp, and each vertex s, find the edge vectors sn, sp, and
    # np, and a vector normal to the triangle at s (sn x sp).
    vert_s_coord: Float[t.Tensor, "tri 3 3"] = vert_coords[tris]

    edge_ns = vert_s_coord[:, [1, 2, 0], :] - vert_s_coord
    edge_ps = vert_s_coord[:, [2, 0, 1], :] - vert_s_coord
    edge_np = vert_s_coord[:, [2, 0, 1], :] - vert_s_coord[:, [1, 2, 0], :]

    norm_s: Float[t.Tensor, "tri 3 3"] = t.cross(edge_ns, edge_ps, dim=-1)
    norm_s_len = t.linalg.norm(norm_s, dim=-1, keepdim=True) + 1e-9

    unorm_s = norm_s / norm_s_len

    # For each triangle snp, the gradient of its area with respect to each vertex
    # s is given by (unorm_s x edge_np)/2
    dAdV = t.cross(unorm_s, edge_np, dim=-1) / 2.0

    return dAdV


# TODO: analytical gradient for the hodge stars


def star_2(simplicial_mesh: Simplicial2Complex) -> Float[t.Tensor, "tri"]:
    """
    The Hodge 2-star operator maps the 2-simplices (triangles) in a mesh to their
    dual 0-cells. This function computes the ratio of the "size" of the dual 0-cells
    (which is 1 by convention) to the area of the primal triangles. The returned tensor
    forms the diagonal of the 2-star tensor.
    """
    return 1.0 / _tri_area(simplicial_mesh.vert_coords, simplicial_mesh.tris)


def star_1(simplicial_mesh: Simplicial2Complex) -> Float[t.Tensor, "edge"]:
    """
    The Hodge 1-star operator maps the 1-simplices (edges) in a mesh to the dual
    1-cells. This function computes the length ratio of the dual 1-cells to the
    primal edges, which is given by the cotan formula. The returned tensor forms
    the diagonal of the 1-star tensor.
    """
    vert_coords: Float[t.Tensor, "vert 3"] = simplicial_mesh.vert_coords
    tris: Integer[t.LongTensor, "tri 3"] = simplicial_mesh.tris

    n_verts = simplicial_mesh.n_verts

    # The off-diagonal elements of the stiff matrix already contains the desired
    # values; i.e., S_ij = -0.5*sum_k[cot_k] for all k that forms a triangle with
    # i and j.
    stiff_off_diag: Float[t.Tensor, "vert vert"] = _compute_cotan_weights_matrix(
        vert_coords, tris, n_verts
    )
    # For the stiff matrix S_ij in COO format, the first row of its index tensor
    # is for the i dimension, and the second row is for the j dimension. We flatten
    # the two dimensions into a 1D index with (i, j) -> i*n_verts + j.
    all_idx = stiff_off_diag.indices()
    all_idx_flat = all_idx[0] * n_verts + all_idx[1]

    # Similarly, we convert the canonical edges represented by a vertex-indexing tuple
    # (i, j) into a flat index with the same formulae.
    edges: Integer[t.Tensor, "edge 2"] = simplicial_mesh.edges
    edge_idx_flat = edges[:, 0] * n_verts + edges[:, 1]

    # Identify the location of the canonical edge ij in the sparse S_ij indices,
    # using the flattened indices, and use the location to extract the cotan values.
    subset_idx = t.searchsorted(all_idx_flat, edge_idx_flat)
    subset_vals = stiff_off_diag.values()[subset_idx]

    return -subset_vals  # note the negative sign to get dual edge lengths


def star_0(simplicial_mesh: Simplicial2Complex) -> Float[t.Tensor, "vert vert"]:
    """
    The Hodge 0-star operator maps the 0-simplices (vertices) in a mesh to their
    dual 2-cells. This function computes the ratio of the area of the dual 2-cells
    to the "size" of the vertices (which is 1 by convention). The returned tensor
    forms the diagonal of the 0-star tensor.

    This function assumes that the area of the dual 2-cell is the barycentric dual
    area for each vertex, which is the sum of 1/3 of the areas of all triangles
    that share the vertex as a face.
    """
    n_verts = simplicial_mesh.n_verts

    tri_area = _tri_area(simplicial_mesh.vert_coords, simplicial_mesh.tris)

    diag = t.zeros(n_verts, device=simplicial_mesh.vert_coords.device)
    diag.scatter_add_(
        dim=0,
        index=simplicial_mesh.tris.flatten(),
        src=t.repeat_interleave(tri_area / 3.0, 3),
    )

    return diag


def d_star_0_d_vert_coords(
    simplicial_mesh: Simplicial2Complex,
) -> Float[t.Tensor, "vert vert 3"]:
    """
    Compute the Jacobian of the Hodge 0-star matrix (diagonal elements) with respect
    to vertex coordinates.
    """
    vert_coords: Float[t.Tensor, "vert 3"] = simplicial_mesh.vert_coords
    tris: Integer[t.LongTensor, "tri 3"] = simplicial_mesh.tris
    n_verts = simplicial_mesh.n_verts

    dAdV: Float[t.Tensor, "tri vert 3"] = _d_tri_area_d_vert_coords(vert_coords, tris)

    # For each triangle ijk and each vertex s, dAdV_ijk_s contributes to the gradient
    # star0_ll wrt s whenever l = s or js is an edge in the mesh. Therefore, each
    # triangle ijk contributes 9 gradient terms, in COO format:
    # [
    #   (i, i, dAdV_ijk_i/3),
    #   (i, j, dAdV_ijk_j/3),
    #   (i, k, dAdV_ijk_k/3),
    #
    #   (j, i, dAdV_ijk_i/3),
    #   (j, j, dAdV_ijk_j/3),
    #   (j, k, dAdV_ijk_k/3),
    #
    #   (k, i, dAdV_ijk_i/3),
    #   (k, j, dAdV_ijk_j/3),
    #   (k, k, dAdV_ijk_k/3),
    # ]

    # Translate the ijk notation to actual indices to access tensor elements.
    i, j, k = 0, 1, 2

    # fmt: off
    dSdV_idx = (
        tris[
            :,
            [
                i, i, i, j, j, j, k, k, k, # first column/index
                i, j, k, i, j, k, i, j, k, # second column/index
            ],
        ]
        .T
        .flatten()
        .reshape(3, -1)
    )
    # fmt: on
    dSdV_val = t.repeat_interleave(dAdV, repeats=2, dim=0).flatten(end_dim=1) / 3.0
    dSdV = t.sparse_coo_tensor(dSdV_idx, dSdV_val, (n_verts, n_verts, 3)).coalesce()

    return dSdV

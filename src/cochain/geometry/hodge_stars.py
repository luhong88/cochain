import torch as t
from jaxtyping import Float, Integer

from ..complex import Simplicial2Complex
from .cotan_laplacian import _compute_cotan_weights_matrix


def _compute_tri_area(
    vert_coords: Float[t.Tensor, "vert 3"], tris: Integer[t.LongTensor, "tri 3"]
) -> Float[t.Tensor, "tri"]:
    vert_s_coord: Float[t.Tensor, "tri 3 3"] = vert_coords[tris]

    edge_ij = vert_s_coord[:, 1] - vert_s_coord[:, 0]
    edge_ik = vert_s_coord[:, 2] - vert_s_coord[:, 0]

    area = 0.5 * t.linalg.norm(t.cross(edge_ij, edge_ik, dim=-1), dim=-1) + 1e-9

    return area


def star_2(simplicial_mesh: Simplicial2Complex) -> Float[t.Tensor, "tri tri"]:
    """
    The Hodge 2-star operator acts on the triangles in a mesh and returns the area
    of the dual 1-cells, which is assigned the area of the primal triangle.
    """
    n_tris = simplicial_mesh.n_tris

    area = _compute_tri_area(simplicial_mesh.vert_coords, simplicial_mesh.tris)

    matrix = (
        t.sparse_coo_tensor(
            t.stack([t.arange(n_tris), t.arange(n_tris)]), area, (n_tris, n_tris)
        )
        .coalesce()
        .to_sparse_csr()
    )

    return matrix


def star_1(simplicial_mesh: Simplicial2Complex) -> Float[t.Tensor, "edge edge"]:
    """
    The Hodge 1-star operator acts on the edges in a mesh and returns the length
    of the dual 1-cells, given by the cotan formula
    """
    vert_coords: Float[t.Tensor, "vert 3"] = simplicial_mesh.vert_coords
    tris: Integer[t.LongTensor, "tri 3"] = simplicial_mesh.tris

    n_verts = simplicial_mesh.n_verts
    n_edges = simplicial_mesh.n_edges

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

    matrix = (
        t.sparse_coo_tensor(
            t.stack([t.arange(n_edges), t.arange(n_edges)]),
            -subset_vals,  # note the negative sign to get dual edge lengths
            (n_edges, n_edges),
        )
        .coalesce()
        .to_sparse_csr()
    )

    return matrix


def star_0(simplicial_mesh: Simplicial2Complex) -> Float[t.Tensor, "vert vert"]:
    """
    The Hodge 0-star operator acts on the vertices in a mesh and returns the area
    of the dual 2-cells; here, we adopt the convention that this area is the
    barycentric dual area for each vertex, which is the sum of 1/3 of the areas
    of all triangles that share the vertex as a face.
    """
    n_verts = simplicial_mesh.n_verts

    tri_area = _compute_tri_area(simplicial_mesh.vert_coords, simplicial_mesh.tris)

    star_0_idx = t.vstack(
        (simplicial_mesh.tris.flatten(), simplicial_mesh.tris.flatten())
    )
    star_0_val = t.repeat_interleave(tri_area / 3.0, 3)

    matrix = (
        t.sparse_coo_tensor(star_0_idx, star_0_val, (n_verts, n_verts))
        .coalesce()
        .to_sparse_csr()
    )

    return matrix

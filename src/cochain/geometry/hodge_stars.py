import torch as t
from jaxtyping import Float, Integer

from ..complex import Simplicial2Complex


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


def star_0(simplicial_mesh: Simplicial2Complex) -> Float[t.Tensor, "vert vert"]:
    """
    The Hoedge 0-star operator acts on the vertices in a mesh and returns the area
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

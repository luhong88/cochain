import torch as t
from jaxtyping import Float, Integer

from .complex import Simplicial2Complex


def cotan_laplacian(
    simp_mesh: Simplicial2Complex,
) -> Float[t.Tensor, "vert vert"]:
    """
    Computes the cotan Laplacian (L0) for a 2D mesh.
    """
    # For each triangle, compute the cotan of the angle at each vertex.
    tri_vert_coord: Float[t.Tensor, "tri 3 3"] = simp_mesh.vert_coords[simp_mesh.tris]

    tri_vert_vec1 = tri_vert_coord[:, [1, 2, 0], :] - tri_vert_coord
    tri_vert_vec2 = tri_vert_coord[:, [2, 0, 1], :] - tri_vert_coord

    tri_vert_ang_dot = t.sum(tri_vert_vec1 * tri_vert_vec2, dim=-1)
    tri_vert_ang_cross = t.linalg.norm(
        t.cross(tri_vert_vec1, tri_vert_vec2, dim=-1), dim=-1
    )
    tri_vert_ang_cotan: Float[t.Tensor, "tri 3"] = tri_vert_ang_dot / (
        1e-9 + tri_vert_ang_cross
    )

    # For each triangle ijk, and each of its vertex i, scatter the cotan at i to
    # edge jk in the laplacian (L_jk); i.e., each triangle ijk contributes the
    # following values to the asym_laplacian (in COO format):
    # [
    #   (j, k, -0.5*cot_i),
    #   (i, k, -0.5*cot_j),
    #   (i, j, -0.5*cot_k),
    # ]
    laplacian_idx = simp_mesh.tris[:, [1, 0, 0, 2, 2, 1]].T.flatten().reshape(2, -1)
    laplacian_val = -0.5 * tri_vert_ang_cotan[:, [0, 1, 2]].T.flatten()
    asym_laplacian = t.sparse_coo_tensor(
        laplacian_idx, laplacian_val, (simp_mesh.n_verts, simp_mesh.n_verts)
    )

    # Symmetrize so that the cotan at i is scattered to both jk and kj.
    sym_laplacian = (asym_laplacian + asym_laplacian.T).coalesce()

    # Compute the diagonal elements of the laplacian.
    laplacian_diag = t.sparse.sum(sym_laplacian, dim=-1)
    diag_idx = t.tile(t.arange(simp_mesh.n_verts), (2, 1)).to(
        simp_mesh.vert_coords.device
    )

    # Generate the final, complete Laplacian operator.
    laplacian = (
        t.sparse_coo_tensor(
            t.hstack((sym_laplacian.indices(), diag_idx)),
            t.concatenate((sym_laplacian.values(), -laplacian_diag.values())),
        )
        .coalesce()
        .to_sparse_csr()
    )

    return laplacian

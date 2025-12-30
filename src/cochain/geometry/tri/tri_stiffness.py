import torch as t
from jaxtyping import Float

from ...complex import SimplicialComplex
from ...sparse.operators import SparseOperator
from .tri_geometry import cotan_weights


def stiffness_matrix(
    tri_mesh: SimplicialComplex,
) -> Float[SparseOperator, "vert vert"]:
    """
    Computes the stiffness matrix for a 2D mesh, sometimes also known as the "cotan
    Laplacian".
    """
    # The cotan weight matrix W gives the stiffness matrix except for the diagonal
    # elements.
    sym_stiffness = cotan_weights(tri_mesh.vert_coords, tri_mesh.tris, tri_mesh.n_verts)

    # Compute the diagonal elements of the stiffness matrix.
    stiffness_diag = t.sparse.sum(sym_stiffness, dim=-1)
    # laplacian_diag.indices() has shape (1, nnz_diag)
    diag_idx = t.concatenate([stiffness_diag.indices(), stiffness_diag.indices()])

    # Generate the final, complete stiffness matrix.
    stiffness = t.sparse_coo_tensor(
        t.hstack((sym_stiffness.indices(), diag_idx)),
        t.concatenate((sym_stiffness.values(), -stiffness_diag.values())),
    ).coalesce()

    return SparseOperator.from_tensor(stiffness)

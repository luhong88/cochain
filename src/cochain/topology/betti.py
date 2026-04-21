__all__ = ["compute_betti_numbers"]

import torch

from ..complex import SimplicialMesh
from .morse import compute_morse_complex
from .spanning_tree import _minimum_spanning_tree
from .topo_laplacians import laplacian_k


def _tri_manifold_betti_via_trees(tri_mesh: SimplicialMesh) -> tuple[int, int, int]:
    """
    Compute the first three Betti numbers for a manifold tri mesh.

    This function uses the tree-cotree decomposition to compute the Betti numbers.
    As such, it inherits the limitation of the cotree decomposition to manifold
    meshes.
    """
    # First, construct the spanning tree on the 1-skeleton, which identifies
    # |V| - b_0 edges. Note that no special considerations are needed for meshes
    # with boundaries.
    l0 = laplacian_k(tri_mesh, k=0, component="up")
    primal_mst = _minimum_spanning_tree(adjacency=l0.triu(diagonal=1).abs())
    n_primal_mst_edges = primal_mst.size(-1)
    b0 = tri_mesh.n_verts - n_primal_mst_edges

    # Next, construct the dual spanning tree over the dual vertices (and dual
    # edges), which identifies |F| - b_2 dual edges. Note that, when the tri mesh
    # has boundaries, due to the Poincare-Lefschetz duality, the absolute
    # homology on the primal complex is isomorphic to the relative homology on the
    # dual complex; therefore, we need to compute the spanning tree on the quotient
    # dual 1-skeleton using the super node method (and the edges connecting to the
    # super node need to be preserved for accounting).
    dual_l0 = laplacian_k(tri_mesh, k=0, component="up", dual_complex=True)

    # This finds the dual vertices (i.e., primal triangles) that contain boundary
    # edges; such dual vertices connect to the super node when computing the dual
    # spanning tree; this same trick is used for computing the cotree decomposition.
    cbd_1_abs = tri_mesh.cbd[1].abs()
    bd_dual_vert_mask = (
        cbd_1_abs @ tri_mesh.bd_edge_mask.to(dtype=cbd_1_abs.dtype)
    ) > 0.0

    dual_mst = _minimum_spanning_tree(
        adjacency=dual_l0.triu(diagonal=1).abs(),
        root_mask=bd_dual_vert_mask,
        keep_super_node=True,
    )
    n_dual_mst_edges = dual_mst.shape[-1]
    b2 = tri_mesh.n_tris - n_dual_mst_edges

    # Note that the Euler characteristic χ can be expressed in two ways for a
    # tri mesh:
    #     χ = |V| - |E| + |F|
    #     χ = b_0 - b_1 + b_2
    # it follows that b_1 = |E| - (|V| - b_0) - (|F| - b_2).
    b1 = tri_mesh.n_edges - n_primal_mst_edges - n_dual_mst_edges

    return b0, b1, b2


def _betti_via_morse(mesh: SimplicialMesh) -> tuple[int, int, int]:
    """
    Compute the first three Betti numbers for a tri or tet mesh.

    This function uses the discrete Morse complex to compute the Betti numbers,
    and works on arbitrary immersed meshes.
    """
    morse_cbd, crit_splx = compute_morse_complex(mesh)

    n_crit_splx = [splx.size(0) for splx in crit_splx]

    cbd_dense = [cbd.to_dense() for cbd in morse_cbd]
    cbd_rank = [torch.linalg.matrix_rank(cbd).item() for cbd in cbd_dense]

    # b_0 = |K_0| - rank(d_0)
    b_0 = n_crit_splx[0] - cbd_rank[0]

    # b_1 = |K_1| - rank(d_0) - rank(d_1)
    b_1 = n_crit_splx[1] - cbd_rank[0] - cbd_rank[1]

    # b_2 = |K_2| - rank(d_1) - rankd(d_2)
    b_2 = n_crit_splx[2] - cbd_rank[1] - cbd_rank[2]

    return b_0, b_1, b_2


def compute_betti_numbers(
    mesh: SimplicialMesh, manifold: bool = False
) -> tuple[int, int, int]:
    """
    Compute the first three Betti numbers for a mesh.

    Parameters
    ----------
    mesh
        A tri or tet mesh.
    manifold
        If the input mesh is 2D and manifold, compute the betti numbers via
        tree-cotree decomposition; otherwise compute the betti numbers via
        discrete Morse complex.

    Returns
    -------
    b_0
        The 0th Betti number, which measures the number of connected components.
    b_1
        The 1st Betti number, which measures the number of holes.
    b_2
        The 2nd Betti number, which measures the number of voids.
    """
    if manifold and (mesh.dim == 2):
        return _tri_manifold_betti_via_trees(mesh)
    else:
        return _betti_via_morse(mesh)

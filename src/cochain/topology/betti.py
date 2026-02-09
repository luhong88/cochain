import torch as t

from ..complex import SimplicialComplex
from .spanning_tree import _minimum_spanning_tree
from .topo_laplacians import laplacian_k


def tri_mesh_betti_numbers(tri_mesh: SimplicialComplex) -> tuple[int, int, int, int]:
    """
    Compute the first three Betti numbers (b_0, b_1, b_2) for a triangular
    mesh using the tree-cotree decomposition.
    """
    # First, construct the spanning tree on the 1-skeleton, which identifies
    # |V| - b_0 edges; the adjacency matrix for the 1-skeleton can be constructed
    # as the absolute value of the off-diagonal elements of the topological
    # up 0-Laplacian (in the up 0-Laplacian, element (i, j) is nonzero iff the
    # vertices i and j form an edge ij).
    _, l0_up, _ = laplacian_k(tri_mesh, k=0)
    primal_mst = _minimum_spanning_tree(l0_up.off_diagonal().abs())
    n_primal_mst_edges = primal_mst.shape[-1]
    b0 = tri_mesh.n_verts - n_primal_mst_edges

    # Next, construct the dual spanning tree over the dual vertices (and dual
    # edges), which identifies |F| - b_2 dual edges; the adjacency matrix for
    # the dual complex can be constructed as before using the dual coboundary
    # operators.
    _, dual_l0_up, _ = laplacian_k(tri_mesh, k=0, dual=True)
    dual_mst = _minimum_spanning_tree(dual_l0_up.off_diagonal().abs())
    n_dual_mst_edges = dual_mst.shape[-1]
    b2 = tri_mesh.n_tris - n_dual_mst_edges

    # Note that the Euler characteristic X can be expressed in two ways for a
    # triangular mesh:
    #     X = |V| - |E| + |F|
    #     X = b_0 - b_1 + b_2
    # Putting together, it follows that b_1 = |E| - (|V| - b_0) - (|F| - b_2).
    b1 = tri_mesh.n_edges - n_primal_mst_edges - n_dual_mst_edges

    return b0, b1, b2

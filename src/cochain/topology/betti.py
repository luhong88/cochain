from ..complex import SimplicialComplex
from .spanning_tree import _minimum_spanning_tree
from .topo_laplacians import laplacian_k


def compute_tri_mesh_betti_numbers(tri_mesh: SimplicialComplex) -> tuple[int, int, int]:
    """
    Compute the first three Betti numbers (b_0, b_1, b_2) for a triangular
    mesh using the tree-cotree decomposition.
    """
    # First, construct the spanning tree on the 1-skeleton, which identifies
    # |V| - b_0 edges. Note that no special considerations are needed for meshes
    # with boundaries.
    l0 = laplacian_k(tri_mesh, k=0, component="up")
    primal_mst = _minimum_spanning_tree(adjacency=l0.triu(diagonal=1).abs())
    n_primal_mst_edges = primal_mst.shape[-1]
    b0 = tri_mesh.n_verts - n_primal_mst_edges

    # Next, construct the dual spanning tree over the dual vertices (and dual
    # edges), which identifies |F| - b_2 dual edges. Note that, when the tri mesh
    # has boundaries, due to the Poincare-Lefschetz duality, the absolute
    # homology on the primal complex is isomorphic to the relative homology on the
    # dual complex; therefore, we need to compute the spanning tree on the quotient
    # dual 1-skeleton using the super node method (and the edges connecting to the
    # super node need to be preserved for accounting).
    dual_l0 = laplacian_k(tri_mesh, k=0, component="up", dual=True)
    # This finds the dual vertices (i.e., primal triangles) that contain boundary
    # edges; such dual vertices connect to the super node when computing the dual
    # spanning tree; this same trick is used for computing the cotree decomposition.
    bd_dual_vert_mask = (
        tri_mesh.cbd[1].abs() @ tri_mesh.bd_edge_mask.to(dtype=tri_mesh.cbd[1].dtype)
    ) > 0.0
    dual_mst = _minimum_spanning_tree(
        adjacency=dual_l0.triu(diagonal=1).abs(),
        root_mask=bd_dual_vert_mask,
        keep_super_node=True,
    )
    n_dual_mst_edges = dual_mst.shape[-1]
    b2 = tri_mesh.n_tris - n_dual_mst_edges

    # Note that the Euler characteristic X can be expressed in two ways for a
    # triangular mesh:
    #     X = |V| - |E| + |F|
    #     X = b_0 - b_1 + b_2
    # Putting together, it follows that b_1 = |E| - (|V| - b_0) - (|F| - b_2).
    b1 = tri_mesh.n_edges - n_primal_mst_edges - n_dual_mst_edges

    return b0, b1, b2

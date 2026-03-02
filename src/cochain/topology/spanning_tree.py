import numpy as np
import torch as t
from jaxtyping import Bool, Float, Integer
from scipy.sparse import coo_array
from scipy.sparse.csgraph import minimum_spanning_tree

from cochain.sparse.operators import SparseOperator
from cochain.utils.search import simplex_search


def _minimum_spanning_tree(
    adjacency: Float[SparseOperator, "node node"],
    root_mask: Bool[t.Tensor, " node"] | None = None,
    exclusion_mask: Bool[t.Tensor, " edge"] | None = None,
    weights: Float[t.Tensor, " edge"] | None = None,
    keep_super_node: bool = False,
) -> Integer[t.LongTensor, "2 mst_node"]:
    """
    the root_mask is a boolean mask that specifies the node(s) that will serve as
    the root(s) of the spanning tree/forest.

    The exclusion_mask is a boolean mask whose elements correspond to the nonzero
    elements of the adjacency matrix; an edge in the adjacency matrix marked as
    True by the exclusion_mask is disallowed in constructing the spanning tree/forest.
    """
    n_nodes = adjacency.shape[0]

    # int32 is required for scipy sparse array indices
    idx_coo_full = adjacency.sp_topo.idx_coo.to(dtype=t.int32)
    idx_coo_rows_full = idx_coo_full[0].detach().cpu().numpy()
    idx_coo_cols_full = idx_coo_full[1].detach().cpu().numpy()

    # If provided, the weights overwrite the adjacency matrix data.
    coo_data_full = (
        adjacency.val.detach().cpu().numpy()
        if weights is None
        else weights.detach().cpu().numpy()
    )

    # If provided, use the exclusion mask to remove banned edges.
    if exclusion_mask is None:
        idx_coo_rows = idx_coo_rows_full
        idx_coo_cols = idx_coo_cols_full
        coo_data = coo_data_full
    else:
        exclusion_mask_np = exclusion_mask.detach().cpu().numpy()

        idx_coo_rows = idx_coo_rows_full[~exclusion_mask_np]
        idx_coo_cols = idx_coo_cols_full[~exclusion_mask_np]
        coo_data = coo_data_full[~exclusion_mask_np]

    if (root_mask is not None) and (root_mask.any()):
        root_mask_np = root_mask.detach().cpu().numpy()

        idx_dtype = idx_coo_rows.dtype
        data_dtype = coo_data.dtype

        # If the root(s) of the tree is specified, add a new "super node" to the
        # graph and connect the super node to all boundary nodes with a weight of 0.
        super_node_idx = n_nodes
        root_idx = np.argwhere(root_mask_np).flatten().astype(idx_dtype)

        # Augment the adjacency matrix by adding in edges connecting the super node.
        new_rows = np.full(len(root_idx), super_node_idx, dtype=idx_dtype)
        new_cols = root_idx

        # Need to ensure that the edges connecting to the super node has a weight
        # that is strictly smaller than all existing weights.
        min_weight = coo_data.min() if len(coo_data) > 0 else 0.0
        super_weight = min_weight - np.abs(min_weight) - 1.0
        new_data = np.full(len(root_idx), super_weight, dtype=data_dtype)

        # The adjacency matrix does not need to be symmetric.
        aug_rows = np.concatenate([idx_coo_rows, new_rows])
        aug_cols = np.concatenate([idx_coo_cols, new_cols])
        aug_data = np.concatenate([coo_data, new_data])

        aug_shape = (n_nodes + 1, n_nodes + 1)
        aug_adjacency = coo_array((aug_data, (aug_rows, aug_cols)), shape=aug_shape)

    else:
        aug_adjacency = coo_array(
            (coo_data, (idx_coo_rows, idx_coo_cols)), shape=adjacency.shape
        )

    mst = minimum_spanning_tree(aug_adjacency)
    mst_coo = mst.tocoo()

    if keep_super_node:
        tree_u = mst_coo.row
        tree_v = mst_coo.col

    else:
        # Filter out edges connected to the super node.
        valid_mask = (mst_coo.row != n_nodes) & (mst_coo.col != n_nodes)

        tree_u = mst_coo.row[valid_mask]
        tree_v = mst_coo.col[valid_mask]

    tree_edges = t.from_numpy(np.stack((tree_u, tree_v))).to(
        dtype=adjacency.sp_topo.dtype, device=adjacency.device
    )

    return tree_edges


# TODO: update to accommodate tet meshes
def compute_tree_mask(
    topo_laplacian_0: Float[SparseOperator, "vert vert"],
    canon_edges: Integer[t.LongTensor, "edge 2"],
    mass_1: Float[SparseOperator, "edge edge"] | None = None,
    vert_rel_bc_mask: Bool[t.Tensor, " vert"] | None = None,
    cotree_mask: Bool[t.Tensor, " edge"] | None = None,
) -> Bool[t.Tensor, " edge"]:
    """
    Compute the spanning tree on the 1-skeleton of a triangular mesh, which
    can be used to fix the gauge freedom of the down/grad-div component of the
    weak 1-Laplacian.

    If the edge masses are provided using the `mass_1` argument (in the form of
    either the Hodge star or the mass matrix), the function will compute a
    maximum spanning tree using the edge masses as weights, which should result
    in better condition number for the gauge fixed linear system.

    If the triangular mesh contains boundaries subject to relative boundary
    conditions, a mask of the boundary vertices should be passed to the
    `vert_rel_bc_mask` argument.

    Note that, if this function is used as part of the tree-cotree decomposition
    for the full 1-Laplacian guage fixing, the `compute_cotree_mask()` function
    needs to be called first, and its result should be passed to the `cotree_mask`
    argument of this function. This order of operation ensures that the tree
    and cotree remain disjoint.
    """
    # Compute the vertex adjacency matrix from the (topological) 0-Laplacian
    # Use the upper diagonal portion of the adjacency matrix since the scipy
    # MinST function interprets the adjacency matrix as an undirected graph.
    adjacency = topo_laplacian_0.triu(diagonal=1).abs()

    # Find the indices of the adjacency edges on the canonical edge list, and
    # use the indices to retrieve the edge weights from the provided mass (or hodge
    # star) matrices.
    edges = adjacency.sp_topo.idx_coo.T
    edge_idx = simplex_search(
        key_simps=canon_edges,
        query_simps=edges,
        sort_key_simp=True,
        sort_key_vert=False,
        sort_query_vert=True,
    )

    if mass_1 is None:
        edge_weights = None
    else:
        # Perform a diagonal approximation for the edge mass (if the input is a
        # Hodge star, then this is exact.)
        diag_mass = mass_1.diagonal()
        # Note that we take the negative mass so that the MinST function performs
        # a MaxST calculation.
        edge_weights = -diag_mass[edge_idx]

    # Compute the MaxST and find the indices of the MaxST edges on the canonical
    # edge list. If
    mst = _minimum_spanning_tree(
        adjacency=adjacency,
        root_mask=vert_rel_bc_mask,
        exclusion_mask=cotree_mask,
        weights=edge_weights,
        keep_super_node=False,
    ).T

    mst_idx = simplex_search(
        key_simps=canon_edges,
        query_simps=mst,
        sort_key_simp=True,
        sort_key_vert=False,
        sort_query_vert=True,
    )

    tree_mask = t.zeros(canon_edges.shape[0], dtype=t.bool, device=adjacency.device)
    tree_mask[mst_idx] = True

    return tree_mask


def _cbd_to_coface(
    cbd, degree: int = 2
) -> tuple[Integer[t.LongTensor, " face"], Integer[t.LongTensor, "face coface"]]:
    """
    For a given k-coboundary operator, find the indices of all k-simplices of
    degree d (i.e., the number of cofaces of the k-simplices), and, for each
    k-simplex of degree d, determine the indices of the d (k+1)-simplices that
    share the k-simplex as a face.

    Note that the returned list of k-simplex indices is in ascending order, and
    the returned list of (k+1)-simplex index tuples are sorted in ascending order
    within each tuple (but the list itself is not necessarily in lex order).
    """
    idx_coo = cbd.sp_topo.idx_coo
    # The row indices correspond to the (k+1)-simplex indices, and the col indices
    # correspond to the k-simplex indices.
    idx_coo_col = idx_coo[-1]

    # Find how many times the index of each k-simplex shows up in the column index
    # For example, if a k-simplex is the face of two (k+1)-simplices, then it
    # will show up as two nonzero elements in the its column in cbd[k].
    face_idx, face_degree = t.unique(idx_coo_col, return_counts=True)

    # Filter for the indices of all k-simplices that have the desired degree, then
    # turn this into a boolean mask for the coo indices.
    shared_face_idx = face_idx[face_degree == degree]
    shared_face_mask = t.isin(idx_coo_col, shared_face_idx)

    # Get the subset of cbd coo indices corresponding to the k-simplices with the
    # desired degree; by sorting the coo col indices (the k-simplex indices), the
    # coface (k+1)-simplices (the coo row indices) for each k-simplex are re-ordered
    # next to each other, which can then be reshaped to generate the coface list.
    idx_coo_subset = idx_coo[:, shared_face_mask]
    sort_idx = t.sort(idx_coo_subset[-1], stable=True).indices
    coface_idx = idx_coo_subset[-2][sort_idx].reshape(-1, degree)

    unique_face_idx = idx_coo_subset[-1, sort_idx][::degree]

    return unique_face_idx, coface_idx


# TODO: update to accommodate tet meshes
def compute_cotree_mask(
    dual_topo_laplacian_0: Float[SparseOperator, "vert vert"],
    cbd_1: Float[SparseOperator, "tri edge"],
    inv_mass_1: Float[SparseOperator, "edge edge"] | None = None,
    edge_rel_bc_mask: Bool[t.Tensor, " edge"] | None = None,
) -> Bool[t.Tensor, " edge"]:
    """
    Compute the dual spanning tree (i.e., cotree) on the dual 1-skeleton of a
    triangular mesh, which can be used to fix the gauge freedom of the up/curl-curl
    component of the weak 1-Laplacian.

    If the dual edge masses are provided using the `inv_mass_1` argument (in the
    form of either the inverse Hodge star or the inverse mass matrix), the function
    will compute a maximum dual spanning tree using the inverse edge masses as
    weights, which should result in better condition number for the gauge fixed
    linear system.

    If the triangular mesh contains boundaries subject to relative boundary
    conditions, a mask of the boundary edges should be passed to the
    `edge_rel_bc_mask` argument.
    """
    # Technically speaking, we should use the dual topological 0-Laplacian restricted
    # to the interior primal edges of the mesh. However, such dual meshes show up
    # in the diagonal of the 0-Laplacian, which are disgarded when taking the upper
    # triangular part of the matrix; thus, this procedure guarantees that the
    # adjacency matrix only contains information on dual vertices and their connections
    # by the interior dual edges.
    adjacency = dual_topo_laplacian_0.triu(diagonal=1).abs()

    # Here, the dual edges are indexed by the indices of the two primal triangles
    # sharing the primal edge; need to convert this representation of the dual
    # edges to the indices of the corresponding primal edges.
    dual_edges = adjacency.sp_topo.idx_coo.T
    edge_face_idx, edge_coface_idx = _cbd_to_coface(cbd_1, degree=2)
    primal_edge_idx = edge_face_idx[
        simplex_search(
            key_simps=edge_coface_idx,
            query_simps=dual_edges,
            sort_key_simp=True,
            sort_key_vert=False,
            sort_query_vert=True,
        )
    ]

    if inv_mass_1 is None:
        edge_weights = None
    else:
        diag_mass = inv_mass_1.diagonal()
        edge_weights = -diag_mass[primal_edge_idx]

    # For the primal edges satisfying relative boundary conditions, the corresponding
    # clipped dual edges corresponding need to connect its dual triangle coface
    # to the super node. To do so, we check whether each row of the 1-coboundary
    # operator contains edges marked with a relative boundary condition.
    if edge_rel_bc_mask is None:
        bd_dual_vert_mask = t.zeros(cbd_1.shape[0], dtype=t.bool, device=cbd_1.device)
    else:
        bd_dual_vert_mask = (cbd_1.abs() @ edge_rel_bc_mask.to(dtype=cbd_1.dtype)) > 0.0

    mst = _minimum_spanning_tree(
        adjacency=adjacency,
        root_mask=bd_dual_vert_mask,
        exclusion_mask=None,
        weights=edge_weights,
        keep_super_node=False,
    ).T

    # Again, need to convert the dual edge representation as a pair of dual vertices
    # (primal triangles) to the corresponding primal edge indices.
    mst_idx = edge_face_idx[
        simplex_search(
            key_simps=edge_coface_idx,
            query_simps=mst,
            sort_key_simp=True,
            sort_key_vert=False,
            sort_query_vert=True,
        )
    ]

    cotree_mask = t.zeros(cbd_1.shape[-1], dtype=t.bool, device=adjacency.device)
    cotree_mask[mst_idx] = True

    return cotree_mask

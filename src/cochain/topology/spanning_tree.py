from typing import Literal

import numpy as np
import torch as t
from jaxtyping import Bool, Float, Integer
from scipy.sparse import coo_array
from scipy.sparse.csgraph import minimum_spanning_tree

from cochain.complex import SimplicialComplex
from cochain.sparse.operators import SparseOperator
from cochain.utils.search import simplex_search


# TODO: fix zero super node edge weights (ignored by scipy)
def _minimum_spanning_tree(
    adjacency: Float[SparseOperator, "node node"],
    root_mask: Bool[t.Tensor, " node"] | None = None,
    exclusion_mask: Bool[t.Tensor, " edge"] | None = None,
    weights: Float[t.Tensor, " edge"] | None = None,
) -> Integer[t.LongTensor, "2 mst_node"]:
    """
    It is assumed that the nonzero elements in the input adjacency matrix are all
    strictly positive.

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
        idx_coo_rows = idx_coo_rows_full[~exclusion_mask]
        idx_coo_cols = idx_coo_cols_full[~exclusion_mask]
        coo_data = coo_data_full[~exclusion_mask]

    if (root_mask is not None) and (root_mask.any()):
        idx_dtype = idx_coo_rows.dtype
        data_dtype = coo_data.dtype

        # If the root(s) of the tree is specified, add a new "super node" to the
        # graph and connect the super node to all boundary nodes with a weight of 0.
        super_node_idx = n_nodes
        root_idx = np.argwhere(root_mask).flatten().astype(idx_dtype)

        # Augment the adjacency matrix by adding in edges connecting the super node.
        new_rows = np.full(len(root_idx), super_node_idx, dtype=idx_dtype)
        new_cols = root_idx
        new_data = np.zeros(len(root_idx), dtype=data_dtype)

        aug_rows = np.concatenate([idx_coo_rows, new_rows, new_cols])
        aug_cols = np.concatenate([idx_coo_cols, new_cols, new_rows])
        aug_data = np.concatenate([coo_data, new_data, new_data])

        aug_shape = (n_nodes + 1, n_nodes + 1)
        aug_adjacency = coo_array((aug_data, (aug_rows, aug_cols)), shape=aug_shape)

    else:
        aug_adjacency = coo_array(
            (coo_data, (idx_coo_rows, idx_coo_cols)), shape=adjacency.shape
        )

    mst = minimum_spanning_tree(aug_adjacency)
    mst_coo = mst.tocoo()

    # Filter out edges connected to the super node.
    valid_mask = (mst_coo.row != n_nodes) & (mst_coo.col != n_nodes)

    # Extract and return the node index pairs corresponding to edges on the MST.
    tree_u = mst_coo.row[valid_mask]
    tree_v = mst_coo.col[valid_mask]

    tree_edges = t.from_numpy(np.stack((tree_u, tree_v))).to(
        dtype=adjacency.sp_topo.dtype, device=adjacency.device
    )

    return tree_edges


def _l1_down_tree_gauge(
    topo_laplacian_0: Float[SparseOperator, "vert vert"],
    canon_edges: Integer[t.LongTensor, "edge 2"],
    mass_1: Float[SparseOperator, "edge edge"] | None = None,
    rel_bc_mask: Bool[t.Tensor, " vert"] | None = None,
    cotree_mask: Bool[t.Tensor, " edge"] | None = None,
) -> Bool[t.Tensor, " edge"]:
    """
    Down/grad-div L-1 Laplacian tree gauge fixing.
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
        sort_key_simp=False,
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
        root_mask=rel_bc_mask,
        exclusion_mask=cotree_mask,
        weights=edge_weights,
    ).T

    mst_idx = simplex_search(
        key_simps=canon_edges,
        query_simps=mst,
        sort_key_simp=False,
        sort_key_vert=False,
        sort_query_vert=True,
    )

    tree_mask = t.zeros(canon_edges.shape[0], dtype=t.bool, device=adjacency.device)
    tree_mask[mst_idx] = True

    return tree_mask


"""
Construct spanning tree/forest for:
* Cotree gauge fixing for up/curl-curl L-1 Laplacian
* Tree-cotree gauge fixing for the full L-1 Laplacian
"""

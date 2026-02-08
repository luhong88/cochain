from typing import Literal

import numpy as np
import torch as t
from jaxtyping import Bool, Float
from scipy.sparse import coo_array
from scipy.sparse.csgraph import minimum_spanning_tree


def _build_minimum_spanning_tree(
    adjacency: Float[coo_array, "node node"],
    root_mask: Bool[np.ndarray, " node"] | None = None,
    weights: Float[np.ndarray, " node"] | None = None,
):
    n_simps = adjacency.shape[0]

    coo_idx_rows = adjacency.row
    coo_idx_cols = adjacency.col
    # If provided, the weights overwrite the adjacency matrix data
    coo_data = adjacency.data if weights is None else weights

    idx_dtype = adjacency._get_index_dtype()
    data_dtype = adjacency.dtype

    if (root_mask is not None) and (root_mask.any()):
        # If the root(s) of the tree is specified, add a new "super node" to the
        # graph and connect the super node to all boundary nodes with a weight of 0.
        super_node_idx = n_simps
        root_idx = np.argwhere(root_mask).flatten().astype(idx_dtype)

        # Augment the adjacency matrix by adding in edges connecting the super node.
        new_rows = np.full(len(root_idx), super_node_idx, dtype=idx_dtype)
        new_cols = root_idx
        new_data = np.zeros(len(root_idx), dtype=data_dtype)

        aug_rows = np.concatenate([coo_idx_rows, new_rows, new_cols])
        aug_cols = np.concatenate([coo_idx_cols, new_cols, new_rows])
        aug_data = np.concatenate([coo_data, new_data, new_data])

        aug_shape = (n_simps + 1, n_simps + 1)
        aug_adjacency = coo_array((aug_data, (aug_rows, aug_cols)), shape=aug_shape)

    else:
        aug_adjacency = coo_array(
            (coo_data, (coo_idx_rows, coo_idx_cols)), shape=adjacency.shape
        )

    mst = minimum_spanning_tree(aug_adjacency)
    mst_coo = mst.tocoo()

    # Filter out edges connected to the super node.
    valid_mask = (mst_coo.row != super_node_idx) & (mst_coo.col != super_node_idx)

    # Extract and return the node index pairs corresponding to edges on the MST.
    tree_u = mst_coo.row[valid_mask]
    tree_v = mst_coo.col[valid_mask]

    return tree_u, tree_v

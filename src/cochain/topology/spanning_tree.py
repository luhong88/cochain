import numpy as np
import torch as t
from jaxtyping import Bool, Float, Integer
from scipy.sparse import coo_array
from scipy.sparse.csgraph import minimum_spanning_tree

from cochain.sparse.operators import SparseOperator


def _minimum_spanning_tree(
    adjacency: Float[SparseOperator, "node node"],
    root_mask: Bool[t.Tensor, " node"] | None = None,
    weights: Float[t.Tensor, " node"] | None = None,
) -> Integer[t.LongTensor, "2 mst_node"]:
    """
    It is assumed that the nonzero elements in the input adjacency matrix are all
    strictly positive.
    """
    n_nodes = adjacency.shape[0]

    # int32 is required for scipy sparse array indices
    idx_coo = adjacency.sp_topo.idx_coo.to(dtype=t.int32)
    coo_idx_rows = idx_coo[0].detach().cpu().numpy()
    coo_idx_cols = idx_coo[1].detach().cpu().numpy()

    # If provided, the weights overwrite the adjacency matrix data.
    coo_data = (
        adjacency.val.detach().cpu().numpy()
        if weights is None
        else weights.detach().cpu().numpy()
    )

    if (root_mask is not None) and (root_mask.any()):
        idx_dtype = coo_idx_rows.dtype
        data_dtype = coo_data.dtype

        # If the root(s) of the tree is specified, add a new "super node" to the
        # graph and connect the super node to all boundary nodes with a weight of 0.
        super_node_idx = n_nodes
        root_idx = np.argwhere(root_mask).flatten().astype(idx_dtype)

        # Augment the adjacency matrix by adding in edges connecting the super node.
        new_rows = np.full(len(root_idx), super_node_idx, dtype=idx_dtype)
        new_cols = root_idx
        new_data = np.zeros(len(root_idx), dtype=data_dtype)

        aug_rows = np.concatenate([coo_idx_rows, new_rows, new_cols])
        aug_cols = np.concatenate([coo_idx_cols, new_cols, new_rows])
        aug_data = np.concatenate([coo_data, new_data, new_data])

        aug_shape = (n_nodes + 1, n_nodes + 1)
        aug_adjacency = coo_array((aug_data, (aug_rows, aug_cols)), shape=aug_shape)

    else:
        aug_adjacency = coo_array(
            (coo_data, (coo_idx_rows, coo_idx_cols)), shape=adjacency.shape
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

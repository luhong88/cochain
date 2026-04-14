__all__ = ["compute_tree_mask", "compute_cotree_mask"]

import numpy as np
import torch
from jaxtyping import Bool, Float, Integer
from scipy.sparse import coo_array
from scipy.sparse.csgraph import minimum_spanning_tree
from torch import LongTensor, Tensor

from cochain.sparse.decoupled_tensor import BaseDecoupledTensor, SparseDecoupledTensor
from cochain.utils.search import splx_search


def _minimum_spanning_tree(
    adjacency: Float[SparseDecoupledTensor, "node node"],
    root_mask: Bool[Tensor, " node"] | None = None,
    exclusion_mask: Bool[Tensor, " edge"] | None = None,
    weights: Float[Tensor, " edge"] | None = None,
    keep_super_node: bool = False,
) -> Integer[LongTensor, "2 mst_node"]:
    r"""
    Compute the minimum spanning forest over a graph.

    Parameters
    ----------
    adjacency : [node, node]
        The adjacency matrix of the undirected graph. The adjacency matrix needs
        not be symmetric; if $A_{ij} \ne A_{ji}$, then the minimum nonzero value
        of the two will be used as the edge weight.
    root_mask : [node,]
        A boolean mask that specifies the node(s) that will serve as the root(s)
        of the spanning tree/forest. If a valid root_mask is provided, this function
        will augment the adjacency matrix with a "super node" that connects to the
        root vertices using a nonzero weight strictly smaller than all other edge
        weights in the graph, such that, once the super node is removed, the
        root vertices are guaranteed to be the roots of the spanning forest.
    exclusion_mask : [edge,]
        A boolean mask whose elements correspond to the nonzero elements of the
        adjacency matrix; an edge in the adjacency matrix marked as True by the
        this mask is disallowed when constructing the spanning forest.
    weights : [edge,]
        A list of weights that, when provided, override the edge weights in the
        adjacency matrix.
    keep_super_node
        Whether to keep the super node in the output spanning forest. If no
        valid `root_mask` is provided, this argument is ignored.

    Returns
    -------
    [2, mst_node]
        The computed minimum spanning forest; each column in the tensor represents
        the indices of a graph node pair that is connected by an edge in the forest.

    Notes
    -----
    This function uses `scipy.sparse.csgraph.minimum_spanning_tree()` to construct
    the minimum spanning forest.
    """
    n_nodes = adjacency.size(0)

    # int32 is required for scipy sparse array indices
    idx_coo_full = adjacency.pattern.idx_coo.to(dtype=torch.int32)
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
        # graph and connect the super node to all root nodes with a small weight.
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
            (coo_data, (idx_coo_rows, idx_coo_cols)), shape=tuple(adjacency.shape)
        )

    mst = minimum_spanning_tree(aug_adjacency)
    mst_coo = mst.tocoo()

    if keep_super_node:
        tree_u = mst_coo.row
        tree_v = mst_coo.col

    else:
        # Filter out edges connected to the super node. Note that, if no super
        # node was added to the graph, this mask does nothing, since no node
        # will be indexed at n_nodes.
        valid_mask = (mst_coo.row != n_nodes) & (mst_coo.col != n_nodes)

        tree_u = mst_coo.row[valid_mask]
        tree_v = mst_coo.col[valid_mask]

    tree_edges = torch.from_numpy(np.stack((tree_u, tree_v))).to(
        dtype=adjacency.pattern.dtype, device=adjacency.device
    )

    return tree_edges


# TODO: update to accommodate tet meshes
def compute_tree_mask(
    topo_laplacian_0: Float[SparseDecoupledTensor, "global_vert global_vert"],
    canon_edges: Integer[LongTensor, "global_edge local_vert=2"],
    mass_1: Float[BaseDecoupledTensor, "global_edge global_edge"] | None = None,
    vert_rel_bc_mask: Bool[Tensor, " global_vert"] | None = None,
    cotree_mask: Bool[Tensor, " global_edge"] | None = None,
) -> Bool[Tensor, " global_edge"]:
    """
    Compute the spanning forest on the 1-skeleton of a tri mesh.

    The forest can be used to fix the gauge freedom of the down/grad-div component
    of the weak 1-Laplacian. Note that, if this function is used as part of the
    tree-cotree decomposition for the full 1-Laplacian guage fixing, the
    `compute_cotree_mask()` function needs to be called first, and its result
    should be passed to the `cotree_mask` argument of this function. This order
    of operation ensures that the tree and cotree remain disjoint.

    Parameters
    ----------
    topo_laplacian_0 : [global_vert, global_vert]
        The topological/combinatorial 0-Laplacian of the tri mesh.
    canon_edges : [global_edge, local_vert=2]
        The list of canonical edges in the mesh (mesh.edges).
    mass_1 : [global_edge, global_edge]
        The Hodge 1-star or consistent 1-mass operator. If provided, this function
        will compute a maximum spanning forest using the the diagonal elements
        of the matrix as weights/edge masses, which should result in a better condition
        number for the gauge fixed linear system.
    vert_rel_bc_mask : [global_vert,]
        A boolean mask that mark vertices subject to relative boundary condition(s).
    cotree_mask : [global_edge,]
        A boolean mask that mark edges that belong to the cotree.

    Returns
    -------
    [global_edge,]
        A boolean mask that mark edges that belong to the spanning forest.
    """
    # Compute the vertex adjacency matrix from the (topological) 0-Laplacian
    # Use the upper diagonal portion of the adjacency matrix.
    adjacency = topo_laplacian_0.triu(diagonal=1).abs()

    # Find the indices of the adjacency edges on the canonical edge list, and
    # use the indices to retrieve the edge weights from the provided mass (or hodge
    # star) matrices.
    edges = adjacency.pattern.idx_coo.T
    edge_idx = splx_search(
        key_splx=canon_edges,
        query_splx=edges,
        sort_key_splx=True,
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
        # a MaxST calculation; in particular, a larger edge mass translates into
        # a more negative edge weight, which translates into a shorter path on
        # the tree.
        edge_weights = -diag_mass[edge_idx]

    # Compute the MaxST and find the indices of the MaxST edges on the canonical
    # edge list.
    mst = _minimum_spanning_tree(
        adjacency=adjacency,
        root_mask=vert_rel_bc_mask,
        exclusion_mask=cotree_mask,
        weights=edge_weights,
        keep_super_node=False,
    ).T

    mst_idx = splx_search(
        key_splx=canon_edges,
        query_splx=mst,
        sort_key_splx=True,
        sort_key_vert=False,
        sort_query_vert=True,
    )

    tree_mask = torch.zeros(
        canon_edges.shape[0], dtype=torch.bool, device=adjacency.device
    )
    tree_mask[mst_idx] = True

    return tree_mask


def _cbd_to_coface(
    cbd: Float[Tensor, "kp1_splx k_splx"], degree: int
) -> tuple[Integer[LongTensor, " face"], Integer[LongTensor, "face degree"]]:
    """
    Find the cofaces of all k-simplices of a given degree.

    For a given k-coboundary operator, find the indices of all k-simplices of
    degree d (i.e., the number of cofaces of the k-simplices), and, for each
    k-simplex of degree d, determine the indices of the d (k+1)-simplices that
    share the k-simplex as a face.

    Parameters
    ----------
    cbd : [kp1_splx, k_splx]
        The k-coboundary operator.
    degree
        The degree of k-simplices.

    Returns
    -------
    unique_face_idx : [face,]
        The indices of all k-simplices of the given degree. the indices are
        sorted in ascending order.
    coface_idx : [face, degree]
        Each row corresponds to a k-simplex of the given degree in the
        `unique_face_idx` list, and each row contains the indices of the (k+1)-
        simplices that contain the k-simplex as a face. The (k+1)-simplex
        indices are sorted in ascending order within each row; however, the
        coface_idx as a whole is not guaranteed to be in lex-order.
    """
    idx_coo = cbd.pattern.idx_coo
    # The row indices correspond to the (k+1)-simplex indices, and the col indices
    # correspond to the k-simplex indices.
    idx_coo_col = idx_coo[-1]

    # Find how many times the index of each k-simplex shows up in the column index
    # For example, if a k-simplex is the face of two (k+1)-simplices, then it
    # will show up as two nonzero elements in the its column in cbd[k].
    face_idx, face_degree = torch.unique(idx_coo_col, return_counts=True)

    # Filter for the indices of all k-simplices that have the desired degree, then
    # turn this into a boolean mask for the coo indices.
    shared_face_idx = face_idx[face_degree == degree]
    shared_face_mask = torch.isin(idx_coo_col, shared_face_idx)

    # Get the subset of cbd coo indices corresponding to the k-simplices with the
    # desired degree; by sorting the coo col indices (the k-simplex indices), the
    # coface (k+1)-simplices (the coo row indices) for each k-simplex are re-ordered
    # next to each other, which can then be reshaped to generate the coface list.
    idx_coo_subset = idx_coo[:, shared_face_mask]
    sort_idx = torch.sort(idx_coo_subset[-1], stable=True).indices
    coface_idx = idx_coo_subset[-2][sort_idx].reshape(-1, degree)

    unique_face_idx = idx_coo_subset[-1, sort_idx][::degree]

    return unique_face_idx, coface_idx


# TODO: check whether applicable to nonmanifold meshes.
# TODO: update to accommodate tet meshes
def compute_cotree_mask(
    dual_topo_laplacian_0: Float[SparseDecoupledTensor, "dual_vert dual_vert"],
    cbd_1: Float[SparseDecoupledTensor, "tri edge"],
    inv_mass_1: Float[SparseDecoupledTensor, "edge edge"] | None = None,
    edge_rel_bc_mask: Bool[Tensor, " edge"] | None = None,
) -> Bool[Tensor, " edge"]:
    """
    Compute the spanning forest on the dual 1-skeleton of a tri mesh.

    The dual spanning tree/forest (also known as the cotree) can be used to fix
    the gauge freedom of the up/curl-curl component of the weak 1-Laplacian.

    Parameters
    ----------
    dual_topo_laplacian_0 : [dual_vert, dual_vert]
        The topological/combinatorial 0-Laplacian of the dual complex of a tri mesh.
    cbd_1 : [tri, edge]
        The 1-coboundary operator of the mesh.
    inv_mass_1 : [edge, edge]
        The inverse of the Hodge 1-star or consistent 1-mass operator. If provided,
        this function will compute a maximum dual spanning forest using the the
        diagonal elements of the matrix as weights/inverse edge masses, which should
        result in a better condition number for the gauge fixed linear system.
    edge_rel_bc_mask : [edge,]
        A boolean mask that mark (primal) edges subject to relative boundary
        condition(s).

    Returns
    -------
    [edge,]
        A boolean mask that mark edges that belong to the dual spanning forest.
    """
    # The dual 1-skeleton of a tri mesh consists of dual vertices that correspond
    # to the primal triangles, and two dual vertices are connected by a dual
    # edge iff they share a primal edge as a face. The dual topological 0-Laplacian
    # encodes the adjacency information for this dual 1-skeleton, plus the
    # truncated/clipped dual edges that correspond to primal boundary edges (if
    # there are any). Such truncated edges show up in the diagonal of the 0-Laplacian,
    # which are disgarded when taking the upper triangular part of the matrix.
    adjacency = dual_topo_laplacian_0.triu(diagonal=1).abs()

    # Here, the dual edges are indexed by the indices of the two primal triangles
    # sharing the primal edge; need to convert this representation of the dual
    # edges to the indices of the corresponding primal edges.
    dual_edges = adjacency.pattern.idx_coo.T
    edge_face_idx, edge_coface_idx = _cbd_to_coface(cbd_1, degree=2)
    primal_edge_idx = edge_face_idx[
        splx_search(
            key_splx=edge_coface_idx,
            query_splx=dual_edges,
            sort_key_splx=True,
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
    # clipped dual edges need to connect the dual vertex corresponding to its
    # primal triangle coface to the super node. To find such dual vertices/primal
    # triangles, we check for rows of the 1-coboundary operator that contain
    # edges marked with a relative boundary condition.
    if edge_rel_bc_mask is None:
        bd_dual_vert_mask = torch.zeros(
            cbd_1.shape[0], dtype=torch.bool, device=cbd_1.device
        )
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
        splx_search(
            key_splx=edge_coface_idx,
            query_splx=mst,
            sort_key_splx=True,
            sort_key_vert=False,
            sort_query_vert=True,
        )
    ]

    cotree_mask = torch.zeros(
        cbd_1.shape[-1], dtype=torch.bool, device=adjacency.device
    )
    cotree_mask[mst_idx] = True

    return cotree_mask

__all__ = ["compute_tree_mask", "compute_cotree_mask"]

import numpy as np
import torch
from jaxtyping import Bool, Float, Int64, Integer
from scipy.sparse import coo_array
from scipy.sparse.csgraph import minimum_spanning_tree
from torch import Tensor

from ..boundary.selection import BoundarySelection
from ..sparse.decoupled_tensor import BaseDecoupledTensor, SparseDecoupledTensor
from ..utils.parsing import to_np
from ..utils.search import splx_search


def _minimum_spanning_tree(
    adjacency: Float[SparseDecoupledTensor, "node node"],
    root_mask: Bool[Tensor, " node"] | None = None,
    exclusion_mask: Bool[Tensor, " edge"] | None = None,
    weights: Float[Tensor, " edge"] | None = None,
    super_weights: Float[Tensor, " root"] | None = None,
    keep_super_node: bool = False,
) -> Integer[Tensor, "mst_node 2"]:
    r"""
    Compute the minimum spanning forest over a graph.

    Parameters
    ----------
    adjacency : [node, node]
        The adjacency matrix of the undirected graph. The adjacency matrix needs
        not be symmetric; if $A_{ij} \ne A_{ji}$, then the minimum nonzero value
        of the two will be used as the edge weight.
    root_mask : [node,]
        A boolean mask marking nodes that will be connected to a virtual "super
        node". When no `super_weights` are supplied, these connections are forced
        into the tree using weights smaller than all ordinary edge weights; removing
        the super node then produces a forest rooted at the marked nodes. If
        `super_weights` are supplied, the connections to the super node compete
        normally with the ordinary edges and need not all be selected.
    exclusion_mask : [edge,]
        A boolean mask whose elements correspond to the nonzero elements of the
        adjacency matrix; an edge in the adjacency matrix marked as True by the
        this mask is disallowed when constructing the spanning forest.
    weights : [edge,]
        A list of weights that, when provided, override the edge weights in the
        adjacency matrix.
    super_weights : [root,]
        A list of weights for connections between the virtual super node and the
        nodes marked by `root_mask`. If omitted, automatically choose a weight
        smaller than all ordinary edge weights. Ignored when no nodes are marked
        in `root_mask`.
    keep_super_node
        Whether to keep the super node in the output spanning forest. If no
        valid `root_mask` is provided, this argument is ignored.

    Returns
    -------
    tree_edges : [mst_node, 2]
        The computed minimum spanning forest; each row in the tensor represents
        the indices of a graph node pair that is connected by an edge in the forest.

    Notes
    -----
    This function uses `scipy.sparse.csgraph.minimum_spanning_tree()` to construct
    the minimum spanning forest.
    """
    n_nodes = adjacency.size(0)

    # int32 is required for scipy sparse array indices
    idx_coo_full = adjacency.pattern.idx_coo.to(dtype=torch.int32)
    idx_coo_rows_full = to_np(idx_coo_full[0])
    idx_coo_cols_full = to_np(idx_coo_full[1])

    # If provided, the weights overwrite the adjacency matrix data.
    coo_data_full = to_np(adjacency.values if weights is None else weights)

    # If provided, use the exclusion mask to remove banned edges.
    if exclusion_mask is None:
        idx_coo_rows = idx_coo_rows_full
        idx_coo_cols = idx_coo_cols_full
        coo_data = coo_data_full
    else:
        exclusion_mask_np = to_np(exclusion_mask)

        idx_coo_rows = idx_coo_rows_full[~exclusion_mask_np]
        idx_coo_cols = idx_coo_cols_full[~exclusion_mask_np]
        coo_data = coo_data_full[~exclusion_mask_np]

    if (root_mask is not None) and (root_mask.any()):
        root_mask_np = to_np(root_mask)

        idx_dtype = idx_coo_rows.dtype
        data_dtype = coo_data.dtype

        # Add a super node connected to all marked nodes.
        super_node_idx = n_nodes
        root_idx = np.flatnonzero(root_mask_np).astype(idx_dtype)

        # Augment the adjacency matrix by adding in edges connecting the super node.
        new_rows = np.full(len(root_idx), super_node_idx, dtype=idx_dtype)
        new_cols = root_idx

        if super_weights is None:
            # Force root connections only when no physical weights are supplied.
            min_weight = coo_data.min() if len(coo_data) > 0 else 0.0
            super_weight = min_weight - np.abs(min_weight) - 1.0
            new_data = np.full(len(root_idx), super_weight, dtype=data_dtype)
        else:
            new_data = to_np(super_weights, dtype=data_dtype)

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

    tree_edges = torch.from_numpy(np.column_stack((tree_u, tree_v))).to(
        dtype=adjacency.pattern.dtype, device=adjacency.device
    )

    return tree_edges


def compute_tree_mask(
    topo_laplacian_0: Float[SparseDecoupledTensor, "global_vert global_vert"],
    canon_edges: Integer[Tensor, "global_edge local_vert=2"],
    mass_1: Float[BaseDecoupledTensor, "global_edge global_edge"] | None = None,
    bc_sel: BoundarySelection | None = None,
    cotree_mask: Bool[Tensor, " global_edge"] | None = None,
) -> Bool[Tensor, " global_edge"]:
    """
    Compute the spanning forest on the 1-skeleton of a tri mesh.

    The forest can be used to fix the gauge freedom of the down/grad-div component
    of the weak 1-Laplacian. Note that, if this function is used as part of the
    tree-cotree decomposition for the full 1-Laplacian gauge fixing, the
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
        will compute a maximum spanning forest using its diagonal as edge masses,
        which may result in a better condition number for the gauge fixed linear system.
    bc_sel
        The tangential boundary subcomplex of the tri mesh. Vertices in the subcomplex
        are connected to a virtual super node, and edges in the subcomplex are
        excluded. If omitted, assume that the mesh has no boundaries or subject
        to absolute boundary conditions.
    cotree_mask : [global_edge,]
        A boolean mask that marks edges that belong to the cotree.

    Returns
    -------
    tree_mask : [global_edge,]
        A boolean mask that mark edges that belong to the spanning forest.
    """
    # Verts in the tangential boundary subcomplex need to be the roots of the
    # spanning forest.
    root_mask = None if bc_sel is None else bc_sel.constrained_mask[0]
    # Edges in the cotree or the tangential boundary subcomplex need to be excluded
    # from the trees.
    match cotree_mask, bc_sel:
        case None, None:
            exclusion_mask = None
        case _, None:
            exclusion_mask = cotree_mask
        case None, _:
            exclusion_mask = bc_sel.constrained_mask[1]
        case _, _:
            exclusion_mask = cotree_mask | bc_sel.constrained_mask[1]

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
        root_mask=root_mask,
        exclusion_mask=None if exclusion_mask is None else exclusion_mask[edge_idx],
        weights=edge_weights,
        keep_super_node=False,
    )

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


def _filter_by_upper_deg(
    cbd: Float[SparseDecoupledTensor, "kp1_splx k_splx"], upper_degree: int
) -> tuple[Int64[Tensor, " face"], Int64[Tensor, "face degree"]]:
    """
    Find the cofaces of all k-simplices of a given upper degree.

    For a given k-coboundary operator, find the indices of all k-simplices of
    upper degree d (i.e., the number of cofaces of the k-simplices), and, for each
    k-simplex of upper degree d, determine the indices of the d (k+1)-simplices
    that share the k-simplex as a face.

    Parameters
    ----------
    cbd : [kp1_splx, k_splx]
        The k-coboundary operator.
    upper_degree
        The upper degree of k-simplices.

    Returns
    -------
    deg_k_splx : [face,]
        The indices of all k-simplices of the given upper degree. the indices are
        sorted in ascending order.
    codim1_cofaces : [face, degree]
        Each row corresponds to a k-simplex of the given upper degree in the
        `deg_k_splx` list, and each row contains the indices of the (k+1)-
        simplices that contain the k-simplex as a face. The (k+1)-simplex
        indices are sorted in ascending order within each row; however, the
        `codim1_cofaces` as a whole is not guaranteed to be in lex-order.
    """
    device = cbd.device
    dtype = torch.int64

    # The CSC idx_ccol index array of d_k encode information on the number of
    # codim-1 cofaces of each k-simplex.
    offsets = cbd.pattern.idx_ccol
    face_degree = offsets[1:] - offsets[:-1]
    deg_k_splx = torch.where(face_degree == upper_degree)[0].to(dtype)

    # Create a 2D index tensor I for the idx_row_csc index array of d_k where
    # I_ij points to the j-th codim-1 coface of the i-th k-simplex of the given
    # upper degree.
    idx = offsets[deg_k_splx].unsqueeze(-1) + torch.arange(
        upper_degree, dtype=dtype, device=device
    ).unsqueeze(0)
    codim1_cofaces = cbd.pattern.idx_row_csc[idx].to(dtype)

    return deg_k_splx, codim1_cofaces


def compute_cotree_mask(
    dual_topo_laplacian_0: Float[SparseDecoupledTensor, "dual_vert dual_vert"],
    cbd_1: Float[SparseDecoupledTensor, "tri edge"],
    inv_mass_1: Float[BaseDecoupledTensor, "edge edge"] | None = None,
    bc_sel: BoundarySelection | None = None,
) -> Bool[Tensor, " edge"]:
    r"""
    Compute the spanning forest on the dual 1-skeleton of an orientable manifold tri mesh.

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
        this function will compute a maximum dual spanning forest using its
        diagonal as inverse edge masses, which may improve the condition number
        of the gauge fixed linear system.
    bc_sel
        The tangential boundary subcomplex of the tri mesh. Edges in the subcomplex
        are excluded. If omitted, assume that the mesh has no boundaries or subject
        to absolute boundary conditions.

    Returns
    -------
    cotree_mask : [edge,]
        A boolean mask marking primal edges belonging to the dual spanning forest,

    Notes
    -----
    The input mesh is assumed to be an orientable manifold triangle mesh. Nonmanifold
    creases shared by more than two triangles are not supported. On nonorientable
    surfaces, the algorithm effectively assumes that the (co)chain complex is defined
    over the field $\mathbb{Z}_2$ rather than $\mathbb{R}$.
    """
    device = dual_topo_laplacian_0.device
    int_dtype = torch.int64
    float_dtype = inv_mass_1.dtype if inv_mass_1 else cbd_1.dtype

    n_tris, n_edges = cbd_1.shape

    # We want to determine the MaxST over a modified dual 1-skeleton of the
    # tri mesh. This 1-skeleton consists of dual verts that correspond to the
    # primal tris, and two dual verts are conntected by a (full) dual edge iff
    # they share a primal interior edge as a face. The boundary edges, if there
    # are any, are represented as dual half edges, and we modify such half edges
    # by "completing" them with connections to a virtual super node.

    # First, determine the connectivity of the interior of the dual 1-skeleton
    # (without the dual half edges) and the weights associated with the dual full
    # edges.

    # The dual topological 0-Laplacian encodes the adjacency information for this
    # dual 1-skeleton, plus the half dual edges, which show up in the diagonal and
    # are discarded when taking the upper triangular part of the matrix.
    adjacency = dual_topo_laplacian_0.triu(diagonal=1).abs()

    # Find the interior edges and the codim-1 cofaces of the interior edges.
    int_edges, int_edge_cofaces = _filter_by_upper_deg(cbd_1, upper_degree=2)

    if inv_mass_1 is None:
        dual_edge_weights = None

    else:
        diag_mass = inv_mass_1.diagonal()

        # The dual edges are indexed by the indices of the two primal triangles
        # sharing the primal edge; need to convert this representation of the dual
        # edges to the indices of the corresponding primal edges in order to extract
        # edge weights from the mass matrices.
        dual_edges = adjacency.pattern.idx_coo.T

        # Search for the (full) dual edges in the list of interior edge cofaces
        # (both of which are represented by primal tri pairs) to find their idx
        # in the list of interior primal edges.
        dual_edge_primal_idx = int_edges[
            splx_search(
                key_splx=int_edge_cofaces,
                query_splx=dual_edges,
                sort_key_splx=True,
                sort_key_vert=False,
                sort_query_vert=True,
            )
        ]

        dual_edge_weights = -diag_mass[dual_edge_primal_idx]

    # Second, identify the dual half edges and their weights.

    # Find the boundary edges outside of the tangential boundary subcomplex and
    # the codim-1 cofaces of such boundary edges.
    bd_edges, bd_edge_cofaces = _filter_by_upper_deg(cbd_1, upper_degree=1)
    if bc_sel is not None:
        retained_mask = bc_sel.retained_mask[1][bd_edges]
        bd_edges = bd_edges[retained_mask]
        bd_edge_cofaces = bd_edge_cofaces[retained_mask]

    # Since each boundary edge by definition only has one codim-1 coface, squeeze
    # out the trivial last dim from bd_edge_cofaces.
    bd_edge_cofaces = bd_edge_cofaces.squeeze(dim=-1)

    if inv_mass_1 is None:
        bd_edge_weights = torch.ones(bd_edges.size(0), dtype=float_dtype, device=device)
    else:
        bd_edge_weights = -diag_mass[bd_edges]

    # The primal boundary edges correspond to dual half edges that will be connected
    # to a virtual super node for the MaxST construction. However, if a primal
    # tri has multiple boundary edges, this can create multiple parallel connections
    # between the super node and a dual vert, which is unnecessary. Therefore, for
    # primal tris with multiple boundary edges, we keep only one representative
    # edge with the lowest weight.
    tri_rep_weights = torch.full((n_tris,), torch.inf, dtype=float_dtype, device=device)
    tri_rep_weights.scatter_reduce_(
        dim=0, index=bd_edge_cofaces, src=bd_edge_weights, reduce="amin"
    )

    # Also determine, for each tri, the index of its boundary edge, if there is any,
    # that was chosen as the representative. If multiple boundary edges of a given
    # triangle have the same lowest weight, then tie-break by picking the one with the
    # smallest index.
    rep_bd_edge_mask = bd_edge_weights == tri_rep_weights[bd_edge_cofaces]
    tri_rep_bd_edge = torch.full((n_tris,), n_edges, dtype=int_dtype, device=device)
    tri_rep_bd_edge.scatter_reduce_(
        dim=0,
        index=bd_edge_cofaces[rep_bd_edge_mask],
        src=bd_edges[rep_bd_edge_mask],
        reduce="amin",
    )

    # Mark the dual verts eligible for connection to the virtual super node, and
    # extract the connection weights from tri_rep_weights.
    bd_dual_vert_mask = torch.zeros(n_tris, dtype=torch.bool, device=device)
    bd_dual_vert_mask[bd_edge_cofaces] = True

    dual_half_edge_weights = tri_rep_weights[bd_dual_vert_mask]

    # Lastly, compute the MaxST and extract the cotree mask.

    # Note that, unlike compute_tree_mask(), connections to the virtual super
    # node are not mandatory.
    mst = _minimum_spanning_tree(
        adjacency=adjacency,
        root_mask=bd_dual_vert_mask,
        exclusion_mask=None,
        weights=dual_edge_weights,
        super_weights=dual_half_edge_weights,
        keep_super_node=True,
    )

    # Any dual edges on the MaxST connecting to the super node (indexed to n_tris)
    # correspond to the dual half edges of the original dual 1-skeleton.
    mst_dual_half_edge_mask = (mst == n_tris).any(dim=1)

    # Again, need to convert the dual edge representation as a pair of dual vertices
    # (primal triangles) to the corresponding primal interior edge indices.
    mst_int_edges = int_edges[
        splx_search(
            key_splx=int_edge_cofaces,
            query_splx=mst[~mst_dual_half_edge_mask],
            sort_key_splx=True,
            sort_key_vert=False,
            sort_query_vert=True,
        )
    ]

    cotree_mask = torch.zeros(n_edges, dtype=torch.bool, device=device)
    cotree_mask[mst_int_edges] = True

    # For a dual half edge selected on the MaxST, one of the indices represent
    # the super node (indexed to n_tris), and the other represent a dual boundary
    # vert (i.e., a primal tri containing a representative boundary edge).
    # Since n_tris is always strictly larger than any actual dual vert index,
    # min(dim=1).values always get the dual boundary vert index, which is then
    # mapped by tri_rep_bd_edge to the index of the representative primal boundary
    # edge.
    mst_bd_verts = mst[mst_dual_half_edge_mask].min(dim=1).values
    cotree_mask[tri_rep_bd_edge[mst_bd_verts]] = True

    return cotree_mask

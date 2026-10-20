__all__ = ["compute_morse_complex"]


import heapq

import numba
import numpy as np
import numpy.typing as npt
import torch
from einops import reduce
from jaxtyping import Bool, Float, Integer
from torch import Tensor

from ..boundary.selection import BoundarySelection
from ..complex import SimplicialMesh
from ..sparse.decoupled_tensor import SparseDecoupledTensor
from ..utils.parsing import to_np

# Define negative sentinel values for the pairing map.
CRITICAL = -1
EXCLUDED = -2
UNPROCESSED = -3
NOT_CRITICAL = -4


def _prepare_inputs(
    mesh: SimplicialMesh,
    scalar_field: Float[Tensor, " vert"],
    constrained_masks: tuple[
        Bool[Tensor, " vert"],
        Bool[Tensor, " edge"],
        Bool[Tensor, " tri"],
        Bool[Tensor, " tet"],
    ],
    int_dtype: np.dtype,
):
    """
    Prepare input mesh and scalar field for discrete Morse theory.

    Parameters
    ----------
    mesh
        A simplicial complex.
    scalar_field
        A scalar function defined/sampled at the verts of the mesh.
    constrained_masks
        A tuple of boolean masks. `constrained_mask[k]` marks the k-simplices
        in the tangential boundary subcomplex.
    int_dtype
        The integer dtype used to represent simplex indices, typically either
        np.int32 or np.int64. Must be a signed dtype because the negative sentinel
        values will be used to flag certain simplices.

    Returns
    -------
    ordered_verts_np
        The indices of mesh verts ordered by their associated scalar field values
        (in ascending order) with tie break by vert indices.
    vert_ranks_np
        Integer ranks assigned to verts based on ordered_verts_np.
    splx_vert_rank_max
        The max vert rank among all vert faces of a simplex.
    splx_scalar_sum
        The scalar value sum over all vert faces for each simplex.
    vert_coface_indices
        The csc nonzero element row indices of the coboundary operators $d_0$,
        $|d_1||d_0|$, and $|d_2||d_1||d_0|$, which contains vert coface global
        indices.
    vert_coface_offset
        The csc nonzero element ccol indices of the coboundary operators $d_0$,
        $|d_1||d_0|$, and $|d_2||d_1||d_0|$, which contains information on the
        number of cofaces for each vert.
    codim1_coface_indices
        The csc nonzero element row indices of the coboundary operators $d_0$,
        $d_1$, and $d_2$, which contains codimension 1 coface global indices.
    codim1_coface_offset
        The csc nonzero element ccol indices of the coboundary operators $d_0$,
        $d_1$, and $d_2$, which contains information on the number of codimension
        1 cofaces for each simplex.
    codim1_face_indices
        The csr nonzero element col indices of the coboundary operators $d_0$,
        $d_1$, and $d_2$, which contains codimension 1 face global indices.
    codim1_face_offset
        The csr nonzero element crow indices of the coboundary operators $d_0$,
        $d_1$, and $d_2$, which contains information on the number of codimension
        1 faces for each simplex.
    codim1_face_signs
        The csr nonzero element values of the coboundary operators $d_0$, $d_1$,
        and $d_2$, which contains face boundary signs.
    max_n_cofaces
        The maximum number of vert, edge, tri, and tet cofaces among the verts
        in the mesh, which is used to allocate a buffer array later for tracking
        lower star membership for acyclic partial matching.
    pairing_map
        An integer array for tracking acyclic partial matching; initialized to
        UNPROCESSED for retained simplices and EXCLUDED for constrained simplices;
        see _process_lower_stars() for more details.
    splx_dim_offsets
        An integer array containing cumulative simplex counts; more explicitly,
        it is [0, V, V+E, V+E+F, V+E+F+T].
    """
    # Attach the scalar field to higher-order simplices
    splx_scalar_fields = (
        to_np(scalar_field.view(-1, 1)),
        to_np(scalar_field[mesh.edges]),
        to_np(scalar_field[mesh.tris]),
        to_np(scalar_field[mesh.tets]),
    )

    splx_scalar_sum = np.concat(
        [
            reduce(splx_scalar_field, "splx vert -> splx", reduction="sum")
            for splx_scalar_field in splx_scalar_fields
        ]
    )

    # Sort the vertices in ascending order by the scalar value; stable=True
    # ensures tiebreak using vertex indices.
    ordered_verts = torch.argsort(scalar_field, stable=True, dim=0)
    ordered_verts_np = to_np(ordered_verts, dtype=int_dtype)

    # Use the vert ordering to assign a ranking to each vert, which establishes
    # a strict total ordering of the verts.
    vert_ranks = torch.empty_like(
        ordered_verts, dtype=ordered_verts.dtype, device=ordered_verts.device
    )
    vert_ranks[ordered_verts] = torch.arange(
        ordered_verts.size(0), dtype=ordered_verts.dtype, device=ordered_verts.device
    )
    vert_ranks_np = to_np(vert_ranks, dtype=int_dtype)

    # Use the max vert rank to assign a rank score per simplex.
    splx_vert_ranks = (
        to_np(vert_ranks.view(-1, 1), dtype=int_dtype),
        to_np(vert_ranks[mesh.edges], dtype=int_dtype),
        to_np(vert_ranks[mesh.tris], dtype=int_dtype),
        to_np(vert_ranks[mesh.tets], dtype=int_dtype),
    )

    splx_vert_rank_max = np.concat(
        [
            reduce(splx_vert_rank, "splx vert -> splx", reduction="max")
            for splx_vert_rank in splx_vert_ranks
        ]
    )

    # Prepare index data for a global simplex indexing scheme, where all 0-, 1-,
    # 2-, and 3-simplices are ordered in a flattened list and assigned a 0-index.
    # In particular, splx_dim_offsets is [0, V, V+E, V+E+F, V+E+F+T].
    splx_dim_offsets = np.array(
        [sum(mesh.n_splx[:k]) for k in range(len(mesh.n_splx) + 1)], dtype=int_dtype
    )

    # Precompute the coface relations between vertices and higher-order simplices.
    cbd_02 = mesh.cbd[1].abs() @ mesh.cbd[0].abs()
    cbd_03 = mesh.cbd[2].abs() @ cbd_02

    vert_coface_indices = (
        to_np(mesh.cbd[0].pattern.idx_row_csc, dtype=int_dtype) + splx_dim_offsets[1],
        to_np(cbd_02.pattern.idx_row_csc, dtype=int_dtype) + splx_dim_offsets[2],
        to_np(cbd_03.pattern.idx_row_csc, dtype=int_dtype) + splx_dim_offsets[3],
    )
    vert_coface_offset = (
        to_np(mesh.cbd[0].pattern.idx_ccol, dtype=int_dtype),
        to_np(cbd_02.pattern.idx_ccol, dtype=int_dtype),
        to_np(cbd_03.pattern.idx_ccol, dtype=int_dtype),
    )

    # Also compute the coface relations between k-simplices and (k+1)-simplices.
    # Note that, for the k-coboundary operator, its idx_row_csc index tensor
    # contains the 0-indices of the k-simplices, and we need to shift this local
    # 0-index scheme to the global simplex index scheme described above.
    codim1_coface_indices = (
        to_np(mesh.cbd[0].pattern.idx_row_csc, dtype=int_dtype) + splx_dim_offsets[1],
        to_np(mesh.cbd[1].pattern.idx_row_csc, dtype=int_dtype) + splx_dim_offsets[2],
        to_np(mesh.cbd[2].pattern.idx_row_csc, dtype=int_dtype) + splx_dim_offsets[3],
    )
    codim1_coface_offset = (
        to_np(mesh.cbd[0].pattern.idx_ccol, dtype=int_dtype),
        to_np(mesh.cbd[1].pattern.idx_ccol, dtype=int_dtype),
        to_np(mesh.cbd[2].pattern.idx_ccol, dtype=int_dtype),
    )

    # Similarly compute the face relations between (k+1)-simplices and k-simplices
    # as well as the boundary signs. Note that, for the k-coboundary operator,
    # its idx_col index tensor contains the 0-indices of the (k-1)-simplices, and
    # we need to shift this local 0-index scheme to the global index scheme
    # described above.
    codim1_face_indices = (
        to_np(mesh.cbd[0].pattern.idx_col, dtype=int_dtype) + splx_dim_offsets[0],
        to_np(mesh.cbd[1].pattern.idx_col, dtype=int_dtype) + splx_dim_offsets[1],
        to_np(mesh.cbd[2].pattern.idx_col, dtype=int_dtype) + splx_dim_offsets[2],
    )
    codim1_face_offset = (
        to_np(mesh.cbd[0].pattern.idx_crow, dtype=int_dtype),
        to_np(mesh.cbd[1].pattern.idx_crow, dtype=int_dtype),
        to_np(mesh.cbd[2].pattern.idx_crow, dtype=int_dtype),
    )
    codim1_face_signs = (
        to_np(mesh.cbd[0].values),
        to_np(mesh.cbd[1].values),
        to_np(mesh.cbd[2].values),
    )

    # Determine the max possible lower star size. To compute this size, we
    # determine, for each vert, the number of its vert, edge, tri, and tet cofaces
    # and sum them together, and take the max of this value over the list of verts.
    # Note that the difference between consecutive elements in the ccol index
    # tensor of a coboundary operator gives the number of nonzero elements in each
    # column of the operator, which is equal to the number of cofaces.
    n_cofaces = np.stack(
        [idx_ccol[1:] - idx_ccol[:-1] for idx_ccol in vert_coface_offset]
    )
    # + 1 to account for vert cofaces (which is just itself).
    max_n_cofaces = 1 + int((n_cofaces.sum(axis=0)).max().item())

    # Prepare the pairing map; if σ is a codim 1 face of τ and they are paired/matched,
    # then p[τ] = σ and p[σ] = τ.
    pairing_map = np.full(sum(mesh.n_splx), UNPROCESSED, dtype=int_dtype)

    # For simplices in the tangential boundary subcomplex, use a different
    # sentinel value to indicate that they will be excluded.
    for i, mask in enumerate(constrained_masks):
        mask_np = to_np(mask, dtype=np.bool)
        pairing_map[splx_dim_offsets[i] : splx_dim_offsets[i + 1]][mask_np] = EXCLUDED

    return (
        ordered_verts_np,
        vert_ranks_np,
        splx_vert_rank_max,
        splx_scalar_sum,
        vert_coface_indices,
        vert_coface_offset,
        codim1_coface_indices,
        codim1_coface_offset,
        codim1_face_indices,
        codim1_face_offset,
        codim1_face_signs,
        max_n_cofaces,
        pairing_map,
        splx_dim_offsets,
    )


@numba.jit(nopython=True, cache=True)
def _process_lower_stars(
    mesh_dim: int,
    ordered_verts: npt.NDArray,
    vert_ranks: npt.NDArray,
    splx_vert_rank_max: npt.NDArray,
    splx_scalar_sum: npt.NDArray,
    vert_coface_indices: tuple[npt.NDArray, ...],
    vert_coface_offset: tuple[npt.NDArray, ...],
    codim1_coface_indices: tuple[npt.NDArray, ...],
    codim1_coface_offset: tuple[npt.NDArray, ...],
    codim1_face_indices: tuple[npt.NDArray, ...],
    codim1_face_offset: tuple[npt.NDArray, ...],
    max_n_cofaces: int,
    pairing_map: npt.NDArray,
    splx_dim_offsets: npt.NDArray,
) -> Integer[np.ndarray, " splx"]:
    r"""
    Construct an acyclic partial matching using lower star filtration.

    Let $f(v)$ be a scalar field associated with each vertex on a mesh. For each
    vert $v$ in the mesh, we define the lower star $L(v)$ of $v$ to be the set of
    all simplices $\sigma$ such that (1) $\sigma$ is not in the tangential boundary
    subcomplex, (2) $v \le \sigma$, and (3) $f(v) > f(v')$ for all other vert faces
    $v'$ of $\sigma$; in case of a tie where $f(v) = f(v')$, $\sigma \in L(v)$ if
    the index of $v$ is higher than that of $v'$.

    To construct the acyclic partial matching, use the two-queue lower-star strategy
    of Robins, Wood & Sheppard, TPAMI (2011). Initially, all simplices in $L(v)$
    are marked as unprocessed, and the goal of the strategy is to mark all simplices
    in $L(v)$ as either paired/matched or critical. A simplex enters the "one-face
    queue" if it has exactly one unprocessed codim-1 face in $L(v)$, and the
    "zero-face queue" if it has no unprocessed codim-1 face in $L(v)$. At each
    iteration, if the one-face queue is not empty, pair a simplex in the one-face
    queue with its unique unprocessed codim-1 face, otherwise mark a simplex from
    the zero-face queue as critical. In either case, matched or critical simplices
    lower the face counts of their cofaces; both queues prioritize processing
    simplices with the lowest sum of $f$ over their verts, with ties broken by
    dimension and global simplex index.

    When $\sigma$ is matched with its face $\tau$, every other face of $\sigma$
    in $L(v)$ has already been processed. Hence along any gradient path, the pair
    (max vertex rank, processing time) strictly decreases lexicographically, so
    the matching is guaranteed to be acyclic.

    This function returns a `pairing_map` array, where `pairing_map[i] = j` for
    `i, j >= 0` indicates that the simplices `i` and `j` are paired,
    `pairing_map[i] = CRITICAL` indicates that simplex `i` is critical, and
    `pairing_map[i] = EXCLUDED` indicates that the simplex `i` is excluded due to
    tangential boundary subcomplex membership. Here, the simplices are 0-indexed
    consecutively, where the lower-dimensional simplices are placed before
    higher-dimensional ones, and simplices of the same dimension maintain the
    ordering in the mesh definition.
    """
    # Set up buffers to be reused across lower stars.

    # Buffers for keeping track of simplices in a given lower star.
    lower_star_idx_buffer = np.empty(max_n_cofaces, dtype=pairing_map.dtype)
    lower_star_dim_buffer = np.empty(max_n_cofaces, dtype=pairing_map.dtype)

    # A buffer that keeps track of how many unprocessed codim 1 faces of a given
    # simplex are in a lower star.
    unprocessed_faces_count = np.zeros(splx_dim_offsets[-1], dtype=np.int8)

    # Buffers for keeping track of simplices to be removed from consideration
    # in unprocessed_faces_count. At any given time, at most two such simplices
    # need to be tracked.
    removed_idx_buffer = np.empty(2, dtype=pairing_map.dtype)
    removed_dim_buffer = np.empty(2, dtype=pairing_map.dtype)

    # Initialize the zero-face and one-face queues; the queues should automatically
    # be empty at the end of processing the L(v) of each vertex v. The zero-length
    # list comprehension allows numba to infer element dtypes at compile time;
    # Scalar field value, simplex dimension, and simplex index are casted to
    # float64, int8, and int64, regardless of the input dtypes.
    zero_face_queue = [(np.float64(0), np.int8(0), np.int64(0)) for _ in range(0)]
    one_face_queue = [(np.float64(0), np.int8(0), np.int64(0)) for _ in range(0)]

    for vert in ordered_verts:
        # First, for each vertex v, determine its lower star L(v).

        # A "pointer" that keeps track of the size of the lower star of a vert.
        lower_star_size = 0
        vert_rank = vert_ranks[vert]

        # For each vert, there is only one coface of codimension 0: the vert itself.
        # Note that, for a v in the tangential boundary subcomplex, even though
        # v cannot be a member of L(v), we still need to determine L(v) itself.
        if pairing_map[vert] == UNPROCESSED:
            lower_star_idx_buffer[lower_star_size] = vert
            lower_star_dim_buffer[lower_star_size] = 0
            lower_star_size += 1

        for coface_dim in [1, 2, 3]:
            # Find the coface indices via the cbd csc indices.
            cbd_idx = vert_coface_indices[coface_dim - 1]
            cbd_offset = vert_coface_offset[coface_dim - 1]
            cofaces = cbd_idx[cbd_offset[vert] : cbd_offset[vert + 1]]

            # Check whether the identified cofaces are in the lower star. By
            # checking that the vert rank is equal to the coface rank score,
            # we ensure that the vert has the max scalar field value (and, in
            # the event of ties, the vert has higher index). This ensures that
            # the lower stars of the verts form a partition of the simplicial
            # complex (excluding the tangential boundary subcomplex).
            cofaces_are_lower = splx_vert_rank_max[cofaces] == vert_rank
            cofaces_are_not_excluded = pairing_map[cofaces] == UNPROCESSED
            lower_star_mask = cofaces_are_lower & cofaces_are_not_excluded

            lower_cofaces = cofaces[lower_star_mask]
            n_lower_cofaces = lower_star_mask.sum()

            # Get the index range for registering the cofaces in the buffer.
            start = lower_star_size
            end = lower_star_size + n_lower_cofaces

            lower_star_idx_buffer[start:end] = lower_cofaces
            lower_star_dim_buffer[start:end] = coface_dim
            lower_star_size += n_lower_cofaces

        # Second, for each L(v), mark each simplex in L(v) as either matched/paired
        # or critical.

        # For each (unprocessed) simplex σ in L(v), first determine how many codim
        # 1 faces of σ in L(v) remains unprocessed. In particular, keep track of
        # σ with either one or zero such faces in the one_face_queue and
        # zero_face_queue, respectively.
        for i in range(lower_star_size):
            splx = lower_star_idx_buffer[i]
            splx_dim = lower_star_dim_buffer[i]
            n_faces = 0

            if splx_dim > 0:
                # Find the face indices via the cbd csr indices. Note that face_idx
                # is in global index, while face_offset is in local, per-dim index.
                face_idx = codim1_face_indices[splx_dim - 1]
                face_offset = codim1_face_offset[splx_dim - 1]
                face_local_idx = splx - splx_dim_offsets[splx_dim]
                for face in face_idx[
                    face_offset[face_local_idx] : face_offset[face_local_idx + 1]
                ]:
                    if (
                        splx_vert_rank_max[face] == vert_rank
                        and pairing_map[face] == UNPROCESSED
                    ):
                        n_faces += 1

            unprocessed_faces_count[splx] = n_faces

            entry = (
                np.float64(splx_scalar_sum[splx]),
                np.int8(splx_dim),
                np.int64(splx),
            )
            if n_faces == 0:
                heapq.heappush(zero_face_queue, entry)
            elif n_faces == 1:
                heapq.heappush(one_face_queue, entry)

        # Start processing simplices in L(v); keep track of how many simplices in
        # L(v) have been processed (i.e., marked as paired or critical in pairing_map);
        # by the end of the while-loop, all simplices in L(v) should be processed.
        n_processed = 0
        while one_face_queue or zero_face_queue:
            # Process one σ from the one_face_queue if it is not empty; otherwise
            # process one σ from the zero_face_queue. Always prioritize σ with the
            # lowest sum of f over the vert faces when picking which simplex to
            # process.
            if one_face_queue:
                _, splx_dim, splx = heapq.heappop(one_face_queue)
                # Make sure that σ is still eligible to be in the one_face_queue.
                # A simplex can become ineligible if (1) it has been paired as a
                # face (it is no longer UNPROCESSED), or (2) its last face was
                # paired with another simplex in L(v) (its unprocessed_faces_count
                # is 0).
                if (
                    pairing_map[splx] != UNPROCESSED
                    or unprocessed_faces_count[splx] != 1
                ):
                    continue

                # Find the index of τ, the only remaining face of σ.
                face_idx = codim1_face_indices[splx_dim - 1]
                face_offset = codim1_face_offset[splx_dim - 1]
                face_local_idx = splx - splx_dim_offsets[splx_dim]
                unique_face = -1
                for face in face_idx[
                    face_offset[face_local_idx] : face_offset[face_local_idx + 1]
                ]:
                    if (
                        splx_vert_rank_max[face] == vert_rank
                        and pairing_map[face] == UNPROCESSED
                    ):
                        unique_face = face
                        break

                if unique_face == -1:
                    raise RuntimeError("One-face queue has no unprocessed face.")

                # register τ and σ as a matched pair.
                pairing_map[splx] = unique_face
                pairing_map[unique_face] = splx

                # register τ and σ for removal from unprocessed_faces_count so that
                # they are no longer considered as unprocessed faces of other
                # unprocessed simplices.
                removed_idx_buffer[0] = unique_face
                removed_dim_buffer[0] = splx_dim - 1
                removed_idx_buffer[1] = splx
                removed_dim_buffer[1] = splx_dim
                n_removed = 2

            else:
                # For σ in the zero_face_queue, check that it has not been paired
                # as the face of another simplex; if so, mark it as critical and
                # remove it from unprocessed_faces_count.
                _, splx_dim, splx = heapq.heappop(zero_face_queue)
                if pairing_map[splx] != UNPROCESSED:
                    continue

                pairing_map[splx] = CRITICAL
                removed_idx_buffer[0] = splx
                removed_dim_buffer[0] = splx_dim
                n_removed = 1

            # For simplices registered for removal, check if they are counted as
            # unprocessed faces of any other unprocessed simplices in L(v); if so,
            # remove them from consideration.
            for i in range(n_removed):
                removed_splx = removed_idx_buffer[i]
                removed_dim = removed_dim_buffer[i]
                # If a simplex being removed is a top-level simplex of the mesh,
                # then by definition it cannot be the face of another simplex;
                # so nothing needs to be done here.
                if removed_dim == mesh_dim:
                    continue

                coface_idx = codim1_coface_indices[removed_dim]
                coface_offset = codim1_coface_offset[removed_dim]
                coface_local_idx = removed_splx - splx_dim_offsets[removed_dim]
                for coface in coface_idx[
                    coface_offset[coface_local_idx] : coface_offset[
                        coface_local_idx + 1
                    ]
                ]:
                    if (
                        splx_vert_rank_max[coface] != vert_rank
                        or pairing_map[coface] != UNPROCESSED
                    ):
                        continue

                    unprocessed_faces_count[coface] -= 1
                    entry = (
                        np.float64(splx_scalar_sum[coface]),
                        np.int8(removed_dim + 1),
                        np.int64(coface),
                    )

                    if unprocessed_faces_count[coface] == 0:
                        heapq.heappush(zero_face_queue, entry)
                    elif unprocessed_faces_count[coface] == 1:
                        heapq.heappush(one_face_queue, entry)

            # Count the removed simplices as having been processed.
            n_processed += n_removed

        if n_processed != lower_star_size:
            raise RuntimeError("Lower star was not fully processed.")

    if np.any(pairing_map == UNPROCESSED):
        raise RuntimeError("Lower-star matching left unprocessed simplices.")

    return pairing_map


def _find_critical_splx(
    pairing_map: npt.NDArray,
    splx_dim_offsets: npt.NDArray,
) -> tuple[tuple[npt.NDArray, ...], npt.NDArray]:
    """
    Identify the critical simplices in an acyclic partial matching.

    This function takes in a `pairing_map` representing an acyclic partial
    matching and performs two processing steps: (1) it finds the "global" indices
    of the critical simplices and order them into separate arrays, one for
    each dimension, and (2) it returns a `crit_splx_reduced_idx` array, where
    `crit_splx_reduced_idx[i] = j` indicates that simplex `i` (indexed globally)
    is the `j`th critical simplex among all critical simplices of the same
    dimension as simplex `i`, and `crit_splx_reduced_idx[i] = NOT_CRITICAL` indicates
    that simplex `i` is not a critical simplex.
    """
    crit_splx = np.flatnonzero(pairing_map == CRITICAL)

    # Split the list of critical simplices by their dimensions.
    split_points = np.searchsorted(crit_splx, splx_dim_offsets[1:-1])
    crit_splx_by_dim = np.split(crit_splx, split_points)

    crit_splx_reduced_idx = np.full(
        splx_dim_offsets[-1], NOT_CRITICAL, dtype=pairing_map.dtype
    )
    for crit_splx in crit_splx_by_dim:
        crit_splx_local_idx = np.arange(crit_splx.size, dtype=pairing_map.dtype)
        crit_splx_reduced_idx[crit_splx] = crit_splx_local_idx

    return tuple(crit_splx_by_dim), crit_splx_reduced_idx


# TODO: option to save the gradient paths
# TODO: consider dynamic programming approach that avoids explicit path enumeration
@numba.jit(nopython=True, cache=True)
def _construct_morse_cbds(
    codim1_face_indices: tuple[npt.NDArray, ...],
    codim1_face_offset: tuple[npt.NDArray, ...],
    codim1_face_signs: tuple[npt.NDArray, ...],
    pairing_map: npt.NDArray,
    splx_dim_offsets: npt.NDArray,
    crit_splx_by_dim: tuple[npt.NDArray, ...],
    crit_splx_reduced_idx: npt.NDArray,
) -> tuple[list[npt.NDArray], list[npt.NDArray]]:
    r"""
    Construct the reduced Morse coboundary operators.

    Consider the $k$-th Morse coboundary operator $d$. The element $d_{ij}$ is
    the signed sum of the weights of all gradient paths outside of the tangential
    boundary subcomplex of the original mesh that connects the critical $(k+1)$-simplex
    $\tau_i$ to the critical $k$-simplex $\sigma_j$; it is zero if no such paths
    exist, or the weights associated with all such paths cancel exactly.

    A gradient path $\rho$ between $\tau_i$ and $\sigma_j$ is a sequence of
    alternating $(k+1)$- and $k$-dimensional simplices that satisfies either of
    the following two conditions:

    * The sequence contains only $\tau_i$ and $\sigma_j$, where $\sigma_j \le \tau_i$, or
    * The $\sigma_j$ is not a face of $\tau_i$ and the sequence contains additional
      simplices outside of the tangential boundary subcomplex, whereby every
      "step-down" satisfies a face relation and every "step-up" is part of an
      acyclic partial matching.

    We assign each step of the gradient path a weight: a step-down is assigned
    a weight equal to the boundary sign of the face, and a step-up is assigned
    a weight equal to the negative boundary sign of the face. Then, the weight
    associated with a gradient path is the product of the weight assigned to
    each step along the path.

    In this function, we perform the path search using a depth-first search
    approach.

    Note that, while the Morse complex is chain homotopic to the original (relative)
    simplicial cochain complex, the Morse complex itself is a CW complex, rather
    than a simplicial complex.
    """
    int_dtype = pairing_map.dtype
    float_dtype = codim1_face_signs[0].dtype

    # Generate LIFO buffer/stack for depth-first simplex search.
    stack_size = 2**15
    stack_splx = np.empty(stack_size, dtype=int_dtype)
    stack_sign = np.empty(stack_size, dtype=float_dtype)
    stack_top = 0

    cbd_idx_coo = []
    cbd_val = []

    for splx_dim in [3, 2, 1]:
        # Pre allocate the memories for the morse coboundary operator coo indices
        # and values. Since we don't actually know how many non-coalesced nonzero
        # elements there are per reduced coboundary operator ahead of time, we
        # use a heuristic that allocates 4*(splx_dim+1)*n_row nonzero slots. If this
        # is not enough, the buffer gets doubled dynamically (we set a minimum
        # buffer size of 10,000 to avoid potential, repeated size adjustments).
        cbd_n_rows = crit_splx_by_dim[splx_dim].size
        cbd_buffer_size = max(10_000, 4 * (splx_dim + 1) * cbd_n_rows)
        idx_coo = np.empty(
            (2, cbd_buffer_size),
            dtype=int_dtype,
        )
        val = np.empty(cbd_buffer_size, dtype=float_dtype)
        nnz = 0

        # Determine the index offset required to map from the global simplex
        # indices to the local, per-dim simplex indices.
        global_splx_offset = splx_dim_offsets[splx_dim]

        # Fetch cbd indices and values for face relation lookup.
        cbd_idx = codim1_face_indices[splx_dim - 1]
        # Note that offset still indexes simplices using a local, per-dim 0-index,
        # so element access from this array requires subtracting global_splx_offset.
        cbd_offset = codim1_face_offset[splx_dim - 1]
        bd_signs = codim1_face_signs[splx_dim - 1]

        for crit_splx in crit_splx_by_dim[splx_dim]:
            # For each critical k-simplex, find all of its codim 1 faces and push
            # them to the search queue.
            start = cbd_offset[crit_splx - global_splx_offset]
            end = cbd_offset[crit_splx + 1 - global_splx_offset]

            all_face_idx = cbd_idx[start:end]
            all_face_sign = bd_signs[start:end]

            # Exclude any faces in the tangential boundary subcomplex. This way,
            # any path that reaches the tangential boundary subcomplex contributes
            # 0, as required by the relative cohomology.
            include_mask = pairing_map[all_face_idx] != EXCLUDED
            face_idx = all_face_idx[include_mask]
            face_sign = all_face_sign[include_mask]

            n_faces = len(face_idx)

            # Before adding the faces to the search queue, check if at the stack
            # size limit; if so, double the stack size before proceeding.
            if stack_top + n_faces > stack_size:
                stack_size *= 2
                new_stack_splx = np.empty(stack_size, dtype=int_dtype)
                new_stack_sign = np.empty(stack_size, dtype=float_dtype)

                new_stack_splx[:stack_top] = stack_splx[:stack_top]
                new_stack_sign[:stack_top] = stack_sign[:stack_top]

                stack_splx = new_stack_splx
                stack_sign = new_stack_sign

            stack_splx[stack_top : stack_top + n_faces] = face_idx
            stack_sign[stack_top : stack_top + n_faces] = face_sign
            stack_top += n_faces

            # When the search queue is not empty.
            while stack_top > 0:
                # Pop the last simplex in the search queue; note that stack_top
                # needs to be decreased first since stack_top points to the
                # location right after the last valid element in the stack.
                stack_top -= 1
                current_face_idx = stack_splx[stack_top]
                current_face_sign = stack_sign[stack_top]

                paired_coface = pairing_map[current_face_idx]

                # Check whether the current_face is paired with one of its faces
                # or one of its cofaces.
                is_paired_up = paired_coface > current_face_idx

                # If the current_face is critical, then the path terminates.
                # Add the current_face_sign to (crit_splx, current_face_idx)
                # position of the coboundary operator.
                if paired_coface == CRITICAL:
                    # Before adding the entry to the coboundary operator, check
                    # if at the buffer size limit; if so, double the buffer size
                    # before proceeding.
                    if nnz + 1 > cbd_buffer_size:
                        cbd_buffer_size *= 2
                        new_idx_coo = np.empty(
                            (2, cbd_buffer_size),
                            dtype=int_dtype,
                        )
                        new_val = np.empty(cbd_buffer_size, dtype=float_dtype)

                        new_idx_coo[:, :nnz] = idx_coo[:, :nnz]
                        new_val[:nnz] = val[:nnz]

                        idx_coo = new_idx_coo
                        val = new_val

                    # Use the crit_splx_reduced_idx to map the global indices
                    # of the critical simplices to the reduced, local, per-dim
                    # 0-indices.
                    idx_coo[0, nnz] = crit_splx_reduced_idx[crit_splx]
                    idx_coo[1, nnz] = crit_splx_reduced_idx[current_face_idx]
                    val[nnz] = current_face_sign
                    nnz += 1

                # If the current_face is paired, add all the new faces of its
                # paired coface to the search queue.
                elif is_paired_up:
                    # For the paired coface, find all of its faces.
                    start = cbd_offset[paired_coface - global_splx_offset]
                    end = cbd_offset[paired_coface + 1 - global_splx_offset]

                    coface_all_faces_idx = cbd_idx[start:end]
                    coface_all_faces_sign = bd_signs[start:end]

                    # Exclude any faces in the tangential boundary subcomplex.
                    include_mask = pairing_map[coface_all_faces_idx] != EXCLUDED
                    coface_faces_idx = coface_all_faces_idx[include_mask]
                    coface_faces_sign = coface_all_faces_sign[include_mask]

                    # As the path moves from current_face to paired_coface, update
                    # the current_sign using the (negative of the) boundary sign
                    # of the current_face in the paired_coface; a negative sign
                    # is required because we are moving up the coface relation.
                    for coface_face_idx, coface_face_sign in zip(
                        coface_faces_idx, coface_faces_sign
                    ):
                        if coface_face_idx == current_face_idx:
                            current_face_sign *= -1.0 * coface_face_sign

                    # Push the new faces of the paired_coface to the search queue.
                    for coface_face_idx, coface_face_sign in zip(
                        coface_faces_idx, coface_faces_sign
                    ):
                        if coface_face_idx != current_face_idx:
                            next_splx_idx = coface_face_idx
                            next_splx_sign = current_face_sign * coface_face_sign

                            # Before adding the face to the search queue, check if
                            # at the stack size limit; if so, double the stack size
                            # before proceeding.
                            if stack_top + 1 > stack_size:
                                stack_size *= 2
                                new_stack_splx = np.empty(stack_size, dtype=int_dtype)
                                new_stack_sign = np.empty(stack_size, dtype=float_dtype)

                                new_stack_splx[:stack_top] = stack_splx[:stack_top]
                                new_stack_sign[:stack_top] = stack_sign[:stack_top]

                                stack_splx = new_stack_splx
                                stack_sign = new_stack_sign

                            stack_splx[stack_top] = next_splx_idx
                            stack_sign[stack_top] = next_splx_sign
                            stack_top += 1

                # If the current_face is paired downward, then the path does not
                # satisfy the definition of a gradient path (i.e., two step down
                # in a row) and is dropped here implicitly. Similarly, if the
                # current_face gets paired with an excluded simplex in the
                # tangential boundary subcomplex, it is dropped here implicitly
                # as well, because is_paired_up evals to False if paired_coface
                # is negative; although this shouldn't happen in practice, since
                # _process_lower_stars() should not allow an excluded simplex to
                # be paired with a retained simplex.

        # Once the search queue has been exhausted, return the reduced cbd.
        cbd_idx_coo.append(idx_coo[:, :nnz])
        cbd_val.append(val[:nnz])

    return cbd_idx_coo, cbd_val


def _construct_sdt(
    cbd_idx_coo: list[npt.NDArray],
    cbd_val: list[npt.NDArray],
    crit_splx_by_dim: tuple[npt.NDArray, ...],
    int_dtype: torch.dtype,
    float_dtype: torch.dtype,
    device: torch.device,
) -> tuple[tuple[SparseDecoupledTensor, ...], tuple[Tensor, ...]]:
    morse_cbd_list = []
    for k, idx_coo, val in zip([3, 2, 1], cbd_idx_coo, cbd_val, strict=True):
        shape = (crit_splx_by_dim[k].size, crit_splx_by_dim[k - 1].size)

        idx_coo_torch = torch.from_numpy(idx_coo).to(dtype=int_dtype, device=device)
        val_torch = torch.from_numpy(val).to(dtype=float_dtype, device=device)

        cbd_coo = torch.sparse_coo_tensor(
            indices=idx_coo_torch, values=val_torch, size=shape
        ).coalesce()

        # Remove entries that are close to zero due to cancellation of gradient
        # path weights.
        nz_mask = ~torch.isclose(cbd_coo.values(), torch.zeros_like(cbd_coo.values()))
        cbd_coo = torch.sparse_coo_tensor(
            indices=cbd_coo.indices()[:, nz_mask],
            values=cbd_coo.values()[nz_mask],
            size=shape,
        ).coalesce()

        cbd_sdt = SparseDecoupledTensor.from_tensor(cbd_coo)

        morse_cbd_list.append(cbd_sdt)

    morse_cbd_list.reverse()

    crit_splx_by_dim_torch = [
        torch.from_numpy(crit_splx).to(dtype=int_dtype, device=device)
        for crit_splx in crit_splx_by_dim
    ]

    return tuple(morse_cbd_list), tuple(crit_splx_by_dim_torch)


@torch.no_grad()
def compute_morse_complex(
    mesh: SimplicialMesh,
    scalar_field: Float[Tensor, " vert"] | None = None,
    bc_sel: BoundarySelection | None = None,
) -> tuple[
    tuple[
        Float[SparseDecoupledTensor, "crit_edge crit_vert"],
        Float[SparseDecoupledTensor, "crit_tri crit_edge"],
        Float[SparseDecoupledTensor, "crit_tet crit_tri"],
    ],
    tuple[
        Integer[Tensor, " crit_vert"],
        Integer[Tensor, " crit_edge"],
        Integer[Tensor, " crit_tri"],
        Integer[Tensor, " crit_tet"],
    ],
]:
    """
    Compute the Morse coboundary operators for a mesh.

    This function constructs an acyclic partial matching using the two-queue
    lower-star strategy of Robins, Wood & Sheppard, TPAMI (2011), then constructs
    the reduced coboundary operators using depth-first search. The core algorithms
    are accelerated by Numba.

    Parameters
    ----------
    mesh
        A simplicial complex; see the Notes on the assumptions made about the
        complex in this function.
    scalar_field : [vert,]
        A scalar field associated/sampled at the mesh vertices. If not provided,
        then associate each vert with its distance to the center of the mesh.
    bc_sel
        An optional BoundarySelection object, used to mark the tangential
        boundary subcomplex of the mesh. If set to `None`, assume that the mesh
        either has no boundaries or is subject to absolute boundary conditions.

    Returns
    -------
    morse_cbd
        A tuple containing the reduced 0-, 1-, and 2-coboundary operators of the
        Morse complex.
    crit_splx
        A tuple of integer tensors containing the indices of the critical 0-, 1-,
        2-, and 3-simplices in the complex.

    Notes
    -----
    This function only makes the following assumptions about the input simplicial
    complex:

    * The vertices are numbered with contiguous, 0-based indices.
    * The simplicial complex is at most three-dimensional.
    * If no `scalar_field` is provided, the vertices must be associated with
      coordinate vectors.
    """
    # If no bc_sel is provided, assuming absolute BC/no boundary.
    if bc_sel is None:
        bc_sel = BoundarySelection.from_absolute_bc(mesh)

    # If no scalar_field is provided, compute a simple one that measures the
    # distance from the center of the mesh.
    if scalar_field is None:
        mesh_center = torch.mean(mesh.vert_coords, dim=0, keepdim=True)
        verts_from_center = mesh.vert_coords - mesh_center
        scalar_field = torch.linalg.norm(verts_from_center, dim=-1)

    # Record the mesh cbd int dtype and determine the safe numpy int dtype.
    torch_int_dtype = mesh.cbd[0].pattern.dtype
    torch_float_dtype = mesh.dtype
    torch_device = mesh.device

    if sum(mesh.n_splx) < np.iinfo(np.int32).max:
        np_int_dtype = np.int32
    else:
        np_int_dtype = np.int64

    # Execute subroutines.
    (
        ordered_verts,
        vert_ranks,
        splx_vert_rank_max,
        splx_scalar_sum,
        vert_coface_indices,
        vert_coface_offset,
        codim1_coface_indices,
        codim1_coface_offset,
        codim1_face_indices,
        codim1_face_offset,
        codim1_face_signs,
        max_n_cofaces,
        pairing_map,
        splx_dim_offsets,
    ) = _prepare_inputs(mesh, scalar_field, bc_sel.constrained_mask, np_int_dtype)

    pairing_map = _process_lower_stars(
        mesh.dim,
        ordered_verts,
        vert_ranks,
        splx_vert_rank_max,
        splx_scalar_sum,
        vert_coface_indices,
        vert_coface_offset,
        codim1_coface_indices,
        codim1_coface_offset,
        codim1_face_indices,
        codim1_face_offset,
        max_n_cofaces,
        pairing_map,
        splx_dim_offsets,
    )

    crit_splx_by_dim, crit_splx_reduced_idx = _find_critical_splx(
        pairing_map, splx_dim_offsets
    )

    cbd_idx_coo, cbd_val = _construct_morse_cbds(
        codim1_face_indices,
        codim1_face_offset,
        codim1_face_signs,
        pairing_map,
        splx_dim_offsets,
        crit_splx_by_dim,
        crit_splx_reduced_idx,
    )

    morse_cbd, crit_splx = _construct_sdt(
        cbd_idx_coo,
        cbd_val,
        crit_splx_by_dim,
        torch_int_dtype,
        torch_float_dtype,
        torch_device,
    )

    return morse_cbd, crit_splx

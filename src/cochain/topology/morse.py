__all__ = ["compute_morse_complex"]


import numba
import numpy as np
import numpy.typing as npt
import torch
from einops import reduce
from jaxtyping import Float, Integer
from torch import Tensor

from ..complex import SimplicialMesh
from ..sparse.decoupled_tensor import SparseDecoupledTensor


def _to_np(tensor: Tensor, dtype: np.dtype | None = None) -> npt.NDArray:
    if dtype is None:
        return tensor.detach().cpu().numpy()
    else:
        return tensor.detach().cpu().numpy().astype(dtype)


def _prepare_inputs(
    mesh: SimplicialMesh,
    scalar_field: Float[Tensor, " vert"],
    int_dtype: np.dtype,
):
    """
    Prepare input mesh and scalar field for discrete Morse theory analysis.

    int dtype must be signed.
    """
    # Attach the scalar field to higher-order simplices
    splx_scalar_fields = (
        _to_np(scalar_field.view(-1, 1)),
        _to_np(scalar_field[mesh.edges]),
        _to_np(scalar_field[mesh.tris]),
        _to_np(scalar_field[mesh.tets]),
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
    ordered_verts_np = _to_np(ordered_verts, int_dtype)

    # Use the vert ordering to assign a ranking to each vert, which establishes
    # a strict total ordering of the verts.
    vert_ranks = torch.empty_like(
        ordered_verts, dtype=ordered_verts.dtype, device=ordered_verts.device
    )
    vert_ranks[ordered_verts] = torch.arange(
        ordered_verts.size(0), dtype=ordered_verts.dtype, device=ordered_verts.device
    )
    vert_ranks_np = _to_np(vert_ranks, int_dtype)

    # Use the max vert rank to assign a rank score per simplex.
    splx_vert_ranks = (
        _to_np(vert_ranks.view(-1, 1), int_dtype),
        _to_np(vert_ranks[mesh.edges], int_dtype),
        _to_np(vert_ranks[mesh.tris], int_dtype),
        _to_np(vert_ranks[mesh.tets], int_dtype),
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
        _to_np(mesh.cbd[0].pattern.idx_row_csc, int_dtype) + splx_dim_offsets[1],
        _to_np(cbd_02.pattern.idx_row_csc, int_dtype) + splx_dim_offsets[2],
        _to_np(cbd_03.pattern.idx_row_csc, int_dtype) + splx_dim_offsets[3],
    )
    vert_coface_offset = (
        _to_np(mesh.cbd[0].pattern.idx_ccol, int_dtype),
        _to_np(cbd_02.pattern.idx_ccol, int_dtype),
        _to_np(cbd_03.pattern.idx_ccol, int_dtype),
    )

    # Also compute the coface relations between k-simplices and (k+1)-simplices.
    # Note that, for the k-coboundary operator, its idx_row_csc index tensor
    # contains the 0-indices of the k-simplices, and we need to shift this local
    # 0-index scheme to the global simplex index scheme described above.
    codim1_coface_indices = (
        _to_np(mesh.cbd[0].pattern.idx_row_csc, int_dtype) + splx_dim_offsets[1],
        _to_np(mesh.cbd[1].pattern.idx_row_csc, int_dtype) + splx_dim_offsets[2],
        _to_np(mesh.cbd[2].pattern.idx_row_csc, int_dtype) + splx_dim_offsets[3],
    )
    codim1_coface_offset = (
        _to_np(mesh.cbd[0].pattern.idx_ccol, int_dtype),
        _to_np(mesh.cbd[1].pattern.idx_ccol, int_dtype),
        _to_np(mesh.cbd[2].pattern.idx_ccol, int_dtype),
    )

    # Similarly compute the face relations between (k+1)-simplices and k-simplices
    # as well as the boundary signs. Note that, for the k-coboundary operator,
    # its idx_col index tensor contains the 0-indices of the (k-1)-simplices, and
    # we need to shift this local 0-index scheme to the global index scheme
    # described above.
    codim1_face_indices = (
        _to_np(mesh.cbd[0].pattern.idx_col, int_dtype) + splx_dim_offsets[0],
        _to_np(mesh.cbd[1].pattern.idx_col, int_dtype) + splx_dim_offsets[1],
        _to_np(mesh.cbd[2].pattern.idx_col, int_dtype) + splx_dim_offsets[2],
    )
    codim1_face_offset = (
        _to_np(mesh.cbd[0].pattern.idx_crow, int_dtype),
        _to_np(mesh.cbd[1].pattern.idx_crow, int_dtype),
        _to_np(mesh.cbd[2].pattern.idx_crow, int_dtype),
    )
    codim1_face_signs = (
        _to_np(mesh.cbd[0].val),
        _to_np(mesh.cbd[1].val),
        _to_np(mesh.cbd[2].val),
    )

    # Determine the max possible lower star size. To compute this size, we
    # determine, for each vert, the number of its vert, edge, tri, and tet cofaces
    # and sum them together, and take the max of this value over the list of verts.
    # Note that the difference between consecutive elements in the ccol index
    # tensor of a coboundary operator gives the number of nonzero elements in each
    # column of the operator, which is equal to the number of faces.
    n_cofaces = np.stack(
        [idx_ccol[1:] - idx_ccol[:-1] for idx_ccol in vert_coface_offset]
    )
    # + 1 to account for vert cofaces (which is just itself).
    max_n_cofaces = 1 + int((n_cofaces.sum(axis=0)).max().item())

    # Prepare the pairing map; if τ is a codim 1 face of σ and τ is paired with
    # σ, then p[τ] = σ and p[σ] = τ.
    pairing_map = np.full(sum(mesh.n_splx), -1, dtype=int_dtype)

    return (
        ordered_verts_np,
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
        splx_scalar_sum,
        vert_ranks_np,
        splx_vert_rank_max,
    )


@numba.jit(nopython=True)
def _process_lower_stars(
    ordered_verts: npt.NDArray,
    vert_coface_indices: tuple[npt.NDArray, ...],
    vert_coface_offset: tuple[npt.NDArray, ...],
    codim1_coface_indices: tuple[npt.NDArray, ...],
    codim1_coface_offset: tuple[npt.NDArray, ...],
    max_n_cofaces: int,
    pairing_map: npt.NDArray,
    splx_dim_offsets: npt.NDArray,
    splx_scalar_sum: npt.NDArray,
    vert_ranks: npt.NDArray,
    splx_vert_rank_max: npt.NDArray,
) -> Integer[Tensor, " splx"]:
    r"""
    Construct an acyclic partial matching using lower star filtrations.

    Let $f(v)$ be a scalar field associated with each vertex on a mesh. For each
    vert $v$ in the mesh, we define the lower star $L(v)$ of $v$ to be the set of
    all simplices $\sigma$ such that $v \le \sigma$ and $f(v) > f(v')$ for all
    other vert faces $v'$ of $\sigma$; in case of a tie where $f(v) = f(v')$,
    $\sigma \in L(v)$ if the index of $v$ is lower than that of $v'$.

    For each simplex $\sigma \in L(v)$, identify all of its cofaces $\tau$ of
    codimension 1. We pair/match $\sigma$ with $\tau$ if all following three
    conditions are met:

    * $\tau \in L(v)$,
    * $\tau$ has not been paired/matched, and
    * The sum of $f$ over the vert faces of $\tau$ is the smallest among all
      codim 1 cofaces of $\sigma$ (in case of a tie, the first coface after a
      lex sort is picked.)

    Simplices that remain unpaired after this process are called critical.

    This function returns a `pairing_map` array, where `pairing_map[i] = j` for
    `i, j >= 0` indicates that the simplices `i` and `j` are paired, and
    `pairing_map[i] = -1` indicates that simplex `i` is critical. Here, the simplices
    are 0-indexed consecutively, where the lower-dimensional simplices are placed
    before higher-dimensional ones, and simplices of the same dimension are
    lex-sorted.
    """
    # For accessing the lower star simplex indices, we use a pre-allocated
    # lower_star_buffer; for checking membership in the lower star, we use a flat
    # bool mask.
    lower_star_idx_buffer = np.empty(max_n_cofaces, dtype=pairing_map.dtype)
    lower_star_dim_buffer = np.empty(max_n_cofaces, dtype=pairing_map.dtype)
    lower_star_mask = np.zeros(splx_dim_offsets[-1], dtype=np.bool)

    for vert in ordered_verts:
        # A "pointer" that keeps track of the size of the lower star of a vert.
        lower_star_size = 0
        vert_rank = vert_ranks[vert]

        # For each vert, there is only one coface of codimension 0: the vert itself.
        lower_star_idx_buffer[lower_star_size] = vert
        lower_star_dim_buffer[lower_star_size] = 0
        lower_star_mask[vert] = True
        lower_star_size += 1

        for coface_dim in [1, 2, 3]:
            # Find the coface indices via the cbd csc indices.
            cbd_idx = vert_coface_indices[coface_dim - 1]
            cbd_offset = vert_coface_offset[coface_dim - 1]
            cofaces = cbd_idx[cbd_offset[vert] : cbd_offset[vert + 1]]

            # Check whether the identified cofaces are in the lower star. By
            # checking that the vert rank is equal to the coface rank score,
            # we ensure that the vert has the max scalar field value (and, in
            # the event of ties, the vert has lower index). This ensures that
            # the lower stars of the verts form a partition of the simplicial
            # complex.
            coface_vert_rank_max = splx_vert_rank_max[cofaces]
            lower_mask = coface_vert_rank_max <= vert_rank

            lower_cofaces = cofaces[lower_mask]
            n_lower_cofaces = lower_mask.sum()

            # Get the index range for registering the cofaces in the buffer.
            start = lower_star_size
            end = lower_star_size + n_lower_cofaces

            lower_star_idx_buffer[start:end] = lower_cofaces
            lower_star_dim_buffer[start:end] = coface_dim
            lower_star_mask[cofaces[lower_mask]] = True
            lower_star_size += n_lower_cofaces

        # For each simplex τ in the lower star of v, attempt to pair it with a
        # coface σ of codimension 1 greedily by finding an unpaired coface σ that
        # has the lowest sum of scalar field over its vertices ("steepest descent").
        for i in range(lower_star_size):
            coface_dim = lower_star_dim_buffer[i]
            vert_coface = lower_star_idx_buffer[i]

            # If a vert coface is already paired, do nothing.
            if pairing_map[vert_coface] != -1:
                continue

            # If a vert coface is a tet, it is the highest possible dimensional
            # simplex and doesn't have cofaces of codim 1.
            if coface_dim == 3:
                continue

            # For each vert coface τ, find all of its cofaces (of codim 1) σ's.
            cbd_idx = codim1_coface_indices[coface_dim]
            cbd_offset = codim1_coface_offset[coface_dim]
            # Note that the offset index still operates on the local index
            # scheme (0-index for all splx of a given dim), so we need to first
            # convert the global vert_coface index back to the local 0-index.
            vert_coface_local = vert_coface - splx_dim_offsets[coface_dim]
            coface_cofaces = cbd_idx[
                cbd_offset[vert_coface_local] : cbd_offset[vert_coface_local + 1]
            ]

            # Iterate over all the σ's, ignore any σ that is not in the
            # lower star or already paired, find the σ with the lowest
            # sum of scalar field and pair this σ with τ.
            best_coface_coface = -1
            min_sum = np.inf
            for coface_coface in coface_cofaces:
                in_lower_star = lower_star_mask[coface_coface]
                is_unpaired = pairing_map[coface_coface] == -1

                if in_lower_star and is_unpaired:
                    coface_coface_sum = splx_scalar_sum[coface_coface]
                    if coface_coface_sum < min_sum:
                        best_coface_coface = coface_coface
                        min_sum = coface_coface_sum

            # If a coface in the lower star is critical, then it cannot be paired.
            if best_coface_coface != -1:
                pairing_map[vert_coface] = best_coface_coface
                pairing_map[best_coface_coface] = vert_coface

        # Reset the lower_star_mask for the next vert.
        lower_star_mask[lower_star_idx_buffer[:lower_star_size]] = False

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
    dimension as simplex `i`, and `crit_splx_reduced_idx[i] = -1` indicates that
    simplex `i` is not a critical simplex.
    """
    # Find all the critical simplices by identifying the unpaired simplices.
    crit_splx = np.argwhere(pairing_map == -1).flatten()

    # Split the list of critical simplices by their dimensions.
    split_points = np.searchsorted(crit_splx, splx_dim_offsets[1:-1])
    crit_splx_by_dim = np.split(crit_splx, split_points)

    crit_splx_reduced_idx = np.full(splx_dim_offsets[-1], -1, dtype=pairing_map.dtype)
    for crit_splx in crit_splx_by_dim:
        crit_splx_local_idx = np.arange(crit_splx.size, dtype=pairing_map.dtype)
        crit_splx_reduced_idx[crit_splx] = crit_splx_local_idx

    return tuple(crit_splx_by_dim), crit_splx_reduced_idx


# TODO: option to save the gradient paths
@numba.jit(nopython=True)
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
    nonzero iff there is at least one gradient path connecting the $(k+1)$-simplex
    $\tau_i$ to the $k$-simplex $\sigma_j$, and $d_{ij}$ is the sum of the weights
    associated with all such gradient paths.

    A gradient path $\rho$ between $\tau_i$ and $\sigma_j$ is a sequence of
    alternating $(k+1)$- and $k$-dimensional simplices that satisfies either of
    the following two conditions:

    * The sequence contains only $\tau_i$ and $\sigma_j$, where $\sigma_j \le \tau_i$, or
    * The $\sigma_j$ is not a face of $\tau_i$ and the sequence contains additional
      simplices, whereby every "step-down" satisfies a face relation and every
      "step-up" is part of an acyclic partial matching.

    We assign each step of the gradient path a weight: a step-down is assigned
    a weight equal to the boundary sign of the face, and a step-up is assigned
    a weight equal to the negative boundary sign of the face. Then, the weight
    associated with a gradient path is the product of the weight assigned to
    each step along the path.

    In this function, we perform the path search using a depth-first search
    approach.

    Note that, while the Morse complex is chain homotopic to the original simplicial
    complex, the Morse complex itself is a CW complex, rather than a simplicial
    complex.
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
        # use a heuristic that allocates 50*n_row*n_col nonzero slots. If this
        # is not enough, the buffer gets doubled dynamically (we set a minimum
        # buffer size of 10,000 to avoid potential, repeated size adjustments).
        cbd_size = crit_splx_by_dim[splx_dim].size * crit_splx_by_dim[splx_dim - 1].size
        cbd_buffer_size = max(10_000, cbd_size * 50)
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
            n_faces = end - start

            face_idx = cbd_idx[start:end]
            face_sign = bd_signs[start:end]

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

                # Check whether the current_face is paired.
                is_paired = paired_coface != -1
                # Check whether the current_face is paired with one of its faces
                # or one of its cofaces.
                is_paired_up = paired_coface > current_face_idx

                # If the current_face is critical, then the path terminates.
                # Add the current_face_sign to (crit_splx, current_face_idx)
                # position of the coboundary operator.
                if not is_paired:
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
                    n_coface_faces = end - start

                    coface_faces_idx = cbd_idx[start:end]
                    coface_faces_sign = bd_signs[start:end]

                    # As the path moves from current_face to paired_coface, update
                    # the current_sign using the (negative of the) boundary sign
                    # of the current_face in the paired_coface; a negative sign
                    # is required because we are moving up the coface relation.
                    for i in range(n_coface_faces):
                        coface_face_idx = coface_faces_idx[i]
                        if coface_face_idx == current_face_idx:
                            coface_face_sign = coface_faces_sign[i]
                            current_face_sign *= -1.0 * coface_face_sign

                    # Push the new faces of the paired_coface to the search queue.
                    for i in range(n_coface_faces):
                        coface_face_idx = coface_faces_idx[i]
                        if coface_face_idx != current_face_idx:
                            coface_face_sign = coface_faces_sign[i]

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
        cbd_sdt = SparseDecoupledTensor.from_tensor(cbd_coo)

        morse_cbd_list.append(cbd_sdt)

    morse_cbd_list.reverse()

    crit_splx_by_dim_torch = [
        torch.from_numpy(crit_splx).to(dtype=int_dtype, device=device)
        for crit_splx in crit_splx_by_dim
    ]

    return tuple(morse_cbd_list), tuple(crit_splx_by_dim_torch)


def compute_morse_complex(
    mesh: SimplicialMesh,
    scalar_field: Float[Tensor, " vert"] | None = None,
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

    This function makes the following assumptions about the mesh:
    * Contiguous, 0-based indexing of vertices.
    * Up to three-dimensional.
    * If no scalar_field is provided, the vertices must be associated with coordinate
      vectors.
    """
    with torch.no_grad():
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
            splx_scalar_sum,
            vert_ranks,
            splx_vert_rank_max,
        ) = _prepare_inputs(mesh, scalar_field, np_int_dtype)

        pairing_map = _process_lower_stars(
            ordered_verts,
            vert_coface_indices,
            vert_coface_offset,
            codim1_coface_indices,
            codim1_coface_offset,
            max_n_cofaces,
            pairing_map,
            splx_dim_offsets,
            splx_scalar_sum,
            vert_ranks,
            splx_vert_rank_max,
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

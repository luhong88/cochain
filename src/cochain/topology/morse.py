from typing import Any

import numba
import numpy as np
import numpy.typing as npt
import torch
from einops import einsum, rearrange, reduce, repeat
from jaxtyping import Float, Integer
from torch import Tensor

from ..complex import SimplicialMesh


def _to_np(tensor: Tensor, dtype: np.dtype | None = None) -> npt.NDArray:
    if dtype is None:
        return tensor.detach().cpu().numpy()
    else:
        return tensor.detach().cpu().numpy().astype(dtype)


def _prepare_inputs(
    mesh: SimplicialMesh,
    scalar_field: Float[Tensor, " vert"],
    int_dtype: np.dtype = np.int32,
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

    splx_scalar_sum = torch.cat(
        [
            reduce(splx_scalar_field, "splx vert -> splx", reduction="sum")
            for splx_scalar_field in splx_scalar_fields
        ]
    )
    splx_scalar_max = torch.cat(
        [
            reduce(splx_scalar_field, "splx vert -> splx", reduction="max")
            for splx_scalar_field in splx_scalar_fields
        ]
    )

    # Sort the vertices in ascending order by the scalar value; stable=True
    # ensures tiebreak using vertex indices.
    ordered_verts = _to_np(torch.argsort(scalar_field, stable=True, dim=0), int_dtype)

    # Prepare index data for a global simplex indexing scheme, where all 0-, 1-,
    # 2-, and 3-simplices are ordered in a flattened list and assigned a 0-index.
    # In particular, splx_dim_offsets is [0, V, V+E, V+E+F, V+E+F+T].
    splx_dim_offsets = np.array(
        [sum(mesh.n_splx[:k]) for k in range(len(mesh.n_splx) + 1)], dtype=int_dtype
    )

    # Precompute the coface relations between vertices and higher-order simplices.
    cbd_02 = mesh.cbd[1].abs() @ mesh.cbd[0].abs()
    cbd_03 = mesh.cbd[2].abs() @ mesh.cbd[0].abs()

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
    # as well as the boundary signs.
    codim1_face_indices = (
        _to_np(mesh.cbd[0].pattern.idx_col, int_dtype) + splx_dim_offsets[1],
        _to_np(mesh.cbd[1].pattern.idx_col, int_dtype) + splx_dim_offsets[2],
        _to_np(mesh.cbd[2].pattern.idx_col, int_dtype) + splx_dim_offsets[3],
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

    # Prepare the pairing map; if τ is a codim 1 face of σ and τ is paired with
    # σ, then p[τ] = σ and p[σ] = τ.
    pairing_map = _to_np(np.full(sum(mesh.n_splx), -1), int_dtype)

    # Additional conversion to numpy arrays
    scalar_field_np = _to_np(scalar_field)

    return (
        ordered_verts,
        vert_coface_indices,
        vert_coface_offset,
        codim1_coface_indices,
        codim1_coface_offset,
        codim1_face_indices,
        codim1_face_offset,
        codim1_face_signs,
        pairing_map,
        splx_dim_offsets,
        scalar_field_np,
        splx_scalar_sum,
        splx_scalar_max,
    )


def _process_lower_stars(
    ordered_verts: npt.NDArray,
    vert_coface_indices: tuple[npt.NDArray, ...],
    vert_coface_offset: tuple[npt.NDArray, ...],
    codim1_coface_indices: tuple[npt.NDArray, ...],
    codim1_coface_offset: tuple[npt.NDArray, ...],
    pairing_map: npt.NDArray,
    splx_dim_offsets: npt.NDArray,
    scalar_field: npt.NDArray,
    splx_scalar_sum: npt.NDArray,
    splx_scalar_max: npt.NDArray,
) -> Integer[Tensor, " splx"]:
    """
    Perform greedy lower star filtration/acyclic pairing.
    """
    # It is safe to assume that the lower star of a vert does not contain more
    # than a few thousand simplices.
    buffer_capacity = 2048

    # We will need to find the lower star of each vert in the mesh; for a vert v,
    # the lower star is the set of all k-simplices σ such that v <= σ and v has
    # the max scalar value among all vertices of σ. For accessing the lower star
    # simplex indices, we use a pre-allocated lower_star_buffer; for checking
    # membership in the lower star, we use a flat bool mask.
    lower_star_idx_buffer = np.empty(buffer_capacity, dtype=pairing_map.dtype)
    lower_star_dim_buffer = np.empty(buffer_capacity, dtype=pairing_map.dtype)
    lower_star_mask = np.zeros(splx_dim_offsets[-1], dtype=bool)

    for vert in ordered_verts:
        # A "pointer" that keeps track of the size of the lower star of a vert.
        lower_star_size = 0
        vert_scalar = scalar_field[vert]

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

            # Check whether the identified cofaces are in the lower star.
            coface_scalar_max = splx_scalar_max[cofaces]
            lower_mask = coface_scalar_max <= vert_scalar

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
            # convert th global vert_coface index back to the local 0-index.
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
) -> tuple[npt.NDArray]:
    # Find all the critical simplices by identifying the unpaired simplices.
    crit_splx = np.argwhere(pairing_map == -1).flatten()

    # Split the list of critical simplices by their dimensions.
    split_points = np.searchsorted(crit_splx, splx_dim_offsets[1:-1])
    crit_splx_by_dim = np.split(crit_splx, split_points)

    return tuple(crit_splx_by_dim)


def _construct_morse_cbds(
    codim1_face_indices: tuple[npt.NDArray, ...],
    codim1_face_offset: tuple[npt.NDArray, ...],
    codim1_face_signs: tuple[npt.NDArray, ...],
    pairing_map: npt.NDArray,
    splx_dim_offsets: npt.NDArray,
    crit_splx_by_dim: tuple[npt.NDArray, ...],
) -> tuple[tuple[npt.NDArray, ...], tuple[npt.NDArray]]:
    """
    Construct the reduced Morse coboundary operators.
    """
    int_dtype = pairing_map.dtype
    float_dtype = codim1_face_signs[0].dtype

    # Generate LIFO buffer/stack for depth-first simplex search.
    buffer_size = 2**15
    stack_splx = np.empty(buffer_size, dtype=int_dtype)
    stack_sign = np.empty(buffer_size, dtype=float_dtype)
    stack_top = 0

    cbd_idx_coo = []
    cbd_val = []

    for splx_dim in [3, 2, 1]:
        # Pre allocate the memories for the morse coboundary operator coo indices
        # and values. Since we don't actually know how many nonzero elements there
        # are per reduced coboundary operator, we assume that the operators are dense
        # in the pre allocation.
        max_nnz = crit_splx_by_dim[splx_dim].size * crit_splx_by_dim[splx_dim - 1].size
        idx_coo = np.empty(
            (2, max_nnz),
            dtype=int_dtype,
        )
        val = np.empty(max_nnz, dtype=float_dtype)
        nnz = 0

        # Determine the index offset required to map from the global simplex
        # indices to the local, per-dim simplex indices.
        global_splx_offset = splx_dim_offsets[splx_dim]
        global_face_offset = splx_dim_offsets[splx_dim - 1]

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

            stack_splx[stack_top : stack_top + n_faces] = face_idx
            stack_sign[stack_top : stack_top + n_faces] = face_sign
            stack_top += n_faces

            # When the search queue is not empty.
            while stack_top > 0:
                # Pop the last simplex in the search queue.
                current_face_idx = stack_splx[stack_top]
                current_face_sign = stack_sign[stack_top]
                stack_top -= 1

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
                    idx_coo[0, nnz] = crit_splx - global_splx_offset
                    idx_coo[1, nnz] = current_face_idx - global_face_offset
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

                            stack_splx[stack_top] = next_splx_idx
                            stack_sign[stack_top] = next_splx_sign
                            stack_top += 1

            # Once the search queue has been exhausted, return the reduced cbd.
            cbd_idx_coo.append(idx_coo[:nnz])
            cbd_val.append(val[:nnz])

    return cbd_idx_coo, cbd_val

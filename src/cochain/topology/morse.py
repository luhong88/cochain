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
        pairing_map,
        splx_dim_offsets,
        scalar_field_np,
        splx_scalar_sum,
        splx_scalar_max,
    )


def _lower_star_filtration(
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
            idx = vert_coface_indices[coface_dim - 1]
            offset = vert_coface_offset[coface_dim - 1]
            cofaces = idx[offset[vert] : offset[vert + 1]]

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
        for idx in range(lower_star_size):
            coface_dim = lower_star_dim_buffer[idx]
            vert_coface = lower_star_idx_buffer[idx]

            # If a vert coface is already paired, do nothing.
            if pairing_map[vert_coface] != -1:
                continue

            # If a vert coface is a tet, it is the highest possible dimensional
            # simplex and doesn't have cofaces of codim 1.
            if coface_dim == 3:
                continue

            # For each vert coface τ, find all of its cofaces (of codim 1) σ's.
            idx = codim1_coface_indices[coface_dim]
            offset = codim1_coface_offset[coface_dim]
            # Note that the offset index still operates on the local index
            # scheme (0-index for all splx of a given dim), so we need to first
            # convert th global vert_coface index back to the local 0-index.
            vert_coface_local = vert_coface - splx_dim_offsets[coface_dim]
            coface_cofaces = idx[
                offset[vert_coface_local] : offset[vert_coface_local + 1]
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

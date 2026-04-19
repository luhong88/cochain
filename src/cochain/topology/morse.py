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

    splx_scalar_sum = [
        reduce(splx_scalar_field, "splx vert -> splx", reduction="sum")
        for splx_scalar_field in splx_scalar_fields
    ]
    splx_scalar_max = [
        reduce(splx_scalar_field, "splx vert -> splx", reduction="max")
        for splx_scalar_field in splx_scalar_fields
    ]

    # Sort the vertices in ascending order by the scalar value; stable=True
    # ensures tiebreak using vertex indices.
    ordered_verts = _to_np(torch.argsort(scalar_field, stable=True, dim=0), int_dtype)

    # Precompute the coface relations between vertices and higher-order simplices.
    cbd_02 = mesh.cbd[1].abs() @ mesh.cbd[0].abs()
    cbd_03 = mesh.cbd[2].abs() @ mesh.cbd[0].abs()

    n_v = mesh.n_verts
    n_ve = mesh.n_verts + mesh.n_edges
    n_vet = mesh.n_verts + mesh.n_edges + mesh.n_tris

    # Use a global simplex index scheme; where all 0-, 1-, 2-, and 3-simplices
    # are ordered in a flattened list and assigned a 0-index.
    vert_coface_indices = (
        _to_np(mesh.cbd[0].pattern.idx_row_csc, int_dtype) + n_v,
        _to_np(cbd_02.pattern.idx_row_csc, int_dtype) + n_ve,
        _to_np(cbd_03.pattern.idx_row_csc, int_dtype) + n_vet,
    )
    vert_coface_offset = (
        _to_np(mesh.cbd[0].pattern.idx_ccol, int_dtype),
        _to_np(cbd_02.pattern.idx_ccol, int_dtype),
        _to_np(cbd_03.pattern.idx_ccol, int_dtype),
    )

    # Also compute the coface relations between k-simplices and (k+1)-simplices.
    codim1_coface_indices = (
        _to_np(mesh.cbd[0].pattern.idx_row_csc, int_dtype) + n_v,
        _to_np(mesh.cbd[1].pattern.idx_row_csc, int_dtype) + n_ve,
        _to_np(mesh.cbd[2].pattern.idx_row_csc, int_dtype) + n_vet,
    )
    codim1_coface_offset = (
        _to_np(mesh.cbd[0].pattern.idx_ccol, int_dtype) + n_v,
        _to_np(mesh.cbd[1].pattern.idx_ccol, int_dtype) + n_ve,
        _to_np(mesh.cbd[2].pattern.idx_row_csc, int_dtype) + n_vet,
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
    scalar_field: npt.NDArray,
    splx_scalar_sum: tuple[npt.NDArray, ...],
    splx_scalar_max: tuple[npt.NDArray, ...],
) -> Integer[Tensor, " splx"]:
    """
    Perform greedy lower star filtration/acyclic pairing.
    """
    for vert in ordered_verts:
        vert_scalar = scalar_field[vert]
        # Find the lower star of the vert; for each vert v, the lower star is the
        # set of all k-simplices σ such that v <= σ and v has the max scalar
        # value among all vertices of σ. We partition and enumerate the lower star
        # as subsets of cofaces of a given codimension.
        lower_star: list[set[int]] = []

        # For each vert, there is only one coface of codimension 0: the vert itself.
        lower_star.append(set([vert]))

        for coface_dim in [1, 2, 3]:
            # Find the coface indices via the cbd csc indices.
            idx = vert_coface_indices[coface_dim - 1]
            offset = vert_coface_offset[coface_dim - 1]
            cofaces = idx[offset[vert] : offset[vert + 1]]

            # Check whether the identified cofaces are in the lower star.
            coface_scalar_max = splx_scalar_max[coface_dim][cofaces]
            lower_mask = coface_scalar_max <= vert_scalar

            lower_star.append(set(cofaces[lower_mask].tolist()))

        for coface_dim, lower_star_subset in enumerate(lower_star[:-1]):
            for vert_coface in lower_star_subset:
                if pairing_map[vert_coface] != -1:
                    # If a vert coface is already paired, do nothing.
                    continue
                else:
                    # For each vert coface τ, find all of its cofaces (of codim 1) σ's.
                    idx = codim1_coface_indices[coface_dim]
                    offset = codim1_coface_offset[coface_dim]
                    coface_cofaces = idx[offset[vert_coface] : offset[vert_coface + 1]]

                    # Iterate over all the σ's, ignore any σ that is not in the
                    # lower star or already paired, find the σ with the lowest
                    # sum of scalar field over its vertices ("steepest descent"),
                    # and pair this σ with τ.
                    best_coface_coface = None
                    min_sum = float("inf")
                    for coface_coface in coface_cofaces:
                        in_lower_star = coface_coface in lower_star[coface_dim + 1]
                        is_unpaired = pairing_map[coface_coface] == -1

                        if in_lower_star and is_unpaired:
                            coface_coface_sum = splx_scalar_sum[coface_dim + 1][
                                coface_coface
                            ]
                            if coface_coface_sum < min_sum:
                                best_coface_coface = coface_coface
                                min_sum = coface_coface_sum

                    pairing_map[vert_coface] = best_coface_coface
                    pairing_map[best_coface_coface] = vert_coface

    return pairing_map

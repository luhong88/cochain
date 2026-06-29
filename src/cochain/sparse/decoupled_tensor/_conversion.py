from __future__ import annotations

import scipy
from jaxtyping import Float, Integer
from torch import Tensor

from ...utils.parsing import to_np
from .pattern import SparsityPattern

try:
    import cupy as cp
    import cupyx.scipy.sparse as cp_sp

    _HAS_CUPY = True

except ImportError:
    _HAS_CUPY = False


# TODO: check whether int64 is allowed.
def sdt_to_scipy_csr(
    val: Float[Tensor, " nnz"],
    pattern: Integer[SparsityPattern, "r c"],
) -> Float[scipy.sparse.csr_array, "r c"]:
    sdt_scipy = scipy.sparse.csr_array(
        (
            to_np(val, contiguous=True),
            to_np(pattern.idx_col, contiguous=True),
            to_np(pattern.idx_crow, contiguous=True),
        ),
        shape=tuple(pattern.shape),
    )
    return sdt_scipy


def sdt_to_scipy_csc(
    val: Float[Tensor, " nnz"],
    pattern: Integer[SparsityPattern, "r c"],
) -> Float[scipy.sparse.csc_array, "r c"]:
    sdt_scipy = scipy.sparse.csc_array(
        (
            to_np(val[pattern.csc_to_coo_map], contiguous=True),
            to_np(pattern.idx_row_csc, contiguous=True),
            to_np(pattern.idx_ccol, contiguous=True),
        ),
        shape=tuple(pattern.shape),
    )
    return sdt_scipy


def sdt_to_cupy_csr(
    val: Float[Tensor, " nnz"],
    pattern: Integer[SparsityPattern, "r c"],
) -> Float[cp_sp.csr_matrix, "r c"]:
    if not _HAS_CUPY:
        raise ImportError("CuPy backend required.")

    sdt_cupy = cp_sp.csr_matrix(
        (
            cp.from_dlpack(val.detach().contiguous()),
            cp.from_dlpack(pattern.idx_col.detach().contiguous()),
            cp.from_dlpack(pattern.idx_crow.detach().contiguous()),
        ),
        shape=tuple(pattern.shape),
    )
    return sdt_cupy


def sdt_to_cupy_csc(
    val: Float[Tensor, " nnz"],
    pattern: Integer[SparsityPattern, "r c"],
) -> Float[cp_sp.csc_matrix, "r c"]:
    if not _HAS_CUPY:
        raise ImportError("CuPy backend required.")

    sdt_cupy = cp_sp.csc_matrix(
        (
            cp.from_dlpack(val[pattern.csc_to_coo_map].detach().contiguous()),
            cp.from_dlpack(pattern.idx_row_csc.detach().contiguous()),
            cp.from_dlpack(pattern.idx_ccol.detach().contiguous()),
        ),
        shape=tuple(pattern.shape),
    )
    return sdt_cupy

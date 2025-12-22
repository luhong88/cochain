from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

import torch as t
from jaxtyping import Integer

from ._utils import (
    coalesced_coo_to_col_idx,
    coalesced_coo_to_compressed_idx,
    coalesced_coo_to_row_idx,
    get_csc_sort_perm,
    validate_coo_idx_shape,
)


@dataclass(frozen=True)
class SparseTopology:
    """
    idx_coo must be coalesced and of shape (2, nnz) or (3, nnz).

    Operations on idx_coo are not guaranteed to give contiguous tensors.
    """

    _idx_coo: Integer[t.LongTensor, "sp nnz"]
    shape: tuple[int, ...] | t.Size

    def __post_init__(self):
        validate_coo_idx_shape(self.idx_coo, self.shape)

    @property
    def idx_coo(self) -> Integer[t.LongTensor, "sp nnz"]:
        return self._idx_coo

    @property
    def device(self) -> t.device:
        return self.idx_coo.device

    @property
    def dtype(self) -> t.dtype:
        return self.idx_coo.dtype

    @property
    def n_batch_dim(self) -> int:
        return self.idx_coo.size(0) - 2

    @property
    def n_sp_dim(self) -> int:
        # For sparse csr/csc tensors, the leading batch dimension(s) do not count
        # towards sparse_dim(); for sparse coo tensors, no such distinction is
        # made. Here we follow the sparse csr/csc convention.
        return 2

    @cached_property
    def coo_to_csc_perm(self) -> Integer[t.LongTensor, " nnz"]:
        return get_csc_sort_perm(self.idx_coo, self.shape)

    @cached_property
    def idx_ccol(self) -> Integer[t.LongTensor, "*b nnz/b"]:
        return coalesced_coo_to_compressed_idx(self.idx_coo, self.shape, target="ccol")

    @cached_property
    def idx_ccol_int32(self) -> Integer[t.IntTensor, "*b nnz/b"]:
        return coalesced_coo_to_compressed_idx(
            self.idx_coo, self.shape, target="ccol", dtype=t.int32
        )

    @cached_property
    def idx_crow(self) -> Integer[t.LongTensor, "*b nnz/b"]:
        return coalesced_coo_to_compressed_idx(self.idx_coo, self.shape, target="crow")

    @cached_property
    def idx_crow_int32(self) -> Integer[t.IntTensor, "*b nnz/b"]:
        return coalesced_coo_to_compressed_idx(
            self.idx_coo, self.shape, target="crow", dtype=t.int32
        )

    @cached_property
    def idx_col(self) -> Integer[t.LongTensor, "*b nnz/b"]:
        return coalesced_coo_to_col_idx(self.idx_coo, self.shape)

    @cached_property
    def idx_col_int32(self) -> Integer[t.IntTensor, "*b nnz/b"]:
        return coalesced_coo_to_col_idx(self.idx_coo, self.shape, dtype=t.int32)

    @cached_property
    def idx_row(self) -> Integer[t.LongTensor, "*b nnz/b"]:
        return coalesced_coo_to_row_idx(self.idx_coo, self.shape, self.coo_to_csc_perm)

    @cached_property
    def idx_row_int32(self) -> Integer[t.IntTensor, "*b nnz/b"]:
        return coalesced_coo_to_row_idx(
            self.idx_coo, self.shape, self.coo_to_csc_perm, dtype=t.int32
        )

    def _nnz(self) -> int:
        return self.idx_coo.size(1)

    def to(self, *args, **kwargs) -> SparseTopology:
        return SparseTopology(self.idx_coo.to(*args, **kwargs), self.shape)

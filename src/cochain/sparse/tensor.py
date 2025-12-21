from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

import torch as t
from jaxtyping import Float, Integer

from .utils import coalesced_coo_to_col_idx, coalesced_coo_to_crow_idx


@dataclass(frozen=True)
class SparseTopology:
    """
    idx_coo must be coalesced and of shape (2, nnz) or (3, nnz).

    Operations on idx_coo are not guaranteed to give contiguous tensors.
    """

    idx_coo: Integer[t.LongTensor, "sp nnz"]
    shape: tuple[int, ...]

    @property
    def device(self) -> t.device:
        return self.idx_coo.device

    @property
    def dtype(self) -> t.dtype:
        return self.idx_coo.dtype

    @property
    def T(self) -> Integer[t.LongTensor, "sp nnz"]:
        shape = self.shape
        match len(self.idx_coo):
            case 2:
                idx_coo_trans = self.idx_coo[[1, 0]]
                shape_trans = (shape[1], shape[0])
                return SparseTopology(idx_coo_trans, shape_trans)

            case 3:
                idx_coo_trans = self.idx_coo[[0, 2, 1]]
                shape_trans = (shape[0], shape[2], shape[1])
                return SparseTopology(idx_coo_trans, shape_trans)

    @cached_property
    def idx_crow(self):
        return coalesced_coo_to_crow_idx(self.idx_coo, self.shape)

    @cached_property
    def idx_crow_int32(self):
        return coalesced_coo_to_crow_idx(self.idx_coo, self.shape, dtype=t.int32)

    @cached_property
    def idx_col(self):
        return coalesced_coo_to_col_idx(self.idx_coo, self.shape)

    @cached_property
    def idx_col_int32(self):
        return coalesced_coo_to_col_idx(self.idx_coo, self.shape, dtype=t.int32)

    def _nnz(self) -> int:
        return self.idx_coo.size(1)

    def to(self, *args, **kwargs) -> SparseTopology:
        return SparseTopology(self.idx_coo.to(*args, **kwargs), self.shape)

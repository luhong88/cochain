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

        # Enforce contiguous memory layout.
        object.__setattr__(self, "_idx_coo", self._idx_coo.contiguous())

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

    @property
    def T(self) -> SparseTopology:
        """
        Use cache injection to preserve cached_property for transpose.
        """
        idx_coo_sorted = self.idx_coo[:, self.coo_to_csc_perm]

        idx_coo_trans = idx_coo_sorted.clone()
        idx_coo_trans[-1] = idx_coo_sorted[-2]
        idx_coo_trans[-2] = idx_coo_sorted[-1]

        shape_trans = self.shape[:-2] + (self.shape[-1], self.shape[-2])

        sp_topo_trans = SparseTopology(idx_coo_trans, shape_trans)

        attr_map = {
            "idx_ccol": "idx_crow",
            "idx_ccol_int32": "idx_crow_int32",
            "idx_crow": "idx_ccol",
            "idx_crow_int32": "idx_ccol_int32",
            "idx_col": "idx_row",
            "idx_col_int32": "idx_row_int32",
            "idx_row": "idx_col",
            "idx_row_int32": "idx_col_int32",
        }

        for attr, attr_trans in attr_map.items():
            if attr in self.__dict__:
                sp_topo_trans.__dict__[attr_trans] = self.__dict__[attr]

        return sp_topo_trans

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
        # idx_coo respect all to() arguments, including dtype, device, non_blocking,
        # copy, and memory_format.
        new_idx_coo = self.idx_coo.to(*args, **kwargs)
        new_sp_topo = SparseTopology(new_idx_coo, self.shape)

        # Handle the cached index tensors. Extract the device and dtype arguments
        # from the new_idx_coo, and get the non_blocking and copy arguments from
        # kwargs. Ignores the memory_format argument, since index tensors will
        # always be contiguous.
        target_device = new_idx_coo.device
        target_dtype = new_idx_coo.dtype

        # Forbid conversion to (most) non-int dtypes.
        if target_dtype.is_floating_point or target_dtype.is_complex:
            raise ValueError("SparseTopology indices cannot be float or complex.")

        cache_kwargs = {
            "non_blocking": kwargs.get("non_blocking", False),
            "copy": kwargs.get("copy", False),
        }

        # All cached index tensors respect the dtype, device, non_blocking, and
        # copy argument, but the coo_to_csc_perm and *_int32 tensors will ignore
        # the dtype argument.
        cached_attrs_dtype_covariant = [
            "idx_ccol",
            "idx_crow",
            "idx_col",
            "idx_row",
        ]
        cached_attrs_dtype_invariant = [
            "coo_to_csc_perm",
            "idx_ccol_int32",
            "idx_crow_int32",
            "idx_col_int32",
            "idx_row_int32",
        ]

        for attr in cached_attrs_dtype_covariant:
            if attr in self.__dict__:
                new_sp_topo.__dict__[attr] = self.__dict__[attr].to(
                    dtype=target_dtype, device=target_device, **cache_kwargs
                )

        for attr in cached_attrs_dtype_invariant:
            if attr in self.__dict__:
                new_sp_topo.__dict__[attr] = self.__dict__[attr].to(
                    device=target_device, **cache_kwargs
                )

        return new_sp_topo

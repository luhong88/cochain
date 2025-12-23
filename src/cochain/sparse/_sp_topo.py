from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

import torch as t
from jaxtyping import Integer

from ._index import (
    coalesced_coo_to_col_idx,
    coalesced_coo_to_compressed_idx,
    coalesced_coo_to_row_idx,
    get_csc_sort_perm,
)


def _validate_coo_idx_shape(coo_idx: Integer[t.LongTensor, "sp nnz"], shape: t.Size):
    match len(shape):
        case 1:
            raise ValueError(
                "The 'idx_coo' tensor must have at least two sparse dimensions."
            )

        case 2:
            if coo_idx.size(0) != 2:
                raise ValueError(
                    "For a 2D sparse coo tensor, 'idx_coo' must be of shape (2, nnz)."
                )

        case 3:
            if coo_idx.size(0) != 3:
                raise ValueError(
                    "For a sparse coo tensor with a batch dimension, 'idx_coo' "
                    + "must be of shape (3, nnz)."
                )

            nnz = coo_idx.size(-1)
            n_batch = shape[0]
            batch_idx = coo_idx[0]

            # For batched sparse tensors, enforce the condition that the tensor
            # has equal nnz along the batch dimension, which is required for
            # conversion to sparse csr format.

            # If the input tensor has equal nnz along the batch dimension, then
            # the nnz per tensor in the batch is given by nnz // batch.
            if nnz % n_batch != 0:
                raise ValueError(
                    f"Total nnz ({nnz}) is not divisible by batch size ({n_batch})."
                )

            nnz_per_batch = nnz // n_batch

            # It is possible for a tensor to have non-equal nnz along the batch
            # dimension but still satisfies nnz % batch = 0 (e.g., if the first
            # tensor has 6 nnz, while the second has 2). This second check rules
            # out this possibility.
            if batch_idx is not None:
                batch_counts = t.bincount(batch_idx, minlength=n_batch)
                if not (batch_counts == nnz_per_batch).all():
                    raise ValueError(
                        "The equal nnz per batch item condition is not met."
                    )

        case _:
            raise NotImplementedError(
                "More than one batch dimensions is not supported."
            )


@dataclass(frozen=True)
class SparseTopology:
    """
    idx_coo must be coalesced and of shape (2, nnz) or (3, nnz).
    This class does not check that idx_coo is coalesced.
    """

    _idx_coo: Integer[t.LongTensor, "sp nnz"]
    shape: t.tuple[int, ...] | t.Size

    def __post_init__(self):
        _validate_coo_idx_shape(self.idx_coo, self.shape)

        # Manual out-of-bound index check
        min_idx = self.idx_coo.amin(dim=1)
        lower_ok = (min_idx >= 0).all()

        max_idx = self.idx_coo.amax(dim=1)
        bounds = t.tensor(
            self.shape, device=self.idx_coo.device, dtype=self.idx_coo.dtype
        )
        upper_ok = (max_idx < bounds).all()

        if not (upper_ok and lower_ok):
            raise ValueError("idx_coo contains out-of-bound indices.")

        # Enforce contiguous memory layout. Use object.__setattr__() to bypass
        # frozen=True.
        object.__setattr__(self, "_idx_coo", self._idx_coo.contiguous())

        # Coerse shape dtype.
        object.__setattr__(self, "shape", t.Size(self.shape))

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
        """
        For sparse csr/csc tensors, the leading batch dimension(s) do not count
        towards sparse_dim(); for sparse coo tensors, no such distinction is
        made. Here we follow the sparse csr/csc convention.
        """
        return 2

    @property
    def T(self) -> SparseTopology:
        """
        Note that the transpose preserves the batch dimension and only operates
        on the sparse dimensions.

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
            "idx_col": "idx_row_csc",
            "idx_col_int32": "idx_row_csc_int32",
            "idx_row_csc": "idx_col",
            "idx_row_csc_int32": "idx_col_int32",
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
        return coalesced_coo_to_compressed_idx(self.idx_coo, self.shape, format="ccol")

    @cached_property
    def idx_ccol_int32(self) -> Integer[t.IntTensor, "*b nnz/b"]:
        return coalesced_coo_to_compressed_idx(
            self.idx_coo, self.shape, format="ccol", dtype=t.int32
        )

    @cached_property
    def idx_crow(self) -> Integer[t.LongTensor, "*b nnz/b"]:
        return coalesced_coo_to_compressed_idx(self.idx_coo, self.shape, format="crow")

    @cached_property
    def idx_crow_int32(self) -> Integer[t.IntTensor, "*b nnz/b"]:
        return coalesced_coo_to_compressed_idx(
            self.idx_coo, self.shape, format="crow", dtype=t.int32
        )

    @cached_property
    def idx_col(self) -> Integer[t.LongTensor, "*b nnz/b"]:
        return coalesced_coo_to_col_idx(self.idx_coo, self.shape)

    @cached_property
    def idx_col_int32(self) -> Integer[t.IntTensor, "*b nnz/b"]:
        return coalesced_coo_to_col_idx(self.idx_coo, self.shape, dtype=t.int32)

    @cached_property
    def idx_row_csc(self) -> Integer[t.LongTensor, "*b nnz/b"]:
        return coalesced_coo_to_row_idx(self.idx_coo, self.shape, self.coo_to_csc_perm)

    @cached_property
    def idx_row_csc_int32(self) -> Integer[t.IntTensor, "*b nnz/b"]:
        return coalesced_coo_to_row_idx(
            self.idx_coo, self.shape, self.coo_to_csc_perm, dtype=t.int32
        )

    def _nnz(self) -> int:
        """
        For batched sparse csr/csc tensors, the _nnz() method returns the number
        of nonzero elements per batch item; for sparse coo tensors, the _nnz()
        method returns the total number of nonzero elements, regardless of batch
        dimensions. Here we follow the sparse coo convention.
        """
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
            "idx_row_csc",
        ]
        cached_attrs_dtype_invariant = [
            "coo_to_csc_perm",
            "idx_ccol_int32",
            "idx_crow_int32",
            "idx_col_int32",
            "idx_row_csc_int32",
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

    def size(self, dim: int | None = None) -> int | t.Size:
        if dim is None:
            return self.shape
        else:
            return self.shape[dim]

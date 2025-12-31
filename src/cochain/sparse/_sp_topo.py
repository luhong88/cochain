from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import Sequence

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


def check_topo_equality(
    self_topo: SparseTopology, other_topo: SparseTopology, msg: str
):
    # Enforce equal topology requirement with three increasingly more expensive
    # checks: 1) same underlying sp_topo object, 2) same sp_topo shape, and 3)
    # same sp_topo.idx_coo elements.
    if self_topo is other_topo:
        pass

    elif self_topo.shape == other_topo.shape and t.equal(
        self_topo.idx_coo, other_topo.idx_coo
    ):
        pass

    else:
        raise ValueError(msg)


@dataclass(frozen=True)
class BlockDiagConfig:
    batch_perm: Integer[t.LongTensor, " nnz"]
    nnzs: list[int]
    sp_topo_shapes: list[t.Size]

    def to(self, *args, **kwargs) -> BlockDiagConfig:
        new_batch_perm = self.batch_perm.to(*args, **kwargs)
        return BlockDiagConfig(new_batch_perm, self.nnzs, self.sp_topo_shapes)


@dataclass(frozen=True)
class SparseTopology:
    """
    idx_coo must be coalesced and of shape (2, nnz) or (3, nnz).
    This class does not check that idx_coo is coalesced.
    """

    _idx_coo: Integer[t.LongTensor, "sp nnz"]
    shape: t.tuple[int, ...] | t.Size
    block_diag_config: BlockDiagConfig | None = None

    @property
    def idx_coo(self) -> Integer[t.LongTensor, "sp nnz"]:
        return self._idx_coo

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

        # Enforce ownership and contiguous memory layout. Use object.__setattr__()
        # to bypass frozen=True.
        object.__setattr__(
            self,
            "_idx_coo",
            self._idx_coo.detach().clone(memory_format=t.contiguous_format),
        )

        # Coerse shape dtype.
        object.__setattr__(self, "shape", t.Size(self.shape))

    @classmethod
    def batch_diag(cls, block_sp_topos: Sequence[SparseTopology]) -> SparseTopology:
        """
        Construct a block diagonal matrix using a list of `SparseTopology`.
        """
        # Pick a representative SparseTopology and use it to determine device,
        # dtype, and batch/dense dimension information.
        rep_sp_topo = block_sp_topos[0]

        device = rep_sp_topo.device
        idx_dtype = rep_sp_topo.dtype

        # Determine the input SparseTopology sparse row/column shapes.
        r_sizes_cum = t.tensor(
            [0] + [sp_topo.size(-2) for sp_topo in block_sp_topos],
            dtype=idx_dtype,
            device=device,
        ).cumsum(dim=0)

        c_sizes_cum = t.tensor(
            [0] + [sp_op.size(-1) for sp_op in block_sp_topos],
            dtype=idx_dtype,
            device=device,
        ).cumsum(dim=0)

        # Compute block offsets and apply to concatenated coo index.
        nnzs = [sp_topo._nnz() for sp_topo in block_sp_topos]
        nnz_concat = t.tensor(nnzs, dtype=idx_dtype, device=device)

        r_offset = t.repeat_interleave(r_sizes_cum[:-1], nnz_concat)
        c_offset = t.repeat_interleave(c_sizes_cum[:-1], nnz_concat)

        idx_coo_concat = t.hstack(
            [
                sp_topo.idx_coo.to(device=device, dtype=idx_dtype)
                for sp_topo in block_sp_topos
            ]
        )
        idx_coo_concat[-2] += r_offset
        idx_coo_concat[-1] += c_offset

        # If there is a batch dimension, the coo index needs to be sorted first
        # by batch item order; find the permutation for this sort. If there is no
        # batch dim, this sort does nothing.
        batch_perm = t.sort(idx_coo_concat[0], stable=True).indices

        # Determine the concatenated SparseTopology shape.
        if rep_sp_topo.n_batch_dim > 0:
            sp_topo_shape_concat = t.Size(
                [rep_sp_topo.size(0), r_sizes_cum[-1], c_sizes_cum[-1]]
            )
        else:
            sp_topo_shape_concat = t.Size([r_sizes_cum[-1], c_sizes_cum[-1]])

        # Record block diag construction information for disassembly
        # block_ptr = t.repeat_interleave(t.arange(len(sp_ops)), nnz_concat)
        sp_topo_shapes = [sp_topo.shape for sp_topo in block_sp_topos]
        config = BlockDiagConfig(batch_perm, nnz_concat, sp_topo_shapes)

        # Construct concatenated SparseTopology.
        sp_topo_concat = SparseTopology(
            idx_coo_concat[:, batch_perm],
            shape=sp_topo_shape_concat,
            block_diag_config=config,
        )

        return sp_topo_concat

    def unbatch_diag(
        self,
    ) -> tuple[list[SparseTopology], Integer[t.LongTensor, " nnz"]]:
        """
        Deconstruct a block diagonal `SparseTopology` into a list of constituent
        `SparseTopology`s.
        """
        if not isinstance(self.block_diag_config, BlockDiagConfig):
            raise ValueError("A valid 'block_diag_config' is required for disassembly.")

        device = self.device
        idx_dtype = self.dtype

        block_perm_inv = t.argsort(self.block_diag_config.batch_perm)

        # Undo the batch dim sort, so that the idx_coo is back in a per-block
        # ordering. Fancy indexing guarantees copying.
        idx_coo_concat = self.idx_coo[:, block_perm_inv]

        # Undo the per-block, cumulative index offsets.
        r_sizes_cum = t.tensor(
            [0] + [shape[-2] for shape in self.block_diag_config.sp_topo_shapes],
            dtype=idx_dtype,
            device=device,
        ).cumsum(dim=0)

        c_sizes_cum = t.tensor(
            [0] + [shape[-1] for shape in self.block_diag_config.sp_topo_shapes],
            dtype=idx_dtype,
            device=device,
        ).cumsum(dim=0)

        nnz_concat = t.tensor(
            self.block_diag_config.nnzs, dtype=idx_dtype, device=device
        )

        r_offset = t.repeat_interleave(r_sizes_cum[:-1], nnz_concat)
        c_offset = t.repeat_interleave(c_sizes_cum[:-1], nnz_concat)

        idx_coo_concat[-2] -= r_offset
        idx_coo_concat[-1] -= c_offset

        # Split the concatenated idx_coo into constituent parts. Note that
        # t.split() creates a view without copying; this is okay because the
        # SparseTopology constructor enforces copying.
        idx_coo_list = t.split(idx_coo_concat, self.block_diag_config.nnzs, dim=-1)

        sp_topo_list = [
            SparseTopology(idx_coo, shape)
            for idx_coo, shape in zip(
                idx_coo_list, self.block_diag_config.sp_topo_shapes, strict=True
            )
        ]

        return sp_topo_list, block_perm_inv

    # TODO: optimize to avoid multiple index tensor copies
    @classmethod
    def to_block(
        cls, sp_topos: Sequence[Sequence[SparseTopology | None]]
    ) -> tuple[SparseTopology, Integer[t.LongTensor, " nnz"]]:
        """
        Construct a block matrix as a `SparseTopology` from a 2D grid of existing
        `SparseTopology`s. None is allowed to represent empty/zero blocks.
        """
        # Pick a representative SparseTopology and use it to determine device,
        # dtype, and batch/dense dimension information.
        rep_sp_topo = None
        for sp_topo_row in sp_topos:
            for sp_topo in sp_topo_row:
                if sp_topo is not None:
                    rep_sp_topo = sp_topo

        if rep_sp_topo is None:
            raise ValueError("At least one block in 'blocks' must be non-null value.")

        device = rep_sp_topo.device
        idx_dtype = rep_sp_topo.dtype

        # Determine the input SparseTopology sparse row/column shapes. The row
        # sizes and col sizes are each represented as a 2D tensor; None is assigned
        # a shape of 0.
        r_sizes = []
        c_sizes = []
        for sp_topo_row in sp_topos:
            row_r_sizes = []
            row_c_sizes = []

            for sp_topo in sp_topo_row:
                if sp_topo is None:
                    row_r_sizes.append(0)
                    row_c_sizes.append(0)
                else:
                    row_r_sizes.append(sp_topo.size(-2))
                    row_c_sizes.append(sp_topo.size(-1))

            r_sizes.append(row_r_sizes)
            c_sizes.append(row_c_sizes)

        r_sizes = t.tensor(r_sizes, dtype=idx_dtype, device=device)
        c_sizes = t.tensor(c_sizes, dtype=idx_dtype, device=device)

        # Sanity checks on the row/col shape tensors.
        for sp_op_dims in [r_sizes, c_sizes]:
            for block_dim in [0, 1]:
                if not (sp_op_dims.sum(dim=block_dim) > 0).any():
                    raise ValueError(
                        "Rows/columns with all None or degenerate blocks are not allowed."
                    )

        r_max_size_per_row = r_sizes.max(dim=-1, keepdim=True).values
        r_min_size_per_row = t.zeros_like(r_max_size_per_row)
        if not (
            (r_sizes == r_max_size_per_row) | (r_sizes == r_min_size_per_row)
        ).all():
            raise ValueError(
                "The blocks in each row (except for None) must have the same number of rows."
            )

        c_max_size_per_col = c_sizes.max(dim=0, keepdim=True).values
        c_min_size_per_col = t.zeros_like(c_max_size_per_col)
        if not (
            (c_sizes == c_max_size_per_col) | (c_sizes == c_min_size_per_col)
        ).all():
            raise ValueError(
                "The blocks in each col (except for None) must have the same number of cols."
            )

        # Construct the concatenated coo index by iterating over the block coo indices.
        idx_coo_list = []
        r_offset = 0
        c_offset = 0
        for r_idx, sp_topo_row in enumerate(sp_topos):
            for c_idx, sp_topo in enumerate(sp_topo_row):
                if sp_topo is not None:
                    idx_coo = (
                        sp_topo.idx_coo.detach()
                        .clone()
                        .to(device=device, dtype=idx_dtype)
                    )
                    idx_coo[-2] += r_offset
                    idx_coo[-1] += c_offset
                    idx_coo_list.append(idx_coo)

                c_offset += c_sizes[r_idx, c_idx]

            r_offset += r_sizes[r_idx, c_idx]
            c_offset = 0

        idx_coo_concat = t.hstack(idx_coo_list)

        # If there is a batch dimension, the coo index needs to be sorted first
        # by batch item order; find the permutation for this sort.
        batch_perm = t.sort(idx_coo_concat[0], stable=True).indices

        # Determine the concatenated SparseTopology shape.
        n_row = r_sizes.sum(dim=0).max().item()
        n_col = c_sizes.sum(dim=-1).max().item()
        if rep_sp_topo.n_batch_dim > 0:
            sp_topo_shape_concat = t.Size([rep_sp_topo.size(0), n_row, n_col])
        else:
            sp_topo_shape_concat = t.Size([n_row, n_col])

        # Construct concatenated coo index.
        sp_topo_concat = SparseTopology(
            idx_coo_concat[:, batch_perm], shape=sp_topo_shape_concat
        )

        return sp_topo_concat, batch_perm

    def size(self, dim: int | None = None) -> int | t.Size:
        if dim is None:
            return self.shape
        else:
            return self.shape[dim]

    def _nnz(self) -> int:
        """
        For batched sparse csr/csc tensors, the _nnz() method returns the number
        of nonzero elements per batch item; for sparse coo tensors, the _nnz()
        method returns the total number of nonzero elements, regardless of batch
        dimensions. Here we follow the sparse coo convention.
        """
        return self.idx_coo.size(1)

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

    @property
    def dtype(self) -> t.dtype:
        return self.idx_coo.dtype

    @property
    def device(self) -> t.device:
        return self.idx_coo.device

    def to(self, *args, **kwargs) -> SparseTopology:
        # idx_coo respect all to() arguments, including dtype, device, non_blocking,
        # copy, and memory_format.
        new_idx_coo = self.idx_coo.to(*args, **kwargs)

        # block_diag_config repsect all to() arguments as well.
        if self.block_diag_config is None:
            new_block_diag_config = None
        else:
            new_block_diag_config = self.block_diag_config.to(*args, **kwargs)

        new_sp_topo = SparseTopology(new_idx_coo, self.shape, new_block_diag_config)

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

    @cached_property
    def idx_crow(self) -> Integer[t.LongTensor, "*b nnz/b"]:
        return coalesced_coo_to_compressed_idx(self.idx_coo, self.shape, format="crow")

    @cached_property
    def idx_crow_int32(self) -> Integer[t.IntTensor, "*b nnz/b"]:
        return coalesced_coo_to_compressed_idx(
            self.idx_coo, self.shape, format="crow", dtype=t.int32
        )

    # TODO: consider renaming this to idx_col_csr to avoid confusion.
    @cached_property
    def idx_col(self) -> Integer[t.LongTensor, "*b nnz/b"]:
        return coalesced_coo_to_col_idx(self.idx_coo, self.shape)

    @cached_property
    def idx_col_int32(self) -> Integer[t.IntTensor, "*b nnz/b"]:
        return coalesced_coo_to_col_idx(self.idx_coo, self.shape, dtype=t.int32)

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
    def idx_row_csc(self) -> Integer[t.LongTensor, "*b nnz/b"]:
        return coalesced_coo_to_row_idx(self.idx_coo, self.shape, self.coo_to_csc_perm)

    @cached_property
    def idx_row_csc_int32(self) -> Integer[t.IntTensor, "*b nnz/b"]:
        return coalesced_coo_to_row_idx(
            self.idx_coo, self.shape, self.coo_to_csc_perm, dtype=t.int32
        )

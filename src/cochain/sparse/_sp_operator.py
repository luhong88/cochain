from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import torch as t
from jaxtyping import Float, Integer

from ._base_operator import BaseOperator, is_scalar, validate_matmul_args
from ._matmul import dense_sp_mm, sp_dense_mm, sp_mv, sp_sp_mm, sp_vm
from ._sp_topo import SparseTopology, check_topo_equality


@dataclass
class SparseOperator(BaseOperator):
    sp_topo: Integer[SparseTopology, "*b r c"]
    val: Float[t.Tensor, " nnz *d"]

    # TODO: optimized intake paths for csc/csr formats
    # TODO: allow additional kwargs, e.g., copy=True
    @classmethod
    def from_tensor(cls, tensor: t.Tensor) -> SparseOperator:
        coalesced_tensor = tensor.to_sparse_coo().coalesce()

        return cls(
            sp_topo=SparseTopology(coalesced_tensor.indices(), coalesced_tensor.shape),
            val=coalesced_tensor.values(),
        )

    @classmethod
    def batch_diag(cls, blocks: Sequence[t.Tensor | BaseOperator]) -> SparseOperator:
        """
        Construct a block diagonal matrix as a SparseOperator from a list of tensor,
        SparseOperator, and DiagOperator objects.
        """
        # Convert all input elements to SparseOperator.
        sp_op_list: list[SparseOperator] = []
        for block in blocks:
            match block:
                case t.Tensor():
                    sp_op_list.append(SparseOperator.from_tensor(block))
                case BaseOperator():
                    sp_op_list.append(block.to_sparse_operator())
                case _:
                    raise TypeError()

        # Pick a representative SparseOperator and use it to determine device and
        # dtype information.
        rep_sp_op = sp_op_list[0]
        device = rep_sp_op.device
        val_dtype = rep_sp_op.dtype

        # Construct concatenated sparse topology.
        sp_topo_list = [sp_op.sp_topo for sp_op in sp_op_list]
        sp_topo_concat = SparseTopology.batch_diag(sp_topo_list)

        # Construct concatenated values tensor.
        val_list = [
            sp_op.val.to(device=device, dtype=val_dtype) for sp_op in sp_op_list
        ]

        batch_perm = sp_topo_concat.block_diag_config.batch_perm

        if rep_sp_op.n_dense_dim == 0:
            val_concat = t.hstack(val_list)[batch_perm]
        else:
            val_concat = t.vstack(val_list)[batch_perm]

        return SparseOperator(sp_topo_concat, val_concat)

    def unbatch_diag(self) -> list[BaseOperator]:
        # Reconstruct the constituent SparseTopology.
        sp_topo_list, block_perm_inv = self.sp_topo.unbatch_diag()

        # Perform similar reconstruction on the values
        val_concat = self.val[:, block_perm_inv]
        val_list = t.split(val_concat, self.sp_topo.block_diag_config.nnzs, dim=0)

        sp_op_list = [
            SparseOperator(sp_topo, val) for sp_topo, val in zip(sp_topo_list, val_list)
        ]

        return sp_op_list

    # TODO: optimize to avoid multiple index tensor copies
    # TODO: preserve cached indices
    @classmethod
    def to_block(cls, blocks: Sequence[Sequence[t.Tensor | BaseOperator | None]]):
        """
        Construct a block matrix as a SparseOperator from a 2D grid of existing
        tensor, SparseOperator, or DiagOperator. None is allowed to represent
        empty/zero blocks.
        """
        # Convert all input blocks except for None to SparseOperator.
        sp_ops: list[list[SparseOperator]] = []
        for block_row in blocks:
            sp_op_row = []
            for block in block_row:
                match block:
                    case t.Tensor():
                        sp_op_row.append(SparseOperator.from_tensor(block))
                    case BaseOperator():
                        sp_op_row.append(block.to_sparse_operator())
                    case None:
                        sp_op_row.append(None)
                    case _:
                        raise TypeError()
            sp_ops.append(sp_op_row)

        # Pick a representative SparseOperator and use it to determine device,
        # dtype, and batch/dense dimension information.
        rep_sp_op = None
        for sp_op_row in sp_ops:
            for sp_op in sp_op_row:
                if sp_op is not None:
                    rep_sp_op = sp_op

        if rep_sp_op is None:
            raise ValueError("At least one block in 'blocks' must be non-null value.")

        device = rep_sp_op.device
        val_dtype = rep_sp_op.dtype
        idx_dtype = rep_sp_op.sp_topo.dtype

        # Determine the input SparseOperator sparse row/column shapes. The row
        # sizes and col sizes are each represented as a 2D tensor; None is assigned
        # a shape of 0.
        r_sizes = []
        c_sizes = []
        for sp_op_row in sp_ops:
            row_r_sizes = []
            row_c_sizes = []

            for sp_op in sp_op_row:
                if sp_op is None:
                    row_r_sizes.append(0)
                    row_c_sizes.append(0)
                else:
                    row_r_sizes.append(sp_op.sp_topo.size(-2))
                    row_c_sizes.append(sp_op.sp_topo.size(-1))

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
        for r_idx, sp_op_row in enumerate(sp_ops):
            for c_idx, sp_op in enumerate(sp_op_row):
                if sp_op is not None:
                    idx_coo = (
                        sp_op.sp_topo.idx_coo.detach()
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
        perm = t.sort(idx_coo_concat[0], stable=True).indices

        # Determine the concatenated SparseTopology shape.
        n_row = r_sizes.sum(dim=0).max().item()
        n_col = c_sizes.sum(dim=-1).max().item()
        if rep_sp_op.sp_topo.n_batch_dim > 0:
            sp_topo_shape_concat = t.Size([rep_sp_op.sp_topo.size(0), n_row, n_col])
        else:
            sp_topo_shape_concat = t.Size([n_row, n_col])

        # Construct concatenated coo index.
        sp_topo_concat = SparseTopology(
            idx_coo_concat[:, perm], shape=sp_topo_shape_concat
        )

        # Construct concatenated values tensor.
        val_list = [
            sp_op.val.to(device=device, dtype=val_dtype)
            for sp_op_row in sp_ops
            for sp_op in sp_op_row
            if sp_op is not None
        ]

        if rep_sp_op.n_dense_dim == 0:
            val_concat = t.hstack(val_list)[perm]
        else:
            val_concat = t.vstack(val_list)[perm]

        return SparseOperator(sp_topo_concat, val_concat)

    def __post_init__(self):
        if self.val.device != self.sp_topo.device:
            raise RuntimeError("'val' and 'sp_topo' must be on the same device.")

        if self.val.size(0) != self.sp_topo._nnz():
            raise ValueError("nnz mismatch between 'val' and 'sp_topo'.")

        if not t.isfinite(self.val).all():
            raise ValueError("SparseOperator values contain NaN or Inf.")

        # Enforce contiguous memory layout.
        self.val = self.val.contiguous()

    def apply(self, fn: Callable, **kwargs) -> SparseOperator:
        """
        Apply a sparsity-preserving function on the values of SparseOperator.
        """
        new_val = fn(self.val, **kwargs)

        if new_val.size(0) != self.val.size(0):
            raise RuntimeError("Function changed the nnz dim of the SparseOperator.")

        return SparseOperator(self.sp_topo, new_val)

    def __neg__(self) -> SparseOperator:
        return SparseOperator(self.sp_topo, -self.val)

    def __add__(self, other) -> SparseOperator:
        """
        Elementwise-addition of two SparseOperators that share the same topology/
        sparsity pattern.
        """
        match other:
            case SparseOperator():
                check_topo_equality(
                    self.sp_topo,
                    other.sp_topo,
                    msg="SparseOperator __add__ only supports operators with identical topologies.",
                )
                return SparseOperator(self.sp_topo, self.val + other.val)
            case _:
                return NotImplemented

    def __sub__(self, other) -> SparseOperator:
        match other:
            case SparseOperator():
                check_topo_equality(
                    self.sp_topo,
                    other.sp_topo,
                    msg="SparseOperator __sub__ only supports operators with identical topologies.",
                )
                return SparseOperator(self.sp_topo, self.val - other.val)
            case _:
                return NotImplemented

    @classmethod
    def assemble(cls, *operators: BaseOperator) -> SparseOperator:
        """
        Efficiently sum multiple SparseOperators and/or DiagOperators with different
        topologies.
        """
        if not operators:
            raise ValueError("No operators to assemble.")

        coo_tensors = [op.to_sparse_coo() for op in operators]

        all_idx = t.hstack([coo.indices() for coo in coo_tensors])
        all_val = t.hstack([coo.values() for coo in coo_tensors])

        return t.sparse_coo_tensor(
            all_idx, all_val, size=coo_tensors[0].size()
        ).coalesce()

    def __mul__(self, other) -> SparseOperator:
        """
        Scalar multiplication.
        """
        return (
            SparseOperator(self.sp_topo, self.val * other)
            if is_scalar(other)
            else NotImplemented
        )

    def __truediv__(self, other) -> SparseOperator:
        """
        Scalar division.
        """
        return (
            SparseOperator(self.sp_topo, self.val / other)
            if is_scalar(other)
            else NotImplemented
        )

    def __matmul__(self, other):
        """
        Implement self @ other
        """
        validate_matmul_args(self, other)

        match other:
            case SparseOperator():
                idx_crow, idx_col, val, shape = sp_sp_mm(
                    self.val, self.sp_topo, other.val, other.sp_topo
                )
                sp_sp = t.sparse_csr_tensor(idx_crow, idx_col, val, shape)
                return self.from_tensor(sp_sp)

            case t.Tensor():
                match other.ndim:
                    case 1:
                        return sp_mv(self.val, self.sp_topo, other)
                    case 2:
                        return sp_dense_mm(self.val, self.sp_topo, other)

            case _:
                return NotImplemented

    def __rmatmul__(self, other):
        """
        Implement other @ self
        """
        validate_matmul_args(self, other)

        match other:
            # Do not check for case SparseOperator(), which is handled by __matmul__
            case t.Tensor():
                match other.ndim:
                    case 1:
                        return sp_vm(other, self.val, self.sp_topo)
                    case 2:
                        return dense_sp_mm(other, self.val, self.sp_topo)

            case _:
                return NotImplemented

    @property
    def shape(self) -> t.Size:
        return self.sp_topo.shape + self.val.shape[1:]

    def _nnz(self) -> int:
        """
        For batched sparse csr/csc tensors, the _nnz() method returns the number
        of nonzero elements per batch item; for sparse coo tensors, the _nnz()
        method returns the total number of nonzero elements, regardless of batch
        dimensions. Here we follow the sparse coo convention.
        """
        return self.sp_topo._nnz()

    @property
    def n_batch_dim(self) -> int:
        return self.sp_topo.n_batch_dim

    @property
    def n_sp_dim(self) -> int:
        """
        For sparse csr/csc tensors, the leading batch dimension(s) do not count
        towards sparse_dim(); for sparse coo tensors, no such distinction is
        made. Here we follow the sparse csr/csc convention.
        """
        return self.sp_topo.n_sp_dim

    @property
    def n_dense_dim(self) -> int:
        return self.val.ndim - 1

    @property
    def T(self) -> SparseOperator:
        """
        Note that the transpose preserves the batch and dense dimensions and only
        operates on the sparse dimensions.
        """
        val_trans = self.val[self.sp_topo.coo_to_csc_perm]
        sp_topo_trans = self.sp_topo.T
        return SparseOperator(sp_topo_trans, val_trans)

    def clone(
        self, memory_format: t.memory_format = t.contiguous_format
    ) -> SparseOperator:
        """
        Create a new SparseOperator with the same `sp_topo` but with the `val`
        cloned (in the contiguous format by default).
        """
        return SparseOperator(self.sp_topo, self.val.clone(memory_format=memory_format))

    def detach(self) -> SparseOperator:
        """
        Create a new SparseOperator with the same `sp_topo` but with the `val` detached.
        """
        return SparseOperator(self.sp_topo, self.val.detach())

    def to(self, *args, **kwargs) -> SparseOperator:
        new_val = self.val.to(*args, **kwargs)

        # The topology object ignores dtype
        new_sp_topo = self.sp_topo.to(
            device=new_val.device,
            copy=kwargs.get("copy", False),
            non_blocking=kwargs.get("non_blocking", False),
        )

        return SparseOperator(new_sp_topo, new_val)

    def to_dense(self) -> Float[t.Tensor, "*b r c *d"]:
        return self.to_sparse_coo().to_dense()

    def to_sparse_operator(self) -> SparseOperator:
        return self

    def to_sparse_coo(self) -> Float[t.Tensor, "*b r c *d"]:
        return t.sparse_coo_tensor(
            self.sp_topo.idx_coo,
            self.val,
            self.shape,
            dtype=self.dtype,
            device=self.device,
        ).coalesce()

    def to_sparse_csr(self, int32: bool = False) -> Float[t.Tensor, "*b r c *d"]:
        if int32:
            idx_crow = self.sp_topo.idx_crow_int32
            idx_col = self.sp_topo.idx_col_int32
        else:
            idx_crow = self.sp_topo.idx_crow
            idx_col = self.sp_topo.idx_col

        if self.n_batch_dim == 0:
            val = self.val
        else:
            val = self.val.view(self.size(0), -1).contiguous()

        return t.sparse_csr_tensor(
            idx_crow,
            idx_col,
            val,
            self.shape,
            dtype=self.dtype,
            device=self.device,
        )

    def _prepare_sparse_csr_components(self, int32: bool):
        if int32:
            idx_ccol = self.sp_topo.idx_ccol_int32
            idx_row_csc = self.sp_topo.idx_row_csc_int32
        else:
            idx_ccol = self.sp_topo.idx_ccol
            idx_row_csc = self.sp_topo.idx_row_csc

        if self.n_batch_dim == 0:
            val = self.val[self.sp_topo.coo_to_csc_perm].contiguous()
        else:
            val = (
                self.val[self.sp_topo.coo_to_csc_perm]
                .view(self.size(0), -1)
                .contiguous()
            )

        return idx_ccol, idx_row_csc, val

    # TODO: add other direct transpose conversion options?
    def to_sparse_csr_transposed(
        self, int32: bool = False
    ) -> Float[t.Tensor, "*b c r *d"]:
        idx_ccol, idx_row_csc, val = self._prepare_sparse_csr_components(int32)

        # (*b, r, c, *d) -> (*b, c, r, *d)
        shape_trans = (
            self.shape[: self.n_batch_dim]
            + self.shape[self.n_batch_dim : self.n_batch_dim + self.n_sp_dim][::-1]
            + self.shape[(self.n_dim - self.n_dense_dim) :]
        )

        return t.sparse_csr_tensor(
            idx_ccol,
            idx_row_csc,
            val,
            shape_trans,
            dtype=self.dtype,
            device=self.device,
        )

    def to_sparse_csc(self, int32: bool = False) -> Float[t.Tensor, "*b r c *d"]:
        idx_ccol, idx_row_csc, val = self._prepare_sparse_csr_components(int32)

        return t.sparse_csc_tensor(
            idx_ccol,
            idx_row_csc,
            val,
            self.shape,
            dtype=self.dtype,
            device=self.device,
        )

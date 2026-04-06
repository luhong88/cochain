from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import torch
from jaxtyping import Float, Integer
from torch import LongTensor, Tensor

from ._base_decoupled_tensor import BaseDecoupledTensor, is_scalar, validate_matmul_args
from ._matmul import dense_sp_mm, sp_dense_mm, sp_mv, sp_sp_mm, sp_vm
from ._pattern import SparsityPattern, check_pattern_equality


@dataclass
class SparseDecoupledTensor(BaseDecoupledTensor):
    """
    A custom sparse tensor representation that explicitly decouples non-zero numerical
    values (`val`) from the sparsity pattern (`pattern`). The class supports
    sparse-sparse and sparse-dense linear algebra operations using the native
    CSR and CSC representations in pytorch, caches different index representations
    whenever safe, and offers some basic matrix algebra and manipulation utils.

    Notes
    -----
    * This class is primarily designed for sparse, 2D matrices; batching, if there
      is any, is assumed to be in the form of block diagonal batching; however,
      this class does allow for (at most) one leading batch dimension and arbitrary
      trailing dense dimensions. Therefore, the shape of a supported tensor is
      (*b, r, c, *d), where *b matches to at most one dimension. The number of these
      dimensions can be queried as follows: `n_batch_dim` gives the number of batch
      dimensions (0 or 1), `n_sp_dim` gives the number of sparse dimensions (r and c;
      must be 2), and `n_dense_dim` gives the number of dimensions matching *d.
      The `_nnz()` method gives the total number of nonzero elements (nnz), rather
      than nnz per batch element.
    * Due to pytorch CSR/CSC requirements, for a tensor with a batch dim, all
      constituent sparse tensors must have the same nnz.
    * If the `SparseDecoupledTensor` is initialized directly via its constructor
      using the `pattern` and `val` arguments, it is assumed that the sparse COO
      tensor used to generate these arguments was already coalesced. On the other
      hand, if the `SparseDecoupledTensor` is initialized via `from_tensor()`, then
      the input tensor is always first converted to a COO tensor and coalesced.
    * The `pattern` attribute and the `SparsityPattern` class enforces strict
      onwership of tensor indices via cloning during init and should be treated
      as immutable.
    * The `SparsityPattern` class caches CSR/CSC indices whenever they are calculated.
      These indices are preserved by the following operations: element-wise and
      unary operators, memory management/casting (`clone()`, `detach()`, `to()`),
      and matrix transpose (`.T`). Operations such as matrix assembly/disassembly,
      subsetting, and matrix multiplication will drop the index caches.
    """

    pattern: Integer[SparsityPattern, "*b r c"]
    val: Float[Tensor, " nnz *d"]

    def __post_init__(self):
        if self.val.device != self.pattern.device:
            raise RuntimeError("'val' and 'pattern' must be on the same device.")

        if self.val.size(0) != self.pattern._nnz():
            raise ValueError("nnz mismatch between 'val' and 'pattern'.")

        if not torch.isfinite(self.val).all():
            raise ValueError("SparseDecoupledTensor values contain NaN or Inf.")

        # Enforce contiguous memory layout.
        self.val = self.val.contiguous()

    # TODO: allow additional kwargs, e.g., copy=True
    @classmethod
    def from_tensor(cls, tensor: Tensor) -> SparseDecoupledTensor:
        coalesced_tensor = tensor.to_sparse_coo().coalesce()

        # If the input coo tensor has more than two dimensions (r, c), need to
        # determine if the extra dimensions are batch or dense dimensions.
        # SparsityPattern does not need to see the dense dimensions.
        n_dense_dim = coalesced_tensor.dense_dim()
        if n_dense_dim > 0:
            pattern_shape = coalesced_tensor.shape[:-n_dense_dim]
        else:
            pattern_shape = coalesced_tensor.shape

        return cls(
            pattern=SparsityPattern(coalesced_tensor.indices(), pattern_shape),
            val=coalesced_tensor.values(),
        )

    @classmethod
    def pack_block_diag(
        cls, blocks: Sequence[Tensor | BaseDecoupledTensor]
    ) -> SparseDecoupledTensor:
        """
        Construct a block diagonal matrix as a `SparseDecoupledTensor` from a list of tensor,
        `SparseDecoupledTensor`, and `DiagDecoupledTensor` objects.
        """
        # Convert all input elements to SparseDecoupledTensor.
        sp_op_list: list[SparseDecoupledTensor] = []
        for block in blocks:
            match block:
                case Tensor():
                    sp_op_list.append(SparseDecoupledTensor.from_tensor(block))
                case BaseDecoupledTensor():
                    sp_op_list.append(block.to_sparse_operator())
                case _:
                    raise TypeError()

        # Pick a representative SparseDecoupledTensor and use it to determine device and
        # dtype information.
        rep_sp_op = sp_op_list[0]
        device = rep_sp_op.device
        val_dtype = rep_sp_op.dtype

        # Construct concatenated sparse topology.
        pattern_list = [sdt.pattern for sdt in sp_op_list]
        pattern_concat = SparsityPattern.pack_block_diag(pattern_list)

        # Construct concatenated values tensor.
        val_list = [sdt.val.to(device=device, dtype=val_dtype) for sdt in sp_op_list]

        batch_perm = pattern_concat.block_diag_config.batch_perm

        if rep_sp_op.n_dense_dim == 0:
            val_concat = torch.hstack(val_list)[batch_perm]
        else:
            val_concat = torch.vstack(val_list)[batch_perm]

        return SparseDecoupledTensor(pattern_concat, val_concat)

    def unpack_block_diag(self) -> list[SparseDecoupledTensor]:
        """
        Deconstruct a block diagonal `SparseDecoupledTensor` into a list of constituent
        `SparseDecoupledTensor`s.
        """
        # Reconstruct the constituent SparsityPattern.
        pattern_list, block_perm_inv = self.pattern.unpack_block_diag()

        # Perform similar reconstruction on the values
        val_concat = self.val[block_perm_inv]
        val_list = torch.split(val_concat, self.pattern.block_diag_config.nnzs, dim=0)

        sp_op_list = [
            SparseDecoupledTensor(pattern, val)
            for pattern, val in zip(pattern_list, val_list)
        ]

        return sp_op_list

    def unpack_by_ptrs(
        self,
        n_blocks: int,
        row_ptrs: Integer[LongTensor, " r"],
        col_ptrs: Integer[LongTensor, " c"],
    ) -> list[SparseDecoupledTensor]:
        """
        Deconstruct a block diagonal `SparseDecoupledTensor` into a list of constituent
        `SparseDecoupledTensor`s using row and col "pointers" that indicate the
        "onwership structures" of constituent blocks. Useful for unpacking a block-
        diagonal structures derived from a MeshBatch object.

        Note that the block indices in `row_ptrs` and `col_ptrs` are assumed to
        be 0-indexed.

        If there is a batch dimension, the requirement that the tensor has equal
        nnz along the batch dimension also needs to be satisfied by the constituent
        blocks for the unpacking to be successful.
        """
        idx_dtype = self.pattern.dtype
        device = self.device

        # Get the row and col sizes for each block.
        block_idx, row_sizes = row_ptrs.unique(return_counts=True)
        row_block_sizes = torch.zeros(n_blocks, dtype=idx_dtype, device=device)
        row_block_sizes[block_idx] = row_sizes

        block_idx, col_sizes = col_ptrs.unique(return_counts=True)
        col_block_sizes = torch.zeros(n_blocks, dtype=idx_dtype, device=device)
        col_block_sizes[block_idx] = col_sizes

        # Sort the indices and values by the row index; this undoes the sort by
        # batch dim (if there is any).
        block_perm_inv = torch.argsort(self.pattern.idx_coo[-2], stable=True)

        val_sorted = self.val[block_perm_inv]
        idx_coo_sorted = self.pattern.idx_coo[:, block_perm_inv]

        # Assign block membership to the nonzero values. By slicing the row_ptrs
        # by the row indices of the nonzero elements, we directly get the block
        # membership assignment (for the row indices). Since the input tensor
        # is assumed to be block diagonal, there is no need to repeat this
        # calculation for the col_ptrs.
        block_membership = row_ptrs[idx_coo_sorted[-2]]

        blocks = []
        for block_idx in range(n_blocks):
            # Determine block size.
            if self.n_batch_dim > 0:
                block_shape = torch.Size(
                    [
                        self.size(0),
                        row_block_sizes[block_idx],
                        col_block_sizes[block_idx],
                    ]
                )

            else:
                block_shape = torch.Size(
                    [
                        row_block_sizes[block_idx],
                        col_block_sizes[block_idx],
                    ]
                )

            # Extract block values and indices.
            block_mask = block_membership == block_idx

            if block_mask.sum() == 0:
                block_val_sorted = torch.empty((0,), dtype=self.dtype, device=device)
                block_idx_coo_sorted = torch.empty(
                    (self.n_batch_dim + 2, 0), dtype=idx_dtype, device=device
                )

            else:
                block_val = val_sorted[block_mask]
                block_idx_coo = idx_coo_sorted[:, block_mask]

                # Redo the sort by batch dim (if there is any) within the block.
                block_perm = torch.argsort(block_idx_coo[0], stable=True)

                block_val_sorted = block_val[block_perm]
                block_idx_coo_sorted = block_idx_coo[:, block_perm]

                # Convert the global row/col indices to per-block indices.
                block_idx_coo_sorted[-2] -= row_block_sizes[:block_idx].sum()
                block_idx_coo_sorted[-1] -= col_block_sizes[:block_idx].sum()

            # Assemble sparse tensor.
            block_pattern = SparsityPattern(block_idx_coo_sorted, block_shape)
            block_tensor = SparseDecoupledTensor(block_pattern, block_val_sorted)

            blocks.append(block_tensor)

        return blocks

    @classmethod
    def bmat(cls, blocks: Sequence[Sequence[Tensor | BaseDecoupledTensor | None]]):
        """
        Construct a block matrix as a `SparseDecoupledTensor` from a 2D grid of existing
        tensor, `SparseDecoupledTensor`, or `DiagDecoupledTensor`. `None` is allowed to represent
        empty/zero blocks.
        """
        # Convert all input blocks except for None to SparseDecoupledTensor, and produce
        # two lists: one flattened list of sdt.val (excluding None), and a nested
        # list of sdt.pattern (including None).
        pattern_list = []
        val_list = []
        rep_sp_op = None
        for block_row in blocks:
            pattern_row = []
            for block in block_row:
                match block:
                    case Tensor():
                        sdt = SparseDecoupledTensor.from_tensor(block)
                        pattern_row.append(sdt.pattern)
                        val_list.append(sdt.val)

                        # Pick a representative SparseDecoupledTensor and use it to
                        # determine device, dtype, and dense dimension information.
                        if rep_sp_op is None:
                            rep_sp_op = sdt

                    case BaseDecoupledTensor():
                        sdt = block.to_sparse_operator()
                        pattern_row.append(sdt.pattern)
                        val_list.append(sdt.val)

                        if rep_sp_op is None:
                            rep_sp_op = sdt

                    case None:
                        pattern_row.append(None)

                    case _:
                        raise TypeError()

            pattern_list.append(pattern_row)

        if rep_sp_op is None:
            raise ValueError("At least one block in 'blocks' must be non-null value.")

        device = rep_sp_op.device
        val_dtype = rep_sp_op.dtype

        # Construct concatenated SparsityPattern.
        pattern_concat, batch_perm = SparsityPattern.bmat(pattern_list)

        # Construct concatenated values tensor.
        val_list_uniform = [val.to(device=device, dtype=val_dtype) for val in val_list]

        if rep_sp_op.n_dense_dim == 0:
            val_concat = torch.hstack(val_list_uniform)[batch_perm]
        else:
            val_concat = torch.vstack(val_list_uniform)[batch_perm]

        return SparseDecoupledTensor(pattern_concat, val_concat)

    def apply(self, fn: Callable, **kwargs) -> SparseDecoupledTensor:
        """
        Apply a sparsity-preserving function on the values of SparseDecoupledTensor.
        """
        new_val = fn(self.val, **kwargs)

        if new_val.size(0) != self.val.size(0):
            raise RuntimeError(
                "Function changed the nnz dim of the SparseDecoupledTensor."
            )

        return SparseDecoupledTensor(self.pattern, new_val)

    def __neg__(self) -> SparseDecoupledTensor:
        return SparseDecoupledTensor(self.pattern, -self.val)

    def abs(self) -> SparseDecoupledTensor:
        return SparseDecoupledTensor(self.pattern, self.val.abs())

    def diagonal(self) -> Float[Tensor, "*b diag"]:
        if self.n_batch_dim == 0:
            return self.val[
                torch.argwhere(
                    self.pattern.idx_coo[0] == self.pattern.idx_coo[1]
                ).flatten()
            ]
        else:
            raise NotImplementedError()

    # TODO: implement for batched operators
    # TODO: write tests
    def off_diagonal(self) -> SparseDecoupledTensor:
        if self.n_batch_dim == 0:
            off_diag_mask = self.pattern.idx_coo[0] != self.pattern.idx_coo[1]
            off_diag_pattern = SparsityPattern(
                self.pattern.idx_coo[:, off_diag_mask], self.pattern.shape
            )
            off_diag_val = self.val[off_diag_mask]
            return SparseDecoupledTensor(off_diag_pattern, off_diag_val)

    # TODO: implement trace for batched operators
    # TODO: write tests fot tr()
    # TODO: implement the same tr and diagonal() functions for DiagDecoupledTensors
    @property
    def tr(self) -> Float[Tensor, "*b"]:
        if self.n_batch_dim == 0:
            return self.diagonal().sum(dim=0)
        else:
            raise NotImplementedError()

    # TODO: implement for batched operators
    # TODO: write tests
    def triu(self, diagonal: int = 0) -> SparseDecoupledTensor:
        if self.n_batch_dim == 0:
            triu_mask = self.pattern.idx_coo[0] <= self.pattern.idx_coo[1] - diagonal
            triu_pattern = SparsityPattern(
                self.pattern.idx_coo[:, triu_mask], self.pattern.shape
            )
            triu_val = self.val[triu_mask]
            return SparseDecoupledTensor(triu_pattern, triu_val)

    def __add__(self, other) -> SparseDecoupledTensor:
        """
        Elementwise-addition of two SparseDecoupledTensors that share the same topology/
        sparsity pattern.
        """
        match other:
            case SparseDecoupledTensor():
                check_pattern_equality(
                    self.pattern,
                    other.pattern,
                    msg="SparseDecoupledTensor __add__ only supports operators with identical topologies.",
                )
                return SparseDecoupledTensor(self.pattern, self.val + other.val)
            case _:
                return NotImplemented

    def __sub__(self, other) -> SparseDecoupledTensor:
        match other:
            case SparseDecoupledTensor():
                check_pattern_equality(
                    self.pattern,
                    other.pattern,
                    msg="SparseDecoupledTensor __sub__ only supports operators with identical topologies.",
                )
                return SparseDecoupledTensor(self.pattern, self.val - other.val)
            case _:
                return NotImplemented

    @classmethod
    def assemble(cls, *operators: BaseDecoupledTensor) -> SparseDecoupledTensor:
        """
        Efficiently sum multiple SparseDecoupledTensors and/or DiagDecoupledTensors with different
        topologies.
        """
        if not operators:
            raise ValueError("No operators to assemble.")

        coo_tensors = [op.to_sparse_coo() for op in operators]

        all_idx = torch.hstack([coo.indices() for coo in coo_tensors])
        all_val = torch.hstack([coo.values() for coo in coo_tensors])

        # from_tensor() handles coalesce.
        return SparseDecoupledTensor.from_tensor(
            torch.sparse_coo_tensor(all_idx, all_val, size=coo_tensors[0].size())
        )

    def __mul__(self, other) -> SparseDecoupledTensor:
        """
        Scalar multiplication.
        """
        return (
            SparseDecoupledTensor(self.pattern, self.val * other)
            if is_scalar(other)
            else NotImplemented
        )

    def __truediv__(self, other) -> SparseDecoupledTensor:
        """
        Scalar division.
        """
        return (
            SparseDecoupledTensor(self.pattern, self.val / other)
            if is_scalar(other)
            else NotImplemented
        )

    def __matmul__(self, other):
        """
        Implement self @ other
        """
        validate_matmul_args(self, other)

        match other:
            case SparseDecoupledTensor():
                idx_crow, idx_col, val, shape = sp_sp_mm(
                    self.val, self.pattern, other.val, other.pattern
                )
                sp_sp = torch.sparse_csr_tensor(idx_crow, idx_col, val, shape)
                return self.from_tensor(sp_sp)

            case Tensor():
                match other.ndim:
                    case 1:
                        return sp_mv(self.val, self.pattern, other)
                    case 2:
                        return sp_dense_mm(self.val, self.pattern, other)

            case _:
                return NotImplemented

    def __rmatmul__(self, other):
        """
        Implement other @ self
        """
        validate_matmul_args(self, other)

        match other:
            # Do not check for case SparseDecoupledTensor(), which is handled by __matmul__
            case Tensor():
                match other.ndim:
                    case 1:
                        return sp_vm(other, self.val, self.pattern)
                    case 2:
                        return dense_sp_mm(other, self.val, self.pattern)

            case _:
                return NotImplemented

    @property
    def shape(self) -> torch.Size:
        return self.pattern.shape + self.val.shape[1:]

    def _nnz(self) -> int:
        """
        For batched sparse csr/csc tensors, the _nnz() method returns the number
        of nonzero elements per batch item; for sparse coo tensors, the _nnz()
        method returns the total number of nonzero elements, regardless of batch
        dimensions. Here we follow the sparse coo convention.
        """
        return self.pattern._nnz()

    @property
    def n_batch_dim(self) -> int:
        return self.pattern.n_batch_dim

    @property
    def n_sp_dim(self) -> int:
        """
        For sparse csr/csc tensors, the leading batch dimension(s) do not count
        towards sparse_dim(); for sparse coo tensors, no such distinction is
        made. Here we follow the sparse csr/csc convention.
        """
        return self.pattern.n_sp_dim

    @property
    def n_dense_dim(self) -> int:
        return self.val.ndim - 1

    @property
    def T(self) -> SparseDecoupledTensor:
        """
        Note that the transpose preserves the batch and dense dimensions and only
        operates on the sparse dimensions.
        """
        val_trans = self.val[self.pattern.coo_to_csc_perm]
        pattern_trans = self.pattern.T
        return SparseDecoupledTensor(pattern_trans, val_trans)

    def clone(
        self, memory_format: torch.memory_format = torch.contiguous_format
    ) -> SparseDecoupledTensor:
        """
        Create a new SparseDecoupledTensor with the same `pattern` but with the `val`
        cloned (in the contiguous format by default).
        """
        return SparseDecoupledTensor(
            self.pattern, self.val.clone(memory_format=memory_format)
        )

    def detach(self) -> SparseDecoupledTensor:
        """
        Create a new SparseDecoupledTensor with the same `pattern` but with the `val` detached.
        """
        return SparseDecoupledTensor(self.pattern, self.val.detach())

    def to(self, *args, **kwargs) -> SparseDecoupledTensor:
        new_val = self.val.to(*args, **kwargs)

        # The topology object ignores dtype
        new_pattern = self.pattern.to(
            device=new_val.device,
            copy=kwargs.get("copy", False),
            non_blocking=kwargs.get("non_blocking", False),
        )

        return SparseDecoupledTensor(new_pattern, new_val)

    def to_dense(self) -> Float[Tensor, "*b r c *d"]:
        return self.to_sparse_coo().to_dense()

    def to_sparse_operator(self) -> SparseDecoupledTensor:
        return self

    def to_sparse_coo(self) -> Float[Tensor, "*b r c *d"]:
        return torch.sparse_coo_tensor(
            self.pattern.idx_coo,
            self.val,
            self.shape,
            dtype=self.dtype,
            device=self.device,
        ).coalesce()

    def to_sparse_csr(self, int32: bool = False) -> Float[Tensor, "*b r c *d"]:
        if int32:
            idx_crow = self.pattern.idx_crow_int32
            idx_col = self.pattern.idx_col_int32
        else:
            idx_crow = self.pattern.idx_crow
            idx_col = self.pattern.idx_col

        if self.n_batch_dim == 0:
            val = self.val
        else:
            val = self.val.view(self.size(0), -1).contiguous()

        return torch.sparse_csr_tensor(
            idx_crow,
            idx_col,
            val,
            self.shape,
            dtype=self.dtype,
            device=self.device,
        )

    def _prepare_sparse_csr_components(self, int32: bool):
        if int32:
            idx_ccol = self.pattern.idx_ccol_int32
            idx_row_csc = self.pattern.idx_row_csc_int32
        else:
            idx_ccol = self.pattern.idx_ccol
            idx_row_csc = self.pattern.idx_row_csc

        if self.n_batch_dim == 0:
            val = self.val[self.pattern.coo_to_csc_perm].contiguous()
        else:
            val = (
                self.val[self.pattern.coo_to_csc_perm]
                .view(self.size(0), -1)
                .contiguous()
            )

        return idx_ccol, idx_row_csc, val

    # TODO: add other direct transpose conversion options?
    def to_sparse_csr_transposed(
        self, int32: bool = False
    ) -> Float[Tensor, "*b c r *d"]:
        idx_ccol, idx_row_csc, val = self._prepare_sparse_csr_components(int32)

        # (*b, r, c, *d) -> (*b, c, r, *d)
        shape_trans = (
            self.shape[: self.n_batch_dim]
            + self.shape[self.n_batch_dim : self.n_batch_dim + self.n_sp_dim][::-1]
            + self.shape[(self.n_dim - self.n_dense_dim) :]
        )

        return torch.sparse_csr_tensor(
            idx_ccol,
            idx_row_csc,
            val,
            shape_trans,
            dtype=self.dtype,
            device=self.device,
        )

    def to_sparse_csc(self, int32: bool = False) -> Float[Tensor, "*b r c *d"]:
        idx_ccol, idx_row_csc, val = self._prepare_sparse_csr_components(int32)

        return torch.sparse_csc_tensor(
            idx_ccol,
            idx_row_csc,
            val,
            self.shape,
            dtype=self.dtype,
            device=self.device,
        )

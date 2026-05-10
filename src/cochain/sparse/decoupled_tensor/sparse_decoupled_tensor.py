from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import torch
from jaxtyping import Bool, Float, Integer
from torch import Tensor

from ._matmul import dense_sp_mm, sp_dense_mm, sp_mv, sp_sp_mm, sp_vm
from ._spgemm_plan import (
    SpSpMMPlan,
    discover_matmul_pattern,
    get_bwd_plan_A,
    get_bwd_plan_B,
    get_fwd_plan,
)
from .base_decoupled_tensor import (
    BaseDecoupledTensor,
    is_scalar_like,
    validate_matmul_args,
)
from .pattern import SparsityPattern, check_pattern_equality


@dataclass
class SparseDecoupledTensor(BaseDecoupledTensor):
    """
    A decoupled representation of sparse matrices.

    A custom sparse tensor representation that explicitly decouples non-zero numerical
    values (`values`) from the sparsity pattern (`pattern`). The class supports
    block matrix (de)construction, sparse matrix multiplications, and other basic
    sparse matrix manipulation utils; see `sparse.linalg` for sparse linear solvers
    and eigensolvers for `SparseDecoupledTensor`s.

    This class is primarily designed for sparse, 2D matrices (with block diagonal
    batching). However, this class does allow for (at most) one leading batch
    dimension and arbitrary trailing dense dimensions. Therefore, the shape of a
    supported tensor is (*b, r, c, *d), where *b matches to at most one dimension.
    Due to pytorch CSR/CSC requirements, if the sparse tensor has a batch dimension,
    then all sparse matrices in the batch must have the same number of nonzero
    elements.

    This class caches CSR/CSC sparse index representations whenever they are
    calculated through its `SparsityPattern`. In addition, this class caches the
    `scatter_add()` indices required to perform (masked) general sparse matrix-matrix
    multiplications using a weakref cache system attached to its `SparsityPattern`.


    Parameters
    ----------
    pattern : [*b, r, c]
        a `SparsityPattern` object representing the sparse topology of the tensor.
    values : [nz, *d]
        A dense tensor representing the nonzero elements of the matrix. If the
        input tensor is not contiguous, this class will create and store a
        contiguous copy of the input. The nonzero elements are assumed to be
        coalesced and in sparse COO/CSR ordering.

    Attributes
    ----------
    tr : [*b, *d]
        The trace of the sparse matrix.
    shape
        The shape of the sparse matrix.
    n_batch_dim
        The number of batch dimensions (either one or zero).
    n_sp_dim
        The number of sparse dimensions (which is always two).
    n_dense_dim
        The number of dense dimensions.
    T
        The matrix transpose along the two sparse dimensions.

    Notes
    -----
    Operations that preserve SparsityPattern.
    Operations that preserve BlockDiagConfig.

    Let `a` be a scalar-like object (float, int, or a 1D/0D tensor with one element),
    and let `A and `B` be two `SparseDecoupledTensor`s with the same sparsity pattern.
    This class supports the following tensor arithmetic operations:

    * Unary: negation (`-A`),
    * Binary: addition (`A + B`), subtraction (`A - B`), sclar multiplication
      (`a*D1`), and scalar division (`D1/a`).

    In addition, this class supports the following types of matrix multiplications:

    * Matmul between two `SparseDecoupledTensor`s,
    * Matmul between a `DiagDecoupledTensor` and a `SparseDecoupledTensor`,
    * Matmul between a `SparseDecoupledTensor` and a dense 2D tensor,
    * Matrix-vector multiplication with a dense 1D tensor.

    Note that batch dimensions are not allowed in all four cases.

    These indices are preserved by the following operations: element-wise and
    unary operators, memory management/casting (`clone()`, `detach()`, `to()`),
    and matrix transpose (`.T`). Operations such as matrix assembly/disassembly,
    subsetting, and matrix multiplication will drop the index caches.
    """

    pattern: Integer[SparsityPattern, "*b r c"]
    values: Float[Tensor, " nz *d"]

    def __post_init__(self):
        if self.values.device != self.pattern.device:
            raise RuntimeError("'values' and 'pattern' must be on the same device.")

        if self.values.size(0) != self.pattern._nnz():
            raise ValueError("nnz mismatch between 'values' and 'pattern'.")

        if not torch.isfinite(self.values).all():
            raise ValueError("SparseDecoupledTensor values contain NaN or Inf.")

        # Enforce contiguous memory layout.
        self.values = self.values.contiguous()

    # TODO: allow additional kwargs, e.g., copy=True
    @classmethod
    def from_tensor(cls, tensor: Tensor) -> SparseDecoupledTensor:
        """
        Construct a `SparseDecoupledTensor` from a PyTorch tensor.

        Parameters
        ----------
        tensor
            A PyTorch tensor. This tensor will be converted to a sparse COO
            tensor and coalesced.

        Returns
        -------
            A `SparseDecoupledTensor` representation of the input tensor.
        """
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
            values=coalesced_tensor.values(),
        )

    @classmethod
    def pack_block_diag(
        cls, blocks: Sequence[Tensor | BaseDecoupledTensor]
    ) -> SparseDecoupledTensor:
        """
        Construct a block-diagonal sparse tensor.

        Parameters
        ----------
        blocks
            A sequence of tensors, `SparseDecoupledTensor`s, and/or
            `DiagDecoupledTensor`s from which to construct the block-diagonal tensor.

        Returns
        -------
            A block-diagonal `SparseDecoupledTensor`.
        """
        # Convert all input elements to SparseDecoupledTensor.
        sp_op_list: list[SparseDecoupledTensor] = []
        for block in blocks:
            match block:
                case Tensor():
                    sp_op_list.append(SparseDecoupledTensor.from_tensor(block))
                case BaseDecoupledTensor():
                    sp_op_list.append(block.to_sdt())
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
        val_list = [sdt.values.to(device=device, dtype=val_dtype) for sdt in sp_op_list]

        batch_perm = pattern_concat.block_diag_config.batch_perm

        if rep_sp_op.n_dense_dim == 0:
            val_concat = torch.hstack(val_list)[batch_perm]
        else:
            val_concat = torch.vstack(val_list)[batch_perm]

        return SparseDecoupledTensor(pattern_concat, val_concat)

    def unpack_block_diag(self) -> list[SparseDecoupledTensor]:
        """
        Deconstruct a block-diagonal `SparseDecoupledTensor`.

        If self is constructed via `pack_block_diag()`, this method undo the
        block-diagonal structure and returns the constituent tensors.

        Returns
        -------
           A list of constituent `SparseDecoupledTensor`s. Note that this function
           always unpacks the block-diagonal structure into `SparseDecoupledTensor`,
           regardless of the representation of the original tensors used to
           construct the block-diagonal matrix.

        Notes
        -----
        If a block-diagonal tensor is constructed via `pack_block_diag()`, the
        block-diagonal structure is encoded in the `BlockDiagConfig` attribute of
        its `SparsityPattern`, which is used by this method to undo the
        block-diagonal construction. If the block-diagonal structure arises
        instead from block-diagonal batching of meshes, then the `BlockDiagConfig`
        attribute will not be available, and `unpack_by_ptrs()` should be used
        instead for unpacking the block-diagonal structure.
        """
        # Reconstruct the constituent SparsityPattern.
        pattern_list, block_perm_inv = self.pattern.unpack_block_diag()

        # Perform similar reconstruction on the values
        val_concat = self.values[block_perm_inv]
        val_list = torch.split(val_concat, self.pattern.block_diag_config.nnzs, dim=0)

        sp_op_list = [
            SparseDecoupledTensor(pattern, val)
            for pattern, val in zip(pattern_list, val_list)
        ]

        return sp_op_list

    def unpack_by_ptrs(
        self,
        n_blocks: int,
        row_ptrs: Integer[Tensor, " r"],
        col_ptrs: Integer[Tensor, " c"],
    ) -> list[SparseDecoupledTensor]:
        """
        Deconstruct a block-diagonal `SparseDecoupledTensor` via pointers.

        Parameters
        ----------
        n_blocks
            The number of blocks in the block-diagonal tensor.
        row_ptrs : [r,]
            A 0-indexed tensor that specifies the block membership of the rows.
        col_ptrs : [c,]
            A 0-indexed tensor that specifies the block membership of the columns.

        Returns
        -------
           A list of constituent `SparseDecoupledTensor`s.

        Notes
        -----
        If a block-diagonal tensor is constructed via `pack_block_diag()`, then
        `unpack_block_diag()` is the preferred method for undoing the block-diagonal
        construction.

        If there is a batch dimension, the requirement that the constituent
        matrices have equal nnz needs to be satisfied within each block for the
        unpacking to be successful.
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

        val_sorted = self.values[block_perm_inv]
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
    def bmat(
        cls, blocks: Sequence[Sequence[Tensor | BaseDecoupledTensor | None]]
    ) -> SparseDecoupledTensor:
        """
        Construct a `SparseDecoupledTensor` from a 2D grid of tensors.

        Parameters
        ----------
        blocks:
            A list of list of tensors, `DiagDecoupledTensor`s, and/or
            `SparseDecoupledTensors`. None is allowed to represent empty/zero
            blocks. Rows/columns with all `None` or degenerate blocks are also
            allowed.

        Returns
        -------
            A block `SparseDecoupledTensor`.

        Notes
        -----
        Note that block matrix construction will not preserve the `BlockDiagConfig`
        of the input `SparsityPattern`s, if there is any.
        """
        # Convert all input blocks except for None to SparseDecoupledTensor, and produce
        # two lists: one flattened list of sdt.values (excluding None), and a nested
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
                        val_list.append(sdt.values)

                        # Pick a representative SparseDecoupledTensor and use it to
                        # determine device, dtype, and dense dimension information.
                        if rep_sp_op is None:
                            rep_sp_op = sdt

                    case BaseDecoupledTensor():
                        sdt = block.to_sdt()
                        pattern_row.append(sdt.pattern)
                        val_list.append(sdt.values)

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

    def submatrix(
        self, row_mask: Bool[Tensor, " r"], col_mask: Bool[Tensor, " c"] | None = None
    ) -> SparseDecoupledTensor:
        """
        Extract a submatrix using row and col masks.

        Parameters
        ----------
        row_mask : [r,]
            A boolean mask marking the rows to be preserved for the submatrix.
        col_mask : [c,]
            A boolean mask marking the columns to be preserved for the submatrix.
            If `None`, then the column mask is assumed to be the same as the
            row mask.

        Returns
        -------
            A `SparseDecoupledTensor` submatrix.

        Notes
        -----
        This method preserves and updates the `BlockDiagConfig` attribute of the
        `SparsityPattern` of the original tensor, if there is any.

        If self has a batch dimension, the submatrix extraction will fail if the
        resulting submatrices in the batch violate the equal nnz per batch element
        assumption.
        """
        idx_coo_submat_mask, submat_pattern = self.pattern.submatrix(row_mask, col_mask)

        submat_val = self.values[idx_coo_submat_mask]

        return SparseDecoupledTensor(submat_pattern, submat_val)

    def constrain(self, mask: Bool[Tensor, " r"]) -> SparseDecoupledTensor:
        """
        Perform a soft masking on a symmetric semidefinite sparse tensor.

        For a symmetric sparse tensor $A$, if index $i$ is marked as `False`
        in the input `mask`, then all off-diagonal nonzero elements $A_{ij}$
        get set to zero, and the diagonal element $A_{ii}$ get set to one. This
        function assumes that the diagonal elements are not structurally zero
        at the masked rows/columns.

        Parameters
        ----------
        mask : [r,]
            A boolean mask marking the rows and columns to be preserved for the
            constrained matrix.

        Returns
        -------
            A constrained sparse tensor.

        Notes
        -----
        This method performs a similar function as the `submatrix()`, in that
        solving the linear system $A x = b$ with a "soft" masked $A$ and a
        "hard" subsetted $A$ should give the same $x$ (at the corresponding
        unmasked positions). The advantage of the soft masking approach implemented
        here is that the underlying `SparsityPattern` is preserved.
        """
        if self.size(-1) != self.size(-2):
            raise ValueError(
                "constrain() is only applicable to (batched) sparse square matrices."
            )

        if not torch.allclose(self.values, self.values[self.pattern.csc_to_coo_map]):
            raise ValueError("constrain() is only applicable to symmetric matrices.")

        r_idx = self.pattern.idx_coo[-2]
        c_idx = self.pattern.idx_coo[-1]

        r_idx_mask = ~mask[r_idx]
        c_idx_mask = ~mask[c_idx]

        diag_mask = r_idx == c_idx

        # Check that none of the masked rows/columns contain zero at the diagonal.
        nnz_masked_diag = (diag_mask & r_idx_mask).sum().item()
        if self.n_batch_dim > 0:
            n_masked_diag = (~mask).sum().item() * self.size(0)
        else:
            n_masked_diag = (~mask).sum().item()

        if nnz_masked_diag != n_masked_diag:
            raise ValueError(
                "constrain() can only be used to mask rows/colums where the "
                "diagonal element is nonzero."
            )

        # A nonzero element is numerically set to zero if (1) it is off-diagonal
        # and (2) its row or col index is masked.
        zero_mask = (~diag_mask) & (r_idx_mask | c_idx_mask)

        # A nonzero element is numerically set to one if (1) it is diagonal and
        # (2) its row/col index is masked.
        one_mask = diag_mask & r_idx_mask

        val_zeroed = torch.where(zero_mask, 0.0, self.values)
        val_oned = torch.where(one_mask, 1.0, val_zeroed)

        return SparseDecoupledTensor(self.pattern, val_oned)

    def apply(self, fn: Callable, **kwargs) -> SparseDecoupledTensor:
        """
        Apply a function to the nonzero elements of self.

        Parameters
        ----------
        fn
            A callable to be applied to the `values` attribute of self. If the
            values attribute has the shape (nz, *d), the callable is allowed to
            modify the *d dimensions, but not the size of the nz dimension.
        kwargs
            Additional keyword arguments to be passed to the callable.

        Returns
        -------
            A new `SparseDecoupedTensor` with transformed values but the same
            sparsity pattern.
        """
        new_val = fn(self.values, **kwargs)

        if new_val.size(0) != self.values.size(0):
            raise RuntimeError(
                "Function changed the nnz dim of the SparseDecoupledTensor."
            )

        return SparseDecoupledTensor(self.pattern, new_val)

    def __neg__(self) -> SparseDecoupledTensor:
        """Unary elementwise negation."""
        return SparseDecoupledTensor(self.pattern, -self.values)

    def abs(self) -> SparseDecoupledTensor:
        """Take the absolute value of the sparse tensor."""
        return SparseDecoupledTensor(self.pattern, self.values.abs())

    # TODO: write tests
    def diagonal(self) -> Float[Tensor, " diag *d"] | Float[Tensor, "b diag *d"]:
        """
        Return the diagonal values.

        Returns
        -------
            If self has the shape (r, c, *d), then this function returns a dense
            diagonal values tensor of shape (min(r, c), *d); if self has the shape
            (b, r, c, *d), then this function returns a sparse COO diagonal
            values tensor of shape (b, min(r, c), *d).
        """
        diag_mask = self.pattern.idx_coo[-1] == self.pattern.idx_coo[-2]
        diag_val = self.values[diag_mask]

        n_diag = min(self.pattern.size(-1), self.pattern.size(-2))
        dense_shape = diag_val.shape[1:]

        if self.n_batch_dim > 0:
            n_batch = self.size(0)
            out_shape = (n_batch, n_diag, *dense_shape)

            diag = torch.sparse_coo_tensor(
                indices=self.pattern.idx_coo[:-1, diag_mask],
                values=diag_val,
                size=out_shape,
                dtype=self.dtype,
                device=self.device,
                is_coalesced=True,
            )

        else:
            out_shape = (n_diag, *dense_shape)

            diag = torch.zeros(out_shape, device=self.device, dtype=self.dtype)
            diag[self.pattern.idx_coo[0, diag_mask]] = diag_val

        return diag

    # TODO: write tests
    def off_diagonal(self) -> SparseDecoupledTensor:
        """
        Return the off-diagonal part of the sparse matrix.

        If self has a batch dimension, this function will raise an error if the
        off-diagonal part does not satisfy the equal nnz per batch element assumption.

        Note that this method will generate a new `SparsityPattern` without any
        of the previously cached properties or the `BlockDiagConfig`.
        """
        off_diag_mask = self.pattern.idx_coo[-1] != self.pattern.idx_coo[-2]
        off_diag_pattern = SparsityPattern(
            self.pattern.idx_coo[:, off_diag_mask], self.pattern.shape
        )
        off_diag_val = self.values[off_diag_mask]
        return SparseDecoupledTensor(off_diag_pattern, off_diag_val)

    # TODO: write tests fot tr()
    @property
    def tr(self) -> Float[Tensor, "*d"] | Float[Tensor, " b *d"]:
        """The trace of the sparse matrix."""
        if self.n_batch_dim > 0:
            return torch.sparse.sum(self.diagonal(), dim=1)
        else:
            return self.diagonal().sum(dim=0)

    # TODO: write tests
    def triu(self, diagonal: int = 0) -> SparseDecoupledTensor:
        """
        Return the upper triangular part of the sparse matrix.

        If self has a batch dimension, this function will raise an error if the
        upper triangular part does not satisfy the equal nnz per batch element
        assumption.

        Parameters
        ----------
        diagonal
            The diagonal to consider. If zero, then all elements on and above
            the main diagonal are preserved; a positive value excludes just as
            many diagonals above the main diagonal, and a negative value includes
            just as many diagonals below the main diagonal.

        Returns
        -------
            The upper triangular part of the sparse matrix.

        Notes
        -----
        Note that this method will generate a new `SparsityPattern` without any
        of the previously cached properties or the `BlockDiagConfig`.
        """
        triu_mask = self.pattern.idx_coo[-2] <= self.pattern.idx_coo[-1] - diagonal
        triu_pattern = SparsityPattern(
            self.pattern.idx_coo[:, triu_mask], self.pattern.shape
        )
        triu_val = self.values[triu_mask]
        return SparseDecoupledTensor(triu_pattern, triu_val)

    def __add__(self, other) -> SparseDecoupledTensor:
        """Elementwise addition of two `SparseDecoupledTensor`s with shared topology."""
        match other:
            case SparseDecoupledTensor():
                check_pattern_equality(
                    self.pattern,
                    other.pattern,
                    msg="SparseDecoupledTensor __add__ only supports operators with identical topologies.",
                )
                return SparseDecoupledTensor(self.pattern, self.values + other.values)
            case _:
                return NotImplemented

    def __sub__(self, other) -> SparseDecoupledTensor:
        """Elementwise subtraction of two `SparseDecoupledTensor`s with identical topology."""
        match other:
            case SparseDecoupledTensor():
                check_pattern_equality(
                    self.pattern,
                    other.pattern,
                    msg="SparseDecoupledTensor __sub__ only supports operators with identical topologies.",
                )
                return SparseDecoupledTensor(self.pattern, self.values - other.values)
            case _:
                return NotImplemented

    @classmethod
    def assemble(cls, *tensors: BaseDecoupledTensor) -> SparseDecoupledTensor:
        """
        Sum multiple sparse tensors with potentially different sparsity patterns.

        Parameters
        ----------
        tensors
            A set of `SparseDecoupledTensor`s and/or `DiagDecoupledTensors` to
            be summed.

        Returns
        -------
            A summed `SparseDecoupledTensor`.

        Notes
        -----
        If the input sparse tensors share the same topology/sparsity pattern,
        then direct elementwise sum (`+`) is preferred over `assemble()`.

        Note that this method will generate a new `SparsityPattern` without any
        of the previously cached properties or the `BlockDiagConfig`.
        """
        if not tensors:
            raise ValueError("No operators to assemble.")

        coo_tensors = [op.to_sparse_coo() for op in tensors]

        all_idx = torch.hstack([coo.indices() for coo in coo_tensors])
        all_val = torch.hstack([coo.values() for coo in coo_tensors])

        # from_tensor() handles coalesce.
        return SparseDecoupledTensor.from_tensor(
            torch.sparse_coo_tensor(all_idx, all_val, size=coo_tensors[0].size())
        )

    def __mul__(self, other) -> SparseDecoupledTensor:
        """Scalar multiplication."""
        return (
            SparseDecoupledTensor(self.pattern, self.values * other)
            if is_scalar_like(other)
            else NotImplemented
        )

    def __truediv__(self, other) -> SparseDecoupledTensor:
        """Scalar division."""
        return (
            SparseDecoupledTensor(self.pattern, self.values / other)
            if is_scalar_like(other)
            else NotImplemented
        )

    # TODO: test sp-sp matmul caching
    def __matmul__(self, other):
        """
        Implement self @ other matmul.

        If both operands are `SparseDecoupledTensor`s, this function caches the
        indices required to perform the general sparse matrix multiplication
        (SpGEMM) via scatter_add(). In addition, if either of the operands
        requires gradient, this function also caches the indices required to
        perform masked SpGEMM for the backward pass.
        """
        validate_matmul_args(self, other)

        match other:
            case SparseDecoupledTensor():
                # Check if there is already a cached sparse-sparse matmul plan
                # (note that the plan is associated with the sparsity pattern objects).
                plan = self.pattern._spsp_matmul_plans.get(other.pattern, None)

                if plan is None:
                    # If there is no cached plan, generate one for the sparsity
                    # pattern pair.
                    c_idx_coo, c_idx_crow, c_idx_col = discover_matmul_pattern(
                        self.shape,
                        self.pattern.idx_crow,
                        self.pattern.idx_col,
                        other.shape,
                        other.pattern.idx_crow,
                        other.pattern.idx_col,
                    )

                    plan_fwd = get_fwd_plan(
                        c_idx_coo=c_idx_coo,
                        c_idx_crow=c_idx_crow,
                        c_idx_col=c_idx_col,
                        a_idx_crow=self.pattern.idx_crow,
                        a_idx_col=self.pattern.idx_col,
                        b_idx_ccol=other.pattern.idx_ccol,
                        b_idx_row=other.pattern.idx_row_csc,
                        b_csc_to_coo_map=other.pattern.csc_to_coo_map,
                    )

                    # Populate the forward plan only.
                    plan = SpSpMMPlan(plan_fwd, bwd_plan_A=None, bwd_plan_B=None)
                    self.pattern._spsp_matmul_plans[other.pattern] = plan

                # Populate the backward plans depending on autograd requirements.
                if (plan.bwd_plan_A is None) and self.requires_grad:
                    plan.bwd_plan_A = get_bwd_plan_A(
                        a_idx_coo=self.pattern.idx_coo,
                        c_idx_crow=plan.fwd_plan.c_idx_crow,
                        c_idx_col=plan.fwd_plan.c_idx_col,
                        b_idx_crow=other.pattern.idx_crow,
                        b_idx_col=other.pattern.idx_col,
                    )

                if (plan.bwd_plan_B is None) and other.requires_grad:
                    plan.bwd_plan_B = get_bwd_plan_B(
                        b_idx_coo=other.pattern.idx_coo,
                        c_idx_crow=plan.fwd_plan.c_idx_crow,
                        c_idx_col=plan.fwd_plan.c_idx_col,
                        c_shape=plan.fwd_plan.c_shape,
                        a_idx_ccol=self.pattern.idx_ccol,
                        a_idx_row=self.pattern.idx_row_csc,
                        a_csc_to_coo_map=self.pattern.csc_to_coo_map,
                    )

                # Perform sparse-sparse matmul with a plan using the custom
                # matmul implementation with masked backward pass.
                c_val, c_idx_coo, c_shape = sp_sp_mm(self.values, other.values, plan)
                c_pattern = SparsityPattern(c_idx_coo, c_shape)
                c_sdt = SparseDecoupledTensor(c_pattern, c_val)

                return c_sdt

            case Tensor():
                match other.ndim:
                    case 1:
                        return sp_mv(self.values, self.pattern, other)
                    case 2:
                        return sp_dense_mm(self.values, self.pattern, other)

            case _:
                return NotImplemented

    def __rmatmul__(self, other):
        """Implement other @ self."""
        validate_matmul_args(self, other)

        match other:
            # Do not check for case SparseDecoupledTensor(), which is handled by __matmul__
            case Tensor():
                match other.ndim:
                    case 1:
                        return sp_vm(other, self.values, self.pattern)
                    case 2:
                        return dense_sp_mm(other, self.values, self.pattern)

            case _:
                return NotImplemented

    @property
    def shape(self) -> torch.Size:
        """The shape of the sparse matrix."""
        return self.pattern.shape + self.values.shape[1:]

    def _nnz(self) -> int:
        """
        Return the total number of nonzero/specified elements in the sparse tensor.

        In PyTorch, for batched sparse CSC/CSR tensors, the _nnz() method returns
        the number of nonzero elements per batch item; for sparse COO tensors,
        the _nnz() method returns the total number of nonzero elements, regardless
        of batch dimensions. Here we follow the sparse COO convention.
        """
        return self.pattern._nnz()

    @property
    def n_batch_dim(self) -> int:
        """The number of batch dimensions (either one or zero)."""
        return self.pattern.n_batch_dim

    @property
    def n_sp_dim(self) -> int:
        """
        The number of sparse dimensions (which is always two).

        In PyTorch, for sparse CSC/CSR tensors, the leading batch dimension(s) do
        not count towards `sparse_dim()`; for sparse COO tensors, no such distinction
        is made. Here we follow the sparse CSC/CSR convention.
        """
        return self.pattern.n_sp_dim

    @property
    def n_dense_dim(self) -> int:
        """The number of dense dimensions."""
        return self.values.ndim - 1

    @property
    def T(self) -> SparseDecoupledTensor:
        """
        The matrix transpose along the two sparse dimensions.

        Note that the transpose preserves the batch and dense dimensions and only
        operates on the sparse dimensions. In addition, any cached sparse index
        tensors in the original `SparsityPattern` of self are preserved (but not
        the `BlockDiagConfig`, if there is any).
        """
        val_trans = self.values[self.pattern.csc_to_coo_map]
        pattern_trans = self.pattern.T

        return SparseDecoupledTensor(pattern_trans, val_trans)

    def clone(
        self, memory_format: torch.memory_format = torch.contiguous_format
    ) -> SparseDecoupledTensor:
        """Return a copy of self with the same `pattern` but cloned `values`."""
        return SparseDecoupledTensor(
            self.pattern, self.values.clone(memory_format=memory_format)
        )

    def detach(self) -> SparseDecoupledTensor:
        """Return a new `SparseDecoupledTensor` with the `values` detached."""
        return SparseDecoupledTensor(self.pattern, self.values.detach())

    def to(self, *args, **kwargs) -> SparseDecoupledTensor:
        """
        Perform dtype and/or device conversion.

        See `SparsityPattern.to()` for information on how the index tensors
        behave under dtype/device conversion.
        """
        new_val = self.values.to(*args, **kwargs)
        new_pattern = self.pattern.to(*args, **kwargs)

        return SparseDecoupledTensor(new_pattern, new_val)

    def to_dense(self) -> Float[Tensor, "r c *d"] | Float[Tensor, "b r c *d"]:
        """Create a dense copy of self."""
        return self.to_sparse_coo().to_dense()

    def to_sdt(self) -> SparseDecoupledTensor:
        """Return self."""
        return self

    def to_sparse_coo(self) -> Float[Tensor, "r c *d"] | Float[Tensor, "b r c *d"]:
        """Convert self to a sparse COO tensor."""
        return torch.sparse_coo_tensor(
            self.pattern.idx_coo,
            self.values,
            self.shape,
            dtype=self.dtype,
            device=self.device,
        ).coalesce()

    def to_sparse_csr(self) -> Float[Tensor, "r c *d"] | Float[Tensor, "b r c *d"]:
        """
        Convert self to a sparse CSR tensor.

        This function will use int32 dtype for the crow and col indices, whenever
        it is safe to do so.
        """
        idx_crow = self.pattern.idx_crow
        idx_col = self.pattern.idx_col

        if self.n_batch_dim == 0:
            val = self.values
        else:
            val = self.values.view(self.size(0), -1).contiguous()

        return torch.sparse_csr_tensor(
            idx_crow,
            idx_col,
            val,
            self.shape,
            dtype=self.dtype,
            device=self.device,
        )

    def _prepare_sparse_csc_components(self):
        idx_ccol = self.pattern.idx_ccol
        idx_row_csc = self.pattern.idx_row_csc

        if self.n_batch_dim == 0:
            val = self.values[self.pattern.csc_to_coo_map].contiguous()
        else:
            val = (
                self.values[self.pattern.csc_to_coo_map]
                .view(self.size(0), -1)
                .contiguous()
            )

        return idx_ccol, idx_row_csc, val

    # TODO: add other direct transpose conversion option.
    def to_sparse_csr_transposed(
        self,
    ) -> Float[Tensor, "c r *d"] | Float[Tensor, "b c r *d"]:
        """
        Convert self to a sparse CSR tensor with the sparse dimensions transposed.

        This function will use int32 dtype for the crow and col indices, whenever
        it is safe to do so.
        """
        idx_ccol, idx_row_csc, val = self._prepare_sparse_csc_components()

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

    def to_sparse_csc(self) -> Float[Tensor, "r c *d"] | Float[Tensor, "b r c *d"]:
        """
        Convert self to a sparse CSC tensor.

        This function will use int32 dtype for the ccol and row indices, whenever
        it is safe to do so.
        """
        idx_ccol, idx_row_csc, val = self._prepare_sparse_csc_components()

        return torch.sparse_csc_tensor(
            idx_ccol,
            idx_row_csc,
            val,
            self.shape,
            dtype=self.dtype,
            device=self.device,
        )

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
from einops import repeat
from jaxtyping import Bool, Float, Integer
from torch import Tensor

from ._matmul import (
    dense_diag_mm,
    diag_dense_mm,
    diag_sp_mm,
    sp_diag_mm,
)
from .base_decoupled_tensor import (
    BaseDecoupledTensor,
    is_scalar_like,
    validate_matmul_args,
)
from .pattern import SparsityPattern
from .sparse_decoupled_tensor import SparseDecoupledTensor


@dataclass
class DiagDecoupledTensor(BaseDecoupledTensor):
    """
    A sparse representation of diagonal matrices.

    Batching is supported, but only up to one leading batch dimension. Unlike
    standard PyTorch sparse tensors, no trailing dense dimensions are allowed.

    Parameters
    ----------
    val : [*b, diag]
        A dense tensor representing the diagonal elements of the matrix. If the
        input tensor is not contiguous, this class will create and store a
        contiguous copy of the input.

    Attributes
    ----------
    inv
        The inverse of the diagonal matrix.
    tr : [*b,]
        The trace of the diagonal matrix.
    shape
        The shape of the diagonal matrix.
    n_batch_dim
        The number of batch dimensions (either one or zero).
    n_sp_dim
        The number of sparse dimensions (which is always two).
    n_dense_dim
        The number of dense dimensions (which is always zero).
    T
        The matrix transpose along the two sparse dimensions.

    Notes
    -----
    Let `a` be a scalar-like object (float, int, or a 1D/0D tensor with one element),
    and let `D1 and `D2` be two `DiagDecoupledTensor`s of the same shape. This class
    supports the following tensor arithmetic operations:

    * Unary: negation (`-D1`), power (`D1**a`)
    * Binary: addition (`D1 + D2`), subtraction (`D1 - D2`), sclar multiplication
      (`a*D1`), scalar division (`D1/a`), elementwise multiplication (`D1*D2`),
      and elementwise division (`D1/D2`).

    In addition, this class supports the following types of matrix multiplications:

    * Matmul between two `DiagDecoupledTensor`s; batch dimensions are allowed,
    * Matmul between a `DiagDecoupledTensor` and a `SparseDecoupledTensor`; batch
      dimensions are not allowed on either operands,
    * Matmul between a `DiagDecoupledTensor` and a dense 2D tensor; batch dimensions
      are not allowed on either operands,
    * Matrix-vector multiplication with a dense 1D tensor; batch dimension on
      the `DiagDecoupledTensor` is allowed.
    """

    val: Float[Tensor, "*b diag"]

    def __post_init__(self):
        if self.val.layout != torch.strided:
            raise TypeError(
                "'val' must be a dense tensor of shape (diag,) or (b, diag)."
            )

        if self.val.ndim < 1 or self.val.ndim > 2:
            raise ValueError("'val' must be either of shape (diag,) or (b, diag).")

        if not torch.isfinite(self.val).all():
            raise ValueError("DiagDecoupledTensor values contain NaN or Inf.")

        # Enforce contiguous memory layout.
        self.val = self.val.contiguous()

    @classmethod
    def from_tensor(cls, tensor: Float[Tensor, "*b diag"]) -> DiagDecoupledTensor:
        """
        Instantiate a `DiagDecoupledTensor` from its diagonals.

        Parameters
        ----------
        tensor : [*b, diag]
            A dense tensor representing the diagonal elements of the matrix.

        Returns
        -------
        [diag, diag]
            A `DiagDecoupledTensor` object.
        """
        return cls(tensor)

    def submatrix(
        self,
        row_mask: Bool[Tensor, " diag"],
        col_mask: Bool[Tensor, " diag"] | None = None,
    ) -> Float[BaseDecoupledTensor, "*b sub_r sub_c"]:
        """
        Extract a submatrix using row and col masks.

        Parameters
        ----------
        row_mask : [diag,]
            A boolean mask marking the rows to keep in the submatrix.
        col_mask : [diag,]
            A boolean mask marking the cols to keep in the submatrix. If `None`,
            then assumed to be identical to the `row_mask`.

        Returns
        -------
        [*b, sub_r, sub_c]
            The extracted submatrix. If the `row_mask` and `col_mask` are identical,
            then the submatrix is still a `DiagDecoupledTensor`; otherwise the
            submatrix is represented as a `SparseDecoupledTensor`.
        """
        if col_mask is None:
            return DiagDecoupledTensor(self.val[..., row_mask])

        else:
            if (row_mask == col_mask).all():
                return DiagDecoupledTensor(self.val[..., row_mask])

            else:
                # Find the mask for the subsetted nonzero elements.
                submat_mask = row_mask & col_mask

                # Find the subsetted diagonal values.
                submat_val = self.val[..., submat_mask].flatten()

                # Determine the subsetted and renumbered coo index, using the
                # same cumsum() method as in SparsityPattern.submatrix().
                r_idx_map = torch.cumsum(row_mask, dim=0) - 1
                c_idx_map = torch.cumsum(col_mask, dim=0) - 1

                diag_idx_subset = torch.arange(self.size(-1), device=self.device)[
                    submat_mask
                ]

                idx_coo_row_submat = r_idx_map[diag_idx_subset]
                idx_coo_col_submat = c_idx_map[diag_idx_subset]

                # Tile the coo index if batched.
                if self.n_batch_dim > 0:
                    batch_size = self.size(0)
                    nnz_per_batch = submat_mask.sum().item()

                    b_idx = repeat(
                        torch.arange(batch_size, device=self.device),
                        "b -> (b nz)",
                        nz=nnz_per_batch,
                    )
                    r_idx = repeat(idx_coo_row_submat, "nz -> (b nz)", b=batch_size)
                    c_idx = repeat(idx_coo_col_submat, "nz -> (b nz)", b=batch_size)

                    submat_idx_coo = torch.vstack((b_idx, r_idx, c_idx))

                else:
                    submat_idx_coo = torch.vstack(
                        (idx_coo_row_submat, idx_coo_col_submat)
                    )

                # Determine the size of the submatrix.
                r_submat_size = row_mask.sum().item()
                c_submat_size = col_mask.sum().item()

                if self.n_batch_dim > 0:
                    submat_shape = torch.Size(
                        [self.size(0), r_submat_size, c_submat_size]
                    )
                else:
                    submat_shape = torch.Size([r_submat_size, c_submat_size])

                # Generate the new SparseDecoupledTensor
                pattern = SparsityPattern(submat_idx_coo, shape=submat_shape)
                sdt = SparseDecoupledTensor(pattern, submat_val)

                return sdt

    @classmethod
    def eye(
        cls, n: int, dtype: torch.dtype, device: torch.device
    ) -> Float[DiagDecoupledTensor, "n n"]:
        """Construct an identity matrix with no batch dimensions."""
        val = torch.ones(n, dtype=dtype, device=device)
        return DiagDecoupledTensor(val)

    def apply(self, fn: Callable, **kwargs) -> DiagDecoupledTensor:
        """
        Apply a callable to the `val` tensor.

        This function returns a new `DiagDecoupledTensor` using the transformed
        `val` tensor as the new diagonal.
        """
        new_val = fn(self.val, **kwargs)
        return DiagDecoupledTensor(new_val)

    def __neg__(self) -> DiagDecoupledTensor:
        return DiagDecoupledTensor(-self.val)

    def __pow__(self, exp: float | int) -> DiagDecoupledTensor:
        return DiagDecoupledTensor(self.val**exp)

    def pow(self, exp: float | int) -> DiagDecoupledTensor:
        """Compute the (elementwise) matrix power."""
        return self.__pow__(exp)

    def abs(self) -> DiagDecoupledTensor:
        """Take the absolute value of the diagonal matrix."""
        return DiagDecoupledTensor(self.val.abs())

    def diagonal(self) -> Float[Tensor, "*b diag"]:
        """
        Return the diagonal values.

        This function is equivalent to accessing the self `val` tensor.
        """
        return self.val

    @property
    def inv(self) -> DiagDecoupledTensor:
        """The inverse of the diagonal matrix."""
        return DiagDecoupledTensor(1.0 / self.val)

    @property
    def tr(self) -> Float[Tensor, "*b"]:
        """The trace of the diagonal matrix."""
        if self.n_batch_dim == 0:
            return self.val.sum()
        else:
            return self.val.sum(dim=-1)

    def __add__(self, other) -> DiagDecoupledTensor:
        match other:
            case DiagDecoupledTensor():
                return DiagDecoupledTensor(self.val + other.val)
            case _:
                return NotImplemented

    def __sub__(self, other) -> DiagDecoupledTensor:
        match other:
            case DiagDecoupledTensor():
                return DiagDecoupledTensor(self.val - other.val)
            case _:
                return NotImplemented

    def __mul__(self, other) -> DiagDecoupledTensor:
        if isinstance(other, DiagDecoupledTensor):
            return self.__matmul(other)
        elif is_scalar_like(other):
            return DiagDecoupledTensor(self.val * other)
        else:
            return NotImplemented

    def __truediv__(self, other) -> DiagDecoupledTensor:
        if isinstance(other, DiagDecoupledTensor):
            return DiagDecoupledTensor(self.val / other.val)
        elif is_scalar_like(other):
            return DiagDecoupledTensor(self.val / other)
        else:
            return NotImplemented

    def __matmul__(self, other):
        """Implement self @ other."""
        validate_matmul_args(self, other)

        match other:
            case DiagDecoupledTensor():
                return DiagDecoupledTensor(self.val * other.val)

            case SparseDecoupledTensor():
                val, pattern = diag_sp_mm(self.val, other.val, other.pattern)
                diag_sp = SparseDecoupledTensor(pattern, val)
                return diag_sp

            case Tensor():
                match other.ndim:
                    case 1:
                        return self.val * other
                    case 2:
                        return diag_dense_mm(self.val, other)

            case _:
                return NotImplemented

    def __rmatmul__(self, other):
        """Implement other @ self."""
        validate_matmul_args(self, other)

        match other:
            # Do not check for case DiagDecoupledTensor(), which is handled by __matmul__
            case SparseDecoupledTensor():
                val, pattern = sp_diag_mm(other.val, other.pattern, self.val)
                sp_diag = SparseDecoupledTensor(pattern, val)
                return sp_diag

            case Tensor():
                match other.ndim:
                    case 1:
                        return other * self.val
                    case 2:
                        return dense_diag_mm(other, self.val)

            case _:
                return NotImplemented

    @property
    def shape(self) -> torch.Size:
        """The shape of the diagonal matrix."""
        return torch.Size(self.val.shape + (self.val.shape[-1],))

    def _nnz(self) -> int:
        """
        Return the total number of nonzero/specified elements in the diagonal matrix.

        In PyTorch, for batched sparse CSC/CSR tensors, the _nnz() method returns
        the number of nonzero elements per batch item; for sparse COO tensors,
        the _nnz() method returns the total number of nonzero elements, regardless
        of batch dimensions. Here we follow the sparse COO convention.
        """
        return self.val.numel()

    @property
    def n_batch_dim(self) -> int:
        """The number of batch dimensions (either one or zero)."""
        return self.val.ndim - 1

    @property
    def n_sp_dim(self) -> int:
        """
        The number of sparse dimensions (which is always two).

        In PyTorch, for sparse CSC/CSR tensors, the leading batch dimension(s) do
        not count towards `sparse_dim()`; for sparse COO tensors, no such distinction
        is made. Here we follow the sparse CSC/CSR convention.
        """
        return 2

    @property
    def n_dense_dim(self) -> int:
        """
        The number of dense dimensions (which is always zero).

        In PyTorch, sparse tensors are allowed to have trailing dense dimensions.
        For a diagonal matrix, this can be achieved using a leading batch dimension;
        therefore, trailing dense dimensions are not allowed.
        """
        return 0

    @property
    def T(self) -> DiagDecoupledTensor:
        """
        The matrix transpose along the two sparse dimensions.

        Note that the transpose preserves the batch dimension.
        """
        return self

    def clone(
        self, memory_format: torch.memory_format = torch.contiguous_format
    ) -> DiagDecoupledTensor:
        """Return a copy of self."""
        return DiagDecoupledTensor(self.val.clone(memory_format=memory_format))

    def detach(self) -> DiagDecoupledTensor:
        """Return a new `DiagDecoupledTensor` with the `val` tensor detached."""
        return DiagDecoupledTensor(self.val.detach())

    def to(self, *args, **kwargs) -> DiagDecoupledTensor:
        """
        Perform dtype and/or device conversion.

        This function calls `.to()` on the self `val` tensor and returns a
        new `DiagDecoupledTensor` using the converted `val` tensor.
        """
        return DiagDecoupledTensor(self.val.to(*args, **kwargs))

    def to_dense(self) -> Float[Tensor, "*b diag diag"]:
        """Create a dense/strided copy of self."""
        if self.n_batch_dim == 0:
            return torch.diagflat(self.val)
        else:
            return torch.diag_embed(self.val)

    # TODO: check int64 dtype.
    def _get_idx_coo(self) -> Integer[Tensor, " sp diag"]:
        """
        Generate the coalesced COO index tensor for the diagonal matrix.

        The COO index tensor will be of int64 dtype and on the same device as self.
        """
        if self.n_batch_dim == 0:
            idx_coo = repeat(
                torch.arange(self._nnz(), device=self.device), "diag -> sp diag", sp=2
            )

        else:
            b = self.size(0)
            diag = self.size(-1)

            b_arange = torch.arange(b, device=self.device)
            diag_arange = torch.arange(diag, device=self.device)

            idx_coo = torch.vstack(
                (
                    repeat(b_arange, "b -> (b diag)", diag=diag),
                    repeat(diag_arange, "diag -> sp (b diag)", sp=self.n_sp_dim, b=b),
                )
            )

        return idx_coo

    def to_sdt(self) -> Float[SparseDecoupledTensor, "*b diag diag"]:
        """Convert self to a `SparseDecoupledTensor`."""
        idx_coo = self._get_idx_coo()
        pattern = SparsityPattern(idx_coo, self.shape)

        val = self.val.flatten()

        return SparseDecoupledTensor(pattern, val)

    def to_sparse_coo(self) -> Float[Tensor, "*b diag diag"]:
        """Convert self to a sparse COO tensor."""
        idx_coo = self._get_idx_coo()

        return torch.sparse_coo_tensor(
            idx_coo,
            self.val.flatten(),
            self.shape,
            dtype=self.dtype,
            device=self.device,
        ).coalesce()

    def _to_compressed_sparse_tensor(
        self, constructor: Callable, idx_dtype: torch.dtype
    ) -> Float[Tensor, "*b diag diag"]:
        """
        Convert self to a sparse CSC/CSR tensor.

        The following code is written with variable names assuming the CSR format,
        but for a diagonal matrix, the two formats are indexed identically.
        """
        if self.n_batch_dim == 0:
            idx_crow = torch.arange(
                self._nnz() + 1, dtype=idx_dtype, device=self.device
            )
            idx_col = torch.arange(self._nnz(), dtype=idx_dtype, device=self.device)

        else:
            b = self.size(0)
            diag = self.size(-1)

            idx_crow = repeat(
                torch.arange(diag + 1, dtype=idx_dtype, device=self.device),
                "crow -> b crow",
                b=b,
            )
            idx_col = repeat(
                torch.arange(diag, dtype=idx_dtype, device=self.device),
                "col -> b col",
                b=b,
            )

        return constructor(
            idx_crow,
            idx_col,
            self.val,
            self.shape,
            dtype=self.dtype,
            device=self.device,
        )

    def to_sparse_csr(self, int32: bool = False) -> Float[Tensor, "*b diag diag"]:
        """
        Convert self to a sparse CSR tensor.

        Parameters
        ----------
        int32
            Whether to use int32 dtype for the crow and col indices.

        Returns
        -------
        [*b, diag, diag]
            A sparse CSR tensor.
        """
        return self._to_compressed_sparse_tensor(
            torch.sparse_csr_tensor, torch.int32 if int32 else torch.int64
        )

    def to_sparse_csc(self, int32: bool = False) -> Float[Tensor, "*b diag diag"]:
        """
        Convert self to a sparse CSC tensor.

        Parameters
        ----------
        int32
            Whether to use int32 dtype for the ccol and row indices.

        Returns
        -------
        [*b, diag, diag]
            A sparse CSC tensor.
        """
        return self._to_compressed_sparse_tensor(
            torch.sparse_csc_tensor, torch.int32 if int32 else torch.int64
        )

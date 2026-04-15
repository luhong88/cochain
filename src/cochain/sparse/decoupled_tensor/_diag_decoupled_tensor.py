from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
from jaxtyping import Bool, Float, Integer
from torch import Tensor

from ._base_decoupled_tensor import BaseDecoupledTensor, is_scalar, validate_matmul_args
from ._matmul import (
    dense_diag_mm,
    diag_dense_mm,
    diag_sp_mm,
    sp_diag_mm,
)
from ._pattern import SparsityPattern
from ._sparse_decoupled_tensor import SparseDecoupledTensor


@dataclass
class DiagDecoupledTensor(BaseDecoupledTensor):
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
    def from_tensor(cls, tensor: Tensor) -> DiagDecoupledTensor:
        return cls(tensor)

    def submatrix(
        self, row_mask: Bool[Tensor, " r"], col_mask: Bool[Tensor, " c"] | None = None
    ) -> BaseDecoupledTensor:
        if col_mask is None:
            return DiagDecoupledTensor(self.val[..., row_mask])

        else:
            if (row_mask == col_mask).all():
                return DiagDecoupledTensor(self.val[..., row_mask])

            else:
                # If both the row_mask and col_mask are provided and they are
                # not identical, then the resulting submatrix is no longer diagonal
                # and a SparseDecoupledTensor needs to be constructed.

                # Find the mask for the subsetted nonzero elements.
                submat_mask = row_mask & col_mask

                # Find the subsetted diagonal values.
                submat_val = self.val[..., submat_mask].flatten()

                # Determine the subsetted and renumbered coo index, using the
                # same cumsum() method as in SparsityPattern.submatrix().
                r_idx_map = torch.cumsum(row_mask, dim=0) - 1
                c_idx_map = torch.cumsum(col_mask, dim=0) - 1

                idx_coo = self._get_idx_coo()
                idx_coo_row_submat = r_idx_map[idx_coo[-2, submat_mask]]
                idx_coo_col_submat = c_idx_map[idx_coo[-1, submat_mask]]

                if self.n_batch_dim > 0:
                    submat_idx_coo = torch.stack(
                        (idx_coo_row_submat, idx_coo_col_submat)
                    )
                else:
                    submat_idx_coo = torch.stack(
                        (
                            idx_coo[0, submat_mask],
                            idx_coo_row_submat,
                            idx_coo_col_submat,
                        )
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
    ) -> DiagDecoupledTensor:
        val = torch.ones(n, dtype=dtype, device=device)
        return DiagDecoupledTensor(val)

    def apply(self, fn: Callable, **kwargs) -> DiagDecoupledTensor:
        new_val = fn(self.val, **kwargs)
        return DiagDecoupledTensor(new_val)

    def __neg__(self) -> DiagDecoupledTensor:
        return DiagDecoupledTensor(-self.val)

    def __pow__(self, exp: float | int) -> DiagDecoupledTensor:
        return DiagDecoupledTensor(self.val**exp)

    def pow(self, exp: float | int) -> DiagDecoupledTensor:
        return self.__pow__(exp)

    def abs(self) -> DiagDecoupledTensor:
        return DiagDecoupledTensor(self.val.abs())

    def diagonal(self) -> Float[Tensor, "*b diag"]:
        return self.val

    @property
    def inv(self) -> DiagDecoupledTensor:
        return DiagDecoupledTensor(1.0 / self.val)

    @property
    def tr(self) -> Float[Tensor, "*b"]:
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
        elif is_scalar(other):
            return DiagDecoupledTensor(self.val * other)
        else:
            return NotImplemented

    def __truediv__(self, other) -> DiagDecoupledTensor:
        if isinstance(other, DiagDecoupledTensor):
            return DiagDecoupledTensor(self.val / other.val)
        elif is_scalar(other):
            return DiagDecoupledTensor(self.val / other)
        else:
            return NotImplemented

    def __matmul__(self, other):
        """
        Implement self @ other
        """
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
        """
        Implement other @ self
        """
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
        return torch.Size(self.val.shape + (self.val.shape[-1],))

    def _nnz(self) -> int:
        """
        For batched sparse csr/csc tensors, the _nnz() method returns the number
        of nonzero elements per batch item; for sparse coo tensors, the _nnz()
        method returns the total number of nonzero elements, regardless of batch
        dimensions. Here we follow the sparse coo convention.
        """
        return self.val.numel()

    @property
    def n_batch_dim(self) -> int:
        return self.val.ndim - 1

    @property
    def n_sp_dim(self) -> int:
        return 2

    @property
    def n_dense_dim(self) -> int:
        return 0

    @property
    def T(self) -> DiagDecoupledTensor:
        """
        Note that the transpose preserves the batch dimensions.
        """
        return self

    def clone(
        self, memory_format: torch.memory_format = torch.contiguous_format
    ) -> DiagDecoupledTensor:
        return DiagDecoupledTensor(self.val.clone(memory_format=memory_format))

    def detach(self) -> DiagDecoupledTensor:
        return DiagDecoupledTensor(self.val.detach())

    def to(self, *args, **kwargs) -> DiagDecoupledTensor:
        return DiagDecoupledTensor(self.val.to(*args, **kwargs))

    def to_dense(self) -> Float[Tensor, "*b d d"]:
        if self.n_batch_dim == 0:
            return torch.diagflat(self.val)
        else:
            return torch.diag_embed(self.val)

    def _get_idx_coo(self) -> Integer[Tensor, " sp nnz"]:
        if self.n_batch_dim == 0:
            idx_coo = torch.tile(torch.arange(self._nnz(), device=self.device), (2, 1))

        else:
            b = self.size(0)
            d = self.size(-1)

            idx_coo = torch.vstack(
                (
                    torch.repeat_interleave(torch.arange(b, device=self.device), d),
                    torch.tile(torch.arange(d, device=self.device), (2, b)),
                )
            )

        return idx_coo

    def to_sparse_operator(self) -> SparseDecoupledTensor:
        idx_coo = self._get_idx_coo()
        pattern = SparsityPattern(idx_coo, self.shape)

        val = self.val.flatten()

        return SparseDecoupledTensor(pattern, val)

    def to_sparse_coo(self) -> Float[Tensor, "*b d d"]:
        idx_coo = self._get_idx_coo()

        return torch.sparse_coo_tensor(
            idx_coo,
            self.val.flatten(),
            self.shape,
            dtype=self.dtype,
            device=self.device,
        ).coalesce()

    def _to_compressed_sparse_tensor(
        self, constructor: Callable, idx_dtype: torch.dtype = torch.int64
    ) -> Float[Tensor, "*b d d"]:
        if self.n_batch_dim == 0:
            idx_crow = torch.arange(
                self._nnz() + 1, dtype=idx_dtype, device=self.device
            )
            idx_col = torch.arange(self._nnz(), dtype=idx_dtype, device=self.device)

        else:
            b = self.size(0)
            d = self.size(-1)

            idx_crow = torch.tile(
                torch.arange(d + 1, dtype=idx_dtype, device=self.device), (b, 1)
            )
            idx_col = torch.tile(
                torch.arange(d, dtype=idx_dtype, device=self.device), (b, 1)
            )

        return constructor(
            idx_crow,
            idx_col,
            self.val,
            self.shape,
            dtype=self.dtype,
            device=self.device,
        )

    def to_sparse_csr(self, int32: bool = False) -> Float[Tensor, "*b d d"]:
        return self._to_compressed_sparse_tensor(
            torch.sparse_csr_tensor, torch.int32 if int32 else torch.int64
        )

    def to_sparse_csc(self, int32: bool = False) -> Float[Tensor, "*b d d"]:
        return self._to_compressed_sparse_tensor(
            torch.sparse_csc_tensor, torch.int32 if int32 else torch.int64
        )

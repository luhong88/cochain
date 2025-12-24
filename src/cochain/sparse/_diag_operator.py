from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch as t
from jaxtyping import Float

from ._base_operator import BaseOperator, is_scalar, validate_matmul_args
from ._matmul import (
    dense_diag_mm,
    diag_dense_mm,
    diag_sp_mm,
    sp_diag_mm,
)
from ._sp_operator import SparseOperator


@dataclass
class DiagOperator(BaseOperator):
    val: Float[t.Tensor, "*b diag"]

    def __post_init__(self):
        if self.val.layout != t.strided:
            raise TypeError(
                "'val' must be a dense tensor of shape (diag,) or (b, diag)."
            )

        if self.val.ndim < 1 or self.val.ndim > 2:
            raise ValueError("'val' must be either of shape (diag,) or (b, diag).")

        if not t.isfinite(self.val).all():
            raise ValueError("DiagOperator values contain NaN or Inf.")

        # Enforce contiguous memory layout.
        self.val = self.val.contiguous()

    @classmethod
    def from_tensor(cls, tensor: t.Tensor) -> DiagOperator:
        return cls(tensor)

    @property
    def shape(self) -> t.Size:
        return t.Size(self.val.shape + (self.val.shape[-1],))

    @property
    def n_dense_dim(self) -> int:
        return 0

    @property
    def n_sp_dim(self) -> int:
        return 2

    @property
    def n_batch_dim(self) -> int:
        return self.val.ndim - 1

    @property
    def T(self) -> DiagOperator:
        """
        Note that the transpose preserves the batch dimensions.
        """
        return self

    def apply(self, fn: Callable, **kwargs) -> SparseOperator:
        new_val = fn(self.val, **kwargs)
        return SparseOperator(self.sp_topo, new_val)

    @property
    def inv(self) -> DiagOperator:
        return DiagOperator(1.0 / self.val)

    def detach(self) -> DiagOperator:
        return DiagOperator(self.val.detach())

    def clone(
        self, memory_format: t.memory_format = t.contiguous_format
    ) -> DiagOperator:
        return DiagOperator(self.val.clone(memory_format=memory_format))

    def _nnz(self) -> int:
        """
        For batched sparse csr/csc tensors, the _nnz() method returns the number
        of nonzero elements per batch item; for sparse coo tensors, the _nnz()
        method returns the total number of nonzero elements, regardless of batch
        dimensions. Here we follow the sparse coo convention.
        """
        return self.val.numel()

    def __matmul__(self, other):
        """
        Implement self @ other
        """
        validate_matmul_args(self, other)

        match other:
            case DiagOperator():
                return DiagOperator(self.val * other.val)

            case SparseOperator():
                val, sp_topo = diag_sp_mm(self.val, other.val, other.sp_topo)
                diag_sp = SparseOperator(sp_topo, val)
                return diag_sp

            case t.Tensor():
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
            # Do not check for case DiagOperator(), which is handled by __matmul__
            case SparseOperator():
                val, sp_topo = sp_diag_mm(other.val, other.sp_topo, self.val)
                sp_diag = SparseOperator(sp_topo, val)
                return sp_diag

            case t.Tensor():
                match other.ndim:
                    case 1:
                        return other * self.val
                    case 2:
                        return dense_diag_mm(other, self.val)

            case _:
                return NotImplemented

    def __neg__(self) -> DiagOperator:
        return DiagOperator(-self.val)

    def __pow__(self, exp: float | int) -> DiagOperator:
        return DiagOperator(self.val**exp)

    def pow(self, exp: float | int) -> DiagOperator:
        return self.__pow__(exp)

    def __add__(self, other) -> DiagOperator:
        match other:
            case DiagOperator():
                return DiagOperator(self.val + other.val)
            case _:
                return NotImplemented

    def __sub__(self, other) -> DiagOperator:
        match other:
            case DiagOperator():
                return DiagOperator(self.val - other.val)
            case _:
                return NotImplemented

    def __mul__(self, other) -> DiagOperator:
        if isinstance(other, DiagOperator):
            return self.__matmul(other)
        elif is_scalar(other):
            return DiagOperator(self.val * other)
        else:
            return NotImplemented

    def __truediv__(self, other) -> DiagOperator:
        if isinstance(other, DiagOperator):
            return DiagOperator(self.val / other.val)
        elif is_scalar(other):
            return DiagOperator(self.val / other)
        else:
            return NotImplemented

    @property
    def tr(self) -> t.Tensor:
        if self.n_batch_dim == 0:
            return self.val.sum()
        else:
            return self.val.sum(dim=-1)

    def to_sparse_coo(self) -> Float[t.Tensor, "*b d d"]:
        if self.n_batch_dim == 0:
            idx_coo = t.tile(t.arange(self._nnz(), device=self.device), (2, 1))

        else:
            b = self.size(0)
            d = self.size(-1)

            idx_coo = t.vstack(
                (
                    t.repeat_interleave(t.arange(b, device=self.device), d),
                    t.tile(t.arange(d, device=self.device), (2, b)),
                )
            )

        return t.sparse_coo_tensor(
            idx_coo,
            self.val.flatten(),
            self.shape,
            dtype=self.dtype,
            device=self.device,
        ).coalesce()

    def _to_compressed_sparse_tensor(
        self, constructor: Callable, idx_dtype: t.dtype = t.int64
    ) -> Float[t.Tensor, "*b d d"]:
        if self.n_batch_dim == 0:
            idx_crow = t.arange(self._nnz() + 1, dtype=idx_dtype, device=self.device)
            idx_col = t.arange(self._nnz(), dtype=idx_dtype, device=self.device)

        else:
            b = self.size(0)
            d = self.size(-1)

            idx_crow = t.tile(
                t.arange(d + 1, dtype=idx_dtype, device=self.device), (b, 1)
            )
            idx_col = t.tile(t.arange(d, dtype=idx_dtype, device=self.device), (b, 1))

        return constructor(
            idx_crow,
            idx_col,
            self.val,
            self.shape,
            dtype=self.dtype,
            device=self.device,
        )

    def to_sparse_csr(self, int32: bool = False) -> Float[t.Tensor, "*b d d"]:
        return self._to_compressed_sparse_tensor(
            t.sparse_csr_tensor, t.int32 if int32 else t.int64
        )

    def to_sparse_csc(self, int32: bool = False) -> Float[t.Tensor, "*b d d"]:
        return self._to_compressed_sparse_tensor(
            t.sparse_csc_tensor, t.int32 if int32 else t.int64
        )

    def to_dense(self) -> Float[t.Tensor, "*b d d"]:
        if self.n_batch_dim == 0:
            return t.diagflat(self.val)
        else:
            return t.diag_embed(self.val)

    def to(self, *args, **kwargs) -> DiagOperator:
        return DiagOperator(self.val.to(*args, **kwargs))

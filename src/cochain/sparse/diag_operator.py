from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch as t
from jaxtyping import Float

from ._matmul import (
    dense_diag_mm,
    diag_dense_mm,
    diag_sp_mm,
    sp_diag_mm,
)
from .sp_operator import SparseOperator


def _validate_diag_matmul_args(
    self: DiagOperator, other: DiagOperator | SparseOperator | t.Tensor
):
    if self.n_batch_dim > 0:
        raise NotImplementedError(
            "__matmul__ with batched DiagOperator is not supported."
        )

    match other:
        case DiagOperator():
            if other.n_batch_dim > 0:
                raise NotImplementedError(
                    "__matmul__ with batched DiagOperator is not supported."
                )

        case SparseOperator():
            if other.n_batch_dim > 0:
                raise NotImplementedError(
                    "__matmul__ with batched SparseOperator is not supported."
                )

            if other.n_dense_dim > 0:
                raise NotImplementedError(
                    "__matmul__ with sparse hybrid SparseOperator is not supported."
                )

        case t.Tensor():
            if (other.ndim < 1) or (other.ndim > 2):
                raise NotImplementedError(
                    f"__matmul__ with tensor of shape {other.shape} is not supported."
                )

        case _:
            raise TypeError(
                f"__matmul__ between DiagOperator and {type(other)} is not supported."
            )


@dataclass
class DiagOperator:
    val: Float[t.Tensor, "*b diag"]

    def __post_init__(self):
        if not t.isfinite(self.val).all():
            raise ValueError("DiagOperator values contain NaN or Inf.")

        # Enforce contiguous memory layout.
        self.val = self.val.contiguous()

    @property
    def shape(self) -> t.Size:
        return t.Size(self.val.shape + (self.val.shape[-1],))

    @property
    def device(self) -> t.device:
        return self.val.device

    @property
    def dtype(self) -> t.dtype:
        return self.val.dtype

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
    def n_dim(self) -> int:
        return self.n_batch_dim + self.n_sp_dim + self.n_dense_dim

    @property
    def T(self) -> DiagOperator:
        """
        Note that the transpose preserves the batch dimensions.
        """
        return self

    @property
    def requires_grad(self) -> bool:
        return self.val.requires_grad

    def requires_grad_(self, requires_grad: bool = True) -> DiagOperator:
        self.val.requires_grad_(requires_grad)
        return self

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
        _validate_diag_matmul_args(self, other)

        match other:
            case DiagOperator():
                return DiagOperator(self.val * other.val)

            case SparseOperator():
                val, sp_topo = diag_sp_mm(self.val, other.val, other.sp_topo)
                diag_sp = SparseOperator(val, sp_topo)
                return diag_sp

            case t.Tensor():
                match other.ndim:
                    case 1:
                        return self.val * other
                    case 2:
                        return diag_dense_mm(self.val, other)

    def __rmatmul__(self, other):
        """
        Implement other @ self
        """
        _validate_diag_matmul_args(self, other)

        match other:
            case DiagOperator():
                return DiagOperator(self.val * other.val)

            case SparseOperator():
                val, sp_topo = sp_diag_mm(other.val, other.sp_topo, self.val)
                sp_diag = SparseOperator(val, sp_topo)
                return sp_diag

            case t.Tensor():
                match other.ndim:
                    case 1:
                        return other * self.val
                    case 2:
                        return dense_diag_mm(other, self.val)

    def to_sparse_coo(self) -> Float[t.Tensor, "*b d d"]:
        if self.n_batch_dim == 0:
            idx_coo = t.tile(t.arange(self._nnz(), device=self.device), (2, 1))

        else:
            b = t.size(0)
            d = t.size(-1)

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
        )

    def _to_compressed_sparse_tensor(
        self, constructor: Callable, idx_dtype: t.dtype = t.int64
    ) -> Float[t.Tensor, "*b d d"]:
        if self.n_batch_dim == 0:
            idx_crow = t.arange(self._nnz() + 1, dtype=idx_dtype, device=self.device)
            idx_col = t.arange(self._nnz(), dtype=idx_dtype, device=self.device)

        else:
            b = t.size(0)
            d = t.size(-1)

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
            t.sparse_csr_tensor, t.int32 if int32 else t.int64
        )

    def to_dense(self) -> Float[t.Tensor, "*b d d"]:
        if self.n_batch_dim == 0:
            return t.diagflat(self.val)
        else:
            return t.diag_embed(self.val)

    def to(self, *args, **kwargs) -> DiagOperator:
        return DiagOperator(self.val.to(*args, **kwargs))

    def size(self, dim: int | None = None) -> int | t.Size:
        if dim is None:
            return self.shape
        else:
            return self.shape[dim]

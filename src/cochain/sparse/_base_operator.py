from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable, Sequence

import torch as t


def is_scalar(other) -> bool:
    return isinstance(other, (float, int)) or (
        isinstance(other, t.Tensor) and other.numel() == 1
    )


def validate_matmul_args(self: BaseOperator, other: BaseOperator | t.Tensor):
    if self.n_batch_dim > 0:
        raise NotImplementedError(
            "__matmul__ with batched sparse BaseOperator is not supported."
        )

    if self.n_dense_dim > 0:
        raise NotImplementedError(
            "__matmul__ with sparse hybrid BaseOperator is not supported."
        )

    match other:
        case BaseOperator():
            if other.n_batch_dim > 0:
                raise NotImplementedError(
                    "__matmul__ with batched sparse BaseOperator is not supported."
                )

            if other.n_dense_dim > 0:
                raise NotImplementedError(
                    "__matmul__ with sparse hybrid BaseOperator is not supported."
                )

        case t.Tensor():
            if (other.ndim < 1) or (other.ndim > 2):
                raise NotImplementedError(
                    f"__matmul__ with tensor of shape {other.shape} is not supported."
                )

        case _:
            raise TypeError(
                f"__matmul__ between BaseOperator and {type(other)} is not supported."
            )


class BaseOperator(ABC):
    val: t.Tensor

    @classmethod
    @abstractmethod
    def from_tensor(cls, tensor: t.Tensor) -> BaseOperator: ...

    @classmethod
    @abstractmethod
    def to_block_diag(
        cls, blocks: Sequence[t.Tensor | BaseOperator]
    ) -> BaseOperator: ...

    @abstractmethod
    def apply(self, fn: Callable, **kwargs) -> BaseOperator: ...

    @abstractmethod
    def __neg__(self) -> BaseOperator: ...

    @abstractmethod
    def __add__(self, other) -> BaseOperator: ...

    @abstractmethod
    def __sub__(self, other) -> BaseOperator: ...

    @abstractmethod
    def __mul__(self, other) -> BaseOperator: ...

    def __rmul__(self, other) -> BaseOperator:
        return self.__mul__(other)

    @abstractmethod
    def __truediv__(self, other) -> BaseOperator: ...

    @abstractmethod
    def __matmul__(self, other): ...

    @abstractmethod
    def __rmatmul__(self, other): ...

    @property
    @abstractmethod
    def shape(self) -> t.Size: ...

    def size(self, dim: int | None = None) -> int | t.Size:
        if dim is None:
            return self.shape
        else:
            return self.shape[dim]

    @abstractmethod
    def _nnz(self) -> int: ...

    @property
    @abstractmethod
    def n_batch_dim(self) -> int: ...

    @property
    @abstractmethod
    def n_sp_dim(self) -> int: ...

    @property
    @abstractmethod
    def n_dense_dim(self) -> int: ...

    @property
    def n_dim(self) -> int:
        return self.n_batch_dim + self.n_sp_dim + self.n_dense_dim

    @property
    @abstractmethod
    def T(self) -> BaseOperator: ...

    @property
    def dtype(self) -> t.dtype:
        return self.val.dtype

    @property
    def device(self) -> t.device:
        return self.val.device

    @abstractmethod
    def clone(
        self, memory_format: t.memory_format = t.contiguous_format
    ) -> BaseOperator: ...

    @abstractmethod
    def detach(self) -> BaseOperator: ...

    @property
    def requires_grad(self) -> bool:
        return self.val.requires_grad

    def requires_grad_(self, requires_grad: bool = True) -> BaseOperator:
        self.val.requires_grad_(requires_grad)
        return self

    @abstractmethod
    def to(self, *args, **kwargs) -> BaseOperator: ...

    @abstractmethod
    def to_dense(self) -> t.Tensor: ...

    @abstractmethod
    def to_sparse_operator(self) -> BaseOperator: ...

    @abstractmethod
    def to_sparse_coo(self) -> t.Tensor: ...

    @abstractmethod
    def to_sparse_csr(self, int32: bool = False) -> t.Tensor: ...

    @abstractmethod
    def to_sparse_csc(self, int32: bool = False) -> t.Tensor: ...

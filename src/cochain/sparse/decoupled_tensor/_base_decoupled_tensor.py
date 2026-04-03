from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable

import torch
from torch import Tensor


def is_scalar(other) -> bool:
    return isinstance(other, (float, int)) or (
        isinstance(other, Tensor) and other.numel() == 1
    )


def validate_matmul_args(
    self: BaseDecoupledTensor, other: BaseDecoupledTensor | Tensor
):
    if self.n_batch_dim > 0:
        raise NotImplementedError(
            "__matmul__ with batched sparse BaseDecoupledTensor is not supported."
        )

    if self.n_dense_dim > 0:
        raise NotImplementedError(
            "__matmul__ with sparse hybrid BaseDecoupledTensor is not supported."
        )

    match other:
        case BaseDecoupledTensor():
            if other.n_batch_dim > 0:
                raise NotImplementedError(
                    "__matmul__ with batched sparse BaseDecoupledTensor is not supported."
                )

            if other.n_dense_dim > 0:
                raise NotImplementedError(
                    "__matmul__ with sparse hybrid BaseDecoupledTensor is not supported."
                )

        case Tensor():
            if (other.ndim < 1) or (other.ndim > 2):
                raise NotImplementedError(
                    f"__matmul__ with tensor of shape {other.shape} is not supported."
                )

        case _:
            raise TypeError(
                f"__matmul__ between BaseDecoupledTensor and {type(other)} is not supported."
            )


class BaseDecoupledTensor(ABC):
    val: Tensor

    @classmethod
    @abstractmethod
    def from_tensor(cls, tensor: Tensor) -> BaseDecoupledTensor: ...

    @abstractmethod
    def apply(self, fn: Callable, **kwargs) -> BaseDecoupledTensor: ...

    @abstractmethod
    def __neg__(self) -> BaseDecoupledTensor: ...

    @property
    @abstractmethod
    def tr(self): ...

    @abstractmethod
    def __add__(self, other) -> BaseDecoupledTensor: ...

    @abstractmethod
    def __sub__(self, other) -> BaseDecoupledTensor: ...

    @abstractmethod
    def abs(self) -> BaseDecoupledTensor: ...

    @abstractmethod
    def diagonal(self) -> Tensor: ...

    @abstractmethod
    def __mul__(self, other) -> BaseDecoupledTensor: ...

    def __rmul__(self, other) -> BaseDecoupledTensor:
        return self.__mul__(other)

    @abstractmethod
    def __truediv__(self, other) -> BaseDecoupledTensor: ...

    @abstractmethod
    def __matmul__(self, other): ...

    @abstractmethod
    def __rmatmul__(self, other): ...

    @property
    @abstractmethod
    def shape(self) -> torch.Size: ...

    def size(self, dim: int | None = None) -> int | torch.Size:
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
    def T(self) -> BaseDecoupledTensor: ...

    @property
    def dtype(self) -> torch.dtype:
        return self.val.dtype

    @property
    def device(self) -> torch.device:
        return self.val.device

    @abstractmethod
    def clone(
        self, memory_format: torch.memory_format = torch.contiguous_format
    ) -> BaseDecoupledTensor: ...

    @abstractmethod
    def detach(self) -> BaseDecoupledTensor: ...

    @property
    def requires_grad(self) -> bool:
        return self.val.requires_grad

    def requires_grad_(self, requires_grad: bool = True) -> BaseDecoupledTensor:
        self.val.requires_grad_(requires_grad)
        return self

    @abstractmethod
    def to(self, *args, **kwargs) -> BaseDecoupledTensor: ...

    @abstractmethod
    def to_dense(self) -> Tensor: ...

    @abstractmethod
    def to_sparse_operator(self) -> BaseDecoupledTensor: ...

    @abstractmethod
    def to_sparse_coo(self) -> Tensor: ...

    @abstractmethod
    def to_sparse_csr(self, int32: bool = False) -> Tensor: ...

    @abstractmethod
    def to_sparse_csc(self, int32: bool = False) -> Tensor: ...

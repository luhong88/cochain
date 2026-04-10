from abc import ABC, abstractmethod

import torch
from jaxtyping import Float
from torch import Tensor


class InvSparseOperator(ABC):
    """
    An ABC for "stateful" sparse linear solver classes.

    This class provides an abstraction to solving the sparse linear system A@x=b
    for x. The tensor A is assumed to have shape (r, c), and the tensor b is
    assumed to have shape (r, *ch), and the output x tensor is assumed to have
    shape (c, *ch); no explicit leading batch dimensions are allowed.
    """

    dtype: torch.dtype
    device: torch.device
    shape: torch.Size

    def size(self, dim: int | None = None) -> int | torch.Size:
        if dim is None:
            return self.shape
        else:
            return self.shape[dim]

    @abstractmethod
    def __del__(self): ...

    @abstractmethod
    def __call__(
        self, b: Float[Tensor, " r *ch"], *args, **kwargs
    ) -> Float[Tensor, " c *ch"]: ...

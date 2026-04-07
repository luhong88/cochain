import torch
from torch import Tensor

EPS = 1e-9


def get_eps(tensor_or_dtype: Tensor | torch.dtype) -> float:
    """Retrieve the machine epsilon for a given tensor or dtype."""
    dtype = (
        tensor_or_dtype.dtype
        if isinstance(tensor_or_dtype, torch.Tensor)
        else tensor_or_dtype
    )

    if not dtype.is_floating_point:
        raise ValueError(f"Epsilon is only defined for float dtypes, got {dtype}.")

    return torch.finfo(dtype).eps

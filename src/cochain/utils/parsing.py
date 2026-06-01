import numpy as np
import numpy.typing as npt
import torch
from torch import Tensor


def parse_to(*args, **kwargs):
    # Pop 'copy' out first, as _parse_to doesn't expect it
    copy_flag = kwargs.pop("copy", False)

    # Parse the other arguments using torch's internal module parser
    device, dtype, non_blocking, memory_format = torch._C._nn._parse_to(*args, **kwargs)

    return device, dtype, copy_flag, non_blocking, memory_format


def to_np(
    tensor: Tensor,
    *,
    dtype: np.dtype | None = None,
    contiguous: bool = False,
) -> npt.NDArray:
    match (dtype, contiguous):
        case (None, False):
            return tensor.detach().cpu().numpy()
        case (None, True):
            return tensor.detach().contiguous().cpu().numpy()
        case (dtype, False):
            return tensor.detach().cpu().numpy().astype(dtype)
        case (dtype, True):
            return tensor.detach().contiguous().cpu().numpy().astype(dtype)

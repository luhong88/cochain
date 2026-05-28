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


def to_np(tensor: Tensor, dtype: np.dtype | None = None) -> npt.NDArray:
    if dtype is None:
        return tensor.detach().cpu().numpy()
    else:
        return tensor.detach().cpu().numpy().astype(dtype)

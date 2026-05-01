import torch


def parse_to(*args, **kwargs):
    # Pop 'copy' out first, as _parse_to doesn't expect it
    copy_flag = kwargs.pop("copy", False)

    # Parse the other arguments using torch's internal module parser
    device, dtype, non_blocking, memory_format = torch._C._nn._parse_to(*args, **kwargs)

    return device, dtype, copy_flag, non_blocking, memory_format

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


def to_col_major(tensor: Tensor, *, batch_first: bool = False) -> Tensor:
    """
    Convert a PyTorch tensor to column-major layout.

    Parameters
    ----------
    tensor
        The input tensor; tensors with up to 3 dimensions are supported, and no
        assumptions are made about the memory layout of the input tensor.
    batch_first
        Whether to interpret the first dimension of the input tensor as a batch
        dimension. If the first dimension is batch, the input tensor will be
        transformed into a batch of column-major tensors (i.e., the first dimension
        will have the slowest/largest stride).

    Returns
    -------
    strided_tensor
        The input tensor transformed to a column-major memory layout.
    """
    if tensor.ndim < 2:
        # 1D tensors don't have col/row major, just ensure it's contiguous.
        return tensor.contiguous()

    if batch_first:
        match tensor.ndim:
            case 2:
                # Input tensor: (b, r)
                # Col-major stride: (r, 1)
                # Note that this is a batch of 1D vectors; identical to row-major stride.
                s0, s1 = tensor.shape
                expected_stride = (s1, 1)

                output_tensor = tensor.contiguous()

            case 3:
                # Input tensor: (b, r, c)
                # Col-major stride: (r*c, 1, r)
                s0, s1, s2 = tensor.shape
                expected_stride = (s1 * s2, 1, s1)

                if tensor.stride() == expected_stride:
                    output_tensor = tensor
                else:
                    output_tensor = (
                        tensor.contiguous()
                        .transpose(-1, -2)
                        .contiguous()
                        .transpose(-1, -2)
                    )

            case _:
                raise ValueError(
                    "Tensors with more than three dimensions are not supported."
                )

    else:
        match tensor.ndim:
            case 2:
                # Input tensor: (r, c)
                # Col-major stride: (1, r)
                s0, s1 = tensor.shape
                expected_stride = (1, s0)

                if tensor.stride() == expected_stride:
                    output_tensor = tensor
                else:
                    output_tensor = tensor.T.contiguous().T

            case 3:
                # Input tensor: (x, y, z)
                # Col-major stride: (1, x, x*y)
                s0, s1, s2 = tensor.shape
                expected_stride = (1, s0, s0 * s1)

                if tensor.stride() == expected_stride:
                    output_tensor = tensor
                else:
                    output_tensor = (
                        tensor.contiguous()
                        .transpose(0, -1)
                        .contiguous()
                        .transpose(0, -1)
                    )

            case _:
                raise ValueError(
                    "Tensors with more than three dimensions are not supported."
                )

    # Explicitly enforce the expected stride; necessary for trivial dimensions
    # to be annotated with the correct stride.
    strided_tensor = output_tensor.as_strided(tensor.shape, expected_stride)

    return strided_tensor

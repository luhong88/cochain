from __future__ import annotations

__all__ = ["SubmatPlan", "SubmatResult"]

from dataclasses import dataclass
from typing import NamedTuple

import torch
from jaxtyping import Bool
from torch import Tensor

from ...utils.parsing import parse_to
from .base_decoupled_tensor import BaseDecoupledTensor
from .pattern import SparsityPattern


@dataclass
class SubmatPlan:
    full_pattern: SparsityPattern | None
    submat_pattern: SparsityPattern | None
    submat_mask: Bool[Tensor, " nz"]

    def __post_init__(self):
        # Check for device consistency.
        ref_device = self.submat_mask.device
        full_device = getattr(self.full_pattern, "device", ref_device)
        submat_device = getattr(self.submat_pattern, "device", ref_device)

        if (full_device != ref_device) or (submat_device != ref_device):
            raise RuntimeError(
                "'submat_mask', 'full_pattern', and 'submat_pattern' must be on the same device."
            )

    def to(self, *args, **kwargs) -> SubmatPlan:
        device, dtype, copy_flag, non_blocking, memory_format = parse_to(
            *args, **kwargs
        )

        # If no device change occurs and copy=False, then return self.
        device_unchanged = (device is None) or (
            torch.device(device) == self.submat_mask.device
        )
        if device_unchanged and not copy_flag:
            return self

        return SubmatPlan(
            self.full_pattern.to(*args, **kwargs) if self.full_pattern else None,
            self.submat_pattern.to(*args, **kwargs) if self.submat_pattern else None,
            self.submat_mask.to(
                device=device,
                dtype=torch.bool,
                copy=copy_flag,
                non_blocking=non_blocking,
                memory_format=memory_format,
            ),
        )


class SubmatResult(NamedTuple):
    tensor: BaseDecoupledTensor
    plan: SubmatPlan

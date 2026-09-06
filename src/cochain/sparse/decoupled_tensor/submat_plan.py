from __future__ import annotations

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

    def to(self, *args, **kwargs) -> "SubmatPlan":
        device, dtype, copy_flag, non_blocking, memory_format = parse_to(
            *args, **kwargs
        )
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

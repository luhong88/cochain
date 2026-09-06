from __future__ import annotations

from dataclasses import dataclass

import torch
from jaxtyping import Bool, Integer
from torch import Tensor

from .pattern import SparsityPattern


@dataclass
class SubmatPlan:
    full_pattern: SparsityPattern
    submat_pattern: SparsityPattern
    idx_coo_submat_mask: Bool[Tensor, " nz"]

    def to(self, *args, **kwargs) -> "SubmatPlan":
        return SubmatPlan(
            self.full_pattern.to(*args, **kwargs),
            self.submat_pattern.to(*args, **kwargs),
            self.idx_coo_submat_mask.to(*args, **kwargs),
        )

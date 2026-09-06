from __future__ import annotations

from dataclasses import dataclass

from jaxtyping import Bool
from torch import Tensor

from .pattern import SparsityPattern


@dataclass
class SubmatPlan:
    full_pattern: SparsityPattern | None
    submat_pattern: SparsityPattern | None
    submat_mask: Bool[Tensor, " nz"]

    def to(self, *args, **kwargs) -> "SubmatPlan":
        # Note that the SparsityPattern.to() method ignores dtype conversions.
        return SubmatPlan(
            self.full_pattern.to(*args, **kwargs) if self.full_pattern else None,
            self.submat_pattern.to(*args, **kwargs) if self.submat_pattern else None,
            self.submat_mask.to(*args, **kwargs),
        )

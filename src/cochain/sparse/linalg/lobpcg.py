from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any, Callable, Literal

import torch as t
from jaxtyping import Float, Integer

from ..operators import SparseOperator, SparseTopology
from ._eigsh_utils import dLdA_backward, dLdA_dLdM_backward

try:
    import nvmath.sparse.advanced as nvmath_sp
    from cuda.core.experimental import Device

    _HAS_NVMATH = True

except ImportError:
    _HAS_NVMATH = False

if TYPE_CHECKING:
    import nvmath.sparse.advanced as nvmath_sp
    from cuda.core.experimental import Device

    from .nvmath_wrapper import DirectSolverConfig


@dataclass
class LOBPCGConfig:
    sigma: float | int | None = None
    v0: Float[t.Tensor, "m n"] | None = None
    n: int | None = None
    maxiter: int | None = None
    tol: float | None = None
    largest: bool = True

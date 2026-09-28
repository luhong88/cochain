from dataclasses import dataclass
from typing import Any

import torch
from jaxtyping import Float
from torch import Tensor

from ...sparse.linalg.eigen.lobpcg_._lobpcg_operators import LinearOp
from ...sparse.linalg.solvers import InvSparseOperator
from ..laplacians import MixedWeakLaplacianBlocks


@dataclass(frozen=True)
class MixedWeakLaplacianOp(LinearOp):
    laplacian: Float[MixedWeakLaplacianBlocks, "k_splx k_splx"]
    mass_km1_solver: Float[InvSparseOperator, "km1_splx km1_splx"]
    solver_kwargs: dict[str, Any] | None = None

    def __post_init__(self):
        if self.solver_kwargs is None:
            object.__setattr__(self, "solver_kwargs", {})

    def __matmul__(
        self, other: Float[Tensor, " k_splx *ch"]
    ) -> Float[Tensor, " k_splx *ch"]:
        _, rhs = self.laplacian.get_codiff_system(other)

        codiff = self.mass_km1_solver(rhs, **self.solver_kwargs)

        prod = self.laplacian.get_forward_pass(x=other, y=codiff)

        return prod

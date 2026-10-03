from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import Any, Literal

import torch
from jaxtyping import Float
from torch import Tensor

from ...sparse.decoupled_tensor import SparseDecoupledTensor
from ...sparse.linalg.eigen.lobpcg_._lobpcg_operators import LinearOp
from ...sparse.linalg.solvers import DirectSolverConfig
from ...sparse.linalg.solvers.nvmath_wrapper import _NVMathSparseSolver
from ...sparse.linalg.solvers.splu_wrapper import _SuperLUSparseSolver
from ...utils.parsing import to_col_major
from ..laplacians import MixedWeakLaplacianBlocks


@dataclass(frozen=True)
class MassKm1Solver:
    mass_km1: Float[SparseDecoupledTensor, "km1_splx km1_splx"]
    solver_type: Literal["scipy_splu", "cupy_splu", "nvmath_direct_solver"]
    solver_config: DirectSolverConfig | dict[str, Any]
    n: int

    @cached_property
    def _nvmath_direct_solver(self) -> _NVMathSparseSolver:
        import nvmath.sparse.advanced as nvmath_sp

        if not isinstance(self.solver_config, DirectSolverConfig):
            raise TypeError("'solver_config' must be a DirectSolverConfig object.")

        # Solve a linear system with a channel dim of at most 3n size.
        b_dummy = to_col_major(
            torch.zeros(
                (self.mass_km1.size(-1), 3 * self.n),
                dtype=self.mass_km1.dtype,
                device=self.mass_km1.device,
            ),
            batch_first=False,
        )

        return _NVMathSparseSolver(
            self.mass_km1.values,
            self.mass_km1.pattern,
            b_dummy,
            matrix_type=nvmath_sp.DirectSolverMatrixType.SPD,
            config=self.solver_config,
        )

    @cached_property
    def _splu(self) -> _SuperLUSparseSolver:
        if "scipy" in self.solver_type:
            backend = "scipy"
        elif "cupy" in self.solver_type:
            backend = "cupy"
        else:
            raise ValueError("Unknown 'backend' argument.")

        if not isinstance(self.solver_config, dict):
            raise TypeError("'solver_config' must be a dict object.")

        return _SuperLUSparseSolver(
            self.mass_km1,
            matrix_type="spd",
            backend=backend,
            **self.solver_config,
        )

    def _solve_via_nvmath_direct_solver(
        self, b: Float[Tensor, " km1_splx *ch"]
    ) -> Float[Tensor, " km1_splx *ch"]:
        # Pad channel dim up to size 3n.
        l = b.size(-1)
        pad = 3 * self.n - l

        rhs_padded_col_major = to_col_major(
            torch.nn.functional.pad(b, (0, pad, 0, 0)), batch_first=False
        )

        x = self._nvmath_direct_solver.solve(rhs_padded_col_major)[:, :l]

        return x

    def _solve_via_splu(
        self, b: Float[Tensor, " km1_splx *ch"]
    ) -> Float[Tensor, " km1_splx *ch"]:
        return self._splu.solve(b)

    def solve(
        self, b: Float[Tensor, " km1_splx *ch"]
    ) -> Float[Tensor, " km1_splx *ch"]:
        match self.solver_type:
            case "nvmath_direct_solver":
                return self._solve_via_nvmath_direct_solver(b)
            case "scipy_splu" | "cupy_splu":
                return self._solve_via_splu(b)
            case _:
                raise ValueError(
                    f"Unrecognized 'solver_type' argument '{self.solver_type}'"
                )


@dataclass(frozen=True)
class MixedWeakLaplacianOp(LinearOp):
    laplacian: Float[MixedWeakLaplacianBlocks, "k_splx k_splx"]
    mass_km1_solver: MassKm1Solver

    @property
    def dtype(self) -> torch.dtype:
        return self.laplacian.dtype

    @property
    def device(self) -> torch.device:
        return self.laplacian.device

    @property
    def shape(self) -> torch.Size:
        return self.laplacian.shape

    def size(self, dim: int | None = None) -> int | torch.Size:
        return self.laplacian.size(dim)

    def __matmul__(
        self, other: Float[Tensor, " k_splx *ch"]
    ) -> Float[Tensor, " k_splx *ch"]:
        _, rhs = self.laplacian.get_codiff_system(other)
        codiff = self.mass_km1_solver.solve(rhs)
        prod = self.laplacian.get_forward_pass(x=other, y=codiff)
        return prod

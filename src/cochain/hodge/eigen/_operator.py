from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import Any, Literal

import torch
from jaxtyping import Float
from torch import Tensor

from ...sparse.linalg.eigen.lobpcg_._lobpcg_operators import LinearOp
from ...sparse.linalg.solvers import DirectSolverConfig, InvSparseOperator
from ...sparse.linalg.solvers.nvmath_wrapper import _NVMathSparseSolver
from ...sparse.linalg.solvers.splu_wrapper import _SuperLUSparseSolver
from ...utils.parsing import to_col_major
from ..laplacians import MixedWeakLaplacianBlocks


@dataclass(frozen=True)
class MixedWeakLaplacianOp(LinearOp):
    laplacian: Float[MixedWeakLaplacianBlocks, "k_splx k_splx"]
    mass_km1_solver: Float[InvSparseOperator, "km1_splx km1_splx"]
    solver_type: Literal["scipy_splu", "cupy_splu", "nvmath_direct_solver"]
    solver_config: DirectSolverConfig | dict[str, Any]
    n: int

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

    @cached_property
    def _nvmath_direct_solver(self) -> _NVMathSparseSolver:
        import nvmath.sparse.advanced as nvmath_sp

        if not isinstance(self.solver_config, DirectSolverConfig):
            raise TypeError("'solver_config' must be a DirectSolverConfig object.")

        # Solve a linear system with a channel dim of at most 3n size.
        b_dummy = to_col_major(
            torch.zeros(
                (self.laplacian.mass_km1.size(-1), 3 * self.n),
                dtype=self.dtype,
                device=self.device,
            ),
            batch_first=False,
        )

        return _NVMathSparseSolver(
            self.laplacian.mass_km1.values,
            self.laplacian.mass_km1.pattern,
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
            self.laplacian.mass_km1,
            matrix_type="spd",
            backend=backend,
            **self.solver_config,
        )

    def __matmul_nvmath_direct_solver__(
        self, other: Float[Tensor, " k_splx *ch"]
    ) -> Float[Tensor, " k_splx *ch"]:
        _, rhs = self.laplacian.get_codiff_system(other)

        # Pad channel dim up to size 3n.
        l = other.size(-1)
        pad = 3 * self.n - l

        rhs_padded_col_major = to_col_major(
            torch.nn.functional.pad(rhs, (0, pad, 0, 0)), batch_first=False
        )

        codiff = self._nvmath_direct_solver(rhs_padded_col_major)

        prod = self.laplacian.get_forward_pass(x=other[:, :l], y=codiff[:, :l])

        return prod

    def __matmul_splu__(
        self, other: Float[Tensor, " k_splx *ch"]
    ) -> Float[Tensor, " k_splx *ch"]:
        other_flat = self._splu._flatten_b(other)
        codiff_flat = self._splu.solve(other_flat)
        prod_flat = self.laplacian.get_forward_pass(x=other_flat, y=codiff_flat)
        prod = _SuperLUSparseSolver._unflatten_x(prod_flat, other)

        return prod

    def __matmul__(
        self, other: Float[Tensor, " k_splx *ch"]
    ) -> Float[Tensor, " k_splx *ch"]:
        match self.solver_type:
            case "nvmath_direct_solver":
                return self.__matmul_nvmath_direct_solver__(other)
            case "scipy_splu" | "cupy_splu":
                return self.__matmul_splu__(other)
            case _:
                raise ValueError(
                    f"Unrecognized 'solver_type' argument '{self.solver_type}'"
                )

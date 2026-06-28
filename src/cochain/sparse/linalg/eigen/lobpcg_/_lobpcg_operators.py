from __future__ import annotations

import torch
from jaxtyping import Float
from torch import Tensor

from .....utils.parsing import to_col_major
from ....decoupled_tensor import DiagDecoupledTensor, SparseDecoupledTensor
from ...solvers import DirectSolverConfig
from ...solvers.nvmath import _NVMathSparseSolver

try:
    import nvmath.sparse.advanced as nvmath_sp

    _HAS_NVMATH = True

except ImportError:
    _HAS_NVMATH = False


class IdOp:
    """An identity operator."""

    def __init__(self):
        pass

    def __matmul__(self, other):
        return other

    def __rmatmul__(self, other):
        return other

    def to(self, *args, **kwargs):
        return self


class ShiftInvSymSpOp:
    r"""
    A linear operator used to solve an eigenvalue problem in the shift-invert mode.

    For the shift-inverted eigenvalue problem

    $$(A - \sigma I)^{-1} x = (\lambda - \sigma)^{-1} x$$

    this class represents the operator $(A - \sigma I)^{-1}$ and implements
    `__matmul__()`. Performing the matrix-vector multiplication
    $(A - \sigma I)^{-1} x = y$ for $y$ is equal to solving the sparse linear system
    $(A - \sigma I) y = x$ for $y$.
    """

    def __init__(
        self,
        a_sdt: Float[SparseDecoupledTensor, "m m"],
        sigma: float,
        n: int,
        config: DirectSolverConfig,
    ):
        if not _HAS_NVMATH:
            raise ImportError("nvmath-python backend required.")

        self.n = n

        eye = DiagDecoupledTensor.eye(
            a_sdt.size(-1), dtype=a_sdt.dtype, device=a_sdt.device
        )
        a_shift_inv = SparseDecoupledTensor.assemble(a_sdt, -sigma * eye)

        # Solve a linear system with a channel dim of at most 3n size.
        b_dummy = to_col_major(
            torch.zeros(
                (a_shift_inv.size(-1), 3 * n),
                dtype=a_shift_inv.dtype,
                device=a_shift_inv.device,
            ),
            batch_first=False,
        )

        self.solver = _NVMathSparseSolver(
            a_shift_inv.values,
            a_shift_inv.pattern,
            b_dummy,
            matrix_type=nvmath_sp.DirectSolverMatrixType.SYMMETRIC,
            config=config,
        )

    def __matmul__(self, x: Float[Tensor, "m k"]) -> Float[Tensor, "m k"]:
        # Pad channel dim up to size 3n.
        k = x.size(-1)
        pad = 3 * self.n - k

        x_padded_col_major = to_col_major(
            torch.nn.functional.pad(x, (0, pad, 0, 0)), batch_first=False
        )

        # Note that there is no need to "unflatten" b since there is only one
        # channel dimension.
        b = self.solver.solve(x_padded_col_major)

        return b[:, :k]


class ShiftInvSymGEPSpOp:
    r"""
    A linear operator used to solve a generalized eigenvalue problem in the shift-invert mode.

    For the shift-inverted generalized eigenvalue problem

    $$(A - \sigma M)^{-1} M x = (\lambda - \sigma)^{-1} x$$

    this class represents the operator $(A - \sigma M)^{-1} M$ and implements
    `__matmul__()`. Performing the matrix-vector multiplication
    $(A - \sigma M)^{-1} M x = y$ for $y$ is equal to solving the sparse linear
    system $(A - \sigma M) y = M x$ for $y$.
    """

    def __init__(
        self,
        a_sdt: Float[SparseDecoupledTensor, "m m"],
        m_sdt: Float[SparseDecoupledTensor, "m m"],
        sigma: float,
        n: int,
        config: DirectSolverConfig,
    ):
        if not _HAS_NVMATH:
            raise ImportError("nvmath-python backend required.")

        self.n = n
        self.m_sdt = m_sdt

        a_shift_inv = SparseDecoupledTensor.assemble(a_sdt, -sigma * m_sdt)

        # Solve a linear system with a channel dim of at most 3n size.
        b_dummy = to_col_major(
            torch.zeros(
                (self.m_sdt.size(-1), 3 * n),
                dtype=m_sdt.dtype,
                device=m_sdt.device,
            ),
            batch_first=False,
        )

        self.solver = _NVMathSparseSolver(
            a_shift_inv.values,
            a_shift_inv.pattern,
            b_dummy,
            matrix_type=nvmath_sp.DirectSolverMatrixType.SYMMETRIC,
            config=config,
        )

    def __matmul__(self, x: Float[Tensor, "m k"]) -> Float[Tensor, "m k"]:
        m_x = self.m_sdt @ x

        # Pad channel dim up to size 3n.
        k = m_x.size(-1)
        pad = 3 * self.n - k

        m_x_padded_col_major = to_col_major(
            torch.nn.functional.pad(m_x, (0, pad, 0, 0)), batch_first=False
        )

        # Note that there is no need to "unflatten" b since there is only one
        # channel dimension.
        b = self.solver.solve(m_x_padded_col_major)

        return b[:, :k]

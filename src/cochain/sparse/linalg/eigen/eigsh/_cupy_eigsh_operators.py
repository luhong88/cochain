from __future__ import annotations

import torch
from jaxtyping import Float, Integer
from torch import Tensor

from ....decoupled_tensor import (
    DiagDecoupledTensor,
    SparseDecoupledTensor,
    SparsityPattern,
)
from ...solvers import DirectSolverConfig
from ...solvers.nvmath_wrapper import _NVMathSparseSolver

try:
    import cupy as cp
    import cupyx.scipy.sparse.linalg as cp_sp_linalg

    _HAS_CUPY = True

except ImportError:
    _HAS_CUPY = False

try:
    import nvmath.sparse.advanced as nvmath_sp

    _HAS_NVMATH = True

except ImportError:
    _HAS_NVMATH = False


class CuPyShiftInvSymOp(cp_sp_linalg.LinearOperator):
    """A CuPy `LinearOperator` for shift-invert mode eigenvalue problem."""

    def __init__(
        self,
        a_val: Float[Tensor, " nz"],
        a_pattern: Integer[SparsityPattern, "r c"],
        sigma: float,
        config: DirectSolverConfig,
    ):
        if not _HAS_NVMATH:
            raise ImportError("nvmath-python backend is required.")
        if not _HAS_CUPY:
            raise ImportError("CuPy backend required.")

        a_sdt = SparseDecoupledTensor(a_pattern, a_val)
        eye = DiagDecoupledTensor.eye(
            a_pattern.size(-1), dtype=a_val.dtype, device=a_val.device
        )
        a_shift_inv = SparseDecoupledTensor.assemble(a_sdt, -sigma * eye)

        b_dummy = torch.zeros(
            a_pattern.size(-1), dtype=a_val.dtype, device=a_val.device
        )

        self.solver = _NVMathSparseSolver(
            a_shift_inv.values,
            a_shift_inv.pattern,
            b_dummy,
            matrix_type=nvmath_sp.DirectSolverMatrixType.SYMMETRIC,
            config=config,
        )

        cp_sp_linalg.LinearOperator.__init__(
            self,
            dtype=cp.dtype(str(a_val.dtype).split(".")[-1]),
            shape=tuple(a_pattern.shape),
        )

    def _matvec(self, x: Float[cp.ndarray, " c"]):
        r"""
        Compute inverse sparse matrix-vector product.

        For the operator $L = (A - \sigma I)^{-1}$, the matrix-vector
        multiplication $L x = b$ can also be written as $(A - \sigma I) b = x$;
        """
        # Get the current active CuPy stream and its representation in PyTorch.
        cp_stream = cp.cuda.get_current_stream()
        torch_stream = torch.cuda.ExternalStream(cp_stream.ptr)

        # Force PyTorch to use the current CuPy stream as its current active stream
        with torch.cuda.stream(torch_stream):
            x_torch = torch.from_dlpack(x)
            # Note that there is no need to "unflatten" b since there are no
            # channel dimensions.
            b_torch = self.solver.solve(x_torch)

        # Force the CPU to wait for safety.
        cp_stream.synchronize()

        b = cp.from_dlpack(b_torch.detach().contiguous())

        return b

    def _matmat(self, X):
        # Cupy eigsh() should not need matrix-matrix multiplications
        raise NotImplementedError()

    def _adjoint(self):
        # The adjoint operator is self since self is symmetric.
        return self

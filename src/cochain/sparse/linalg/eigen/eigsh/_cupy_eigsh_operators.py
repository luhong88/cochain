from __future__ import annotations

import torch
from jaxtyping import Float, Integer
from torch import Tensor

from .....utils.stream import cupy_in_torch_stream
from ....decoupled_tensor import SparsityPattern
from ....decoupled_tensor._conversion import sdt_to_cupy_csr
from ...solvers import DirectSolverConfig
from ..base._inv_operator import BaseNVMathInvSymSpOp

try:
    import cupy as cp
    import cupyx.scipy.sparse as cp_sp
    import cupyx.scipy.sparse.linalg as cp_sp_linalg

    _HAS_CUPY = True

except ImportError:
    _HAS_CUPY = False

try:
    import nvmath.sparse.advanced

    _HAS_NVMATH = True

except ImportError:
    _HAS_NVMATH = False

if _HAS_NVMATH:
    try:
        from cuda.core import Device
    except ImportError:
        from cuda.core.experimental import Device


if _HAS_NVMATH and _HAS_CUPY:

    class CuPyShiftInvSymOp(BaseNVMathInvSymSpOp, cp_sp_linalg.LinearOperator):
        """A CuPy `LinearOperator` for shift-invert mode eigenvalue problem."""

        def __init__(
            self,
            a_val: Float[Tensor, " nz"],
            a_pattern: Integer[SparsityPattern, "r c"],
            sigma: float,
            config: DirectSolverConfig,
        ):
            if not a_pattern._is_int32_safe:
                raise ValueError(
                    "The sparse indices of the input tensor 'A' cannot be safely "
                    "cast to int32 dtype."
                )

            # Prepare Cupy arrays.
            with cupy_in_torch_stream():
                a_cp = sdt_to_cupy_csr(a_val, a_pattern)

                diag_cp = sigma * cp_sp.identity(
                    a_cp.shape[0], dtype=a_cp.dtype, format="csr"
                )

                a_shift_inv_cp = a_cp - diag_cp

                b_dummy = cp.zeros(a_cp.shape[0], dtype=a_cp.dtype)

                BaseNVMathInvSymSpOp.__init__(
                    self,
                    a=a_shift_inv_cp,
                    b=b_dummy,
                    config=config,
                    stream=cp.cuda.get_current_stream(),
                )

            cp_sp_linalg.LinearOperator.__init__(
                self, dtype=a_cp.dtype, shape=a_cp.shape
            )

        def _matvec(self, x: Float[cp.ndarray, " c"]):
            r"""
            Compute inverse sparse matrix-vector product.

            For the operator $L = (A - \sigma I)^{-1}$, the matrix-vector
            multiplication $L x = b$ can also be written as $(A - \sigma I) b = x$;
            """
            cp_stream = cp.cuda.get_current_stream()
            Device(x.device.id).set_current()

            self.solver.reset_operands(b=x, stream=cp_stream)
            b = self.solver.solve(stream=cp_stream)

            return b

        def _matmat(self, X):
            # Cupy eigsh() should not need matrix-matrix multiplications
            raise NotImplementedError()

        def _adjoint(self):
            # The adjoint operator is self since self is symmetric.
            return self

else:

    class CuPyShiftInvSymOp:
        def __init__(self, *args, **kwargs):
            raise ImportError("CuPy and nvmath-python backends required.")

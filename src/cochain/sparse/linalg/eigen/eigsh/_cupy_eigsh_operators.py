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
        """
        A CuPy LinearOperator object used to solve an eigenvalue problem in the
        shift inverse mode.
        """

        def __init__(
            self,
            a_val: Float[Tensor, " nnz"],
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
                A_cp = sdt_to_cupy_csr(a_val, a_pattern)

                diag_cp = sigma * cp_sp.identity(
                    A_cp.shape[0], dtype=A_cp.dtype, format="csr"
                )

                A_shift_inv_cp = A_cp - diag_cp

                b_dummy = cp.zeros(A_cp.shape[0], dtype=A_cp.dtype)

                BaseNVMathInvSymSpOp.__init__(
                    self,
                    a=A_shift_inv_cp,
                    b=b_dummy,
                    config=config,
                    stream=cp.cuda.get_current_stream(),
                )

            cp_sp_linalg.LinearOperator.__init__(
                self, dtype=A_cp.dtype, shape=A_cp.shape
            )

        def _matvec(self, x: Float[cp.ndarray, " c"]):
            """
            For the operator L = inv(A - σI), the matrix-vector multiplication L@x = b
            can also be written as x = (A - σI)@b, or solve(A - σI, x).
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

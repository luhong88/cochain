from __future__ import annotations

from typing import TYPE_CHECKING

import torch as t
from jaxtyping import Float

from ..operators import SparseOperator

try:
    import cupy as cp
    import cupyx.scipy.sparse as cp_sp
    import cupyx.scipy.sparse.linalg as cp_sp_linalg

    from .nvmath_wrapper import DirectSolverConfig, sp_literal_to_matrix_type

    _HAS_CUPY = True

except ImportError:
    _HAS_CUPY = False

try:
    import nvmath.sparse.advanced as nvmath_sp
    from cuda.core.experimental import Device

    _HAS_NVMATH = True

except ImportError:
    _HAS_NVMATH = False

if TYPE_CHECKING:
    import cupyx.scipy.sparse as cp_sp
    import cupyx.scipy.sparse.linalg as cp_sp_linalg
    import nvmath.sparse.advanced as nvmath_sp
    from cuda.core.experimental import Device

    from .nvmath_wrapper import DirectSolverConfig, sp_literal_to_matrix_type


def _sp_op_to_cp_csr(sp_op: SparseOperator) -> cp_sp.csr_matrix:
    return cp_sp.csr_matrix(
        (
            cp.from_dlpack(sp_op.val.detach().contiguous()),
            cp.from_dlpack(sp_op.sp_topo.idx_col_int32.detach().contiguous()),
            cp.from_dlpack(sp_op.sp_topo.idx_crow_int32.detach().contiguous()),
        ),
        shape=tuple(sp_op.shape),
    )


if _HAS_CUPY and _HAS_NVMATH:

    class _CuPySymOp:
        def __init__(
            self,
            a: cp_sp.csr_matrix,
            config: DirectSolverConfig | None,
            stream: t.Stream,
        ):
            if config is None:
                config = DirectSolverConfig()

            config.options.sparse_system_type = sp_literal_to_matrix_type["symmetric"]

            b_dummy = cp.zeros(a.size(0), dtype=a.dtype, device=a.device)

            self.solver = nvmath_sp.DirectSolver(
                a, b_dummy, options=config.options, execution=config.execution
            )

            # force blocking operation to make it memory-safe to potentially call
            # free() immediately after solve().
            self.solver.options.blocking = True

            # Amortize planning and factorization costs upfront in __init__()
            for k, v in config.plan_kwargs.items():
                setattr(self.solver.plan_config, k, v)
            self.solver.plan(stream=stream)

            for k, v in config.factorization_kwargs.items():
                setattr(self.solver.factorization_config, k, v)
            self.solver.factorize(stream=stream)

            for k, v in config.solution_kwargs.items():
                setattr(self.solver.solution_config, k, v)

        def _matmat(self, X):
            # Cupy eigsh() should not need matrix-matrix multiplications
            raise NotImplementedError()

        def _adjoint(self):
            # The adjoint operator is self since self is symmetric.
            return self

        def __del__(self):
            # DirectSolver needs an explicit free() step to free up memory/resources.
            if hasattr(self, "solver"):
                if hasattr(self.solver, "free"):
                    self.solver.free()

    class CuPyGenSymOp(_CuPySymOp, cp_sp_linalg.LinearOperator):
        """
        A CuPy LinearOperator object used to solve a generalized eigenvalue problem.
        """

        def __init__(
            self,
            A: Float[SparseOperator, "r c"],
            M: Float[SparseOperator, "r c"],
            config: DirectSolverConfig = None,
        ):
            t_stream = t.cuda.current_stream()

            with cp.cuda.ExternalStream(t_stream.cuda_stream, t_stream.device_index):
                self.A_cp = _sp_op_to_cp_csr(A)
                M_cp = _sp_op_to_cp_csr(M)

            _CuPySymOp.__init__(self, a=M_cp, config=config, stream=t_stream)

            cp_sp_linalg.LinearOperator.__init__(
                self, dtype=self.A_cp.dtype, shape=self.A_cp.shape
            )

        def _matvec(self, x: Float[cp.ndarray, " c"]):
            """
            For the operator L = inv(M)@A, the matrix-vector multiplication L@x = b
            can also be written as A@x = M@b, or solve(M, A@x).
            """
            cp_stream = cp.cuda.get_current_stream()
            Device(cp_stream.device_id).set_current()

            Ax = self.A_cp @ x
            self.solver.reset_operands(b=Ax, stream=cp_stream)
            b = self.solver.solve(stream=cp_stream)

            return b

    class CuPyShiftInvSymOp(_CuPySymOp, cp_sp_linalg.LinearOperator):
        """
        A CuPy LinearOperator object used to solve an eigenvalue problem in the
        shift inverse mode.
        """

        def __init__(
            self,
            A: Float[SparseOperator, "r c"],
            sigma: float,
            config: DirectSolverConfig = None,
        ):
            t_stream = t.cuda.current_stream()

            with cp.cuda.ExternalStream(t_stream.cuda_stream, t_stream.device_index):
                A_cp = _sp_op_to_cp_csr(A)

                diag_cp = -sigma * cp_sp.identity(
                    A.shape[0], dtype=A.dtype, format="csr"
                )

                A_shift_inv_cp = A_cp - diag_cp

            _CuPySymOp.__init__(self, a=A_shift_inv_cp, config=config, stream=t_stream)

            cp_sp_linalg.LinearOperator.__init__(
                self, dtype=A_cp.dtype, shape=A_cp.shape
            )

        def _matvec(self, x: Float[cp.ndarray, " c"]):
            """
            For the operator L = inv(A - σI), the matrix-vector multiplication L@x = b
            can also be written as x = (A - σI)@b, or solve(A - σI, x).
            """
            cp_stream = cp.cuda.get_current_stream()
            Device(cp_stream.device_id).set_current()

            self.solver.reset_operands(b=x, stream=cp_stream)
            b = self.solver.solve(stream=cp_stream)

            return b

    class CuPyGenShiftInvSymOp(_CuPySymOp, cp_sp_linalg.LinearOperator):
        """
        A CuPy LinearOperator object used to solve a generalized eigenvalue problem
        in the shift inverse mode.
        """

        def __init__(
            self,
            A: Float[SparseOperator, "r c"],
            M: Float[SparseOperator, "r c"],
            sigma: float,
            config: DirectSolverConfig = None,
        ):
            t_stream = t.cuda.current_stream()

            with cp.cuda.ExternalStream(t_stream.cuda_stream, t_stream.device_index):
                A_cp = _sp_op_to_cp_csr(A)
                self.M_cp = _sp_op_to_cp_csr(M)

                A_sM = A_cp - sigma * self.M_cp

            _CuPySymOp.__init__(self, a=A_sM, config=config, stream=t_stream)

            cp_sp_linalg.LinearOperator.__init__(
                self, dtype=A_cp.dtype, shape=A_cp.shape
            )

        def _matvec(self, x: Float[cp.ndarray, " c"]):
            """
            For the operator L = inv(A - σM)@M, the matrix-vector multiplication L@x = b
            can also be written as M@x = (A - σM)@b, or solve(A - σM, M@x).
            """
            cp_stream = cp.cuda.get_current_stream()
            Device(cp_stream.device_id).set_current()

            Mx = self.M_cp @ x
            self.solver.reset_operands(b=Mx, stream=cp_stream)
            b = self.solver.solve(stream=cp_stream)

            return b

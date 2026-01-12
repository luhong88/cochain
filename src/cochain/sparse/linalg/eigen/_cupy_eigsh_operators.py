import cupy as cp
import cupyx.scipy.sparse as cp_sp
import cupyx.scipy.sparse.linalg as cp_sp_linalg
import torch as t
from cuda.core.experimental import Device
from jaxtyping import Float, Integer

from ...operators import SparseTopology
from ..solvers.nvmath_wrapper import DirectSolverConfig
from ._inv_operator import BaseNVMathInvSymSpOp


def sp_op_comps_to_cp_csr(
    A_val: Float[t.Tensor, " nnz"],
    A_sp_topo: Integer[SparseTopology, "r c"],
) -> Float[cp_sp.csr_matrix, "r c"]:
    return cp_sp.csr_matrix(
        (
            cp.from_dlpack(A_val.detach().contiguous()),
            cp.from_dlpack(A_sp_topo.idx_col_int32.detach().contiguous()),
            cp.from_dlpack(A_sp_topo.idx_crow_int32.detach().contiguous()),
        ),
        shape=tuple(A_sp_topo.shape),
    )


class CuPyShiftInvSymOp(BaseNVMathInvSymSpOp, cp_sp_linalg.LinearOperator):
    """
    A CuPy LinearOperator object used to solve an eigenvalue problem in the
    shift inverse mode.
    """

    def __init__(
        self,
        A_val: Float[t.Tensor, " nnz"],
        A_sp_topo: Integer[SparseTopology, "r c"],
        sigma: float,
        config: DirectSolverConfig,
    ):
        t_stream = t.cuda.current_stream()

        # Prepare Cupy arrays.
        with cp.cuda.ExternalStream(t_stream.cuda_stream, t_stream.device_index):
            A_cp = sp_op_comps_to_cp_csr(A_val, A_sp_topo)

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

        cp_sp_linalg.LinearOperator.__init__(self, dtype=A_cp.dtype, shape=A_cp.shape)

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

    def _matmat(self, X):
        # Cupy eigsh() should not need matrix-matrix multiplications
        raise NotImplementedError()

    def _adjoint(self):
        # The adjoint operator is self since self is symmetric.
        return self

from collections import ChainMap
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import torch as t
from cuda.core.experimental import Device
from jaxtyping import Float

from ...operators import DiagOperator, SparseOperator
from ..solvers.nvmath_wrapper import DirectSolverConfig
from ._inv_operator import BaseNVMathInvSymSpOp
from ._lobpcg_operators import IdentityOperator

try:
    import cupy as cp
    import cupyx.scipy.sparse as cp_sp
    import cupyx.scipy.sparse.linalg as cp_sp_linalg

    _HAS_CUPY = True

except ImportError:
    _HAS_CUPY = False

if TYPE_CHECKING:
    import cupy as cp
    import cupyx.scipy.sparse as cp_sp
    import cupyx.scipy.sparse.linalg as cp_sp_linalg


@dataclass
class LOBPCGPrecondConfig:
    """
    diag_damp only applicable if method is ilu or cholesky.
    nvmath_config only applicable if method is cholesky.
    spilu_kwargs only applicable if method is ilu.
    """

    method: Literal["identity", "jacobi", "ilu", "cholesky"] = "cholesky"
    diag_damp: float | int | None = None
    nvmath_config: DirectSolverConfig | None = None
    spilu_kwargs: dict[str, Any] | None = None

    def __post_init__(self):
        if self.spilu_kwargs is None:
            self.spilu_kwargs = {}
        if self.nvmath_config is None:
            self.nvmath_config = DirectSolverConfig()


IdentityPrecond = IdentityOperator


class JacobiPrecond:
    """
    Jacobi preconditioner for LOBPCG.

    While the Jacobi preconditioner is computationally much cheaper than the ILU
    and Cholesky preconditioner, it is only recommended when the initial guess
    is already very good.
    """

    def __init__(self, A_op: Float[SparseOperator, "m m"]):
        eps = 1e-4 * A_op.tr / A_op.size(0)
        self.diag_op = DiagOperator(1 / (A_op.diagonal() + eps))

    def __matmul__(self, res: Float[t.Tensor, "m k"]) -> Float[t.Tensor, "m k"]:
        return self.diag_op @ res


class ILUPrecond:
    """
    Diagonally damped incomplete LU preconditioner for LOBPCG.

    For the standard eigenvalue problem A@x = λ*x and the generalized eigenvalue
    problem A@x = λ*M@x, if r is a residual vector, applying the preconditioner
    to r is equivalent to solving a sparse linear system (A + ϵI)@w = r for w
    using incomplete LU factorization.

    This preconditioner uses CuPy to perform the incomplete LU factorization.
    By default, it uses the cusparse backend to perform the factorization with
    zero fill-in and no pivoting. This default config tends to result in smaller
    memory usage than the Cholesky preconditioner.
    """

    def __init__(
        self,
        A_op: Float[SparseOperator, "m m"],
        diag_damp: float | int | None,
        spilu_kwargs: dict[str, Any],
    ):
        if not _HAS_CUPY:
            raise ImportError("cupy backend required.")

        if diag_damp == 0:
            op = A_op.to_sparse_csc(int32=True)
        else:
            if diag_damp is None:
                eps = 1e-4 * A_op.tr / A_op.size(0)
            else:
                eps = diag_damp

            # Pytorch currently does not support operations like A - I on sparse
            # CSC tensors.
            eye = DiagOperator.eye(A_op.size(-1), dtype=A_op.dtype, device=A_op.device)

            op = SparseOperator.assemble(A_op, eps * eye).to_sparse_csc(int32=True)

        stream = t.cuda.current_stream()
        with cp.cuda.ExternalStream(stream.cuda_stream, stream.device_index):
            op_cp: Float[cp_sp.csc_matrix, "r c"] = cp_sp.csc_matrix(
                (
                    cp.from_dlpack(op.values()),
                    cp.from_dlpack(op.row_indices()),
                    cp.from_dlpack(op.ccol_indices()),
                ),
                shape=tuple(op.shape),
            )

            spilu_default = {"fill_factor": 1}
            spilu_config = ChainMap(spilu_kwargs, spilu_default)

            self.solver = cp_sp_linalg.spilu(A=op_cp, **spilu_config)

    def __matmul__(self, res: Float[t.Tensor, "m k"]) -> Float[t.Tensor, "m k"]:
        stream = t.cuda.current_stream()
        with cp.cuda.ExternalStream(stream.cuda_stream, stream.device_index):
            res_cp = cp.from_dlpack(res.detach().contiguous())
            sol = t.from_dlpack(self.solver.solve(res_cp, trans="N"))

        return sol


class ChoPrecond(BaseNVMathInvSymSpOp):
    """
    Diagonally damped Cholesky preconditioner for LOBPCG.

    For the standard eigenvalue problem A@x = λ*x and the generalized eigenvalue
    problem A@x = λ*M@x, if r is a residual vector, applying the preconditioner
    to r is equivalent to solving a sparse linear system (A + ϵI)@w = r for w
    using Cholesky factorization.

    This preconditioner uses `nvmath-python` DirectSolver to perform Cholesky
    factorization.
    """

    def __init__(
        self,
        A_op: Float[SparseOperator, "m m"],
        n: int,
        diag_damp: float | int | None,
        nvmath_config: DirectSolverConfig,
    ):
        self.n = n

        # Solve a linear system with at most 3n channel dims
        b_dummy = (
            t.zeros((A_op.size(-1), 3 * n), dtype=A_op.dtype, device=A_op.device)
            .transpose(-1, -2)
            .contiguous()
            .transpose(-1, -2)
        )

        if diag_damp == 0:
            op = A_op.to_sparse_csr(int32=True)
        else:
            if diag_damp is None:
                eps = 1e-4 * A_op.tr / A_op.size(0)
            else:
                eps = diag_damp

            # Pytorch currently does not support operations like A - I on sparse
            # CSR tensors.
            eye = DiagOperator.eye(A_op.size(-1), dtype=A_op.dtype, device=A_op.device)

            op = SparseOperator.assemble(A_op, eps * eye).to_sparse_csr(int32=True)

        super().__init__(op, b_dummy, nvmath_config, t.cuda.current_stream())

    def __matmul__(self, res: Float[t.Tensor, "m k"]) -> Float[t.Tensor, "m k"]:
        stream = t.cuda.current_stream()
        Device(res.device.index).set_current()

        # Pad up to 3n channel dims
        k = res.size(-1)
        pad = 3 * self.n - k

        res_padded_col_major = (
            t.nn.functional.pad(res, (0, pad, 0, 0))
            .transpose(-1, -2)
            .contiguous()
            .transpose(-1, -2)
        )

        self.solver.reset_operands(b=res_padded_col_major, stream=stream)
        sol = self.solver.solve(stream=stream)

        return sol[:, :k]

from collections import ChainMap
from dataclasses import dataclass
from typing import Any, Literal

import torch
from jaxtyping import Float
from torch import Tensor

from .....utils.parsing import to_col_major
from .....utils.stream import cupy_in_torch_stream
from ....decoupled_tensor import DiagDecoupledTensor, SparseDecoupledTensor
from ....decoupled_tensor._conversion import sdt_to_cupy_csc
from ...solvers import DirectSolverConfig
from ...solvers.nvmath import _NVMathSparseSolver
from ._lobpcg_operators import IdentityOperator

try:
    import cupy as cp
    import cupyx.scipy.sparse as cp_sp
    import cupyx.scipy.sparse.linalg as cp_sp_linalg

    _HAS_CUPY = True

except ImportError:
    _HAS_CUPY = False

try:
    import nvmath.sparse.advanced as nvmath_sp

    _HAS_NVMATH = True

except ImportError:
    _HAS_NVMATH = False

if _HAS_NVMATH:
    try:
        from cuda.core import Device
    except ImportError:
        from cuda.core.experimental import Device


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
        if not _HAS_NVMATH:
            raise ImportError("nvmath-python backend required.")

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

    def __init__(self, A_op: Float[SparseDecoupledTensor, "m m"]):
        eps = 1e-4 * A_op.tr / A_op.size(0)
        self.ddt = DiagDecoupledTensor(1 / (A_op.diagonal() + eps))

    def __matmul__(self, res: Float[Tensor, "m k"]) -> Float[Tensor, "m k"]:
        return self.ddt @ res


if _HAS_CUPY:

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
            a_sdt: Float[SparseDecoupledTensor, "m m"],
            diag_damp: float | int | None,
            spilu_kwargs: dict[str, Any],
        ):
            if diag_damp == 0:
                op_sdt = a_sdt

            else:
                if diag_damp is None:
                    eps = 1e-4 * a_sdt.tr / a_sdt.size(0)
                else:
                    eps = diag_damp

                eye = DiagDecoupledTensor.eye(
                    a_sdt.size(-1), dtype=a_sdt.dtype, device=a_sdt.device
                )

                op_sdt = SparseDecoupledTensor.assemble(a_sdt, eps * eye)

            with cupy_in_torch_stream():
                op_cp: Float[cp_sp.csc_matrix, "r c"] = sdt_to_cupy_csc(
                    op_sdt.values, op_sdt.pattern
                )

                spilu_default = {"fill_factor": 1}
                spilu_config = ChainMap(spilu_kwargs, spilu_default)

                self.solver = cp_sp_linalg.spilu(A=op_cp, **spilu_config)

        def __matmul__(self, res: Float[Tensor, "m k"]) -> Float[Tensor, "m k"]:
            with cupy_in_torch_stream():
                res_cp = cp.from_dlpack(res.detach().contiguous())
                sol = torch.from_dlpack(self.solver.solve(res_cp))

            return sol

else:

    class ILUPrecond:
        def __init__(self, *args, **kwargs):
            raise ImportError("CuPy backend required.")


if _HAS_NVMATH:

    class ChoPrecond:
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
            a_sdt: Float[SparseDecoupledTensor, "m m"],
            n: int,
            diag_damp: float | int | None,
            nvmath_config: DirectSolverConfig,
        ):
            self.n = n

            # Solve a linear system with a channel dim of at most 3n size.
            b_dummy = to_col_major(
                torch.zeros(
                    (a_sdt.size(-1), 3 * n), dtype=a_sdt.dtype, device=a_sdt.device
                ),
                batch_first=False,
            )

            if diag_damp == 0:
                op_sdt = a_sdt

            else:
                if diag_damp is None:
                    eps = 1e-4 * a_sdt.tr / a_sdt.size(0)
                else:
                    eps = diag_damp

                eye = DiagDecoupledTensor.eye(
                    a_sdt.size(-1), dtype=a_sdt.dtype, device=a_sdt.device
                )

                op_sdt = SparseDecoupledTensor.assemble(a_sdt, eps * eye)

            self.solver = _NVMathSparseSolver(
                op_sdt.values,
                op_sdt.pattern,
                b_dummy,
                matrix_type=nvmath_sp.DirectSolverMatrixType.SYMMETRIC,
                config=nvmath_config,
            )

        def __matmul__(self, res: Float[Tensor, "m k"]) -> Float[Tensor, "m k"]:
            # Pad up to 3n channel dims
            k = res.size(-1)
            pad = 3 * self.n - k

            res_padded_col_major = to_col_major(
                torch.nn.functional.pad(res, (0, pad, 0, 0)), batch_first=False
            )

            # Note that there is no need to "unflatten" b since there is only one
            # channel dimension.
            sol = self.solver.solve(res_padded_col_major)

            return sol[:, :k]

else:

    class ChoPrecond:
        def __init__(self, *args, **kwargs):
            raise ImportError("nvmath-python backend required.")

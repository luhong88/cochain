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
from ...solvers.nvmath_wrapper import _NVMathSparseSolver
from ._lobpcg_operators import IdOp

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

# A machine eps scaling factor for controlling the strength of diagonal damping.
ALPHA = 1e3


@dataclass
class LOBPCGPrecondConfig:
    """
    A dataclass encapsulating LOBPCG preconditioner configuration.

    Parameters
    ----------
    method
        The preconditioning method; set to "identity" to disable preconditioning.
    diag_damp
        The strength of diagonal damping/Tikhonov regularization. This parameter
        is only applicable if `method` is 'ilu' or 'cholesky'; set to integer 0
        to disable diagonal damping.
    nvmath_config
        Optional configurations for the `nvmath-python` `DirectSolver()`; only
        applicable if `method` is 'cholesky'.
    spilu_kwargs
        Optional configurations for the `CuPy` `spilu()`; only applicable if
        `method` is 'ilu'.
    """

    method: Literal["identity", "jacobi", "ilu", "cholesky"] = "identity"
    diag_damp: float | int | None = None
    nvmath_config: DirectSolverConfig | None = None
    spilu_kwargs: dict[str, Any] | None = None

    def __post_init__(self):
        if self.spilu_kwargs is None:
            self.spilu_kwargs = {}
        if self.nvmath_config is None:
            self.nvmath_config = DirectSolverConfig()


IdentityPrecond = IdOp


class JacobiPrecond:
    r"""
    Jacobi preconditioner for LOBPCG.

    Consider the standard eigenvalue problem $A x = \lambda x$. Let
    $r = A x - \lambda M x$ be the residual vector. The Jacobi preconditioner
    approximates the inverse of $A$ as

    $$P = \text{diag}(A)^{-1}$$

    to solve for the search direction $w = P r$.

    For the generalized eigenvalue problem $A x = \lambda M x$, this $P$ is still
    a good approximation for the inverse of the shifted operator
    $(A - \lambda M)^{-1}$ when the target eigenvalues are small.

    While the Jacobi preconditioner is computationally much cheaper than the ILU
    and Cholesky preconditioner, it is only recommended when the initial guess
    is already very good.

    This preconditioner does not require `CuPy` or `nvmath-python`.
    """

    def __init__(self, a_sdt: Float[SparseDecoupledTensor, "m m"]):
        # Add a small eps to prevent division by zero.
        eps = ALPHA * torch.finfo(a_sdt.dtype).eps * a_sdt.tr / a_sdt.size(0)
        self.ddt = DiagDecoupledTensor(1 / (a_sdt.diagonal() + eps))

    def __matmul__(self, res: Float[Tensor, "m k"]) -> Float[Tensor, "m k"]:
        return self.ddt @ res


class ILUPrecond:
    r"""
    Diagonally damped incomplete LU preconditioner for LOBPCG.

    Consider the standard eigenvalue problem $A x = \lambda x$. Let
    $r = A x - \lambda M x$ be the residual vector. This preconditioner computes
    the incomplete LU factorization of the diagonally damped $A$; i.e.,

    $$A + \epsilon I \approx L U$$

    which is used to approximate the inverse of $A$ and compute the search direction
    $w$ as the solution to $L U w = r$. The diagonal damping provided by $\epsilon I$
    is important to increase the diagonal dominance of $A$ to ensure that the ILU
    is successful and well-conditioned.

    For the generalized eigenvalue problem $A x = \lambda M x$, this $P = (L U)^{-1}$
    is still a good approximation for the inverse of the shifted operator
    $(A - \lambda M)^{-1}$ when the target eigenvalues are small.

    This preconditioner uses `CuPy` to perform the incomplete LU factorization.
    By default, it uses the `cusparse` backend to perform the factorization with
    zero fill-in and no pivoting. This default config tends to result in smaller
    memory usage than the Cholesky preconditioner.
    """

    def __init__(
        self,
        a_sdt: Float[SparseDecoupledTensor, "m m"],
        diag_damp: float | int | None,
        spilu_kwargs: dict[str, Any],
    ):
        if not _HAS_CUPY:
            raise ImportError("CuPy backend required.")

        if diag_damp == 0:
            op_sdt = a_sdt

        else:
            if diag_damp is None:
                # If no eps is given, set eps to be proportional to the average
                # diagonal entry size, scaled by the dtype machine eps.
                eps = ALPHA * torch.finfo(a_sdt.dtype).eps * a_sdt.tr / a_sdt.size(0)
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


class ChoPrecond:
    r"""
    Diagonally damped Cholesky preconditioner for LOBPCG.

    Consider the standard eigenvalue problem $A x = \lambda x$. Let
    $r = A x - \lambda M x$ be the residual vector. This preconditioner computes
    the exact Cholesky factorization of the diagonally damped $A$; i.e.,

    $$A + \epsilon I = L L^T$$

    which is used to compute the inverse of $A$ and the search direction $w$ as
    the solution to $L L^T w = r$. The diagonal damping provided by $\epsilon I$
    is important to increase the diagonal dominance of $A$ to ensure that the
    matrix is strictly symmetric positive definite.

    For the generalized eigenvalue problem $A x = \lambda M x$, this $P = (L L^T)^{-1}$
    is still a good approximation for the inverse of the shifted operator
    $(A - \lambda M)^{-1}$ when the target eigenvalues are small.

    This preconditioner uses `nvmath-python` `DirectSolver` to perform Cholesky
    factorization.
    """

    def __init__(
        self,
        a_sdt: Float[SparseDecoupledTensor, "m m"],
        n: int,
        diag_damp: float | int | None,
        nvmath_config: DirectSolverConfig,
    ):
        if not _HAS_NVMATH:
            raise ImportError("nvmath-python backend required.")

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
                # If no eps is given, set eps to be proportional to the average
                # diagonal entry size, scaled by the dtype machine eps.
                eps = ALPHA * torch.finfo(a_sdt.dtype).eps * a_sdt.tr / a_sdt.size(0)
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

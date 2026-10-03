from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from jaxtyping import Float
from torch import Tensor

from ...sparse.decoupled_tensor import (
    BaseDecoupledTensor,
    DiagDecoupledTensor,
    SparseDecoupledTensor,
)
from ...sparse.linalg.eigen.lobpcg_._lobpcg_preconditioners import LOBPCGPreconditioner
from ...sparse.linalg.solvers import DirectSolverConfig
from ...sparse.linalg.solvers.nvmath_wrapper import _NVMathSparseSolver
from ...utils.parsing import to_col_major

try:
    import nvmath.sparse.advanced as nvmath_sp

    _HAS_NVMATH = True

except ImportError:
    _HAS_NVMATH = False


@dataclass
class LaplacianLOBPCGPrecondConfig:
    """
    A dataclass encapsulating mixed weak Laplacian LOBPCG preconditioner configuration.

    Parameters
    ----------
    method
        The preconditioning method; set to "identity" to disable preconditioning.
    tau
        The strength of shift/regularization by the mass matrix and has the same
        unit as the eigenvalues. When `tau` is set to "auto", it is computed as
        the approximate mean of the generalized eigenvalues, scaled down by a
        factor of 0.01. This is applicable to both the "shifted_up" and "shifted_lumped"
        methods.
    star_km1 : [km1_splx, km1_splx]
        An optional, diagonal Hodge (k-1)-star operator used by the "shifted_lumped"
        method to approximate the inverse of the corresponding consistent mass
        matrix. It is recommended to compute the Hodge star with a barycentric
        dual so that it is guaranteed to be SPD regardless of mesh quality.
    star_k : [k_splx, k_splx]
        An optional, diagonal Hodge k-star operator used when `tau` is "auto".
    nvmath_config
        Optional configurations for the `nvmath-python` `DirectSolver()`; applicable
        to both the "shifted_up" and "shifted_lumped" methods.
    """

    method: Literal["identity", "shifted_up", "shifted_lumped"] = "identity"
    tau: float | Literal["auto"] = "auto"
    star_km1: Float[DiagDecoupledTensor, "km1_splx km1_splx"] | None = None
    star_k: Float[DiagDecoupledTensor, "k_splx k_splx"] | None = None
    nvmath_config: DirectSolverConfig | None = None

    def __post_init__(self):
        if self.method in ["shifted_up", "shifted_lumped"]:
            if self.tau == "auto":
                if self.star_k is None:
                    raise ValueError(
                        "'star_k' is required for automatic scale estimation for tau."
                    )
            else:
                if self.tau <= 0:
                    raise ValueError("'tau' must be a positive float.")

            if self.method == "shifted_lumped":
                if self.star_km1 is None:
                    raise ValueError(
                        "'star_km1' is required for method = 'shifted_lumped'."
                    )

            if self.nvmath_config is None:
                self.nvmath_config = DirectSolverConfig()


def _mean_generalized_eigenvalues(
    weak_laplacian: Float[SparseDecoupledTensor, "k_splx k_splx"],
    inv_star_k: Float[DiagDecoupledTensor, "k_splx k_splx"],
    scale: float = 1e-2,
):
    """
    Approximate the mean generalized eigenvalues of a weak Hodge Laplacian.

    For a weak Laplacian $S_k$ and associated mass matrix $M_k$, the sum of the
    generalized eigenvalues of $S_k$ is $tr(M_k^{-1}S_k)$. This function computes
    an approximate mean of the generalized eigenvalues by replacing the mass
    matrix with a Hodge star, and then scale the mean with the `scale` scalar.
    """
    n_splx = inv_star_k.size(0)
    mean = (inv_star_k @ weak_laplacian).tr / n_splx
    return scale * mean


class ShiftedUpPrecond(LOBPCGPreconditioner):
    r"""
    A mixed Hodge Laplacian LOBPCG preconditioner using the sparse up-component.

    This preconditioner computes the search direction by effectively applying
    $(S_k^\up + \tau M_k)^{-1}$ to the residual vector. Because the down-component
    is omitted, this is most useful in cases where the desired eigenmode is
    dominated by the up component.

    Here, `tau` controls the strength of regularization and has the same unit as
    the eigenvalues. When `tau` is not provided, it is computed as the approximate
    mean of the generalized eigenvalues of $S_k^\up$, scaled down by a factor of
    0.01; specifically, `star_k` is used to approximate $M_k^{-1}$ when `tau` is
    not provided.
    """

    def __init__(
        self,
        weak_up_laplacian: Float[SparseDecoupledTensor, "k_splx k_splx"],
        mass_k: Float[BaseDecoupledTensor, "k_splx k_splx"],
        star_k: Float[DiagDecoupledTensor, "k_splx k_splx"] | None,
        tau: float | Literal["auto"],
        n: int,
        nvmath_config: DirectSolverConfig,
    ):
        if not _HAS_NVMATH:
            raise ImportError("nvmath-python backend required.")

        if tau == "auto":
            tau = _mean_generalized_eigenvalues(weak_up_laplacian, star_k.inv)

        self.n = n

        # Solve a linear system with a channel dim of at most 3n size.
        b_dummy = to_col_major(
            torch.zeros(
                (weak_up_laplacian.size(-1), 3 * n),
                dtype=weak_up_laplacian.dtype,
                device=weak_up_laplacian.device,
            ),
            batch_first=False,
        )

        op = SparseDecoupledTensor.assemble(weak_up_laplacian, tau * mass_k)

        self.solver = _NVMathSparseSolver(
            op.values,
            op.pattern,
            b_dummy,
            matrix_type=nvmath_sp.DirectSolverMatrixType.SPD,
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


class ShiftedLumpedPrecond(LOBPCGPreconditioner):
    r"""
    A mixed Hodge Laplacian LOBPCG preconditioner using the mass-lumped down-component.

    It is recommended to compute `inv_star_km1` using the barycentric dual so
    that the resulting operator remains strictly SPD regardless of mesh quality.

    This preconditioner computes the search direction by effectively applying
    $(S_k' + \tau M_k)^{-1}$ to the residual vector, where $S_k'$ is a mass-lumped
    version of the weak Hodge Laplacian $S_k$ constructed by replacing $M_{k-1}^{-1}$
    with the inverse of the corresponding Hodge star. Here, `tau` controls the strength
    of regularization and has the same unit as the eigenvalues. When `tau` is not
    provided, it is computed as the approximate mean of the generalized eigenvalues
    of $S_k^\up$, scaled down by a factor of 0.01; specifically, `star_k` is used
    to approximate $M_k^{-1}$ when `tau` is not provided.
    """

    def __init__(
        self,
        cbd_km1: Float[SparseDecoupledTensor, "k_splx km1_splx"],
        cbd_k: Float[SparseDecoupledTensor, "kp1_splx k_splx"] | None,
        star_km1: Float[DiagDecoupledTensor, "km1_splx km1_splx"],
        mass_k: Float[BaseDecoupledTensor, "k_splx k_splx"],
        star_k: Float[DiagDecoupledTensor, "k_splx k_splx"] | None,
        mass_kp1: Float[BaseDecoupledTensor, "kp1_splx kp1_splx"] | None,
        tau: float | Literal["auto"],
        n: int,
        nvmath_config: DirectSolverConfig,
    ):
        if not _HAS_NVMATH:
            raise ImportError("nvmath-python backend required.")

        if (cbd_k is None) != (mass_kp1 is None):
            raise ValueError(
                "'cbd_k' and 'mass_kp1' must both be None or neither be None."
            )

        self.n = n

        weak_down_laplacian = mass_k @ cbd_km1 @ star_km1.inv @ cbd_km1.T @ mass_k

        if mass_kp1 is None:
            if tau == "auto":
                tau = _mean_generalized_eigenvalues(weak_down_laplacian, star_k.inv)

            op = SparseDecoupledTensor.assemble(weak_down_laplacian, tau * mass_k)

        else:
            weak_up_laplacian = cbd_k.T @ mass_kp1 @ cbd_k
            weak_laplacian = SparseDecoupledTensor.assemble(
                weak_down_laplacian, weak_up_laplacian
            )

            if tau == "auto":
                tau = _mean_generalized_eigenvalues(weak_laplacian, star_k.inv)

            op = SparseDecoupledTensor.assemble(weak_laplacian, tau * mass_k)

        # Solve a linear system with a channel dim of at most 3n size.
        b_dummy = to_col_major(
            torch.zeros(
                (op.size(-1), 3 * n),
                dtype=op.dtype,
                device=op.device,
            ),
            batch_first=False,
        )

        self.solver = _NVMathSparseSolver(
            op.values,
            op.pattern,
            b_dummy,
            matrix_type=nvmath_sp.DirectSolverMatrixType.SPD,
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

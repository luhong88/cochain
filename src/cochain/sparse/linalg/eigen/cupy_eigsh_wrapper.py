from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass, replace
from typing import TYPE_CHECKING, Any, Literal

import torch as t
from jaxtyping import Float, Integer

from ...operators import SparseOperator, SparseTopology
from ._backward import dLdA_backward

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

if TYPE_CHECKING:
    import cupyx.scipy.sparse.linalg as cp_sp_linalg

    from ..solvers.nvmath_wrapper import DirectSolverConfig


if _HAS_CUPY:

    @dataclass
    class CuPyEigshConfig:
        sigma: float | None = None
        which: Literal["LM", "LA", "SA"] = "LM"
        v0: (
            Float[t.Tensor | cp.ndarray, " c"]
            | Sequence[Float[t.Tensor | cp.ndarray, " c"] | None]
            | None
        ) = None
        ncv: int | None = None
        maxiter: int | None = None
        tol: float | int = 0

        def __post_init__(self):
            match self.v0:
                case t.Tensor():
                    self.v0 = cp.from_dlpack(self.v0.detach().contiguous())
                case Sequence():
                    v0_list = []
                    for v0 in self.v0:
                        match v0:
                            case t.Tensor():
                                v0_list.append(
                                    cp.from_dlpack(self.v0.detach().contiguous())
                                )
                            case _:
                                v0_list.append(v0)
                    self.v0 = v0_list

        def expand(self, n: int) -> list[CuPyEigshConfig]:
            config_list = []
            if isinstance(self.v0, list):
                config_list.append(replace(self, v0=v0) for v0 in self.v0)
                if len(config_list) != n:
                    raise ValueError("Inconsistent v0 specification.")
            else:
                config_list = [self] * n

            return config_list


class _CuPyEigshAutogradFunction(t.autograd.Function):
    @staticmethod
    def forward(
        A_val: Float[t.Tensor, " nnz"],
        A_sp_topo: Integer[SparseTopology, "r c"],
        k: int,
        eps: float | int,
        compute_eig_vecs: bool,
        cp_config: CuPyEigshConfig,
        nvmath_config: DirectSolverConfig,
    ) -> tuple[Float[t.Tensor, " k"], Float[t.Tensor, "c k"] | None]:
        from ._cupy_eigsh_operators import CuPyShiftInvSymOp, sp_op_comps_to_cp_csr

        # Force CuPy to use the current Pytorch stream.
        stream = t.cuda.current_stream()
        with cp.cuda.ExternalStream(stream.cuda_stream, stream.device_index):
            if cp_config.sigma is None:
                A_cp = sp_op_comps_to_cp_csr(A_val, A_sp_topo)
            else:
                A_cp = CuPyShiftInvSymOp(
                    A_val, A_sp_topo, cp_config.sigma, nvmath_config
                )

            results = cp_sp_linalg.eigsh(
                a=A_cp,
                k=k,
                return_eigenvectors=compute_eig_vecs,
                **asdict(cp_config),
            )

            if compute_eig_vecs:
                eig_vals_cp, eig_vecs_cp = results
                eig_vecs = t.from_dlpack(eig_vecs_cp)
            else:
                eig_vals_cp = results
                eig_vecs = None

            eig_vals = t.from_dlpack(eig_vals_cp)

        if cp_config.sigma is None:
            eig_vals_true = eig_vals
        else:
            # In shift-invert mode, λ_returned = 1 / (λ_true - σ)
            eig_vals_true = cp_config.sigma + (1.0 / eig_vals)

        return eig_vals_true, eig_vecs

    @staticmethod
    def setup_context(ctx, inputs, output):
        A_val, A_sp_topo, k, eps, compute_eig_vecs, cp_config, nvmath_config = inputs
        eig_vals, eig_vecs = output

        ctx.save_for_backward(eig_vals, eig_vecs)
        ctx.A_sp_topo = A_sp_topo
        ctx.k = k
        ctx.eps = eps

    @staticmethod
    def backward(
        ctx, dLdl: Float[t.Tensor, " k"], dLdv: Float[t.Tensor, "c k"] | None
    ) -> tuple[
        Float[t.Tensor, " nnz"] | None,
        None,
        None,
        None,
        None,
        None,
        None,
    ]:
        needs_grad_A_val = ctx.needs_input_grad[0]

        dLdA_val = dLdA_backward(ctx, dLdl, dLdv) if needs_grad_A_val else None

        return dLdA_val, None, None, None, None, None, None


def _cupy_eigsh_no_batch(
    A: Float[SparseOperator, "r c"],
    k: int,
    eps: float | int,
    compute_eig_vecs: bool,
    cp_config: CuPyEigshConfig,
    nvmath_config: DirectSolverConfig,
) -> tuple[Float[t.Tensor, " k"], Float[t.Tensor, "c k"]]:
    eig_vals, eig_vecs = _CuPyEigshAutogradFunction.apply(
        A.val, A.sp_topo, k, eps, compute_eig_vecs, cp_config, nvmath_config
    )

    return eig_vals, eig_vecs


def _cupy_eigsh_batch(
    A_batched: Float[SparseOperator, "r c"],
    k: int,
    eps: float | int,
    compute_eig_vecs: bool,
    cp_config_batched: CuPyEigshConfig,
    nvmath_config: DirectSolverConfig,
) -> tuple[Float[t.Tensor, "b k"], Float[t.Tensor, "c k"]]:
    A_list = A_batched.unpack_block_diag()

    cp_config_list = cp_config_batched.expand(n=len(A_list))

    eig_val_list = []
    eig_vec_list = []
    for A, cp_config in zip(A_list, cp_config_list):
        eig_val, eig_vec = _cupy_eigsh_no_batch(
            A, k, eps, compute_eig_vecs, cp_config, nvmath_config
        )
        eig_val_list.append(eig_val)
        eig_vec_list.append(eig_vec)

    eig_vals = t.vstack(eig_val_list)

    if eig_vec_list[0] is None:
        eig_vecs = None
    else:
        eig_vecs = t.vstack(eig_vec_list)

    return eig_vals, eig_vecs


def cupy_eigsh(
    A: Float[SparseOperator, "r c"],
    block_diag_batch: bool = False,
    k: int = 6,
    eps: float | int = 1e-6,
    return_eigenvectors: bool = True,
    cp_config: CuPyEigshConfig | None = None,
    nvmath_config: DirectSolverConfig | None = None,
) -> tuple[Float[t.Tensor, "*b k"], Float[t.Tensor, "c k"] | None]:
    """
    This function provides a differentiable wrapper for the GPU-based
    `cupyx.scipy.sparse.linalg.eigsh()` method.

    The API for `eigsh()` is almost reproduced one-to-one, with the following
    exceptions:

    * The arguments `which`, `v0`, `ncv`, `maxiter`, and `tol`, are collected in
      a `CuPyEigshConfig` dataclass object, whilc the rest of the arguments are
      exposed as direct arguments to this function.
    * The `A` matrix must be a `SparseOperator` object and will be converted to
      CuPy CSR matrix. The `v0` argument can be a torch tensor, but will be converted
      to a cupy array and copied. The use of CuPy `LinearOperator` objects for `A`
      is not supported.
    * The `eps` argument is used for Lorentzian broadening/regularization in the
      gradient calculation to prevent gradient explosion when the eigenvalues are
      (near) degenerate.

    Note that the CuPy `eigsh()` and SciPy `eigsh()` differ in some major aspects.

    * The CuPy `eigsh()` does not support generalized eigenvalue problems.
    * The CuPy `eigsh()` does not natively support the shift-invert mode. Here,
      the shift-invert mode is implemented by constructing a CuPy `LinearOperator`
      object for the L = inv(A - σI) matrix; computing the matrix-vector multiplication
      L@x = b is then equivalent to solving a sparse linear system with matrix
      A - σI and vector x. This approach corresonds to the `mode='normal'` option
      in SciPy `eigsh()`. Currently, the sparse solver is implemented using the
      `DirectSolver()` class of `nvmath-python`; the behavior of this solver can
      be changed via the `nvmath_config` argument.


    Note on performance:

    * If `A` requires gradient tracking, eigenvectors will be computed; in this
      case, if `return_eigenvectors=False`, the computed eigenvectors are not
      returned.
    * The `eigsh()` function does not natively support batching. if
      `block_diag_batch` is True, the `A` `SparseOperator` will be split into
      individual sparse matrices and solved sequentially. The resulting eigenvalue
      tensor will contain a leading batch dimension, but the resulting eigenvector
      tensor will respect the original concatenated/packed format. This requires
      that `A` has a valid `BlockDiagConfig` for unpacking the block diagonal
      batch structure.
    """
    if not (_HAS_CUPY and _HAS_NVMATH):
        raise ImportError("cupy and nvmath-python backends required.")

    from ..solvers.nvmath_wrapper import DirectSolverConfig

    # Eigenvectors are required for backward().
    compute_eig_vecs = return_eigenvectors
    if A.requires_grad:
        compute_eig_vecs = True

    if cp_config is None:
        cp_config = CuPyEigshConfig()
    if nvmath_config is None:
        nvmath_config = DirectSolverConfig()

    if block_diag_batch:
        eig_vals, eig_vecs = _cupy_eigsh_batch(
            A, k, eps, compute_eig_vecs, cp_config, nvmath_config
        )
    else:
        eig_vals, eig_vecs = _cupy_eigsh_no_batch(
            A, k, eps, compute_eig_vecs, cp_config, nvmath_config
        )

    if return_eigenvectors:
        return eig_vals, eig_vecs
    else:
        return eig_vals

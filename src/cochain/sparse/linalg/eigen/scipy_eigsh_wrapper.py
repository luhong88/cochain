from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Any, Literal, Sequence

import numpy as np
import scipy.sparse
import scipy.sparse.linalg
import torch as t
from jaxtyping import Float, Integer

from ...operators import SparseOperator, SparseTopology
from ._backward import dLdA_backward, dLdA_dLdM_backward


@dataclass
class SciPyEigshConfig:
    sigma: float | None = None
    which: Literal["LM", "SM", "LA", "SA", "BE"] = "LM"
    v0: (
        Float[t.Tensor | np.typing.NDArray, " c"]
        | Sequence[Float[t.Tensor | np.typing.NDArray, " c"] | None]
        | None
    ) = None
    ncv: int | None = None
    maxiter: int | None = None
    tol: float | int = 0
    mode: Literal["normal", "buckling", "cayley"] = "normal"

    def __post_init__(self):
        match self.v0:
            case t.Tensor():
                self.v0 = self.v0.detach().contiguous().cpu().numpy()
            case Sequence():
                v0_list = []
                for v0 in self.v0:
                    match v0:
                        case t.Tensor():
                            v0_list.append(v0.detach().contiguous().cpu().numpy())
                        case _:
                            v0_list.append(v0)
                self.v0 = v0_list

    def expand(self, n: int) -> list[SciPyEigshConfig]:
        config_list = []
        if isinstance(self.v0, list):
            config_list.append(replace(self, v0=v0) for v0 in self.v0)
            if len(config_list) != n:
                raise ValueError("Inconsistent v0 specification.")
        else:
            config_list = [self] * n

        return config_list


def _sp_op_comps_to_scipy_csr(
    val: Float[t.Tensor, " nnz"],
    sp_topo: Integer[SparseTopology, "r c"],
) -> Float[scipy.sparse.csr_array, "r c"]:
    sp_op_scipy = scipy.sparse.csr_array(
        (
            val.detach().contiguous().cpu().numpy(),
            sp_topo.idx_col_int32.detach().contiguous().cpu().numpy(),
            sp_topo.idx_crow_int32.detach().contiguous().cpu().numpy(),
        ),
        shape=sp_topo.shape,
    )
    return sp_op_scipy


def _sp_op_comps_to_scipy_csc(
    val: Float[t.Tensor, " nnz"],
    sp_topo: Integer[SparseTopology, "r c"],
) -> Float[scipy.sparse.csc_array, "r c"]:
    sp_op_scipy = scipy.sparse.csc_array(
        (
            val[sp_topo.coo_to_csc_perm].detach().contiguous().cpu().numpy(),
            sp_topo.idx_row_csc_int32.detach().contiguous().cpu().numpy(),
            sp_topo.idx_ccol_int32.detach().contiguous().cpu().numpy(),
        ),
        shape=sp_topo.shape,
    )
    return sp_op_scipy


class _SciPyEigshStandardAutogradFunction(t.autograd.Function):
    @staticmethod
    def forward(
        A_val: Float[t.Tensor, " nnz"],
        A_sp_topo: Integer[SparseTopology, "r c"],
        k: int,
        eps: float | int,
        compute_eig_vecs: bool,
        config: SciPyEigshConfig,
    ) -> tuple[Float[t.Tensor, " k"], Float[t.Tensor, "c k"] | None]:
        # When solving the standard A@x=λx, the CSR format is preferred for
        # matrix-vector multiplication.
        if config.sigma is None:
            A_scipy = _sp_op_comps_to_scipy_csr(A_val, A_sp_topo)

        # In the shift-invert mode, an LU factorization of A + σI is required,
        # and therefore the CSC format is preferred.
        else:
            A_scipy = _sp_op_comps_to_scipy_csc(A_val, A_sp_topo)

        results = scipy.sparse.linalg.eigsh(
            A=A_scipy,
            k=k,
            return_eigenvectors=compute_eig_vecs,
            **asdict(config),
        )

        if compute_eig_vecs:
            eig_vals_np, eig_vecs_np = results

            eig_vecs = t.from_numpy(eig_vecs_np).to(
                dtype=A_val.dtype, device=A_val.device
            )

        else:
            eig_vals_np = results
            eig_vecs = None

        eig_vals = t.from_numpy(eig_vals_np).to(dtype=A_val.dtype, device=A_val.device)

        return eig_vals, eig_vecs

    @staticmethod
    def setup_context(ctx, inputs, output):
        A_val, A_sp_topo, k, eps, compute_eig_vecs, config = inputs
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
    ]:
        needs_grad_A_val = ctx.needs_input_grad[0]

        dLdA_val = dLdA_backward(ctx, dLdl, dLdv) if needs_grad_A_val else None

        return dLdA_val, None, None, None, None, None


class _SciPyEigshGEPAutogradFunction(t.autograd.Function):
    @staticmethod
    def forward(
        A_val: Float[t.Tensor, " A_nnz"],
        A_sp_topo: Integer[SparseTopology, "r c"],
        M_val: Float[t.Tensor, " M_nnz"],
        M_sp_topo: Integer[SparseTopology, "r c"],
        k: int,
        eps: float | int,
        compute_eig_vecs: bool,
        config: SciPyEigshConfig,
    ) -> tuple[Float[t.Tensor, " k"], Float[t.Tensor, "c k"] | None]:
        # When solving the standard A@x=λx, the CSR format is preferred for
        # matrix-vector multiplication.
        if config.sigma is None:
            A_scipy = _sp_op_comps_to_scipy_csr(A_val, A_sp_topo)

        # In the shift-invert mode, an LU factorization of A + σI is required,
        # and therefore the CSC format is preferred.
        else:
            A_scipy = _sp_op_comps_to_scipy_csc(A_val, A_sp_topo)

        # M should always be in CSC format for LU factorization.
        M_scipy = _sp_op_comps_to_scipy_csc(M_val, M_sp_topo)

        results = scipy.sparse.linalg.eigsh(
            A=A_scipy,
            k=k,
            M=M_scipy,
            return_eigenvectors=compute_eig_vecs,
            **asdict(config),
        )

        if compute_eig_vecs:
            eig_vals_np, eig_vecs_np = results

            eig_vecs = t.from_numpy(eig_vecs_np).to(
                dtype=A_val.dtype, device=A_val.device
            )

        else:
            eig_vals_np = results
            eig_vecs = None

        eig_vals = t.from_numpy(eig_vals_np).to(dtype=A_val.dtype, device=A_val.device)

        return eig_vals, eig_vecs

    @staticmethod
    def setup_context(ctx, inputs, output):
        A_val, A_sp_topo, M_val, M_sp_topo, k, eps, compute_eig_vecs, config = inputs
        eig_vals, eig_vecs = output

        ctx.save_for_backward(eig_vals, eig_vecs)
        ctx.A_sp_topo = A_sp_topo
        ctx.M_sp_topo = M_sp_topo
        ctx.k = k
        ctx.eps = eps

    @staticmethod
    def backward(
        ctx, dLdl: Float[t.Tensor, " k"], dLdv: Float[t.Tensor, "c k"] | None
    ) -> tuple[
        Float[t.Tensor, " A_nnz"] | None,
        None,
        Float[t.Tensor, " M_nnz"] | None,
        None,
        None,
        None,
        None,
        None,
    ]:
        needs_grad_A_val = ctx.needs_input_grad[0]
        needs_grad_M_val = ctx.needs_input_grad[2]

        dLdA_val, dLdM_val = dLdA_dLdM_backward(
            ctx, dLdl, dLdv, needs_grad_A_val, needs_grad_M_val
        )

        return dLdA_val, None, dLdM_val, None, None, None, None, None


def _scipy_eigsh_no_batch(
    A: Float[SparseOperator, "r c"],
    M: Float[SparseOperator, "r c"] | None,
    config: SciPyEigshConfig,
    kwargs: dict[str, Any],
) -> tuple[Float[t.Tensor, " k"], Float[t.Tensor, "c k"]]:
    if M is None:
        eig_vals, eig_vecs = _SciPyEigshStandardAutogradFunction.apply(
            A.val, A.sp_topo, config=config, **kwargs
        )
    else:
        eig_vals, eig_vecs = _SciPyEigshGEPAutogradFunction.apply(
            A.val, A.sp_topo, M.val, M.sp_topo, config=config, **kwargs
        )

    return eig_vals, eig_vecs


def _scipy_eigsh_batch(
    A_batched: Float[SparseOperator, "r c"],
    M_batched: Float[SparseOperator, "r c"] | None,
    config_batched: SciPyEigshConfig,
    kwargs: dict[str, Any],
) -> tuple[Float[t.Tensor, "b k"], Float[t.Tensor, "c k"]]:
    # Unpack the A, M (if not None), and v0 (if not None) for looping
    A_list = A_batched.unpack_block_diag()
    if M_batched is None:
        M_list = [None] * len(A_list)
    else:
        M_list = M_batched.unpack_block_diag()

    config_list = config_batched.expand(n=len(A_list))

    eig_val_list = []
    eig_vec_list = []
    for A, M, config in zip(A_list, M_list, config_list, strict=True):
        eig_val, eig_vec = _scipy_eigsh_no_batch(A, M, config, kwargs)
        eig_val_list.append(eig_val)
        eig_vec_list.append(eig_vec)

    eig_vals = t.vstack(eig_val_list)

    if eig_vec_list[0] is None:
        eig_vecs = None
    else:
        eig_vecs = t.vstack(eig_vec_list)

    return eig_vals, eig_vecs


def scipy_eigsh(
    A: Float[SparseOperator, "r c"],
    M: Float[SparseOperator, "r c"] | None = None,
    block_diag_batch: bool = False,
    k: int = 6,
    eps: float | int = 1e-6,
    return_eigenvectors: bool = True,
    config: SciPyEigshConfig | None = None,
) -> tuple[Float[t.Tensor, "*b k"], Float[t.Tensor, "c k"] | None]:
    """
    This function provides a differentiable wrapper for the CPU-based
    `scipy.sparse.linalg.eigsh()` method.

    The API for `eigsh()` is almost reproduced one-to-one, with the following
    exceptions:

    * The arguments `sigma`, `which`, `v0`, `ncv`, `maxiter`, `tol`, and `mode`
      are collected in a `ScipyEigshConfig` dataclass object, whilc the rest of
      the arguments are exposed as direct arguments to this function.
    * The `A` and `M` matrices must be `SparseOperator` objects and will be
      converted to SciPy CSR/CSC arrays and copied to CPU. The `v0` argument
      can be a torch tensor, but will be converted to a numpy array and copied
      to CPU. The use of SciPy `LinearOperator` objects for `A` and `M` is not
      supported.
    * The `Minv` and `OPinv` arguments are not supported.
    * The `eps` argument is used for Lorentzian broadening/regularization in the
      gradient calculation to prevent gradient explosion when the eigenvalues are
      (near) degenerate.

    Note on performance:

    * If either `A` or `M` requires gradient tracking, eigenvectors will be
      computed; in this case, if `return_eigenvectors=False`, the computed
      eigenvectors are not returned.
    * The `eigsh()` function does not natively support batching. if
      `block_diag_batch` is True, the `A` `SparseOperator` (and `M` if not `None`)
      will be split into individual sparse matrices and solved sequentially. The
      resulting eigenvalue tensor will contain a leading batch dimension, but the
      resulting eigenvector tensor will respect the original concatenated/packed
      format. This requires that `A` (and `M` if not `None`) has a valid
      `BlockDiagConfig` for unpacking the block diagonal batch structure.

    Note on shift-invert mode: for finding the lowest non-zero eigenvalues of a
    positive semi-definite matrix, one can either use `which='SM'` or `which='LM'`
    with a `sigma` close to zero. While the latter approach typically lead to
    faster convergence, it has higher memory requirement due to the need to factorize
    the matrix.
    """
    # Eigenvectors are required for backward().
    compute_eig_vecs = return_eigenvectors
    if A.requires_grad:
        compute_eig_vecs = True
    if (M is not None) and M.requires_grad:
        compute_eig_vecs = True

    if config is None:
        config = SciPyEigshConfig()

    kwargs = {
        "k": k,
        "eps": eps,
        "compute_eig_vecs": compute_eig_vecs,
    }

    if block_diag_batch:
        eig_vals, eig_vecs = _scipy_eigsh_batch(A, M, config, kwargs)
    else:
        eig_vals, eig_vecs = _scipy_eigsh_no_batch(A, M, config, kwargs)

    if return_eigenvectors:
        return eig_vals, eig_vecs
    else:
        return eig_vals

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Literal

import scipy.sparse
import scipy.sparse.linalg
import torch as t
from jaxtyping import Float, Integer

try:
    import cupy as cp
    import cupyx.scipy.sparse as cp_sp
    import cupyx.scipy.sparse.linalg as cp_sp_linalg

    _HAS_CUPY = True

except ImportError:
    _HAS_CUPY = False

if TYPE_CHECKING:
    import cupyx.scipy.sparse as cp_sp
    import cupyx.scipy.sparse.linalg as cp_sp_linalg


def _csc_torch_to_cupy(torch_sp_csc: t.Tensor) -> cp_sp.csc_matrix:
    src = torch_sp_csc.detach()

    shape = tuple(src.shape)

    # Note that cupy will force downcasting from int64 to int32.
    ccol_idx = cp.from_dlpack(src.ccol_indices())
    row_idx = cp.from_dlpack(src.row_indices())

    val = cp.from_dlpack(src.values())

    cupy_sp_csc = cp_sp.csc_matrix((val, row_idx, ccol_idx), shape=shape)

    return cupy_sp_csc


class _CuPySuperLUWrapper(t.autograd.Function):
    @staticmethod
    def forward(
        A: Float[t.Tensor, "r c"],
        A_coo_idx: Integer[t.LongTensor, "2 nnz"],
        b: Float[t.Tensor, " r"],
        splu_kwargs: dict[str, Any],
    ) -> tuple[Float[t.Tensor, " c"], cp_sp_linalg.SuperLU]:
        # A must be 1) in sparse csc format and 2) moved to the cuda device
        A_cp = _csc_torch_to_cupy(A)
        solver = cp_sp_linalg.splu(A_cp, **splu_kwargs)
        x = t.from_dlpack(solver.solve(cp.from_dlpack(b.detach()), trans="N"))
        return x, solver

    @staticmethod
    def setup_context(ctx, inputs, output):
        A, A_coo_idx, b, splu_kwargs = inputs

        x, solver = output
        ctx.mark_non_differentiable(solver)

        ctx.save_for_backward(A_coo_idx, x)
        ctx.solver = solver
        ctx.A_shape = tuple(A.shape)

    @staticmethod
    def backward(
        ctx, dLdx: Float[t.Tensor, " c"], _
    ) -> tuple[Float[t.Tensor, "r c"] | None, None, Float[t.Tensor, " r"] | None, None]:
        """
        Let x be the solution to the linear system A@x = b. To find the gradient
        of some loss L wrt A and b, the adjoint method first defines a Lagrangian
        function F = L - lambda.T@(A@x - b), where lambda is the Lagrangian multiplier.
        By definition, F and L (as well as their total differentials) must be
        equal whenver A@x = b. From this condition, it follows that dLdb is given
        by lambda, and dLdA is given by the outer product -lambda@x.T.
        """
        needs_grad_A, _, needs_grad_b, _ = ctx.needs_input_grad

        if not (needs_grad_A or needs_grad_b):
            return (None, None, None, None)

        A_coo_idx, x = ctx.saved_tensors

        lambda_cp: Float[t.Tensor, " r"] = ctx.solver.solve(
            cp.from_dlpack(dLdx.detach()), trans="T"
        )
        lambda_ = t.from_dlpack(lambda_cp)

        if needs_grad_b and not needs_grad_A:
            return (None, None, lambda_, None)

        # dLdA will have the same sparsity pattern as A
        dLdA_val = -lambda_[A_coo_idx[0]] * x[A_coo_idx[1]]

        # The gradient for A will be in the uncoalesced COO format.
        dLdA = t.sparse_coo_tensor(A_coo_idx, dLdA_val, ctx.A_shape)

        if needs_grad_A and not needs_grad_b:
            return (dLdA, None, None, None)

        if needs_grad_A and needs_grad_b:
            return (dLdA, None, lambda_, None)


def _csc_torch_to_scipy(torch_sp_csc: t.Tensor) -> scipy.sparse.csc_array:
    src = torch_sp_csc.detach().cpu()

    shape = tuple(src.shape)

    ccol_idx = src.ccol_indices().numpy()
    row_idx = src.row_indices().numpy()

    val = src.values().numpy()

    scipy_sp_csc = scipy.sparse.csc_array((val, row_idx, ccol_idx), shape=shape)

    return scipy_sp_csc


class _SciPySuperLUWrapper(t.autograd.Function):
    @staticmethod
    def forward(
        A: Float[t.Tensor, "r c"],
        A_coo_idx: Integer[t.LongTensor, "2 nnz"],
        b: Float[t.Tensor, " r"],
        splu_kwargs: dict[str, Any],
    ) -> tuple[Float[t.Tensor, " c"], cp_sp_linalg.SuperLU]:
        A_scipy = _csc_torch_to_scipy(A)
        solver = scipy.sparse.linalg.splu(A_scipy, **splu_kwargs)
        x = t.from_numpy(solver.solve(b.detach().cpu().numpy(), trans="N")).to(
            dtype=A.dtype, device=A.device
        )
        return x, solver

    @staticmethod
    def setup_context(ctx, inputs, output):
        A, A_coo_idx, b, splu_kwargs = inputs

        x, solver = output
        ctx.mark_non_differentiable(solver)

        ctx.save_for_backward(A_coo_idx, x)
        ctx.solver = solver
        ctx.A_shape = tuple(A.shape)

    @staticmethod
    def backward(
        ctx, dLdx: Float[t.Tensor, " c"], _
    ) -> tuple[Float[t.Tensor, "r c"] | None, None, Float[t.Tensor, " r"] | None, None]:
        needs_grad_A, _, needs_grad_b, _ = ctx.needs_input_grad

        if not (needs_grad_A or needs_grad_b):
            return (None, None, None, None)

        A_coo_idx, x = ctx.saved_tensors

        lambda_scipy: Float[t.Tensor, " r"] = ctx.solver.solve(
            dLdx.detach().cpu().numpy(), trans="T"
        )
        lambda_ = t.from_numpy(lambda_scipy).to(dtype=dLdx.dtype, device=dLdx.device)

        if needs_grad_b and not needs_grad_A:
            return (None, None, lambda_, None)

        # dLdA will have the same sparsity pattern as A
        dLdA_val = -lambda_[A_coo_idx[0]] * x[A_coo_idx[1]]

        # The gradient for A will be in the uncoalesced COO format.
        dLdA = t.sparse_coo_tensor(A_coo_idx, dLdA_val, ctx.A_shape)

        if needs_grad_A and not needs_grad_b:
            return (dLdA, None, None, None)

        if needs_grad_A and needs_grad_b:
            return (dLdA, None, lambda_, None)


def solve(
    A: Float[t.Tensor, "r c"],
    A_coo_idx: Integer[t.LongTensor, "2 nnz"],
    b: Float[t.Tensor, " r"],
    method: Literal["splu_cupy", "splu_scipy", "nvmath", "cg", "minres"],
    **kwargs,
) -> Float[t.Tensor, " c"]:
    """
    This function provides a differentiable wrapper for sparse linear solvers.

    Here, A is a sparse tensor/matrix and b is a dense, 1D tensor. The A_coo_idx
    is required for the backward pass to enforce the sparsity pattern of A.

    If method is 'splu_cupy', solve the linear system using CuPy's SuperLU wrapper
    This requires that `A` and `b` must be on the CUDA device and that `A` is
    already in sparse CSC format. Note that the factorization step itself is not
    accelerated on GPU, which necessitates data transfer with the host.

    If method is 'splu_scipy', solve the linear system using SciPy's SuperLU
    wrapper. This requires that `A` is already in sparse CSC format; `A` and `b`
    will be copied to CPU.
    """
    match method:
        case "splu_cupy":
            if not _HAS_CUPY:
                raise ImportError("CuPy backend required for method 'splu_cupy'.")

            x, solver = _CuPySuperLUWrapper.apply(A, A_coo_idx, b, kwargs)
            return x

        case "splu_scipy":
            x, solver = _SciPySuperLUWrapper.apply(A, A_coo_idx, b, kwargs)

        case _:
            raise NotImplementedError()

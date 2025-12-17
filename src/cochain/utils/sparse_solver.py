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


class _CuPySuperLUWrapper(t.autograd.Function):
    @staticmethod
    def forward(
        A_val: Float[t.Tensor, " nnz"],
        A_coo_idx: Integer[t.LongTensor, "2 nnz"],
        A_shape: tuple[int, int],
        b: Float[t.Tensor, " r"],
        splu_kwargs: dict[str, Any],
    ) -> tuple[Float[t.Tensor, " c"], cp_sp_linalg.SuperLU]:
        # A must be 1) in sparse csc format and 2) moved to the cuda device
        A_cp: Float[cp_sp.csc_matrix, "r c"] = cp_sp.coo_matrix(
            (
                cp.from_dlpack(A_val.detach()),
                (
                    cp.from_dlpack(A_coo_idx.detach()[0]),
                    cp.from_dlpack(A_coo_idx.detach()[1]),
                ),
            ),
            shape=A_shape,
        ).tocsc()
        solver = cp_sp_linalg.splu(A_cp, **splu_kwargs)
        x = t.from_dlpack(solver.solve(cp.from_dlpack(b.detach()), trans="N"))
        return x, solver

    @staticmethod
    def setup_context(ctx, inputs, output):
        A_val, A_coo_idx, A_shape, b, splu_kwargs = inputs

        x, solver = output
        ctx.mark_non_differentiable(solver)

        ctx.save_for_backward(A_coo_idx, x)
        ctx.solver = solver
        ctx.A_shape = A_shape

    @staticmethod
    def backward(
        ctx, dLdx: Float[t.Tensor, " c"], _
    ) -> tuple[
        Float[t.Tensor, " nnz"] | None, None, None, Float[t.Tensor, " r"] | None, None
    ]:
        """
        Let x be the solution to the linear system A@x = b. To find the gradient
        of some loss L wrt A and b, the adjoint method first defines a Lagrangian
        function F = L - lambda.T@(A@x - b), where lambda is the Lagrangian multiplier.
        By definition, F and L (as well as their total differentials) must be
        equal whenver A@x = b. From this condition, it follows that dLdb is given
        by lambda, and dLdA is given by the outer product -lambda@x.T.
        """
        needs_grad_A_val = ctx.needs_input_grad[0]
        needs_grad_b = ctx.needs_input_grad[-2]

        if not (needs_grad_A_val or needs_grad_b):
            return (None,) * 5

        A_coo_idx, x = ctx.saved_tensors

        lambda_cp: Float[t.Tensor, " r"] = ctx.solver.solve(
            cp.from_dlpack(dLdx.detach()), trans="T"
        )
        lambda_ = t.from_dlpack(lambda_cp)

        if needs_grad_b and not needs_grad_A_val:
            return (None, None, None, lambda_, None)

        # dLdA will have the same sparsity pattern as A
        dLdA_val = -lambda_[A_coo_idx[0]] * x[A_coo_idx[1]]

        if needs_grad_A_val and not needs_grad_b:
            return (dLdA_val, None, None, None, None)

        if needs_grad_A_val and needs_grad_b:
            return (dLdA_val, None, None, lambda_, None)


class _SciPySuperLUWrapper(t.autograd.Function):
    @staticmethod
    def forward(
        A_val: Float[t.Tensor, " nnz"],
        A_coo_idx: Integer[t.LongTensor, "2 nnz"],
        A_shape: tuple[int, int],
        b: Float[t.Tensor, " r"],
        splu_kwargs: dict[str, Any],
    ) -> tuple[Float[t.Tensor, " c"], scipy.sparse.linalg.SuperLU]:
        A_scipy: Float[scipy.sparse.csc_array, "r c"] = scipy.sparse.coo_array(
            (
                A_val.detach().cpu().numpy(),
                A_coo_idx.detach().cpu().numpy(),
            ),
            shape=A_shape,
        ).tocsc()
        solver = scipy.sparse.linalg.splu(A_scipy, **splu_kwargs)
        x = t.from_numpy(solver.solve(b.detach().cpu().numpy(), trans="N")).to(
            dtype=A_val.dtype, device=A_val.device
        )
        return x, solver

    @staticmethod
    def setup_context(ctx, inputs, output):
        A_val, A_coo_idx, A_shape, b, splu_kwargs = inputs

        x, solver = output
        ctx.mark_non_differentiable(solver)

        ctx.save_for_backward(A_coo_idx, x)
        ctx.solver = solver
        ctx.A_shape = A_shape

    @staticmethod
    def backward(
        ctx, dLdx: Float[t.Tensor, " c"], _
    ) -> tuple[
        Float[t.Tensor, " nnz"] | None,
        None,
        None,
        Float[t.Tensor, " r"] | None,
        None,
    ]:
        needs_grad_A_val = ctx.needs_input_grad[0]
        needs_grad_b = ctx.needs_input_grad[-2]

        if not (needs_grad_A_val or needs_grad_b):
            return (None,) * 5

        A_coo_idx, x = ctx.saved_tensors

        lambda_scipy: Float[t.Tensor, " r"] = ctx.solver.solve(
            dLdx.detach().cpu().numpy(), trans="T"
        )
        lambda_ = t.from_numpy(lambda_scipy).to(dtype=dLdx.dtype, device=dLdx.device)

        if needs_grad_b and not needs_grad_A_val:
            return (None, None, None, lambda_, None)

        # dLdA will have the same sparsity pattern as A
        dLdA_val = -lambda_[A_coo_idx[0]] * x[A_coo_idx[1]]

        if needs_grad_A_val and not needs_grad_b:
            return (dLdA_val, None, None, None, None)

        if needs_grad_A_val and needs_grad_b:
            return (dLdA_val, None, None, lambda_, None)


def solve(
    A: Float[t.Tensor, "r c"],
    b: Float[t.Tensor, " r"],
    method: Literal["splu_cupy", "splu_scipy", "nvmath", "cg", "minres"],
    **kwargs,
) -> Float[t.Tensor, " c"]:
    """
    This function provides a differentiable wrapper for sparse linear solvers.

    Here, A is a sparse coo tensor and b is a dense, 1D tensor.

    If method is 'splu_cupy', solve the linear system using CuPy's SuperLU wrapper
    This requires that `A` and `b` must be on the CUDA device. Note that the
    factorization step itself is not accelerated on GPU, which necessitates data
    transfer with the host.

    If method is 'splu_scipy', solve the linear system using SciPy's SuperLU
    wrapper. This requires that `A` is in the sparse COO format (which is converted
    to sparse CSC format in SciPy for the solver); `A` and `b` will be copied to CPU.
    """
    match method:
        case "splu_cupy":
            if not _HAS_CUPY:
                raise ImportError("CuPy backend required for method 'splu_cupy'.")

            x, solver = _CuPySuperLUWrapper.apply(A.values(), A.indices(), b, kwargs)
            return x

        case "splu_scipy":
            x, solver = _SciPySuperLUWrapper.apply(A.values(), A.indices(), b, kwargs)
            return x

        case _:
            raise NotImplementedError()

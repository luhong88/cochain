from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import numpy as np
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
        b: Float[t.Tensor, " r *ch"],
        splu_kwargs: dict[str, Any],
    ) -> tuple[Float[t.Tensor, " c *ch"], cp_sp_linalg.SuperLU]:
        val = A_val.detach().contiguous()
        idx = A_coo_idx.detach().to(dtype=t.int32).contiguous()

        # Force CuPy to use the current Pytorch stream.
        stream = t.cuda.current_stream()
        with cp.cuda.ExternalStream(stream.cuda_stream, stream.device):
            A_cp: Float[cp_sp.csc_matrix, "r c"] = cp_sp.coo_matrix(
                (
                    cp.from_dlpack(val),
                    (
                        cp.from_dlpack(idx[0]),
                        cp.from_dlpack(idx[1]),
                    ),
                ),
                shape=A_shape,
            ).tocsc()
            x_cp = cp.from_dlpack(b.detach().contiguous())

            solver = cp_sp_linalg.splu(A_cp, **splu_kwargs)
            x = t.from_dlpack(solver.solve(x_cp, trans="N"))

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
        ctx, dLdx: Float[t.Tensor, " c *ch"], _
    ) -> tuple[
        Float[t.Tensor, " nnz"] | None,
        None,
        None,
        Float[t.Tensor, " r *ch"] | None,
        None,
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

        if ctx.solver is None:
            raise RuntimeError(
                "Solver was released. Calling backward() twice with retain_graph=True "
                + "is currently not supported."
            )
        else:
            solver: cp_sp_linalg.SuperLU = ctx.solver

        stream = t.cuda.current_stream()
        with cp.cuda.ExternalStream(stream.cuda_stream, stream.device):
            lambda_cp: Float[t.Tensor, " r"] = solver.solve(
                cp.from_dlpack(dLdx.detach().contiguous()), trans="T"
            )
            lambda_ = t.from_dlpack(lambda_cp)

        # Free up solver memory usage.
        ctx.solver = None

        if needs_grad_b and not needs_grad_A_val:
            return (None, None, None, lambda_, None)

        # dLdA will have the same sparsity pattern as A.
        if lambda_.dim() > 1:
            # If there is a channel dimension, sum over it.
            dLdA_val = t.sum(-lambda_[A_coo_idx[0]] * x[A_coo_idx[1]], dim=-1)
        else:
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
        b: Float[t.Tensor, " r *ch"],
        splu_kwargs: dict[str, Any],
    ) -> tuple[Float[t.Tensor, " c *ch"], scipy.sparse.linalg.SuperLU]:
        A_scipy: Float[scipy.sparse.csc_array, "r c"] = scipy.sparse.coo_array(
            (
                A_val.detach().contiguous().cpu().numpy(),
                A_coo_idx.detach().contiguous().cpu().numpy().astype(np.int32),
            ),
            shape=A_shape,
        ).tocsc()
        x_np = b.detach().contiguous().cpu().numpy()

        solver = scipy.sparse.linalg.splu(A_scipy, **splu_kwargs)
        x = t.from_numpy(solver.solve(x_np, trans="N")).to(
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
        ctx, dLdx: Float[t.Tensor, " c *ch"], _
    ) -> tuple[
        Float[t.Tensor, " nnz"] | None,
        None,
        None,
        Float[t.Tensor, " r *ch"] | None,
        None,
    ]:
        needs_grad_A_val = ctx.needs_input_grad[0]
        needs_grad_b = ctx.needs_input_grad[-2]

        if not (needs_grad_A_val or needs_grad_b):
            return (None,) * 5

        A_coo_idx, x = ctx.saved_tensors

        if ctx.solver is None:
            raise RuntimeError(
                "Solver was released. Calling backward() twice with retain_graph=True "
                + "is currently not supported."
            )
        else:
            solver: scipy.sparse.linalg.SuperLU = ctx.solver

        lambda_np = solver.solve(dLdx.detach().contiguous().cpu().numpy(), trans="T")
        lambda_: Float[t.Tensor, " r *ch"] = t.from_numpy(lambda_np).to(
            dtype=dLdx.dtype, device=dLdx.device
        )

        # Free up solver memory usage.
        ctx.solver = None

        if needs_grad_b and not needs_grad_A_val:
            return (None, None, None, lambda_, None)

        # dLdA will have the same sparsity pattern as A.
        if lambda_.ndim > 1:
            # If there is a channel dimension, sum over it.
            dLdA_val = t.sum(-lambda_[A_coo_idx[0]] * x[A_coo_idx[1]], dim=-1)
        else:
            dLdA_val = -lambda_[A_coo_idx[0]] * x[A_coo_idx[1]]

        if needs_grad_A_val and not needs_grad_b:
            return (dLdA_val, None, None, None, None)

        if needs_grad_A_val and needs_grad_b:
            return (dLdA_val, None, None, lambda_, None)


def splu(
    A: Float[t.Tensor, "r c"],
    b: Float[t.Tensor, "*ch r"],
    *,
    backend: Literal["cupy", "scipy"],
    channel_first: bool = True,
    **splu_kwargs,
) -> Float[t.Tensor, "*ch c"]:
    """
    This function provides a differentiable wrapper for SuperLU.

    Here, A is a sparse coo tensor and b is a dense tensor with optional channel
    dimensions. If `channel_first` is `True`, all but the last dimension of `b`
    is treated as channel dimensions; if `channel_first` is `True`, all but the first
    dimension of `b` is treated as channel dimensions.

    If backend is 'cupy', `A` and `b` must be on the CUDA device. If backend is
    'scipy', `A` and `b` will be copied to CPU.

    If either `A` or `b` requires gradient, then a `SuperLU` solver object will be
    cached in memory to accelerate the backward pass; this memory will not be
    cleaned up until one of the following conditions is met:

    1) a backward() call with `retain_graph=False` has been made through the
    computation graph containing `A` or `b`, or
    2) all references to the output tensor (and its `grad_fn` and `ctx` attributes)
    from this function (and any derived tensors thereof) has gone out of scope/been
    detached from the computation graph.

    This function currently has the following limitations:

    * Double backward through this function is currently not supported.
    * The sparse indices of `A` will be downcasted to `int32` for compatibility with
    the solver; this necessitates copying of the index tensor.
    * SuperLU handles the factorization step on the host CPU, regardless of the
    location of `A` and `b`.
    * The SuperLU solver supports batching/channel dimensions of the `b` tensor,
    but there can only be one channel dimension, and the channel dimension must
    be the last dimension (i.e., `b` is either of shape `(r,)` or `(r, ch)`).
    Therefore, if the input `b` tensor does not conform to this layout, the function
    will have to create a reshaped and memory-contiguous copy of `b`. Batching of
    the `A` tensor is not supported by the solver.
    """
    requires_reshape = channel_first and b.ndim > 1

    if requires_reshape:
        # (*ch, r) -> (r, *ch_flat)
        b_ready = t.movedim(b, -1, 0).reshape(b.shape[-1], -1).contiguous()
    else:
        b_ready = b

    match backend:
        case "cupy":
            if not _HAS_CUPY:
                raise ImportError("CuPy backend required.")

            x, solver = _CuPySuperLUWrapper.apply(
                A.values(), A.indices(), A.shape, b_ready, splu_kwargs
            )

        case "scipy":
            x, solver = _SciPySuperLUWrapper.apply(
                A.values(), A.indices(), A.shape, b_ready, splu_kwargs
            )

        case _:
            raise ValueError()

    if requires_reshape:
        # (c, *ch_flat) -> (*ch, c)
        x_reshaped = x.transpose(0, 1).reshape(*b.shape[:-1], -1)
    else:
        x_reshaped = x

    return x_reshaped

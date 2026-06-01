from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import scipy.sparse
import scipy.sparse.linalg
import torch
from jaxtyping import Float, Integer
from torch import Tensor
from torch.autograd.function import once_differentiable

from .....utils.parsing import to_np
from ....decoupled_tensor import SparseDecoupledTensor, SparsityPattern

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


class _CuPySuperLUAutogradFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        a_val: Float[Tensor, " nz"],
        a_pattern: Integer[SparsityPattern, "r c"],
        b: Float[Tensor, " r *ch"],
        splu_kwargs: dict[str, Any],
    ) -> tuple[Float[Tensor, " c *ch"], cp_sp_linalg.SuperLU]:
        val = a_val[a_pattern.csc_to_coo_map].detach().contiguous()
        idx_ccol = a_pattern.idx_ccol.detach().contiguous()
        idx_row = a_pattern.idx_row_csc.detach().contiguous()

        # Force CuPy to use the current Pytorch stream.
        stream = torch.cuda.current_stream()
        with cp.cuda.ExternalStream(stream.cuda_stream, stream.device_index):
            A_cp: Float[cp_sp.csc_matrix, "r c"] = cp_sp.csc_matrix(
                (
                    cp.from_dlpack(val),
                    cp.from_dlpack(idx_row),
                    cp.from_dlpack(idx_ccol),
                ),
                shape=tuple(a_pattern.shape),
            )
            b_cp = cp.from_dlpack(b.detach().contiguous())

            solver = cp_sp_linalg.splu(A_cp, **splu_kwargs)
            x = torch.from_dlpack(solver.solve(b_cp, trans="N"))

        return x, solver

    @staticmethod
    def setup_context(ctx, inputs, output):
        a_val, a_pattern, b, splu_kwargs = inputs

        x, solver = output

        ctx.save_for_backward(x)
        ctx.solver = solver
        ctx.a_pattern = a_pattern

    @staticmethod
    @once_differentiable
    def backward(
        ctx, dLdx: Float[Tensor, " c *ch"], _
    ) -> tuple[
        Float[Tensor, " nz"] | None,
        None,
        Float[Tensor, " r *ch"] | None,
        None,
    ]:
        # Let x be the solution to the linear system A@x = b. To find the gradient
        # of some loss L wrt A and b, the adjoint method first defines a Lagrangian
        # function F = L - lambda.T@(A@x - b), where lambda is the Lagrangian multiplier.
        # By definition, F and L (as well as their total differentials) must be
        # equal whenever A@x = b. From this condition, it follows that dLdb is given
        # by lambda, and dLdA is given by the outer product -lambda@x.T.

        needs_grad_A_val = ctx.needs_input_grad[0]
        needs_grad_b = ctx.needs_input_grad[2]

        if not (needs_grad_A_val or needs_grad_b):
            return (None,) * 4

        (x,) = ctx.saved_tensors
        a_pattern: SparsityPattern = ctx.a_pattern

        if ctx.solver is None:
            raise RuntimeError(
                "Solver was released. Calling backward() twice with retain_graph=True "
                + "is currently not supported."
            )
        else:
            solver: cp_sp_linalg.SuperLU = ctx.solver

        stream = torch.cuda.current_stream()
        with cp.cuda.ExternalStream(stream.cuda_stream, stream.device_index):
            lambda_cp: Float[Tensor, " r"] = solver.solve(
                cp.from_dlpack(dLdx.detach().contiguous()), trans="T"
            )
            lambda_ = torch.from_dlpack(lambda_cp)

        # Free up solver memory usage.
        ctx.solver = None

        if needs_grad_b and not needs_grad_A_val:
            return (None, None, lambda_, None)

        # dLdA will have the same sparsity pattern as A.
        if lambda_.dim() > 1:
            # If there is a channel dimension, sum over it.
            dLdA_val = torch.sum(
                -lambda_[a_pattern.idx_coo[0]] * x[a_pattern.idx_coo[1]], dim=-1
            )
        else:
            dLdA_val = -lambda_[a_pattern.idx_coo[0]] * x[a_pattern.idx_coo[1]]

        if needs_grad_A_val and not needs_grad_b:
            return (dLdA_val, None, None, None)

        if needs_grad_A_val and needs_grad_b:
            return (dLdA_val, None, lambda_, None)


class _SciPySuperLUAutogradFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        a_val: Float[Tensor, " nz"],
        a_pattern: Integer[SparsityPattern, "r c"],
        b: Float[Tensor, " r *ch"],
        splu_kwargs: dict[str, Any],
    ) -> tuple[Float[Tensor, " c *ch"], scipy.sparse.linalg.SuperLU]:
        val = to_np(a_val[a_pattern.csc_to_coo_map], contiguous=True)
        idx_ccol = to_np(a_pattern.idx_ccol, contiguous=True)
        idx_row = to_np(a_pattern.idx_row_csc, contiguous=True)

        a_scipy: Float[scipy.sparse.csc_array, "r c"] = scipy.sparse.csc_array(
            (val, idx_row, idx_ccol),
            shape=a_pattern.shape,
        )
        b_np = to_np(b, contiguous=True)

        solver = scipy.sparse.linalg.splu(a_scipy, **splu_kwargs)
        x = torch.from_numpy(solver.solve(b_np, trans="N")).to(
            dtype=a_val.dtype, device=a_val.device
        )

        return x, solver

    @staticmethod
    def setup_context(ctx, inputs, output):
        a_val, a_pattern, b, splu_kwargs = inputs

        x, solver = output

        ctx.save_for_backward(x)
        ctx.solver = solver
        ctx.a_pattern = a_pattern

    @staticmethod
    @once_differentiable
    def backward(
        ctx, dLdx: Float[Tensor, " c *ch"], _
    ) -> tuple[
        Float[Tensor, " nz"] | None,
        None,
        Float[Tensor, " r *ch"] | None,
        None,
    ]:
        needs_grad_A_val = ctx.needs_input_grad[0]
        needs_grad_b = ctx.needs_input_grad[2]

        if not (needs_grad_A_val or needs_grad_b):
            return (None,) * 4

        (x,) = ctx.saved_tensors
        a_pattern: SparsityPattern = ctx.a_pattern

        if ctx.solver is None:
            raise RuntimeError(
                "Solver was released. Calling backward() twice with retain_graph=True "
                + "is currently not supported."
            )
        else:
            solver: scipy.sparse.linalg.SuperLU = ctx.solver

        lambda_np = solver.solve(to_np(dLdx, contiguous=True), trans="T")
        lambda_: Float[Tensor, " r *ch"] = torch.from_numpy(lambda_np).to(
            dtype=dLdx.dtype, device=dLdx.device
        )

        # Free up solver memory usage.
        ctx.solver = None

        if needs_grad_b and not needs_grad_A_val:
            return (None, None, lambda_, None)

        # dLdA will have the same sparsity pattern as A.
        if lambda_.ndim > 1:
            # If there is a channel dimension, sum over it.
            dLdA_val = torch.sum(
                -lambda_[a_pattern.idx_coo[0]] * x[a_pattern.idx_coo[1]], dim=-1
            )
        else:
            dLdA_val = -lambda_[a_pattern.idx_coo[0]] * x[a_pattern.idx_coo[1]]

        if needs_grad_A_val and not needs_grad_b:
            return (dLdA_val, None, None, None)

        if needs_grad_A_val and needs_grad_b:
            return (dLdA_val, None, lambda_, None)


def splu(
    a: Float[SparseDecoupledTensor, "r c"],
    b: Float[Tensor, " r *ch"],
    *,
    backend: Literal["cupy", "scipy"],
    channel_first: bool = False,
    **splu_kwargs,
) -> Float[Tensor, " c *ch"]:
    """
    Differentiable wrapper for SuperLU.

    Parameters
    ----------
    a : [r, c]
        A 2D matrix represented as a `SparseDecoupledTensor`.
    b : [r, *ch]
        The RHS vector as a dense tensor with optional channel dimensions.

    Here, A is a SparseDecoupledTensor and b is a dense tensor with optional channel
    dimensions. If `channel_first` is `True`, all but the last dimension of `b`
    is treated as channel dimensions; if it is `False`, all but the first dimension
    of `b` is treated as channel dimensions.

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
    if not a.pattern._is_int32_safe:
        raise ValueError(
            "The sparse indices of the input tensor 'A' cannot be safely cast to int32 dtype."
        )

    match b.ndim:
        case 1:
            b_ready = b
        case 2:
            b_ready = b.transpose(0, 1) if channel_first else b
        case _:
            if channel_first:
                # (*ch, r) -> (r, ch_flat)
                b_ready = torch.movedim(b, -1, 0).reshape(b.shape[-1], -1)
            else:
                # (r, *ch) -> (r, ch_flat)
                b_ready = b.reshape(b.shape[0], -1)

    match backend:
        case "cupy":
            if not _HAS_CUPY:
                raise ImportError("CuPy backend required.")

            x, solver = _CuPySuperLUAutogradFunction.apply(
                a.values, a.pattern, b_ready, splu_kwargs
            )

        case "scipy":
            x, solver = _SciPySuperLUAutogradFunction.apply(
                a.values, a.pattern, b_ready, splu_kwargs
            )

        case _:
            raise ValueError()

    if x.ndim == 1:
        x_reshaped = x
    else:
        if channel_first:
            # (c, ch_flat) -> (*ch, c)
            x_reshaped = x.transpose(0, 1).reshape(*b.shape[:-1], -1)
        else:
            # (c, ch_flat) -> (c, *ch)
            x_reshaped = x.reshape(-1, *b.shape[1:])

    return x_reshaped

from __future__ import annotations

__all__ = ["splu"]

from typing import Any, Literal

import scipy.sparse
import scipy.sparse.linalg
import torch
from jaxtyping import Float, Integer
from torch import Tensor
from torch.autograd.function import once_differentiable

from .....utils.parsing import to_np
from ....decoupled_tensor import SparseDecoupledTensor, SparsityPattern
from ....decoupled_tensor._conversion import sdt_to_cupy_csc, sdt_to_scipy_csc

try:
    import cupy as cp
    import cupyx.scipy.sparse as cp_sp
    import cupyx.scipy.sparse.linalg as cp_sp_linalg

    _HAS_CUPY = True

except ImportError:
    _HAS_CUPY = False


class _SciPySuperLUAutogradFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        a_val: Float[Tensor, " nz"],
        a_pattern: Integer[SparsityPattern, "r c"],
        b: Float[Tensor, " r *ch"],
        splu_kwargs: dict[str, Any],
    ) -> tuple[Float[Tensor, " c *ch"], scipy.sparse.linalg.SuperLU]:
        a_scipy: Float[scipy.sparse.csc_array, "r c"] = sdt_to_scipy_csc(
            a_val, a_pattern
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
        i, j = a_pattern.idx_coo.unbind(0)
        if lambda_.ndim > 1:
            # If there is a channel dimension, sum over it.
            # dLdA_ij = Σ_c[λ_ic * x_jc]
            dLdA_val = torch.sum(-lambda_[i] * x[j], dim=-1)
        else:
            # dLdA_ij = λ_i*x_j
            dLdA_val = -lambda_[i] * x[j]

        if needs_grad_A_val and not needs_grad_b:
            return (dLdA_val, None, None, None)

        if needs_grad_A_val and needs_grad_b:
            return (dLdA_val, None, lambda_, None)


if _HAS_CUPY:

    class _CuPySuperLUAutogradFunction(torch.autograd.Function):
        @staticmethod
        def forward(
            a_val: Float[Tensor, " nz"],
            a_pattern: Integer[SparsityPattern, "r c"],
            b: Float[Tensor, " r *ch"],
            splu_kwargs: dict[str, Any],
        ) -> tuple[Float[Tensor, " c *ch"], cp_sp_linalg.SuperLU]:
            # Force CuPy to use the current Pytorch stream.
            stream = torch.cuda.current_stream()
            with cp.cuda.ExternalStream(stream.cuda_stream, stream.device_index):
                A_cp: Float[cp_sp.csc_matrix, "r c"] = sdt_to_cupy_csc(a_val, a_pattern)
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
            i, j = a_pattern.idx_coo.unbind(0)
            if lambda_.dim() > 1:
                # If there is a channel dimension, sum over it.
                # dLdA_ij = Σ_c[λ_ic * x_jc]
                dLdA_val = torch.sum(-lambda_[i] * x[j], dim=-1)
            else:
                # dLdA_ij = λ_i*x_j
                dLdA_val = -lambda_[i] * x[j]

            if needs_grad_A_val and not needs_grad_b:
                return (dLdA_val, None, None, None)

            if needs_grad_A_val and needs_grad_b:
                return (dLdA_val, None, lambda_, None)

else:

    class _CuPySuperLUAutogradFunction:
        def apply(*args, **kwargs):
            raise ImportError("CuPy backend required.")


def splu(
    a: Float[SparseDecoupledTensor, "r c"],
    b: Float[Tensor, " r *ch"],
    *,
    backend: Literal["cupy", "scipy"],
    **splu_kwargs,
) -> Float[Tensor, " c *ch"]:
    """
    "Stateless" differentiable wrapper for SuperLU.

    Given a sparse 2D matrix `a` and a vector `b`, solve the linear system
    `a @ x = b` for `x`.

    Parameters
    ----------
    a : [r, c]
        A sparse 2D matrix represented as a `SparseDecoupledTensor`. Note that
        the `SuperLU` solver does not allow for batch dimensions in `a`.
    b : [r, *ch]
        The RHS vector as a dense tensor with arbitrary channel dimensions. Internally,
        the solver expects `b` to be a contiguous tensor of shape `[r,]` or
        `[r, ch]`; if the input tensor `b` does not conform to this requirement,
        a reshaped copy will be created.
    backend
        Whether to use the CuPy (`"cupy"`) or SciPy (`"scipy"`) implementation of
        `SuperLU`. If the backend is CuPy, `a` and `b` must be on the CUDA device.
        If backend is SciPy, `a` and `b` will be copied to CPU. Note that `SuperLU`
        handles the LU factorization step on the host CPU, regardless of the backend.
    splu_kwargs
        Additional keyword arguments to be passed to `cupyx.scipy.sparse.linalg.splu()`
        if the backend is CuPy or `scipy.sparse.linalg.splu()` if the backend is
        SciPy.

    Returns
    -------
    [c, *ch]
        The unknown vector `x` with channel dimensions matching those of `b`.

    Notes
    -----
    If the linear system `a @ x = b` does not have a unique solution, then both
    the forward pass and backward gradient will fail.

    If either `a` or `b` requires gradient, then a `SuperLU` solver object will be
    cached in memory to accelerate the backward pass; this memory will not be
    cleaned up until one of the following conditions is met:

    * a backward() call with `retain_graph=False` has been made through the
    computation graph containing `a` or `b`, or
    * all references to the output tensor (and its `grad_fn` and `ctx` attributes)
    from this function (and any derived tensors thereof) has gone out of scope/been
    detached from the computation graph.

    In addition, note that this function has the following limitations:

    * Currently, double backward through this function is not supported.
    * The sparse indices of `a` must be of datatype `int32` for compatibility with
    the solver.
    """
    if a.n_batch_dim > 0:
        raise ValueError("Batch dimension in 'a' is not supported.")
    if a.n_dense_dim > 0:
        raise ValueError("Dense dimension in 'a' is not supported.")
    if not a.pattern._is_int32_safe:
        raise ValueError(
            "The sparse indices of the input tensor 'A' cannot be safely cast to int32 dtype."
        )

    match b.ndim:
        case 0:
            raise ValueError("'b' must have at least one dimension.")
        case 1 | 2:
            b_ready = b
        case _:
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
            raise ValueError(f"Unknwon 'backend' argument '{backend}'.")

    if x.ndim == 1:
        x_reshaped = x
    else:
        # (c, ch_flat) -> (c, *ch)
        x_reshaped = x.reshape(-1, *b.shape[1:])

    return x_reshaped

from __future__ import annotations

__all__ = ["SuperLU"]

from typing import Literal

import scipy.sparse
import scipy.sparse.linalg
import torch
from jaxtyping import Float, Integer
from torch import Tensor
from torch.autograd.function import once_differentiable

from .....utils.parsing import to_np
from ....decoupled_tensor import SparseDecoupledTensor, SparsityPattern
from ....decoupled_tensor._conversion import sdt_to_cupy_csc, sdt_to_scipy_csc
from .._inv_sparse_operator import InvSparseOperator

try:
    import cupy as cp
    import cupyx.scipy.sparse as cp_sp
    import cupyx.scipy.sparse.linalg as cp_sp_linalg

    _HAS_CUPY = True

except ImportError:
    _HAS_CUPY = False


class _PersistentSciPySuperLUAutogradFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        a_val: Float[Tensor, " nz"],
        a_pattern: Integer[SparsityPattern, "r c"],
        b: Float[Tensor, " r *ch"],
        solver: scipy.sparse.linalg.SuperLU,
        trans: Literal["N", "T"],
    ) -> Float[Tensor, " c *ch"]:
        # The a_val and a_pattern are still required for gradient tracking
        # purposes, even though they are not used in the forward pass.
        b_np = to_np(b, contiguous=True)
        x = torch.from_numpy(solver.solve(b_np, trans=trans)).to(
            dtype=a_val.dtype, device=a_val.device
        )

        return x

    @staticmethod
    def setup_context(ctx, inputs, output):
        a_val, a_pattern, b, solver, trans = inputs

        ctx.save_for_backward(output)
        ctx.solver = solver
        ctx.a_pattern = a_pattern
        ctx.trans = trans

    @staticmethod
    @once_differentiable
    def backward(
        ctx, dLdx: Float[Tensor, " c *ch"]
    ) -> tuple[
        Float[Tensor, " nz"] | None,
        None,
        Float[Tensor, " r *ch"] | None,
        None,
        None,
    ]:
        needs_grad_a_val = ctx.needs_input_grad[0]
        needs_grad_b = ctx.needs_input_grad[2]

        if not (needs_grad_a_val or needs_grad_b):
            return (None,) * 5

        (x,) = ctx.saved_tensors
        a_pattern: SparsityPattern = ctx.a_pattern
        solver: scipy.sparse.linalg.SuperLU = ctx.solver

        # If the forward pass is "N", then the backward requires "T" and vice versa.
        lambda_np = solver.solve(
            to_np(dLdx, contiguous=True),
            trans="T" if ctx.trans == "N" else "N",
        )
        lambda_: Float[Tensor, " r *ch"] = torch.from_numpy(lambda_np).to(
            dtype=dLdx.dtype, device=dLdx.device
        )

        if needs_grad_b and not needs_grad_a_val:
            return (None, None, lambda_, None, None)

        # dLdA will have the same sparsity pattern as A.
        i, j = a_pattern.idx_coo.unbind(0)
        if lambda_.ndim > 1:
            # If there is a channel dimension, sum over it.
            # dLdA_ij = Σ_c[λ_ic * x_jc]
            dLda_val = torch.sum(-lambda_[i] * x[j], dim=-1)
        else:
            # dLdA_ij = λ_i*x_j
            dLda_val = -lambda_[i] * x[j]

        if needs_grad_a_val and not needs_grad_b:
            return (dLda_val, None, None, None, None)

        if needs_grad_a_val and needs_grad_b:
            return (dLda_val, None, lambda_, None, None)


if _HAS_CUPY:

    class _PersistentCuPySuperLUAutogradFunction(torch.autograd.Function):
        @staticmethod
        def forward(
            a_val: Float[Tensor, " nz"],
            a_pattern: Integer[SparsityPattern, "r c"],
            b: Float[Tensor, " r *ch"],
            solver: cp_sp_linalg.SuperLU,
            trans: Literal["N", "T"],
        ) -> Float[Tensor, " c *ch"]:
            # The a_val and a_pattern are still required for gradient tracking
            # purposes, even though they are not used in the forward pass.

            # Force CuPy to use the current Pytorch stream.
            stream = torch.cuda.current_stream()
            with cp.cuda.ExternalStream(stream.cuda_stream, stream.device_index):
                b_cp = cp.from_dlpack(b.detach().contiguous())
                x = torch.from_dlpack(solver.solve(b_cp, trans=trans))

            return x

        @staticmethod
        def setup_context(ctx, inputs, output):
            a_val, a_pattern, b, solver, trans = inputs

            ctx.save_for_backward(output)
            ctx.solver = solver
            ctx.a_pattern = a_pattern
            ctx.trans = trans

        @staticmethod
        @once_differentiable
        def backward(
            ctx, dLdx: Float[Tensor, " c *ch"]
        ) -> tuple[
            Float[Tensor, " nz"] | None,
            None,
            Float[Tensor, " r *ch"] | None,
            None,
            None,
        ]:
            needs_grad_a_val = ctx.needs_input_grad[0]
            needs_grad_b = ctx.needs_input_grad[2]

            if not (needs_grad_a_val or needs_grad_b):
                return (None,) * 5

            (x,) = ctx.saved_tensors
            a_pattern: SparsityPattern = ctx.a_pattern
            solver: cp_sp_linalg.SuperLU = ctx.solver

            stream = torch.cuda.current_stream()
            with cp.cuda.ExternalStream(stream.cuda_stream, stream.device_index):
                # If the forward pass is "N", then the backward requires "T" and
                # vice versa.
                lambda_cp: Float[Tensor, " r"] = solver.solve(
                    cp.from_dlpack(dLdx.detach().contiguous()),
                    trans="T" if ctx.trans == "N" else "N",
                )
                lambda_ = torch.from_dlpack(lambda_cp)

            if needs_grad_b and not needs_grad_a_val:
                return (None, None, lambda_, None, None)

            # dLdA will have the same sparsity pattern as A.
            i, j = a_pattern.idx_coo.unbind(0)
            if lambda_.dim() > 1:
                # If there is a channel dimension, sum over it.
                # dLdA_ij = Σ_c[λ_ic * x_jc]
                dLda_val = torch.sum(-lambda_[i] * x[j], dim=-1)
            else:
                # dLdA_ij = λ_i*x_j
                dLda_val = -lambda_[i] * x[j]

            if needs_grad_a_val and not needs_grad_b:
                return (dLda_val, None, None, None, None)

            if needs_grad_a_val and needs_grad_b:
                return (dLda_val, None, lambda_, None, None)

else:

    class _PersistentCuPySuperLUAutogradFunction:
        def apply(*args, **kwargs):
            raise ImportError("CuPy backend required.")


class SuperLU(InvSparseOperator):
    """
    "Stateful" differentiable wrapper for SuperLU.

    Given a sparse 2D matrix `a` and a vector `b`, this class computes and caches
    the LU factorization of `a` for the purpose of solving the linear system
    `a @ x = b` for `x`.

    Parameters
    ----------
    a : [r, c]
        A sparse 2D matrix represented as a `SparseDecoupledTensor`. Note that
        the `SuperLU` solver does not allow for batch dimensions in `a`.
    backend
        Whether to use the CuPy (`"cupu"`) or SciPy (`"scipy"`) implementation of
        `SuperLU`. If the backend is CuPy, `a` and `b` must be on the CUDA device.
        If backend is SciPy, `a` and `b` will be copied to CPU. Note that `SuperLU`
        handles the factorization step on the host CPU, regardless of the backend.
    splu_kwargs
        Additional keyword arguments to be passed to `cupyx.scipy.sparse.linalg.splu()`
        if the backend is CuPy or `scipy.sparse.linalg.splu()` if the backend is
        SciPy.

    Notes
    -----
    This class implements a similar `SuperLU` wrapper as the functional `splu()`.
    Both wrappers cache the solver object and the factorized matrix for backward
    passes; however, in the functional wrapper, the cached solver is tied to the
    computation graph and gc'ed after the backward pass, whereas, in this class,
    the cached solver is tied to the lifecycle of the class instances, which allows
    the same factorized matrix to be re-used to solve different linear systems.
    See the `splu()` function for more details on the requirements and limitations
    of this wrapper.
    """

    def __init__(
        self,
        a: Float[SparseDecoupledTensor, "r c"],
        *,
        backend: Literal["cupy", "scipy"],
        **splu_kwargs,
    ):
        if a.n_batch_dim > 0:
            raise ValueError("Batch dimension in 'a' is not supported.")
        if a.n_dense_dim > 0:
            raise ValueError("Dense dimension in 'a' is not supported.")
        if not a.pattern._is_int32_safe:
            raise ValueError(
                "The sparse indices of the input tensor 'A' cannot be safely "
                "cast to int32 dtype."
            )

        if (backend == "cupy") and not _HAS_CUPY:
            raise ImportError("CuPy backend required.")

        self.a_val = a.values
        self.a_pattern = a.pattern
        self.backend = backend

        self.dtype = a.dtype
        self.device = a.device
        self.shape = a.shape

        match self.backend:
            case "cupy":
                # Force CuPy to use the current Pytorch stream.
                stream = torch.cuda.current_stream()
                with cp.cuda.ExternalStream(stream.cuda_stream, stream.device_index):
                    A_cp: Float[cp_sp.csc_matrix, "r c"] = sdt_to_cupy_csc(
                        self.a_val, self.a_pattern
                    )

                    self.solver = cp_sp_linalg.splu(A_cp, **splu_kwargs)

            case "scipy":
                A_scipy: Float[scipy.sparse.csc_array, "r c"] = sdt_to_scipy_csc(
                    self.a_val, self.a_pattern
                )

                self.solver = scipy.sparse.linalg.splu(A_scipy, **splu_kwargs)

            case _:
                raise ValueError(f"Unknown backend argument '{backend}'.")

    def __del__(self):
        if hasattr(self, "solver") and self.solver is not None:
            # Force device sync before gc.
            try:
                import torch

                if torch.cuda.is_initialized():
                    torch.cuda.synchronize(self.device)

            except Exception:
                pass

            self.solver = None

    def __call__(
        self,
        b: Float[Tensor, " r *ch"],
        *,
        trans: Literal["N", "T"] = "N",
    ) -> Float[Tensor, " c *ch"]:
        """
        Solve a sparse linear system with the given RHS vector using `SuperLU`.

        Parameters
        ----------
        b : [r, *ch]
            The RHS vector as a dense tensor with arbitrary channel dimensions.
            Internally, the solver expects `b` to be a contiguous tensor of shape
            `[r,]` or `[r, ch]`; if the input tensor `b` does not conform to this
            requirement, a reshaped copy will be created.
        trans
            If "N", solve the normal system `a @ x = b`; if "T", solve the transposed
            system `a.T @ x = b`.

        Returns
        -------
        [c, *ch]
            The unknown vector `x` with channel dimensions matching those of `b`.
        """
        if b.ndim > 2:
            # Flatten multiple channel dims, (r, *ch) -> (r, ch_flat)
            b_flat = b.reshape(b.size(0), -1)
        else:
            b_flat = b

        match self.backend:
            case "cupy":
                x = _PersistentCuPySuperLUAutogradFunction.apply(
                    self.a_val, self.a_pattern, b_flat, self.solver, trans
                )
            case "scipy":
                x = _PersistentSciPySuperLUAutogradFunction.apply(
                    self.a_val, self.a_pattern, b_flat, self.solver, trans
                )

        if x.ndim == 1:
            x_reshaped = x
        else:
            # (c, ch_flat) -> (c, *ch)
            x_reshaped = x.reshape(-1, *b.shape[1:])

        return x_reshaped

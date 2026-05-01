from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import scipy.sparse
import scipy.sparse.linalg
import torch
from jaxtyping import Float, Integer
from torch import Tensor
from torch.autograd.function import once_differentiable

from ....decoupled_tensor import SparseDecoupledTensor, SparsityPattern
from .._inv_sparse_operator import InvSparseOperator

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

__all__ = ["SuperLU"]


class _PersistentCuPySuperLUAutogradFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        A_val: Float[Tensor, " nnz"],
        A_pattern: Integer[SparsityPattern, "r c"],
        b: Float[Tensor, " r *ch"],
        solver: cp_sp_linalg.SuperLU,
        trans: Literal["N", "T"],
    ) -> Float[Tensor, " c *ch"]:
        # The A_val and A_pattern are still required for gradient tracking
        # purposes, even though they are not used in the forward pass.

        # Force CuPy to use the current Pytorch stream.
        stream = torch.cuda.current_stream()
        with cp.cuda.ExternalStream(stream.cuda_stream, stream.device_index):
            b_cp = cp.from_dlpack(b.detach().contiguous())
            x = torch.from_dlpack(solver.solve(b_cp, trans=trans))

        return x

    @staticmethod
    def setup_context(ctx, inputs, output):
        A_val, A_pattern, b, solver, trans = inputs

        ctx.save_for_backward(output)
        ctx.solver = solver
        ctx.A_pattern = A_pattern
        ctx.trans = trans

    @staticmethod
    @once_differentiable
    def backward(
        ctx, dLdx: Float[Tensor, " c *ch"]
    ) -> tuple[
        Float[Tensor, " nnz"] | None,
        None,
        Float[Tensor, " r *ch"] | None,
        None,
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
        needs_grad_b = ctx.needs_input_grad[2]

        if not (needs_grad_A_val or needs_grad_b):
            return (None,) * 5

        (x,) = ctx.saved_tensors
        A_pattern: SparsityPattern = ctx.A_pattern
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

        if needs_grad_b and not needs_grad_A_val:
            return (None, None, lambda_, None, None)

        # dLdA will have the same sparsity pattern as A.
        if lambda_.dim() > 1:
            # If there is a channel dimension, sum over it.
            dLdA_val = torch.sum(
                -lambda_[A_pattern.idx_coo[0]] * x[A_pattern.idx_coo[1]], dim=-1
            )
        else:
            dLdA_val = -lambda_[A_pattern.idx_coo[0]] * x[A_pattern.idx_coo[1]]

        if needs_grad_A_val and not needs_grad_b:
            return (dLdA_val, None, None, None, None)

        if needs_grad_A_val and needs_grad_b:
            return (dLdA_val, None, lambda_, None, None)


class _PersistentSciPySuperLUAutogradFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        A_val: Float[Tensor, " nnz"],
        A_pattern: Integer[SparsityPattern, "r c"],
        b: Float[Tensor, " r *ch"],
        solver: scipy.sparse.linalg.SuperLU,
        trans: Literal["N", "T"],
    ) -> Float[Tensor, " c *ch"]:
        # The A_val and A_pattern are still required for gradient tracking
        # purposes, even though they are not used in the forward pass.

        b_np = b.detach().contiguous().cpu().numpy()
        x = torch.from_numpy(solver.solve(b_np, trans=trans)).to(
            dtype=A_val.dtype, device=A_val.device
        )

        return x

    @staticmethod
    def setup_context(ctx, inputs, output):
        A_val, A_pattern, b, solver, trans = inputs

        ctx.save_for_backward(output)
        ctx.solver = solver
        ctx.A_pattern = A_pattern
        ctx.trans = trans

    @staticmethod
    @once_differentiable
    def backward(
        ctx, dLdx: Float[Tensor, " c *ch"]
    ) -> tuple[
        Float[Tensor, " nnz"] | None,
        None,
        Float[Tensor, " r *ch"] | None,
        None,
        None,
    ]:
        needs_grad_A_val = ctx.needs_input_grad[0]
        needs_grad_b = ctx.needs_input_grad[2]

        if not (needs_grad_A_val or needs_grad_b):
            return (None,) * 5

        (x,) = ctx.saved_tensors
        A_pattern: SparsityPattern = ctx.A_pattern
        solver: scipy.sparse.linalg.SuperLU = ctx.solver

        # If the forward pass is "N", then the backward requires "T" and vice versa.
        lambda_np = solver.solve(
            dLdx.detach().contiguous().cpu().numpy(),
            trans="T" if ctx.trans == "N" else "N",
        )
        lambda_: Float[Tensor, " r *ch"] = torch.from_numpy(lambda_np).to(
            dtype=dLdx.dtype, device=dLdx.device
        )

        if needs_grad_b and not needs_grad_A_val:
            return (None, None, lambda_, None, None)

        # dLdA will have the same sparsity pattern as A.
        if lambda_.ndim > 1:
            # If there is a channel dimension, sum over it.
            dLdA_val = torch.sum(
                -lambda_[A_pattern.idx_coo[0]] * x[A_pattern.idx_coo[1]], dim=-1
            )
        else:
            dLdA_val = -lambda_[A_pattern.idx_coo[0]] * x[A_pattern.idx_coo[1]]

        if needs_grad_A_val and not needs_grad_b:
            return (dLdA_val, None, None, None, None)

        if needs_grad_A_val and needs_grad_b:
            return (dLdA_val, None, lambda_, None, None)


class SuperLU(InvSparseOperator):
    """
    A "statefull" version of the cupy/scipy `splu()` wrapper.

    This class implements a similar wrapper to cupy/scipy `splu()` as the functional
    `splu()` wrapper. Both wrappers cache the solver object and the factorized matrix
    for backward passes; however, in the functional wrapper, the cached solver is
    tied to the computation graph and gc'ed after the backward pass, whereas, in
    this class, the cached solver is tied to the lifecycle of the class instances,
    which allows the same factorized matrix to be re-used to solve different linear
    systems.

    See the `splu()` function for more details on the requirements and limitations
    of this wrapper. This wrapper also differs from the functional version in the
    following ways:

    * This class is callable with a "trans" option that supports directly solving
    a transposed linear system.
    * This class assumes that the input vector b is always of the shape (r, *ch).
    """

    def __init__(
        self,
        A: Float[SparseDecoupledTensor, "r c"],
        *,
        backend: Literal["cupy", "scipy"],
        **splu_kwargs,
    ):
        if not A.pattern._is_int32_safe:
            raise ValueError(
                "The sparse indices of the input tensor 'A' cannot be safely "
                "cast to int32 dtype."
            )

        self.A_val = A.values
        self.A_pattern = A.pattern
        self.backend = backend

        self.dtype = A.dtype
        self.device = A.device
        self.shape = A.shape

        val = A.values[A.pattern.coo_to_csc_perm].detach().contiguous()
        idx_ccol = A.pattern.idx_ccol.detach().contiguous()
        idx_row = A.pattern.idx_row_csc.detach().contiguous()

        match backend:
            case "cupy":
                # Force CuPy to use the current Pytorch stream.
                stream = torch.cuda.current_stream()
                with cp.cuda.ExternalStream(stream.cuda_stream, stream.device_index):
                    A_cp: Float[cp_sp.csc_matrix, "r c"] = cp_sp.csc_matrix(
                        (
                            cp.from_dlpack(val),
                            cp.from_dlpack(idx_row),
                            cp.from_dlpack(idx_ccol),
                        ),
                        shape=tuple(A.pattern.shape),
                    )

                    self.solver = cp_sp_linalg.splu(A_cp, **splu_kwargs)

            case "scipy":
                val_np = val.cpu().numpy()
                idx_ccol_np = idx_ccol.cpu().numpy()
                idx_row_np = idx_row.contiguous().cpu().numpy()

                A_scipy: Float[scipy.sparse.csc_array, "r c"] = scipy.sparse.csc_array(
                    (val_np, idx_row_np, idx_ccol_np),
                    shape=A.pattern.shape,
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
        if b.ndim > 2:
            # Flatten multiple channel dims, (r, *ch) -> (r, ch_flat)
            b_flat = b.reshape(b.size(0), -1)
        else:
            b_flat = b

        match self.backend:
            case "cupy":
                x = _PersistentCuPySuperLUAutogradFunction.apply(
                    self.A_val, self.A_pattern, b_flat, self.solver, trans
                )
            case "scipy":
                x = _PersistentSciPySuperLUAutogradFunction.apply(
                    self.A_val, self.A_pattern, b_flat, self.solver, trans
                )

        if x.ndim == 1:
            x_reshaped = x
        else:
            # (c, ch_flat) -> (c, *ch)
            x_reshaped = x.reshape(-1, *b.shape[1:])

        return x_reshaped

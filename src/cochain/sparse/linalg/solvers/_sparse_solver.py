from abc import ABC, abstractmethod
from typing import Literal

import torch
from jaxtyping import Float, Integer
from torch import Tensor
from torch.autograd.function import once_differentiable

from ...decoupled_tensor import SparsityPattern


class BaseSparseSolver(ABC):
    """
    An ABC for sparse linear solver wrapper classes.

    Parameters
    ----------
    matrix_type
        The type of the sparse matrix.
    dtype
        The dtype of the sparse matrix.
    device
        The device of the sparse matrix.
    shape
        The shape of the sparse matrix.
    """

    matrix_type: Literal["general", "symmetric", "spd"]
    dtype: torch.dtype
    device: torch.device
    shape: torch.Size

    def size(self, dim: int | None = None) -> int | torch.Size:
        """Return the shape of the sparse matrix."""
        if dim is None:
            return self.shape
        else:
            return self.shape[dim]

    @abstractmethod
    def solve(self, b: Tensor, trans: Literal["N", "T"] = "N") -> Tensor: ...

    @abstractmethod
    def free(self): ...


class InvSparseOperator(ABC):
    """
    An ABC for "stateful" differentiable sparse linear solver classes.

    Parameters
    ----------
    dtype
        The dtype of the tensor `a`.
    device
        The device of the tensor `a`.
    shape
        The shape of the tensor `a`.
    """

    dtype: torch.dtype
    device: torch.device
    shape: torch.Size
    solver: BaseSparseSolver

    def size(self, dim: int | None = None) -> int | torch.Size:
        """Return the shape of the sparse matrix."""
        if dim is None:
            return self.shape
        else:
            return self.shape[dim]

    @abstractmethod
    def __call__(
        self, b: Float[Tensor, " r *ch"], trans: Literal["N", "T"], *args, **kwargs
    ) -> Float[Tensor, " c *ch"]: ...


class SparseSolverAutogradFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        a_val: Float[Tensor, " nz"],
        a_pattern: Integer[SparsityPattern, "*b r c"],
        b_flat: Float[Tensor, " r *ch_flat"] | Float[Tensor, "b r *ch_flat"],
        solver: BaseSparseSolver,
        trans: Literal["N", "T"],
        persistent: bool,
    ) -> Float[Tensor, " c *ch_flat"] | Float[Tensor, "b c *ch_flat"]:
        x_flat = solver.solve(b_flat, trans=trans)
        return x_flat

    @staticmethod
    def setup_context(ctx, inputs, output):
        a_val, a_pattern, b, solver, trans, persistent = inputs

        ctx.save_for_backward(output)  # x_flat
        ctx.a_pattern = a_pattern
        ctx.solver = solver
        ctx.trans = trans
        ctx.persistent = persistent

    @staticmethod
    @once_differentiable
    def backward(
        ctx,
        dLdx_flat: Float[Tensor, " c *ch_flat"] | Float[Tensor, "b c *ch_flat"],
    ) -> tuple[
        Float[Tensor, " nz"] | None,
        None,
        Float[Tensor, " r *ch_flat"] | Float[Tensor, "b r *ch_flat"] | None,
        None,
        None,
        None,
    ]:
        needs_grad_a_val = ctx.needs_input_grad[0]
        needs_grad_b_flat = ctx.needs_input_grad[2]

        if not (needs_grad_a_val or needs_grad_b_flat):
            return (None,) * 6

        if ctx.solver is None:
            raise RuntimeError("Solver was released.")
        else:
            solver: BaseSparseSolver = ctx.solver

        (x_flat,) = ctx.saved_tensors
        a_pattern: SparsityPattern = ctx.a_pattern

        # The backward pass solves the adjoint system.
        trans = "T" if ctx.trans == "N" else "N"

        dLda_val = None
        dLdb_flat = None

        # Currently supported A @ x = b shape combinations:
        #
        # For splu():
        # -------------------------------------------
        # a.shape    b_flat.shape     x_flat.shape
        # -------------------------------------------
        # [r, c]     [r,]             [c,]
        # [r, c]     [r, ch_flat]     [c, ch_flat]
        # -------------------------------------------
        #
        # For DirectSolver():
        # -------------------------------------------
        # a.shape    b_flat.shape     x_flat.shape
        # -------------------------------------------
        # [r, c]     [r,]             [c,]
        # [r, c]     [r, ch_flat]     [c, ch_flat]
        # [b, r, c]  [b, r, 1]        [b, c, 1]
        # [b, r, c]  [b, r, ch_flat]  [b, c, ch_flat]
        # -------------------------------------------

        if needs_grad_b_flat or needs_grad_a_val:
            # lambda_ has the same shape as b_flat.
            lambda_ = solver.solve(dLdx_flat, trans=trans)

        if needs_grad_b_flat:
            dLdb_flat = lambda_

        if needs_grad_a_val:
            # Compute the nonzero values of dLdA using the outer product between
            # λ and x; dLdA will have the same sparsity pattern as A.

            a_has_batch_dim = a_pattern.n_batch_dim > 0
            b_has_ch_dim = lambda_.ndim > 1

            a_nz_idx = a_pattern.idx_coo.unbind(dim=0)

            match (b_has_ch_dim, a_has_batch_dim):
                case (True, True):
                    b, r, c = a_nz_idx
                    dLda_val = -torch.sum(lambda_[b, r] * x_flat[b, c], dim=-1)

                case (True, False):
                    r, c = a_nz_idx
                    dLda_val = -torch.sum(lambda_[r] * x_flat[c], dim=-1)

                case (False, False):
                    r, c = a_nz_idx
                    dLda_val = -lambda_[r] * x_flat[c]

                case (False, True):
                    raise NotImplementedError()

        # Free solver memory if not persistent.
        if not ctx.persistent:
            solver.free()
            ctx.solver = None

        return (dLda_val, None, dLdb_flat, None, None, None)

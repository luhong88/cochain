from abc import ABC, abstractmethod
from typing import Literal

import torch
from jaxtyping import Float, Integer
from torch import Tensor
from torch.autograd.function import once_differentiable

from ...decoupled_tensor import SparsityPattern


class BaseSparseSolver(ABC):
    matrix_type: Literal["general", "symmetric", "spd"]
    dtype: torch.dtype
    device: torch.device
    shape: torch.Size

    def size(self, dim: int | None = None) -> int | torch.Size:
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
    An ABC for "stateful" sparse linear solver classes.

    This class provides an abstraction to solving the sparse linear system `A@x=b`
    for `x`. The tensor `A` is assumed to have shape `(r, c)`, and the tensor `b` is
    assumed to have shape `(r, *ch)`, and the output `x` tensor is assumed to have
    shape `(c, *ch)`; no explicit leading batch dimensions are allowed.

    Parameters
    ----------
    dtype
        The dtype of the tensor `A`.
    device
        The device of the tensor `A`.
    shape
        The shape of the tensor `A`.
    """

    dtype: torch.dtype
    device: torch.device
    shape: torch.Size
    solver: BaseSparseSolver

    def size(self, dim: int | None = None) -> int | torch.Size:
        if dim is None:
            return self.shape
        else:
            return self.shape[dim]

    @abstractmethod
    def __call__(
        self, b: Float[Tensor, " r *ch"], *args, **kwargs
    ) -> Float[Tensor, " c *ch"]: ...


class SparseSolverAutogradFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        a_val: Float[Tensor, " nz"],
        a_pattern: Integer[SparsityPattern, "*b r c"],
        b: Float[Tensor, " r *ch_flat"] | Float[Tensor, "b r *ch_flat"],
        solver: BaseSparseSolver,
        trans: Literal["N", "T"],
        persistent: bool,
    ) -> Float[Tensor, " c *ch_flat"] | Float[Tensor, "b c *ch_flat"]:
        x = solver.solve(b, trans=trans)
        return x

    @staticmethod
    def setup_context(ctx, inputs, output):
        a_val, a_pattern, b, solver, trans, persistent = inputs

        ctx.save_for_backward(output)  # x
        ctx.a_pattern = a_pattern
        ctx.solver = solver
        ctx.trans = trans
        ctx.persistent = persistent

    @staticmethod
    @once_differentiable
    def backward(
        ctx,
        dLdx: Float[Tensor, " c *ch_flat"] | Float[Tensor, "b c *ch_flat"],
    ) -> tuple[
        Float[Tensor, " nz"] | None,
        None,
        Float[Tensor, " r *ch_flat"] | Float[Tensor, "b r *ch_flat"] | None,
        None,
        None,
        None,
    ]:
        needs_grad_a_val = ctx.needs_input_grad[0]
        needs_grad_b = ctx.needs_input_grad[2]

        if not (needs_grad_a_val or needs_grad_b):
            return (None,) * 6

        if ctx.solver is None:
            raise RuntimeError("Solver was released.")
        else:
            solver: BaseSparseSolver = ctx.solver

        (x,) = ctx.saved_tensors
        a_pattern: SparsityPattern = ctx.a_pattern

        dLda_val = None
        dLdb = None

        if needs_grad_b or needs_grad_a_val:
            # lambda_ has the same shape as b.
            # For splu(), b has shape [r, *ch]
            # For nvmath DirectSolver(), b has shape [r, *ch] or [b, r, *ch]
            lambda_ = solver.solve(dLdx, trans="T" if ctx.trans == "N" else "N")

        if needs_grad_b:
            dLdb = lambda_

        if needs_grad_a_val:
            # dLdA will have the same sparsity pattern as A.
            if lambda_.ndim > 1:
                # Sum over the channel dimension while accounting for zero or more
                # batch dimensions.

                # Recall that A.pattern.idx_coo has shape (sp, nnz), where sp has size
                # equal to the number of batch dims plus 2 (i.e., sp = len(*b) + 2).
                n_batch = a_pattern.n_batch_dim

                # Extract the nonzero dLdA element row and col indices, accounting for
                # batch dimensions, and use them to extract the corresponding elements
                # from lambda_ and x to construct the nonzero outer product elements.
                r_idx: Integer[Tensor, "n_batch+1 nnz"] = a_pattern.idx_coo[
                    list(range(n_batch)) + [n_batch]
                ]
                c_idx: Integer[Tensor, "n_batch+1 nnz"] = a_pattern.idx_coo[
                    list(range(n_batch)) + [n_batch + 1]
                ]
                # Note that r_idx.unbind(0) is equivalent to *r_idx for indexing.
                dLda_val = torch.sum(
                    -lambda_[r_idx.unbind(0)] * x[c_idx.unbind(0)], dim=-1
                )

            else:
                # if there are no batch dimensions, then the a_pattern.idx_coo is of
                # shape (sp=2, nnz).
                dLda_val = -lambda_[a_pattern.idx_coo[0]] * x[a_pattern.idx_coo[1]]

        # Free solver memory if not persistent.
        if not ctx.persistent:
            solver.free()
            ctx.solver = None

        return (dLda_val, None, dLdb, None, None, None)

from typing import Callable

import torch as t
from jaxtyping import Float, Integer


class SparseSolverWrapper(t.autograd.Function):
    @staticmethod
    def forward(
        A: Float[t.Tensor, "r c"],
        A_coo_idx: Integer[t.LongTensor, "2 nnz"],
        b: Float[t.Tensor, " r"],
        solver: Callable,
        transpose_solver: Callable,
    ) -> Float[t.Tensor, " c"]:
        x = solver(A, b)
        return x

    @staticmethod
    def setup_context(ctx, inputs, output):
        A, A_coo_idx, b, solver, transpose_solver = inputs

        ctx.save_for_backward(A, A_coo_idx, output)
        ctx.transpose_solver = transpose_solver

    @staticmethod
    def backward(
        ctx, dLdx: Float[t.Tensor, " c"]
    ) -> tuple[
        Float[t.Tensor, "r c"] | None, None, Float[t.Tensor, " r"] | None, None, None
    ]:
        """
        Let x be the solution to the linear system A@x = b. To find the gradient
        of some loss L wrt A and b, the adjoint method first defines a Lagrangian
        function F = L - lambda.T@(A@x - b), where lambda is the Lagrangian multiplier.
        By definition, F and L (as well as their total differentials) must be
        equal whenver A@x = b. From this condition, it follows that dLdb is given
        by lambda, and dLdA is given by the outer product -lambda@x.T.
        """
        needs_grad_A, _, needs_grad_b, _, _ = ctx.needs_input_grad

        if not (needs_grad_A or needs_grad_b):
            return (None, None, None, None, None)

        A, A_coo_idx, x = ctx.saved_tensors

        lambda_: Float[t.Tensor, " r"] = ctx.transpose_solver(A, dLdx)

        if needs_grad_b and not needs_grad_A:
            return (None, None, lambda_, None, None)

        # dLdA will have the same sparsity pattern as A
        dLdA_val = -lambda_[A_coo_idx[0]] * x[A_coo_idx[1]]

        # The gradient for A will be in the uncoalesced COO format.
        dLdA = t.sparse_coo_tensor(dLdA_val, A_coo_idx, A.shape)

        if needs_grad_A and not needs_grad_b:
            return (dLdA, None, None, None, None)

        if needs_grad_A and needs_grad_b:
            return (dLdA, None, lambda_, None, None)


def sparse_solver_wrapper(
    A: Float[t.Tensor, "r c"],
    A_coo_idx: Integer[t.LongTensor, "2 nnz"],
    b: Float[t.Tensor, " r"],
    solver: Callable,
    transpose_solver: Callable,
) -> Float[t.Tensor, " c"]:
    """
    This wrapper provides a differentiable wrapper for a sparse linear solver.

    Here, A is a sparse tensor/matrix and b is a dense, 1D tensor. The sparse
    matrix A should already be in the format/layout expected by the solver.
    The A_coo_idx is required for the backward pass to enforce the sparsity
    pattern of A.

    The solver should be a function that takes in A and b as its arguments
    and returns a dense, 1D tensor x that is the solution to A@x = b. The
    transpose_solver is similar to the solver function, but solves the
    A.T@x = b system.
    """
    return SparseSolverWrapper.apply(A, A_coo_idx, b, solver, transpose_solver)

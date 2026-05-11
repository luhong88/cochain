from dataclasses import dataclass, field
from typing import Any

import nvmath.sparse.advanced as nvmath_sp
import torch
from jaxtyping import Float, Integer
from torch import Tensor

from ....decoupled_tensor import SparsityPattern

__all__ = ["DirectSolverConfig"]

sp_literal_to_matrix_type = {
    "general": nvmath_sp.DirectSolverMatrixType.GENERAL,
    "symmetric": nvmath_sp.DirectSolverMatrixType.SYMMETRIC,
    "SPD": nvmath_sp.DirectSolverMatrixType.SPD,
}


@dataclass
class DirectSolverConfig:
    """
    Encapsulates all nvmath DirectSolver configuration.

    The `options` and `execution` arguments are directly passed to the arguments
    of the same names to the `DirectSolver` constructor. Note that direct control
    of stream is not allowed to prevent potential stream mismatch during backward().
    Finer grained control over the `plan_config`, `factorization_config`, and
    `solution_config` attributes of the `DirectSolver` is also possible through
    `plan_kwargs`, `factorization_kwargs`, and `solution_kwargs`; the dicts
    passed to these arguments should contain specific attributes of `plan_config`,
    `factorization_config`, and `solution_config` as keys, respectively.
    """

    options: nvmath_sp.DirectSolverOptions | None = None
    execution: nvmath_sp.ExecutionCUDA | nvmath_sp.ExecutionHybrid | None = None

    plan_kwargs: dict[str, Any] = field(default_factory=dict)
    factorization_kwargs: dict[str, Any] = field(default_factory=dict)
    solution_kwargs: dict[str, Any] = field(default_factory=dict)


def nvmath_adjoint_method(
    A_pattern: Integer[SparsityPattern, "*b r c"],
    x: Float[Tensor, "*b c"] | Float[Tensor, "*b c ch"],
    lambda_: Float[Tensor, "*b r"] | Float[Tensor, "*b r ch"],
) -> Float[Tensor, " nnz"]:
    """Use the adjoint method to compute the gradient for the nvmath direct solver."""
    if lambda_.ndim > 1:
        # Sum over the channel dimension while accounting for zero or more
        # batch dimensions.

        # Recall that A.pattern.idx_coo has shape (sp, nnz), where sp has size
        # equal to the number of batch dims plus 2 (i.e., sp = len(*b) + 2).
        n_batch = A_pattern.n_batch_dim

        # Extract the nonzero dLdA element row and col indices, accounting for
        # batch dimensions, and use them to extract the corresponding elements
        # from lambda_ and x to construct the nonzero outer product elements.
        r_idx: Integer[Tensor, "n_batch+1 nnz"] = A_pattern.idx_coo[
            list(range(n_batch)) + [n_batch]
        ]
        c_idx: Integer[Tensor, "n_batch+1 nnz"] = A_pattern.idx_coo[
            list(range(n_batch)) + [n_batch + 1]
        ]
        # Note that r_idx.unbind(0) is equivalent to *r_idx for indexing.
        dLdA_val = torch.sum(-lambda_[r_idx.unbind(0)] * x[c_idx.unbind(0)], dim=-1)

    else:
        # if there are no batch dimensions, then the A_pattern.idx_coo is of
        # shape (sp=2, nnz).
        dLdA_val = -lambda_[A_pattern.idx_coo[0]] * x[A_pattern.idx_coo[1]]

    return dLdA_val

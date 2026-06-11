from __future__ import annotations

__all__ = ["DirectSolverConfig"]

from dataclasses import dataclass, field
from typing import Any

import torch
from jaxtyping import Float, Integer
from torch import Tensor

from ....decoupled_tensor import SparsityPattern

try:
    import nvmath.sparse.advanced as nvmath_sp

    _HAS_NVMATH = True

except ImportError:
    _HAS_NVMATH = False


if _HAS_NVMATH:
    sp_literal_to_matrix_type = {
        "general": nvmath_sp.DirectSolverMatrixType.GENERAL,
        "symmetric": nvmath_sp.DirectSolverMatrixType.SYMMETRIC,
        "spd": nvmath_sp.DirectSolverMatrixType.SPD,
    }

else:
    sp_literal_to_matrix_type = {}


@dataclass
class DirectSolverConfig:
    """
    A dataclass for all nvmath `DirectSolver` configurations.

    Parameters
    ----------
    options
        Options for the direct sparse solver as a `DirectSolverOptions` object.
        This parameter will be pased as an argument of the same name to the
        `DirectSolver` constructor.
    execution
        Execution space options for the direct solver as a `ExecutionCUDA`
        or `ExecutionHybrid` object. This parameter will be passed as an argument
        of the same name to the `DirectSolver` constructor.
    plan_kwargs
        A dict for modifying attributes of `DirectSolver.plan_config`; the keys
        of this dict must match the attributes of `plan_config`.
    factorization_kwargs
        A dict for modifying attributes of `DirectSolver.factorization_config`;
        the keys of this dict must match the attributes of `factorization_config`.
    solution_kwargs
        A dict for modifying attributes of `DirectSolver.solution_config`; the keys
        of this dict must match the attributes of `solution_config`.

    Notes
    -----
    THe `DirectSolver` constructor allows for direct control of the CUDA execution
    stream, which is not allowed here to prevent potential stream mismatch during
    backward passes.
    """

    options: nvmath_sp.DirectSolverOptions | None = None
    execution: nvmath_sp.ExecutionCUDA | nvmath_sp.ExecutionHybrid | None = None

    plan_kwargs: dict[str, Any] = field(default_factory=dict)
    factorization_kwargs: dict[str, Any] = field(default_factory=dict)
    solution_kwargs: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not _HAS_NVMATH:
            raise ImportError("nvmath-python backend required.")


def nvmath_adjoint_method(
    a_pattern: Integer[SparsityPattern, "*b r c"],
    x: Float[Tensor, " c *ch_flat"] | Float[Tensor, "b c *ch_flat"],
    lambda_: Float[Tensor, " r *ch_flat"] | Float[Tensor, "b r *ch_flat"],
) -> Float[Tensor, " nz"]:
    """Use the adjoint method to compute the dLdA gradient for the nvmath direct solver."""
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
        dLdA_val = torch.sum(-lambda_[r_idx.unbind(0)] * x[c_idx.unbind(0)], dim=-1)

    else:
        # if there are no batch dimensions, then the A_pattern.idx_coo is of
        # shape (sp=2, nnz).
        dLdA_val = -lambda_[a_pattern.idx_coo[0]] * x[a_pattern.idx_coo[1]]

    return dLdA_val

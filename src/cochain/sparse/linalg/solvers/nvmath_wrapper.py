from __future__ import annotations

__all__ = ["DirectSolverConfig", "nvmath_direct_solver", "NVMathDirectSolver"]
import warnings
import weakref
from dataclasses import dataclass, field
from typing import Any, Literal

import torch
from jaxtyping import Float, Integer
from torch import Tensor

from ....utils.parsing import to_col_major
from ...decoupled_tensor import SparseDecoupledTensor, SparsityPattern
from ._sparse_solver import (
    BaseSparseSolver,
    InvSparseOperator,
    SparseSolverAutogradFunction,
)

try:
    import nvmath.sparse.advanced as nvmath_sp

    _HAS_NVMATH = True

except ImportError:
    _HAS_NVMATH = False

if _HAS_NVMATH:
    try:
        from cuda.core import Device
    except ImportError:
        from cuda.core.experimental import Device


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
    The `DirectSolver` constructor allows for direct control of the CUDA execution
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
            warnings.warn(
                "A DirectSolverConfig is defined but the nvmath-python backend is not detected.",
                UserWarning,
            )


if _HAS_NVMATH:
    sp_literal_to_matrix_type = {
        "general": nvmath_sp.DirectSolverMatrixType.GENERAL,
        "symmetric": nvmath_sp.DirectSolverMatrixType.SYMMETRIC,
        "spd": nvmath_sp.DirectSolverMatrixType.SPD,
    }

else:
    sp_literal_to_matrix_type = {}


class _NVMathSparseSolver(BaseSparseSolver):
    """A wrapper for the nvmath-python DirectSolver() sparse linear solver."""

    def __init__(
        self,
        a_val: Float[Tensor, " nz"],
        a_pattern: Integer[SparseDecoupledTensor, "*b r c"],
        b_flat: Float[Tensor, " r *ch_flat"] | Float[Tensor, "b r *ch_flat"],
        *,
        matrix_type: nvmath_sp.DirectSolverMatrixType = nvmath_sp.DirectSolverMatrixType.GENERAL,
        config: DirectSolverConfig | None = None,
    ):
        if not _HAS_NVMATH:
            raise ImportError("nvmath-python backend is required.")

        if a_val.ndim > 1:
            raise ValueError("Dense dimension in 'a' is not supported.")

        if not (a_val.is_cuda and b_flat.is_cuda):
            raise RuntimeError("Both a and b must be on the CUDA device.")

        if not a_pattern._is_int32_safe:
            warnings.warn(
                "The sparse indices of the input tensor 'a' cannot be safely cast "
                "to int32 dtype. This may cause downstream errors with the cuDSS backend.",
                UserWarning,
            )

        self.matrix_type = matrix_type
        self.a_val = a_val
        self.a_pattern = a_pattern

        self.dtype = a_val.dtype
        self.device = a_val.device
        self.shape = a_pattern.shape

        # Adjust solver config based on matrix_type.
        if config is None:
            config = DirectSolverConfig()

        if config.options is None:
            config.options = nvmath_sp.DirectSolverOptions(
                sparse_system_type=self.matrix_type
            )
        else:
            config.options.sparse_system_type = self.matrix_type

        self.config = config

        # Prepare the matrix and vector inputs to the solver.
        a_csr = SparseDecoupledTensor(self.a_pattern, self.a_val).to_sparse_csr()

        b_ready = to_col_major(
            b_flat,
            batch_first=self.a_pattern.n_batch_dim > 0,
        )

        # Do not give DirectSolver constructor the current stream to prevent
        # possible stream mismatch in subsequent calls to the solver by other
        # methods; instead, pass the stream to individual solver methods to ensure
        # sync between pytorch and nvmath.
        solver = nvmath_sp.DirectSolver(
            a=a_csr, b=b_ready, options=config.options, execution=config.execution
        )

        # Execute the plan() and factorize() phase upfront and cache the solver.
        stream = torch.cuda.current_stream()

        for k, v in config.plan_kwargs.items():
            setattr(solver.plan_config, k, v)
        solver.plan(stream=stream)

        for k, v in config.factorization_kwargs.items():
            setattr(solver.factorization_config, k, v)
        solver.factorize(stream=stream)

        for k, v in config.solution_kwargs.items():
            setattr(solver.solution_config, k, v)

        self.solver = solver

        # Keep track of the RHS vector and the trans argument from the last
        # time plan() and factorize() was called; so that these steps can be
        # skipped if the same RHS vector/trans argument are given again to the
        # solver. Note that, we track the RHS vector both through a weakref
        # as well as through its pytorch tensor version (this is to detect if
        # the tensor has been modified in-place).
        self._last_b_ref = weakref.ref(b_flat)
        self._last_b_version = b_flat._version
        self._last_trans = "N"

    @staticmethod
    def _flatten_b(
        b: Float[Tensor, " r *ch"] | Float[Tensor, "b r *ch"],
        a_pattern: Integer[SparsityPattern, "*b r c"],
    ) -> Float[Tensor, " r *ch_flat"] | Float[Tensor, "b r *ch_flat"]:
        """Flatten the channel dimensions of b, if there are any."""
        # Get the number of batch and non-channel dimensions in b and x.
        n_batch_dim = a_pattern.n_batch_dim
        n_non_ch_dims = n_batch_dim + 1

        # Put b in the correct shape and then the col-major layout.
        if b.ndim > n_non_ch_dims:
            # If b has channel dimension(s), flatten them.
            b_flat = b.flatten(start_dim=n_non_ch_dims)
        elif n_batch_dim > 0:
            # If b has no channel dimension but a batch dimension, add in a trivial
            # channel dimension.
            b_flat = b.unsqueeze(-1)
        else:
            # Otherwise b is a 1D vector and no reshaping is required.
            b_flat = b

        return b_flat

    @staticmethod
    def _unflatten_x(
        x_flat: Float[Tensor, " c *ch_flat"] | Float[Tensor, "b c *ch_flat"],
        a_pattern: Integer[SparsityPattern, "*b r c"],
        b: Float[Tensor, " r *ch"] | Float[Tensor, "b r *ch"],
    ) -> Float[Tensor, " r *ch"] | Float[Tensor, "b r *ch"]:
        """Unflatten the channel dimensions of x, if there are any."""
        # Get the expected non-channel dimension shapes from x_flat itself, and
        # get the expected channel dimension shapes from b.
        n_non_ch_dims = a_pattern.n_batch_dim + 1
        x = x_flat.view(*x_flat.shape[:n_non_ch_dims], *b.shape[n_non_ch_dims:])

        return x

    def solve(
        self,
        b_flat: Float[Tensor, " r *ch_flat"] | Float[Tensor, "b r *ch_flat"],
        *,
        trans: Literal["N", "T"] = "N",
    ) -> Float[Tensor, " c *ch_flat"] | Float[Tensor, "b c *ch_flat"]:
        """Solve a sparse linear system with the given RHS vector using `DirectSolver()`."""
        if not b_flat.is_cuda:
            raise RuntimeError("The input b must be on the CUDA device.")

        # Torch may call solve() on a separate thread where nvmath's internal
        # cuda state is uninitialized. We must explicitly call Device(id).set_current()
        # to bind the active CUDA context to this thread's local state, ensuring
        # nvmath operations recognize the device.
        Device(b_flat.device.index).set_current()

        # Determine whether a and/or b need to be reset with reset_operand().

        # For general matrices, if the trans argument is the same as the
        # previous, then a does not need to be reset.
        if (self.matrix_type == nvmath_sp.DirectSolverMatrixType.GENERAL) and (
            trans == self._last_trans
        ):
            reset_a = False
        # Symmetric matrices do not need to be reset, regardless of the
        # trans argument.
        elif self.matrix_type in [
            nvmath_sp.DirectSolverMatrixType.SYMMETRIC,
            nvmath_sp.DirectSolverMatrixType.SPD,
        ]:
            reset_a = False
        else:
            reset_a = True

        # If the input b_flat is the same object as the previous b_flat,
        # then b does not need to be reset.
        last_b_flat = getattr(self, "_last_b_ref", lambda: None)()
        reset_b = not (
            (last_b_flat is b_flat) and (self._last_b_version == b_flat._version)
        )

        if reset_a:
            match trans:
                case "N":
                    new_a = SparseDecoupledTensor(
                        self.a_pattern, self.a_val
                    ).to_sparse_csr()

                case "T":
                    new_a = SparseDecoupledTensor(
                        self.a_pattern, self.a_val
                    ).to_sparse_csr_transposed()

                case _:
                    raise ValueError(f"Unknown 'trans' argument '{trans}'.")

        if reset_b:
            b_ready = to_col_major(
                b_flat,
                batch_first=self.a_pattern.n_batch_dim > 0,
            )

        stream = torch.cuda.current_stream()

        # Whenever a is reset, needs to retrigger plan() and factorize(), which
        # is expensive.
        match (reset_a, reset_b):
            case (True, True):
                self.solver.reset_operands(a=new_a, b=b_ready, stream=stream)
                self.solver.plan(stream=stream)
                self.solver.factorize(stream=stream)

            case (True, False):
                self.solver.reset_operands(a=new_a, stream=stream)
                self.solver.plan(stream=stream)
                self.solver.factorize(stream=stream)

            case (False, True):
                self.solver.reset_operands(b=b_ready, stream=stream)

            case (False, False):
                pass

        x_flat = self.solver.solve(stream=stream)

        # Update the RHS/trans argument tracking.
        self._last_b_ref = weakref.ref(b_flat)
        self._last_b_version = b_flat._version
        self._last_trans = trans

        return x_flat

    def free(self):
        """Garbage collect the solver object."""
        if hasattr(self, "solver") and self.solver is not None:
            if getattr(self.solver, "valid_state", False) and hasattr(
                self.solver, "free"
            ):
                # Force device sync before gc.
                try:
                    import torch

                    if torch.cuda.is_initialized():
                        torch.cuda.synchronize(self.device)

                except Exception:
                    pass

                self.solver.free()
                self.solver = None

    def __del__(self):
        self.free()


def nvmath_direct_solver(
    a: Float[SparseDecoupledTensor, "*b r c"],
    b: Float[Tensor, " r *ch"] | Float[Tensor, "b r *ch"],
    *,
    sparse_system_type: Literal["general", "symmetric", "spd"] = "general",
    config: DirectSolverConfig | None = None,
) -> Float[Tensor, " c *ch"] | Float[Tensor, "b c *ch"]:
    """
    "Stateless" differentiable wrapper for `nvmath.sparse.advanced.DirectSolver`.

    Given a (batch of) sparse 2D matrix `a` and a (batch of) vector `b`, solve
    the linear system `a @ x = b` for `x`. Note that both `a` and `b` must be
    on the CUDA device.

    Parameters
    ----------
    a : [*b, r, c]
        A sparse 2D matrix represented as a `SparseDecoupledTensor`; at most
        one batch dimenion is supported.
    b : [*b, r, *ch]
        The RHS vector as a dense tensor; `b` can have at most one batch dimension
        but arbitrary channel dimensions. Internally, the `DirectSolver` expects
        `b` to be in column-major memory layout (i.e., the `r` dimension has
        stride 1) and have at most one channel dimension. If the input tensor `b`
        does not conform to this requirement, a reshaped copy will be created; see
        the Notes section for more details.
    sparse_system_type
        Whether the matrix `a` is symmetric, symmetric positive definite, or
        a general 2D matrix. It is recommended to specify the `sparse_system_type`
        argument to exploit the symmetry and spectral properties of `a`. Note that,
        `sparse_system_type` will override the `options.sparse_system_type`
        attribute in the `config`.
    config
        Additional config arguments to the `DirectSolver`; see the `DirectSolverConfig`
        class for more information.

    Returns
    -------
    x : [*b, c, *ch]
        The unknwon vector `x` in column-major memory layout; the batch and channel
        dimensions, if there are any, match those of the input `b`.

    Notes
    -----
    If the linear system `a @ x = b` does not have a unique solution, then both
    the forward pass and backward gradient will fail.

    The `DirectSolver` class supports batching/channel dimensions in both `a` and
    `b`. More specifically, this class supports the following four configurations:

    | Batch | Channel | `a.shape`   | `b.shape`     | `x.shape`     |
    |-------|---------|-------------|---------------|---------------|
    | False | False   | `[r, c]`    | `[r,]`        | `[c,]`        |
    | False | True    | `[r, c]`    | `[r, *ch]`    | `[c, *ch]`    |
    | True  | False   | `[b, r, c]` | `[b, r, (1)]` | `[b, c, (1)]` |
    | True  | True    | `[b, r, c]` | `[b, r, *ch]` | `[b, c, *ch]` |

    Here, `(1)` indicates an optional trivial channel dimension. When both `b`
    and `ch` dimensions are "active", the solver effectively solves `b*ch` linear
    systems of the form `a_i@x_ij = b_ij` for matrices `a_i` and vectors `x_ij`
    and `b_ij`, where `i` iterates over `b` and `j` iterates over the flattened
    channel dimensions.

    Internally, `DirectSolver` expects that `b` has one of the following shape and
    memory layout configurations:

    | Batch | Channel | `b.shape`         | `b.stride()`        |
    |-------|---------|-------------------|---------------------|
    | False | False   | `[r,]`            | `(1,)`              |
    | False | True    | `[r, ch_flat]`    | `(1, r)`            |
    | True  | False   | `[b, r, 1]`       | `(r, 1, r)`         |
    | True  | True    | `[b, r, ch_flat]` | `(r*ch_flat, 1, r)` |

    Specifically, `DirectSolver` expects `b` to be in the column-major memory layout
    with at most one batch and/or channel dimension. If the input tensor `b` does
    not have the correct shape and stride, this function will create a reshaped
    copy of `b`; note that, by default, PyTorch constructs tensors in row-major
    ordering.

    Note that `DirectSolver` also returns an `x` tensor in a similar column-major
    memory layout as `b`. This function returns a view of `x` that conforms to the
    shape of the input `b`.

    If either `a` or `b` requires gradient, then a `DirectSolver` object will be
    cached in memory to accelerate the backward pass; this memory will not be
    cleaned up until one of the following conditions is met:

    1) a backward() call with `retain_graph=False` has been made through the
    computation graph containing `a` or `b`, or
    2) all references to the output tensor (and its `grad_fn` and `ctx` attributes)
    from this function (and any derived tensors thereof) has gone out of scope/been
    detached from the computation graph.

    This function currently has the following limitations:

    * Double backward through this function is currently not supported.
    * `DirectSolver` also supports explicit batching, where a tensor with a batch
    dimension is represented as a list of tensors; this method is not supported
    in this function.
    * The `DirectSolver` class supports `a` as a batched sparse CSR tensor with an
    arbitrary number of batch dimensions, but this function only supports `a` with
    at most one batch dimension.
    * The sparse indices of `a` will be downcasted to `int32` for compatibility with
    the cuDSS backend.
    * If `a` is a general, non-symmetric matrix, the solver will need to redo
    the factorization step in `backward()` for the transposed matrix, because the
    `DirectSolver` class currently does not expose an option to solve the adjoint
    system directly.
    """
    b_flat = _NVMathSparseSolver._flatten_b(b, a.pattern)

    solver = _NVMathSparseSolver(
        a.values,
        a.pattern,
        b_flat,
        matrix_type=sp_literal_to_matrix_type[sparse_system_type],
        config=config,
    )

    x_flat = SparseSolverAutogradFunction.apply(
        a.values, a.pattern, b_flat, solver, "N", False
    )
    x = _NVMathSparseSolver._unflatten_x(x_flat, a.pattern, b)

    return x


class NVMathDirectSolver(InvSparseOperator):
    """
    "Stateful" differentiable wrapper for `nvmath.sparse.advanced.DirectSolver`.

    Given a (batch of) sparse 2D matrix `a` and a (batch of) vector `b`, this class
    computes and caches the LU factorization of `a` for the purpose of solving
    the linear system `a @ x = b` for `x`. Once initialized, call the class
    instance to with a (new) RHS `b` to perform the linear solve. Note that both
    `a` and `b` must be on the CUDA device.

    Parameters
    ----------
    a : [*b, r, c]
        A sparse 2D matrix represented as a `SparseDecoupledTensor`; at most
        one batch dimenion is supported.
    b : [*b, r, *ch]
        A dummy RHS vector; b` can have at most one batch dimension but arbitrary
        channel dimensions. This argument is required for the purpose of setting
        the size of the RHS vector; all subsequent calls to the solver must use
        RHS with the same shape as this tensor.
    sparse_system_type
        Whether the matrix `a` is symmetric, symmetric positive definite, or
        a general 2D matrix. It is recommended to specify the `sparse_system_type`
        argument to exploit the symmetry and spectral properties of `a`. Note that,
        `sparse_system_type` will override the `options.sparse_system_type`
        attribute in the `config`.
    config
        Additional config arguments to the `DirectSolver`; see the `DirectSolverConfig`
        class for more information.

    Notes
    -----
    This class implements a similar nvmath `DirectSolver` wrapper as the functional
    `nvmath_direct_solver()`. Both wrappers cache the solver object and the factorized
    matrix for backward passes; however, in the functional wrapper, the cached solver
    is tied to the computation graph and gc'ed after the backward pass, whereas,
    in this class, the cached solver is tied to the lifecycle of the class instances,
    which allows the same factorized matrix to be re-used to solve different linear
    systems. See the `nvmath_direct_solver()` function for more details on the
    requirements and limitations of this wrapper.
    """

    def __init__(
        self,
        a: Float[SparseDecoupledTensor, "r c"],
        b: Float[Tensor, " r *ch"],
        *,
        sparse_system_type: Literal["general", "symmetric", "spd"] = "general",
        config: DirectSolverConfig | None = None,
    ):
        self.dtype = a.dtype
        self.shape = a.shape
        self.device = a.device

        b_flat = _NVMathSparseSolver._flatten_b(b, a.pattern)

        self.solver = _NVMathSparseSolver(
            a.values,
            a.pattern,
            b_flat,
            matrix_type=sp_literal_to_matrix_type[sparse_system_type],
            config=config,
        )

    def __call__(
        self, b: Float[Tensor, " r *ch"], trans: Literal["N", "T"] = "N"
    ) -> Float[Tensor, " c *ch"]:
        """
        Solve a sparse linear system with the given RHS vector using `DirectSolver`.

        Parameters
        ----------
        b : [r, *ch]
            The RHS vector as a dense tensor; `b` can have at most one batch
            dimension but arbitrary channel dimensions. Internally, the `DirectSolver`
            expects `b` to be in column-major memory layout (i.e., the `r` dimension
            has stride 1) and have at most one channel dimension. If the input
            tensor `b` does not conform to this requirement, a reshaped copy will
            be created; see the `nvmath_direct_solver()` function for more details.

        Returns
        -------
        x : [c, *ch]
            The unknown vector `x` with channel dimensions matching those of `b`.
        """
        b_flat = _NVMathSparseSolver._flatten_b(b, self.solver.a_pattern)
        x_flat = SparseSolverAutogradFunction.apply(
            self.solver.a_val, self.solver.a_pattern, b_flat, self.solver, trans, True
        )
        x = _NVMathSparseSolver._unflatten_x(x_flat, self.solver.a_pattern, b)

        return x

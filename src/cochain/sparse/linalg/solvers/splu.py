from __future__ import annotations

__all__ = ["splu", "SuperLU"]

import warnings
from typing import Literal

import scipy.sparse
import scipy.sparse.linalg
import torch
from jaxtyping import Float
from torch import Tensor

from ....utils.parsing import to_np
from ....utils.stream import cupy_in_torch_stream
from ...decoupled_tensor import SparseDecoupledTensor
from ...decoupled_tensor._conversion import sdt_to_cupy_csc, sdt_to_scipy_csc
from ._sparse_solver import SparseSolver, SparseSolverAutogradFunction

try:
    import cupy as cp
    import cupyx.scipy.sparse as cp_sp
    import cupyx.scipy.sparse.linalg as cp_sp_linalg

    _HAS_CUPY = True

except ImportError:
    _HAS_CUPY = False


class _SuperLUSparseSolver(SparseSolver):
    def __init__(
        self,
        a: Float[SparseDecoupledTensor, "r c"],
        *,
        matrix_type: Literal["general", "symmetric", "spd"] = "general",
        backend: Literal["cupy", "scipy"],
        **splu_kwargs,
    ):
        if a.n_batch_dim > 0:
            raise ValueError("Batch dimension in 'a' is not supported.")
        if a.n_dense_dim > 0:
            raise ValueError("Dense dimension in 'a' is not supported.")

        if not a.pattern._is_int32_safe:
            warnings.warn(
                "The sparse indices of the input tensor 'A' cannot be safely "
                "cast to int32 dtype.",
                UserWarning,
            )

        if (backend == "cupy") and not _HAS_CUPY:
            raise ImportError("CuPy backend required.")

        self.matrix_type = matrix_type
        self.a_val = a.values
        self.a_pattern = a.pattern
        self.backend = backend

        self.dtype = a.dtype
        self.device = a.device
        self.shape = a.shape

        match self.backend:
            case "cupy":
                # Force CuPy to use the current Pytorch stream.
                with cupy_in_torch_stream():
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

    @staticmethod
    def _flatten_b(b: Float[Tensor, " r *ch"]) -> Float[Tensor, " r *ch_flat"]:
        if b.ndim > 2:
            # Flatten multiple channel dims, (r, *ch) -> (r, ch_flat)
            b_flat = b.reshape(b.size(0), -1)
        else:
            b_flat = b

        return b_flat

    @staticmethod
    def _unflatten_x(
        x_flat: Float[Tensor, " c *ch_flat"], b: Float[Tensor, " r *ch"]
    ) -> Float[Tensor, " c *ch"]:
        if x_flat.ndim == 1:
            x = x_flat
        else:
            # (c, ch_flat) -> (c, *ch)
            x = x_flat.reshape(-1, *b.shape[1:])

        return x

    def solve(
        self,
        b_flat: Float[Tensor, " r *ch_flat"],
        *,
        trans: Literal["N", "T"] = "N",
    ) -> Float[Tensor, " c *ch_flat"]:
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
        match self.backend:
            case "cupy":
                # Force CuPy to use the current Pytorch stream.
                with cupy_in_torch_stream():
                    b_cp = cp.from_dlpack(b_flat.detach().contiguous())
                    x_flat = torch.from_dlpack(self.solver.solve(b_cp, trans=trans))

            case "scipy":
                b_np = to_np(b_flat, contiguous=True)
                x_flat = torch.from_numpy(self.solver.solve(b_np, trans=trans)).to(
                    dtype=self.dtype, device=self.device
                )

        return x_flat

    def free(self):
        if (
            (self.backend == "cupy")
            and hasattr(self, "solver")
            and (self.solver is not None)
        ):
            # Force device sync before gc.
            try:
                import torch

                if torch.cuda.is_initialized():
                    torch.cuda.synchronize(self.device)

            except Exception:
                pass

        self.solver = None

    def __del__(self):
        self.free()


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
    solver = _SuperLUSparseSolver(
        a, matrix_type="general", backend=backend, **splu_kwargs
    )

    b_flat = _SuperLUSparseSolver._flatten_b(b)
    x_flat = SparseSolverAutogradFunction.apply(
        a.values, a.pattern, b_flat, solver, "N", False
    )
    x = _SuperLUSparseSolver._unflatten_x(x_flat, b)

    return x


class SuperLU:
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
        self.solver = _SuperLUSparseSolver(
            a, matrix_type="general", backend=backend, **splu_kwargs
        )

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
        b_flat = _SuperLUSparseSolver._flatten_b(b)
        x_flat = SparseSolverAutogradFunction.apply(
            self.solver.a_val, self.solver.a_pattern, b_flat, self.solver, trans, True
        )
        x = _SuperLUSparseSolver._unflatten_x(x_flat, b)

        return x

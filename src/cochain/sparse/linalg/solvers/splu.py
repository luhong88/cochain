from __future__ import annotations

import warnings
from typing import Literal

import scipy.sparse
import scipy.sparse.linalg
import torch
from jaxtyping import Float, Integer
from torch import Tensor

from ....utils.parsing import to_np
from ....utils.stream import cupy_in_torch_stream
from ...decoupled_tensor import SparseDecoupledTensor, SparsityPattern
from ...decoupled_tensor._conversion import sdt_to_cupy_csc, sdt_to_scipy_csc
from ._sparse_solver import SparseSolver

try:
    import cupy as cp
    import cupyx.scipy.sparse as cp_sp
    import cupyx.scipy.sparse.linalg as cp_sp_linalg

    _HAS_CUPY = True

except ImportError:
    _HAS_CUPY = False


class SuperLUSparseSolver(SparseSolver):
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

    def solve(
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
                # Force CuPy to use the current Pytorch stream.
                with cupy_in_torch_stream():
                    b_cp = cp.from_dlpack(b_flat.detach().contiguous())
                    x = torch.from_dlpack(self.solver.solve(b_cp, trans=trans))

            case "scipy":
                b_np = to_np(b_flat, contiguous=True)
                x = torch.from_numpy(self.solver.solve(b_np, trans=trans)).to(
                    dtype=self.dtype, device=self.device
                )

        if x.ndim == 1:
            x_reshaped = x
        else:
            # (c, ch_flat) -> (c, *ch)
            x_reshaped = x.reshape(-1, *b.shape[1:])

        return x_reshaped

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

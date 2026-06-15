from __future__ import annotations

import torch
from jaxtyping import Float
from torch import Tensor

from .....utils.parsing import to_col_major
from ....decoupled_tensor import DiagDecoupledTensor, SparseDecoupledTensor
from ...solvers import DirectSolverConfig
from ...solvers.nvmath import _NVMathSparseSolver
from ..base._inv_operator import BaseNVMathInvSymSpOp

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


class IdentityOperator:
    def __init__(self):
        pass

    def __matmul__(self, other):
        return other

    def __rmatmul__(self, other):
        return other

    def to(self, *args, **kwargs):
        return self


if _HAS_NVMATH:

    class ShiftInvSymSpOp:
        """
        A linear operator used to solve an eigenvalue problem in the shift-invert mode.

        For the shift-inverted eigenvalue problem inv(A - σI)@x = (λ - σ)^(-1)*x,
        this class represents the operator inv(A - σI) and implements __matmul__().
        Performing the matrix-vector multiplication inv(A - σI)@x = y is equal to
        solving the sparse linear system (A - σI)@y = x.
        """

        def __init__(
            self,
            a_sdt: Float[SparseDecoupledTensor, "m m"],
            sigma: float,
            n: int,
            config: DirectSolverConfig,
        ):
            self.n = n

            eye = DiagDecoupledTensor.eye(
                a_sdt.size(-1), dtype=a_sdt.dtype, device=a_sdt.device
            )
            a_shift_inv = SparseDecoupledTensor.assemble(a_sdt, -sigma * eye)

            # Solve a linear system with a channel dim of at most 3n size.
            b_dummy = to_col_major(
                torch.zeros(
                    (a_shift_inv.size(-1), 3 * n),
                    dtype=a_shift_inv.dtype,
                    device=a_shift_inv.device,
                ),
                batch_first=False,
            )

            self.solver = _NVMathSparseSolver(
                a_shift_inv.values,
                a_shift_inv.pattern,
                b_dummy,
                matrix_type=nvmath_sp.DirectSolverMatrixType.SYMMETRIC,
                config=config,
            )

        def __matmul__(self, x: Float[Tensor, "m k"]) -> Float[Tensor, "m k"]:
            # Pad up to 3n channel dims
            k = x.size(-1)
            pad = 3 * self.n - k

            x_padded_col_major = to_col_major(
                torch.nn.functional.pad(x, (0, pad, 0, 0)), batch_first=False
            )

            # Note that there is no need to "unflatten" b since there is only one
            # channel dimension.
            b = self.solver.solve(x_padded_col_major)

            return b[:, :k]

    class ShiftInvSymGEPSpOp(BaseNVMathInvSymSpOp):
        """
        A linear operator used to solve a generalized eigenvalue problem in the
        shift-invert mode.

        For the shift-inverted generalized eigenvalue problem
        inv(A - σM)@M@x = (λ - σ)^(-1)*x, this class represents the operator inv(A - σM)@M
        and implements __matmul__(). Performing the matrix-vector multiplication
        inv(A - σM)@M@x = y is equal to solving the sparse linear system (A - σM)@y = M@x.
        """

        def __init__(
            self,
            A_op: Float[SparseDecoupledTensor, "m m"],
            M_op: Float[SparseDecoupledTensor, "m m"],
            sigma: float,
            n: int,
            config: DirectSolverConfig,
        ):
            if not A_op.pattern._is_int32_safe:
                raise ValueError(
                    "The sparse indices of the input tensor 'A' cannot be safely "
                    "cast to int32 dtype."
                )

            if not M_op.pattern._is_int32_safe:
                raise ValueError(
                    "The sparse indices of the input tensor 'M' cannot be safely "
                    "cast to int32 dtype."
                )

            self.n = n

            # Pytorch currently does not support operations like A - M on sparse CSR
            # tensors.
            A_shift_inv = SparseDecoupledTensor.assemble(
                A_op, -sigma * M_op
            ).to_sparse_csr()
            self.M_csr = M_op.to_sparse_csr()

            # Solve a linear system with at most 3n channel dims
            b_dummy = (
                torch.zeros(
                    (self.M_csr.size(-1), 3 * n),
                    dtype=self.M_csr.dtype,
                    device=self.M_csr.device,
                )
                .transpose(-1, -2)
                .contiguous()
                .transpose(-1, -2)
            )

            super().__init__(A_shift_inv, b_dummy, config, torch.cuda.current_stream())

        def __matmul__(self, x: Float[Tensor, "m k"]) -> Float[Tensor, "m k"]:
            stream = torch.cuda.current_stream()
            torch.cuda.set_device(x.device)
            Device(x.device.index).set_current()

            Mx = self.M_csr @ x

            # Pad up to 3n channel dims
            k = Mx.size(-1)
            pad = 3 * self.n - k

            Mx_padded_col_major = (
                torch.nn.functional.pad(Mx, (0, pad, 0, 0))
                .transpose(-1, -2)
                .contiguous()
                .transpose(-1, -2)
            )

            self.solver.reset_operands(b=Mx_padded_col_major, stream=stream)
            b = self.solver.solve(stream=stream)

            return b[:, :k]

else:

    class ShiftInvSymSpOp:
        def __init__(self, *args, **kwargs):
            raise ImportError("nvmath-python backend required.")

    class ShiftInvSymGEPSpOp:
        def __init__(self, *args, **kwargs):
            raise ImportError("nvmath-python backend required.")

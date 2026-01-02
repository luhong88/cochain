import torch as t
from cuda.core.experimental import Device
from jaxtyping import Float

from ...operators import SparseOperator
from ..solvers.nvmath_wrapper import DirectSolverConfig
from ._inv_operator import BaseInvSymSpOp


def _batched_csr_eye(
    n: int, b: int, val_dtype: t.dtype, idx_dtype: t.dtype, device: t.device
) -> Float[t.Tensor, "*b n n"]:
    if b == 0:
        identity = t.sparse_csr_tensor(
            crow_indices=t.arange(n + 1, dtype=idx_dtype, device=device),
            col_indices=t.arange(n, dtype=idx_dtype, device=device),
            values=t.ones(n, dtype=val_dtype, device=device),
        )
    else:
        identity = t.sparse_csr_tensor(
            crow_indices=t.tile(
                t.arange(n + 1, dtype=idx_dtype, device=device), (b, 1)
            ),
            col_indices=t.tile(t.arange(n, dtype=idx_dtype, device=device), (b, 1)),
            values=t.tile(t.ones(n, dtype=val_dtype, device=device), (b, 1)),
        )

    return identity


class SpPrecond(BaseInvSymSpOp):
    """
    Exact preconditioner for solving a standard or generalized eigenvalue problem
    with LOBPCG.

    For the standard eigenvalue problem A@x = λ*x and the generalized eigenvalue
    problem A@x = λ*M@x, this exact preconditioner is inv(A). If r is a residual
    vector, applying the preconditioner to r is equivalent to solving a sparse
    linear system A@w = r for w.
    """

    def __init__(
        self,
        A_op: Float[SparseOperator, "m m"],
        n: int,
        diag_damp: float | int | None,
        config: DirectSolverConfig,
    ):
        A_csr = A_op.to_sparse_csr(int32=True)

        b_dummy = t.zeros((A_csr.size(-1), n), dtype=A_csr.dtype, device=A_csr.device)

        if diag_damp == 0:
            op = A_csr
        else:
            if diag_damp is None:
                eps = 1e-4 * A_op.tr / A_op.size(0)
            else:
                eps = diag_damp

            eye = _batched_csr_eye(
                n=A_csr.size(-1),
                b=0,
                val_dtype=A_csr.dtype,
                idx_dtype=t.int32,
                device=A_csr.device,
            )
            op = A_csr + eps * eye

        super().__init__(op, b_dummy, config)

    def __matmul__(self, res: Float[t.Tensor, "m n"]) -> Float[t.Tensor, "m n"]:
        stream = t.cuda.current_stream()
        Device(res.device.index).set_current()

        res_col_major = res.transpose(-1, -2).contiguous().transpose(-1, -2)

        self.solver.reset_operands(b=res_col_major, stream=stream)

        return self.solver.solve(stream=stream)


class IdentityPrecond:
    def __init__(self):
        pass

    def __matmal__(self, res):
        return res


class ShiftInvSymSpOp(BaseInvSymSpOp):
    """
    A linear operator used to solve an eigenvalue problem in the shift-invert mode.

    For the shift-inverted eigenvalue problem inv(A - σI)@x = (λ - σ)^(-1)*x,
    this class represents the operator inv(A - σI) and implements __matmul__().
    Performing the matrix-vector multiplication inv(A - σI)@x = y is equal to
    solving the sparse linear system (A - σI)@y = x.
    """

    def __init__(
        self,
        A_op: Float[SparseOperator, "m m"],
        sigma: float,
        n: int,
        config: DirectSolverConfig,
    ):
        A_csr = A_op.to_sparse_csr(int32=True)
        eye = _batched_csr_eye(
            n=A_csr.size(-1),
            b=0,
            val_dtype=A_csr.dtype,
            idx_dtype=t.int32,
            device=A_csr.device,
        )
        A_shift_inv = A_csr - sigma * eye

        b_dummy = t.zeros((A_csr.size(-1), n), dtype=A_csr.dtype, device=A_csr.device)

        super().__init__(A_shift_inv, b_dummy, config)

    def __matmul__(self, x: Float[t.Tensor, "m n"]) -> Float[t.Tensor, "m n"]:
        stream = t.cuda.current_stream()
        t.cuda.set_device(x.device)
        Device(x.device.index).set_current()

        x_col_major = x.transpose(-1, -2).contiguous().transpose(-1, -2)

        self.solver.reset_operands(b=x_col_major, stream=stream)
        b = self.solver.solve(stream=stream)

        return b


class ShiftInvSymGEPSpOp(BaseInvSymSpOp):
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
        A_op: Float[SparseOperator, "m m"],
        M_op: Float[SparseOperator, "m m"],
        sigma: float,
        n: int,
        config: DirectSolverConfig,
    ):
        A_csr = A_op.to_sparse_csr(int32=True)
        self.M_csr = M_op.to_sparse_csr(int32=True)

        A_shift_inv = A_csr - sigma * self.M_csr

        b_dummy = t.zeros((A_csr.size(-1), n), dtype=A_csr.dtype, device=A_csr.device)

        super().__init__(A_shift_inv, b_dummy, config)

    def __matmul__(self, x: Float[t.Tensor, "m n"]) -> Float[t.Tensor, "m n"]:
        stream = t.cuda.current_stream()
        t.cuda.set_device(x.device)
        Device(x.device.index).set_current()

        Mx = self.M_csr @ x
        Mx_col_major = Mx.transpose(-1, -2).contiguous().transpose(-1, -2)

        self.solver.reset_operands(b=Mx_col_major, stream=stream)
        b = self.solver.solve(stream=stream)

        return b


type SparseOperatorLike = SparseOperator | ShiftInvSymSpOp | ShiftInvSymGEPSpOp

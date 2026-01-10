import torch as t
from cuda.core.experimental import Device
from jaxtyping import Float

from ...operators import DiagOperator, SparseOperator
from ..solvers.nvmath_wrapper import DirectSolverConfig
from ._inv_operator import BaseInvSymSpOp


class SpPrecond(BaseInvSymSpOp):
    """
    Diagonally damped preconditioner for solving a standard or generalized
    eigenvalue problem with LOBPCG.

    For the standard eigenvalue problem A@x = λ*x and the generalized eigenvalue
    problem A@x = λ*M@x, this preconditioner is inv(A + ϵI). If r is a residual
    vector, applying the preconditioner to r is equivalent to solving a sparse
    linear system (A + ϵI)@w = r for w.
    """

    def __init__(
        self,
        A_op: Float[SparseOperator, "m m"],
        n: int,
        diag_damp: float | int | None,
        config: DirectSolverConfig,
    ):
        self.n = n

        if self.n > 1:
            b_dummy = (
                t.zeros((A_op.size(-1), n), dtype=A_op.dtype, device=A_op.device)
                .transpose(-1, -2)
                .contiguous()
                .transpose(-1, -2)
            )
        else:
            b_dummy = t.zeros(A_op.size(-1), dtype=A_op.dtype, device=A_op.device)

        if diag_damp == 0:
            op = A_op.to_sparse_csr(int32=True)
        else:
            if diag_damp is None:
                eps = 1e-4 * A_op.tr / A_op.size(0)
            else:
                eps = diag_damp

            # Pytorch currently does not support operations like A - I on sparse
            # CSR tensors.
            eye = DiagOperator.eye(A_op.size(-1), dtype=A_op.dtype, device=A_op.device)

            op = SparseOperator.assemble(A_op, eps * eye).to_sparse_csr(int32=True)

        super().__init__(op, b_dummy, config, t.cuda.current_stream())

    def __matmul__(self, res: Float[t.Tensor, "m n"]) -> Float[t.Tensor, "m n"]:
        stream = t.cuda.current_stream()
        Device(res.device.index).set_current()

        if self.n > 1:
            res_col_major = res.transpose(-1, -2).contiguous().transpose(-1, -2)
        else:
            res_col_major = res.squeeze()

        self.solver.reset_operands(b=res_col_major, stream=stream)

        return self.solver.solve(stream=stream)


class IdentityOperator:
    def __init__(self):
        pass

    def __matmul__(self, res):
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
        self.n = n

        # Pytorch currently does not support operations like A - I on sparse CSR
        # tensors.
        eye = DiagOperator.eye(A_op.size(-1), dtype=A_op.dtype, device=A_op.device)
        A_shift_inv = SparseOperator.assemble(A_op, -sigma * eye).to_sparse_csr(
            int32=True
        )

        if n > 1:
            b_dummy = (
                t.zeros(
                    (A_shift_inv.size(-1), n),
                    dtype=A_shift_inv.dtype,
                    device=A_shift_inv.device,
                )
                .transpose(-1, -2)
                .contiguous()
                .transpose(-1, -2)
            )
        else:
            b_dummy = t.zeros(
                A_shift_inv.size(-1),
                dtype=A_shift_inv.dtype,
                device=A_shift_inv.device,
            )

        super().__init__(A_shift_inv, b_dummy, config, t.cuda.current_stream())

    def __matmul__(self, x: Float[t.Tensor, "m n"]) -> Float[t.Tensor, "m n"]:
        stream = t.cuda.current_stream()
        t.cuda.set_device(x.device)
        Device(x.device.index).set_current()

        if self.n > 1:
            x_col_major = x.transpose(-1, -2).contiguous().transpose(-1, -2)
        else:
            x_col_major = x.squeeze()

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
        self.n = n

        # Pytorch currently does not support operations like A - M on sparse CSR
        # tensors.
        A_shift_inv = SparseOperator.assemble(A_op, -sigma * M_op).to_sparse_csr(
            int32=True
        )
        self.M_csr = M_op.to_sparse_csr(int32=True)

        if self.n > 1:
            b_dummy = (
                t.zeros(
                    (self.M_csr.size(-1), n),
                    dtype=self.M_csr.dtype,
                    device=self.M_csr.device,
                )
                .transpose(-1, -2)
                .contiguous()
                .transpose(-1, -2)
            )
        else:
            b_dummy = t.zeros(
                self.M_csr.size(-1),
                dtype=self.M_csr.dtype,
                device=self.M_csr.device,
            )

        super().__init__(A_shift_inv, b_dummy, config, t.cuda.current_stream())

    def __matmul__(self, x: Float[t.Tensor, "m n"]) -> Float[t.Tensor, "m n"]:
        stream = t.cuda.current_stream()
        t.cuda.set_device(x.device)
        Device(x.device.index).set_current()

        Mx = self.M_csr @ x

        if self.n > 1:
            Mx_col_major = Mx.transpose(-1, -2).contiguous().transpose(-1, -2)
        else:
            Mx_col_major = Mx.squeeze()

        self.solver.reset_operands(b=Mx_col_major, stream=stream)
        b = self.solver.solve(stream=stream)

        return b

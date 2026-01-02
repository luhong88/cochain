import pytest
import torch as t

from cochain.sparse.linalg.solvers import splu
from cochain.sparse.operators import SparseOperator

itemize_backend = pytest.mark.parametrize(
    "backend",
    [
        pytest.param("scipy", marks=[]),
        pytest.param("cupy", marks=[pytest.mark.gpu_only]),
    ],
)


@itemize_backend
def test_splu_forward(A, backend, device):
    A_op = SparseOperator.from_tensor(A).to(device)
    A_dense = A_op.to_dense()

    n_dim = A_op.size(0)

    x_true = t.randn(n_dim).to(device)
    b = A_dense @ x_true

    x = splu(A_op, b, backend=backend)

    t.testing.assert_close(x, x_true)


@itemize_backend
def test_splu_forward_with_channel_dim(A, backend, device):
    A_op = SparseOperator.from_tensor(A).to(device)
    A_dense = A_op.to_dense()

    n_dim = A_op.size(0)
    n_ch = 2

    x_true = t.randn(n_dim, n_ch).to(device)
    b = A_dense @ x_true

    x = splu(A_op, b, backend=backend)
    x_T = splu(A_op, b.T, backend=backend, channel_first=True)

    t.testing.assert_close(x.T, x_T)
    t.testing.assert_close(x, x_true)


@itemize_backend
@pytest.mark.parametrize(
    "n_ch1, n_ch2",
    [(2, 3), (2, 1), (1, 2)],
)
def test_splu_forward_with_complex_channel_dim(A, n_ch1, n_ch2, backend, device):
    A_op = SparseOperator.from_tensor(A).to(device)
    A_dense = A_op.to_dense()

    n_dim = A_op.size(0)

    x_true = t.randn(n_dim, n_ch1, n_ch2).to(device)
    b = t.einsum("ij,jkl->ikl", A_dense, x_true)

    x = splu(A_op, b, backend="scipy")
    x_T = splu(A_op, b.movedim(0, -1), backend=backend, channel_first=True)

    t.testing.assert_close(x.movedim(0, -1), x_T)
    t.testing.assert_close(x, x_true)


@itemize_backend
def test_splu_backward(A, backend, device):
    """
    Let A@x=b and define the loss function as L = <x, v>. Check that the gradients
    dLdA and dLdb computed through the adjoint method matches the autograd gradients
    from t.linalg.solve() (using dense A).
    """
    A_op = SparseOperator.from_tensor(A).to(device)
    A_dense = A_op.to_dense()
    n_dim = A_op.size(0)

    # Compute b and v
    b = t.randn(n_dim).to(device)
    v = t.randn(n_dim).to(device)

    # Compute the dLdA and dLdb gradients via the adjoint method.
    A_op.requires_grad_()
    b.requires_grad_()
    x_via_sp = splu(A_op, b, backend=backend)
    loss = t.sum(x_via_sp * v)
    loss.backward()

    A_sp_grad = A_op.val.grad.detach().clone()
    b_sp_grad = b.grad.detach().clone()

    # Compute the dLdA and dLdb gradients via autograd using a dense A.
    A_dense.requires_grad_()
    b.grad = None  # clear the existing gradient on b
    x_via_dense = t.linalg.solve(A_dense, b)
    loss = t.sum(x_via_dense * v)
    loss.backward()

    # Extract the nonzero elements of dLdA computed using a dense A.
    A_dense_grad = (
        A_dense.grad[A_op.sp_topo.idx_coo[0], A_op.sp_topo.idx_coo[1]].detach().clone()
    )
    b_dense_grad = b.grad.detach().clone()

    # Assert that the adjoint method gradients agree with autograd.
    t.testing.assert_close(A_sp_grad, A_dense_grad)
    t.testing.assert_close(b_sp_grad, b_dense_grad)

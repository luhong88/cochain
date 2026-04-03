import pytest
import torch

from cochain.sparse.decoupled_tensor import SparseDecoupledTensor
from cochain.sparse.linalg.solvers import nvmath_direct_solver


@pytest.mark.gpu_only
def test_direct_solver_forward(A, device):
    A_op = SparseDecoupledTensor.from_tensor(A).to(device)
    A_dense = A_op.to_dense()

    n_dim = A_op.size(0)

    x_true = torch.randn(n_dim).to(device)
    b = A_dense @ x_true

    x = nvmath_direct_solver(A_op, b)

    torch.testing.assert_close(x, x_true)


@pytest.mark.gpu_only
def test_direct_solver_with_channel_dim(A, device):
    A_op = SparseDecoupledTensor.from_tensor(A).to(device)
    A_dense = A_op.to_dense()

    n_dim = A_op.size(0)
    n_ch = 2

    x_true = torch.randn(n_ch, n_dim).to(device)
    b = torch.einsum("ij,kj->ki", A_dense, x_true)

    x = nvmath_direct_solver(A_op, b)

    torch.testing.assert_close(x, x_true.T)


@pytest.mark.gpu_only
def test_direct_solver_with_batch_dim(A_batched, device):
    A_op = SparseDecoupledTensor.from_tensor(A_batched).to(device)
    A_dense = A_op.to_dense()

    n_dim = A_op.size(-1)
    n_batch = A_op.size(0)

    x_true = torch.randn(n_batch, n_dim).to(device)
    b = torch.einsum("bij,bj->bi", A_dense, x_true).view(n_batch, 1, n_dim)

    x = nvmath_direct_solver(A_op, b)
    x_true_shaped = x_true.view(n_batch, n_dim, 1)

    torch.testing.assert_close(x, x_true_shaped)


@pytest.mark.gpu_only
def test_direct_solver_with_batch_channel_dim(A_batched, device):
    A_op = SparseDecoupledTensor.from_tensor(A_batched).to(device)
    A_dense = A_op.to_dense()

    n_dim = A_op.size(-1)
    n_batch = A_op.size(0)
    n_ch = 2

    x_true = torch.randn(n_batch, n_dim, n_ch).to(device)
    b = torch.einsum("bij,bjc->bci", A_dense, x_true)

    x = nvmath_direct_solver(A_op, b)

    torch.testing.assert_close(x, x_true)


@pytest.mark.gpu_only
def test_direct_solver_backward(A, device):
    """
    Let A@x=b and define the loss function as L = <x, v>. Check that the gradients
    dLdA and dLdb computed through the adjoint method matches the autograd gradients
    from torch.linalg.solve() (using dense A).
    """
    A_op = SparseDecoupledTensor.from_tensor(A).to(device)
    A_dense = A_op.to_dense()
    n_dim = A_op.size(0)

    # Compute b and v
    b = torch.randn(n_dim).to(device)
    v = torch.randn(n_dim).to(device)

    # Compute the dLdA and dLdb gradients via the adjoint method.
    A_op.requires_grad_()
    b.requires_grad_()
    x_via_sp = nvmath_direct_solver(A_op, b)
    loss = torch.sum(x_via_sp * v)
    loss.backward()

    A_sp_grad = A_op.val.grad.detach().clone()
    b_sp_grad = b.grad.detach().clone()

    # Compute the dLdA and dLdb gradients via autograd using a dense A.
    A_dense.requires_grad_()
    b.grad = None  # clear the existing gradient on b
    x_via_dense = torch.linalg.solve(A_dense, b)
    loss = torch.sum(x_via_dense * v)
    loss.backward()

    # Extract the nonzero elements of dLdA computed using a dense A.
    A_dense_grad = (
        A_dense.grad[A_op.pattern.idx_coo[0], A_op.pattern.idx_coo[1]].detach().clone()
    )
    b_dense_grad = b.grad.detach().clone()

    # Assert that the adjoint method gradients agree with autograd.
    torch.testing.assert_close(A_sp_grad, A_dense_grad)
    torch.testing.assert_close(b_sp_grad, b_dense_grad)

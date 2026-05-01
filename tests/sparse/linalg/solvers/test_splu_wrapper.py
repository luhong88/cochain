import pytest
import torch

from cochain.sparse.decoupled_tensor import SparseDecoupledTensor
from cochain.sparse.linalg.solvers import SuperLU, splu

itemize_backend = pytest.mark.parametrize(
    "backend",
    [
        pytest.param("scipy", marks=[]),
        pytest.param("cupy", marks=[pytest.mark.gpu_only]),
    ],
)


@itemize_backend
def test_splu_forward(A, backend, device):
    A_op = SparseDecoupledTensor.from_tensor(A).to(device)
    A_dense = A_op.to_dense()

    n_dim = A_op.size(0)

    x_true = torch.randn(n_dim).to(device)
    b = A_dense @ x_true

    x = splu(A_op, b, backend=backend)

    torch.testing.assert_close(x, x_true)


@itemize_backend
def test_splu_forward_with_channel_dim(A, backend, device):
    A_op = SparseDecoupledTensor.from_tensor(A).to(device)
    A_dense = A_op.to_dense()

    n_dim = A_op.size(0)
    n_ch = 2

    x_true = torch.randn(n_dim, n_ch).to(device)
    b = A_dense @ x_true

    x = splu(A_op, b, backend=backend)
    x_T = splu(A_op, b.T, backend=backend, channel_first=True)

    torch.testing.assert_close(x.T, x_T)
    torch.testing.assert_close(x, x_true)


@itemize_backend
@pytest.mark.parametrize(
    "n_ch1, n_ch2",
    [(2, 3), (2, 1), (1, 2)],
)
def test_splu_forward_with_complex_channel_dim(A, n_ch1, n_ch2, backend, device):
    A_op = SparseDecoupledTensor.from_tensor(A).to(device)
    A_dense = A_op.to_dense()

    n_dim = A_op.size(0)

    x_true = torch.randn(n_dim, n_ch1, n_ch2).to(device)
    b = torch.einsum("ij,jkl->ikl", A_dense, x_true)

    x = splu(A_op, b, backend="scipy")
    x_T = splu(A_op, b.movedim(0, -1), backend=backend, channel_first=True)

    torch.testing.assert_close(x.movedim(0, -1), x_T)
    torch.testing.assert_close(x, x_true)


# TODO: backward test should also test channel dims
@itemize_backend
def test_splu_backward(A, backend, device):
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
    x_via_sp = splu(A_op, b, backend=backend)
    loss = torch.sum(x_via_sp * v)
    loss.backward()

    A_sp_grad = A_op.values.grad.detach().clone()
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


@itemize_backend
def test_persistent_splu_forward(A, backend, device):
    A_op = SparseDecoupledTensor.from_tensor(A).to(device)
    A_dense = A_op.to_dense()

    n_dim = A_op.size(0)
    n_ch = 2

    x1_true = torch.randn(n_dim, n_ch).to(device)
    x2_true = torch.randn(n_dim, n_ch).to(device)

    b1 = A_dense @ x1_true
    b2 = A_dense @ x2_true

    solver = SuperLU(A_op, backend=backend)

    x1 = solver(b1)
    x2 = solver(b2)

    torch.testing.assert_close(x1, x1_true)
    torch.testing.assert_close(x2, x2_true)


@itemize_backend
def test_persistent_splu_sequential_backward_pattern_1(A, backend, device):
    """
    Test persistent solver sequential backward passes.

    Test that the gradients are correctly handled when the solver object is
    applied sequentially to two RHS vectors, and the gradient is cleared in between
    the two applications.
    """
    A_op = SparseDecoupledTensor.from_tensor(A).to(device)
    n_dim = A_op.size(0)

    A_dense = A_op.to_dense()
    A_dense.requires_grad_()

    # Compute b and v
    b1 = torch.randn(n_dim).to(device)
    v1 = torch.randn(n_dim).to(device)

    # Define solver
    A_op.requires_grad_()
    solver = SuperLU(A_op, backend=backend)

    # Compute the dLdA and dLdb gradients via the adjoint method.
    b1.requires_grad_()
    x1_via_sp = solver(b1)
    loss1 = torch.sum(x1_via_sp * v1)
    loss1.backward()

    A_sp_grad1 = A_op.values.grad.detach().clone()
    b1_sp_grad = b1.grad.detach().clone()

    # Repeat this process with a different b and v.
    A_op.values.grad = None  # clear the existing gradient on A_op.

    b2 = torch.randn(n_dim).to(device)
    v2 = torch.randn(n_dim).to(device)

    b2.requires_grad_()
    x2_via_sp = solver(b2)
    loss2 = torch.sum(x2_via_sp * v2)
    loss2.backward()

    A_sp_grad2 = A_op.values.grad.detach().clone()
    b2_sp_grad = b2.grad.detach().clone()

    # Compute the dLdA and dLdb gradients via autograd using a dense A.
    b1.grad = None  # clear the existing gradient on b1
    x1_via_dense = torch.linalg.solve(A_dense, b1)
    loss1 = torch.sum(x1_via_dense * v1)
    loss1.backward()

    # Extract the nonzero elements of dLdA computed using a dense A.
    A_dense_grad1 = (
        A_dense.grad[A_op.pattern.idx_coo[0], A_op.pattern.idx_coo[1]].detach().clone()
    )
    b1_dense_grad = b1.grad.detach().clone()

    # Assert that the adjoint method gradients agree with autograd.
    torch.testing.assert_close(A_sp_grad1, A_dense_grad1)
    torch.testing.assert_close(b1_sp_grad, b1_dense_grad)

    # Repeat the same process for b2 and v2.
    b2.grad = None
    A_dense.grad = None

    x2_via_dense = torch.linalg.solve(A_dense, b2)
    loss2 = torch.sum(x2_via_dense * v2)
    loss2.backward()

    A_dense_grad2 = (
        A_dense.grad[A_op.pattern.idx_coo[0], A_op.pattern.idx_coo[1]].detach().clone()
    )
    b2_dense_grad = b2.grad.detach().clone()

    torch.testing.assert_close(A_sp_grad2, A_dense_grad2)
    torch.testing.assert_close(b2_sp_grad, b2_dense_grad)


@itemize_backend
def test_persistent_splu_sequential_backward_pattern_2(A, backend, device):
    """
    Test persistent solver sequential backward passes.

    Test that the gradients are correctly handled when the solver object is
    applied sequentially to two RHS vectors, and a single loss is computed using
    the results from both operations.
    """
    A_sym = A + A.T
    A_op = SparseDecoupledTensor.from_tensor(A_sym).to(device)
    n_dim = A_op.size(0)

    A_dense = A_op.to_dense()
    A_dense.requires_grad_()

    # Compute two different b's.
    b1 = torch.randn(n_dim).to(device)
    b2 = torch.randn(n_dim).to(device)

    # Define solver
    A_op.requires_grad_()
    solver = SuperLU(A_op, backend=backend)

    # Compute the dLdA and dLdb gradients via the adjoint method.
    b1.requires_grad_()
    b2.requires_grad_()

    x1_via_sp = solver(b1)
    x2_via_sp = solver(b2)

    loss = torch.sum(x1_via_sp * x2_via_sp)
    loss.backward()

    A_sp_grad = A_op.values.grad.detach().clone()
    b1_sp_grad = b1.grad.detach().clone()
    b2_sp_grad = b2.grad.detach().clone()

    # Compute the dLdA and dLdb gradients via autograd using a dense A.
    b1.grad = None  # clear the existing gradient on b1
    b2.grad = None
    x1_via_dense = torch.linalg.solve(A_dense, b1)
    x2_via_dense = torch.linalg.solve(A_dense, b2)
    loss = torch.sum(x1_via_dense * x2_via_dense)
    loss.backward()

    # Extract the nonzero elements of dLdA computed using a dense A.
    A_dense_grad = (
        A_dense.grad[A_op.pattern.idx_coo[0], A_op.pattern.idx_coo[1]].detach().clone()
    )
    b1_dense_grad = b1.grad.detach().clone()
    b2_dense_grad = b2.grad.detach().clone()

    # Assert that the adjoint method gradients agree with autograd.
    torch.testing.assert_close(A_sp_grad, A_dense_grad)
    torch.testing.assert_close(b1_sp_grad, b1_dense_grad)
    torch.testing.assert_close(b2_sp_grad, b2_dense_grad)

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
def test_splu_forward(a, backend, device):
    a_sdt = SparseDecoupledTensor.from_tensor(a).to(device)
    a_dense = a_sdt.to_dense()

    n_dim = a_sdt.size(0)

    x_true = torch.randn(n_dim).to(device)
    b = a_dense @ x_true

    x = splu(a_sdt, b, backend=backend)

    torch.testing.assert_close(x, x_true)


@itemize_backend
def test_splu_forward_with_channel_dim(a, backend, device):
    a_sdt = SparseDecoupledTensor.from_tensor(a).to(device)
    a_dense = a_sdt.to_dense()

    n_dim = a_sdt.size(0)
    n_ch = 2

    x_true = torch.randn(n_dim, n_ch).to(device)
    b = a_dense @ x_true

    x = splu(a_sdt, b, backend=backend)

    torch.testing.assert_close(x, x_true)


@itemize_backend
@pytest.mark.parametrize(
    "n_ch1, n_ch2",
    [(2, 3), (2, 1), (1, 2)],
)
def test_splu_forward_with_complex_channel_dim(a, n_ch1, n_ch2, backend, device):
    a_sdt = SparseDecoupledTensor.from_tensor(a).to(device)
    a_dense = a_sdt.to_dense()

    n_dim = a_sdt.size(0)

    x_true = torch.randn(n_dim, n_ch1, n_ch2).to(device)
    b = torch.einsum("ij,jkl->ikl", a_dense, x_true)

    x = splu(a_sdt, b, backend=backend)

    torch.testing.assert_close(x, x_true)


@itemize_backend
def test_splu_backward(a, backend, device):
    """
    Test the splu() backward pass.

    Let A @ x = b and define the loss function as L = <x, v>. Check that the gradients
    dLdA and dLdb computed through the adjoint method matches the autograd gradients
    from torch.linalg.solve() (using dense A).
    """
    a_sdt = SparseDecoupledTensor.from_tensor(a).to(device)
    a_dense = a_sdt.to_dense()
    n_dim = a_sdt.size(0)

    # Compute b and v
    b = torch.randn(n_dim).to(device)
    v = torch.randn(n_dim).to(device)

    # Compute the dLdA and dLdb gradients via the adjoint method.
    a_sdt.requires_grad_()
    b.requires_grad_()
    x_via_sp = splu(a_sdt, b, backend=backend)
    loss = torch.sum(x_via_sp * v)
    loss.backward()

    a_sp_grad = a_sdt.values.grad.detach().clone()
    b_sp_grad = b.grad.detach().clone()

    # Compute the dLdA and dLdb gradients via autograd using a dense A.
    a_dense.requires_grad_()
    b.grad = None  # clear the existing gradient on b
    x_via_dense = torch.linalg.solve(a_dense, b)
    loss = torch.sum(x_via_dense * v)
    loss.backward()

    # Extract the nonzero elements of dLdA computed using a dense A.
    a_dense_grad = (
        a_dense.grad[a_sdt.pattern.idx_coo[0], a_sdt.pattern.idx_coo[1]]
        .detach()
        .clone()
    )
    b_dense_grad = b.grad.detach().clone()

    # Assert that the adjoint method gradients agree with autograd.
    torch.testing.assert_close(a_sp_grad, a_dense_grad)
    torch.testing.assert_close(b_sp_grad, b_dense_grad)


@itemize_backend
def test_splu_backward_with_channel_dim(a, backend, device):
    a_sdt = SparseDecoupledTensor.from_tensor(a).to(device)
    a_dense = a_sdt.to_dense()
    n_dim = a_sdt.size(0)
    n_ch = 3

    # Compute b and v with channel dimensions
    b = torch.randn(n_dim, n_ch).to(device)
    v = torch.randn(n_dim, n_ch).to(device)

    # Compute the dLdA and dLdb gradients via the adjoint method
    a_sdt.requires_grad_()
    b.requires_grad_()
    x_via_sp = splu(a_sdt, b, backend=backend)
    loss = torch.sum(x_via_sp * v)
    loss.backward()

    a_sp_grad = a_sdt.values.grad.detach().clone()
    b_sp_grad = b.grad.detach().clone()

    # Compute the dLdA and dLdb gradients via autograd using a dense A
    a_dense.requires_grad_()
    b.grad = None
    x_via_dense = torch.linalg.solve(a_dense, b)
    loss = torch.sum(x_via_dense * v)
    loss.backward()

    # Extract the nonzero elements of dLdA computed using a dense A
    a_dense_grad = (
        a_dense.grad[a_sdt.pattern.idx_coo[0], a_sdt.pattern.idx_coo[1]]
        .detach()
        .clone()
    )
    b_dense_grad = b.grad.detach().clone()

    torch.testing.assert_close(a_sp_grad, a_dense_grad)
    torch.testing.assert_close(b_sp_grad, b_dense_grad)


@itemize_backend
@pytest.mark.parametrize(
    "n_ch1, n_ch2",
    [(2, 3), (2, 1), (1, 2)],
)
def test_splu_backward_with_complex_channel_dim(a, n_ch1, n_ch2, backend, device):
    a_sdt = SparseDecoupledTensor.from_tensor(a).to(device)
    a_dense = a_sdt.to_dense()
    n_dim = a_sdt.size(0)

    # Compute b and v with complex channel dimensions
    b = torch.randn(n_dim, n_ch1, n_ch2).to(device)
    v = torch.randn(n_dim, n_ch1, n_ch2).to(device)

    # Compute the dLdA and dLdb gradients via the adjoint method
    a_sdt.requires_grad_()
    b.requires_grad_()
    x_via_sp = splu(a_sdt, b, backend=backend)
    loss = torch.sum(x_via_sp * v)
    loss.backward()

    a_sp_grad = a_sdt.values.grad.detach().clone()
    b_sp_grad = b.grad.detach().clone()

    # Compute the dLdA and dLdb gradients via autograd using a dense A
    a_dense.requires_grad_()
    b.grad = None
    b_flat = b.view(n_dim, -1)
    x_via_dense_flat = torch.linalg.solve(a_dense, b_flat)
    x_via_dense = x_via_dense_flat.view(n_dim, n_ch1, n_ch2)
    loss = torch.sum(x_via_dense * v)
    loss.backward()

    # Extract the nonzero elements of dLdA computed using a dense A
    a_dense_grad = (
        a_dense.grad[a_sdt.pattern.idx_coo[0], a_sdt.pattern.idx_coo[1]]
        .detach()
        .clone()
    )
    b_dense_grad = b.grad.detach().clone()

    torch.testing.assert_close(a_sp_grad, a_dense_grad)
    torch.testing.assert_close(b_sp_grad, b_dense_grad)


@itemize_backend
def test_persistent_splu_forward(a, backend, device):
    a_sdt = SparseDecoupledTensor.from_tensor(a).to(device)
    a_dense = a_sdt.to_dense()

    n_dim = a_sdt.size(0)
    n_ch = 2

    x1_true = torch.randn(n_dim, n_ch).to(device)
    x2_true = torch.randn(n_dim, n_ch).to(device)

    b1 = a_dense @ x1_true
    b2 = a_dense @ x2_true

    solver = SuperLU(a_sdt, backend=backend)

    x1 = solver(b1)
    x2 = solver(b2)

    torch.testing.assert_close(x1, x1_true)
    torch.testing.assert_close(x2, x2_true)


@itemize_backend
@pytest.mark.parametrize(
    "n_ch1, n_ch2",
    [(2, 3), (2, 1), (1, 2)],
)
def test_persistent_splu_forward_with_complex_channel_dim(
    a, n_ch1, n_ch2, backend, device
):
    a_sdt = SparseDecoupledTensor.from_tensor(a).to(device)
    a_dense = a_sdt.to_dense()

    n_dim = a_sdt.size(0)

    x1_true = torch.randn(n_dim, n_ch1, n_ch2).to(device)
    x2_true = torch.randn(n_dim, n_ch1, n_ch2).to(device)

    b1 = torch.einsum("ij,jkl->ikl", a_dense, x1_true)
    b2 = torch.einsum("ij,jkl->ikl", a_dense, x2_true)

    solver = SuperLU(a_sdt, backend=backend)

    x1 = solver(b1)
    x2 = solver(b2)

    torch.testing.assert_close(x1, x1_true)
    torch.testing.assert_close(x2, x2_true)


@itemize_backend
def test_persistent_splu_sequential_backward_pattern_1(a, backend, device):
    """
    Test persistent solver sequential backward passes.

    Test that the gradients are correctly handled when the solver object is
    applied sequentially to two RHS vectors, and the gradient is cleared in between
    the two applications.
    """
    a_sdt = SparseDecoupledTensor.from_tensor(a).to(device)
    n_dim = a_sdt.size(0)

    a_dense = a_sdt.to_dense()
    a_dense.requires_grad_()

    # Compute b and v
    b1 = torch.randn(n_dim).to(device)
    v1 = torch.randn(n_dim).to(device)

    # Define solver
    a_sdt.requires_grad_()
    solver = SuperLU(a_sdt, backend=backend)

    # Compute the dLdA and dLdb gradients via the adjoint method.
    b1.requires_grad_()
    x1_via_sp = solver(b1)
    loss1 = torch.sum(x1_via_sp * v1)
    loss1.backward()

    a_sp_grad1 = a_sdt.values.grad.detach().clone()
    b1_sp_grad = b1.grad.detach().clone()

    # Repeat this process with a different b and v.
    a_sdt.values.grad = None  # clear the existing gradient on A_op.

    b2 = torch.randn(n_dim).to(device)
    v2 = torch.randn(n_dim).to(device)

    b2.requires_grad_()
    x2_via_sp = solver(b2)
    loss2 = torch.sum(x2_via_sp * v2)
    loss2.backward()

    a_sp_grad2 = a_sdt.values.grad.detach().clone()
    b2_sp_grad = b2.grad.detach().clone()

    # Compute the dLdA and dLdb gradients via autograd using a dense A.
    b1.grad = None  # clear the existing gradient on b1
    x1_via_dense = torch.linalg.solve(a_dense, b1)
    loss1 = torch.sum(x1_via_dense * v1)
    loss1.backward()

    # Extract the nonzero elements of dLdA computed using a dense A.
    a_dense_grad1 = (
        a_dense.grad[a_sdt.pattern.idx_coo[0], a_sdt.pattern.idx_coo[1]]
        .detach()
        .clone()
    )
    b1_dense_grad = b1.grad.detach().clone()

    # Assert that the adjoint method gradients agree with autograd.
    torch.testing.assert_close(a_sp_grad1, a_dense_grad1)
    torch.testing.assert_close(b1_sp_grad, b1_dense_grad)

    # Repeat the same process for b2 and v2.
    b2.grad = None
    a_dense.grad = None

    x2_via_dense = torch.linalg.solve(a_dense, b2)
    loss2 = torch.sum(x2_via_dense * v2)
    loss2.backward()

    a_dense_grad2 = (
        a_dense.grad[a_sdt.pattern.idx_coo[0], a_sdt.pattern.idx_coo[1]]
        .detach()
        .clone()
    )
    b2_dense_grad = b2.grad.detach().clone()

    torch.testing.assert_close(a_sp_grad2, a_dense_grad2)
    torch.testing.assert_close(b2_sp_grad, b2_dense_grad)


@itemize_backend
def test_persistent_splu_sequential_backward_pattern_2(a, backend, device):
    """
    Test persistent solver sequential backward passes.

    Test that the gradients are correctly handled when the solver object is
    applied sequentially to two RHS vectors, and a single loss is computed using
    the results from both operations.
    """
    a_sym = a + a.T
    a_sdt = SparseDecoupledTensor.from_tensor(a_sym).to(device)
    n_dim = a_sdt.size(0)

    a_dense = a_sdt.to_dense()
    a_dense.requires_grad_()

    # Compute two different b's.
    b1 = torch.randn(n_dim).to(device)
    b2 = torch.randn(n_dim).to(device)

    # Define solver
    a_sdt.requires_grad_()
    solver = SuperLU(a_sdt, backend=backend)

    # Compute the dLdA and dLdb gradients via the adjoint method.
    b1.requires_grad_()
    b2.requires_grad_()

    x1_via_sp = solver(b1)
    x2_via_sp = solver(b2)

    loss = torch.sum(x1_via_sp * x2_via_sp)
    loss.backward()

    a_sp_grad = a_sdt.values.grad.detach().clone()
    b1_sp_grad = b1.grad.detach().clone()
    b2_sp_grad = b2.grad.detach().clone()

    # Compute the dLdA and dLdb gradients via autograd using a dense A.
    b1.grad = None  # clear the existing gradient on b1
    b2.grad = None
    x1_via_dense = torch.linalg.solve(a_dense, b1)
    x2_via_dense = torch.linalg.solve(a_dense, b2)
    loss = torch.sum(x1_via_dense * x2_via_dense)
    loss.backward()

    # Extract the nonzero elements of dLdA computed using a dense A.
    a_dense_grad = (
        a_dense.grad[a_sdt.pattern.idx_coo[0], a_sdt.pattern.idx_coo[1]]
        .detach()
        .clone()
    )
    b1_dense_grad = b1.grad.detach().clone()
    b2_dense_grad = b2.grad.detach().clone()

    # Assert that the adjoint method gradients agree with autograd.
    torch.testing.assert_close(a_sp_grad, a_dense_grad)
    torch.testing.assert_close(b1_sp_grad, b1_dense_grad)
    torch.testing.assert_close(b2_sp_grad, b2_dense_grad)


@itemize_backend
def test_persistent_splu_backward_with_channel_dim(a, backend, device):
    a_sdt = SparseDecoupledTensor.from_tensor(a).to(device)
    a_dense = a_sdt.to_dense()
    n_dim = a_sdt.size(0)
    n_ch = 3

    # Compute b and v with channel dimensions.
    b = torch.randn(n_dim, n_ch).to(device)
    v = torch.randn(n_dim, n_ch).to(device)

    # Define solver and compute gradients via adjoint method.
    a_sdt.requires_grad_()
    b.requires_grad_()
    solver = SuperLU(a_sdt, backend=backend)
    x_via_sp = solver(b)
    loss = torch.sum(x_via_sp * v)
    loss.backward()

    a_sp_grad = a_sdt.values.grad.detach().clone()
    b_sp_grad = b.grad.detach().clone()

    # Compute dense autograd baseline.
    a_dense.requires_grad_()
    b.grad = None
    x_via_dense = torch.linalg.solve(a_dense, b)
    loss = torch.sum(x_via_dense * v)
    loss.backward()

    a_dense_grad = (
        a_dense.grad[a_sdt.pattern.idx_coo[0], a_sdt.pattern.idx_coo[1]]
        .detach()
        .clone()
    )
    b_dense_grad = b.grad.detach().clone()

    # Assert that the adjoint method gradients agree with autograd.
    torch.testing.assert_close(a_sp_grad, a_dense_grad)
    torch.testing.assert_close(b_sp_grad, b_dense_grad)

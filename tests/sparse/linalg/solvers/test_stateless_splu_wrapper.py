import pytest
import torch

from cochain.sparse.decoupled_tensor import SparseDecoupledTensor
from cochain.sparse.linalg.solvers import splu

itemize_backend = pytest.mark.parametrize(
    "backend",
    [
        pytest.param("scipy", marks=[]),
        pytest.param("cupy", marks=[pytest.mark.gpu_only, pytest.mark.requires_cupy]),
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

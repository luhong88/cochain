import pytest
import torch

from cochain.sparse.decoupled_tensor import SparseDecoupledTensor
from cochain.sparse.linalg.solvers import nvmath_direct_solver

pytest.importorskip("nvmath")


@pytest.mark.parametrize(
    "matrix, matrix_type", [("a", "general"), ("a_sym", "symmetric"), ("a_spd", "spd")]
)
def test_direct_solver_forward(matrix, matrix_type, request, device):
    a = request.getfixturevalue(matrix)

    a_sdt = SparseDecoupledTensor.from_tensor(a).to(device)
    a_dense = a_sdt.to_dense()

    n_dim = a_sdt.size(0)

    x_true = torch.randn(n_dim).to(device)
    b = a_dense @ x_true

    x = nvmath_direct_solver(a_sdt, b, sparse_system_type=matrix_type)

    torch.testing.assert_close(x, x_true)


@pytest.mark.parametrize(
    "matrix, matrix_type", [("a", "general"), ("a_sym", "symmetric"), ("a_spd", "spd")]
)
def test_direct_solver_with_channel_dim(matrix, matrix_type, request, device):
    a = request.getfixturevalue(matrix)

    a_sdt = SparseDecoupledTensor.from_tensor(a).to(device)
    a_dense = a_sdt.to_dense()

    n_dim = a_sdt.size(0)
    n_ch = 2

    x_true = torch.randn(n_dim, n_ch).to(device)
    b = torch.einsum("ij,jk->ik", a_dense, x_true)

    x = nvmath_direct_solver(a_sdt, b, sparse_system_type=matrix_type)

    torch.testing.assert_close(x, x_true)


@pytest.mark.parametrize(
    "n_ch1, n_ch2",
    [(2, 3), (2, 1), (1, 2)],
)
@pytest.mark.parametrize(
    "matrix, matrix_type", [("a", "general"), ("a_sym", "symmetric"), ("a_spd", "spd")]
)
def test_direct_solver_with_complex_channel_dim(
    matrix, matrix_type, n_ch1, n_ch2, request, device
):
    a = request.getfixturevalue(matrix)
    a_sdt = SparseDecoupledTensor.from_tensor(a).to(device)
    a_dense = a_sdt.to_dense()

    n_dim = a_sdt.size(0)

    x_true = torch.randn(n_dim, n_ch1, n_ch2).to(device)
    b = torch.einsum("ij,jkl->ikl", a_dense, x_true)

    x = nvmath_direct_solver(a_sdt, b, sparse_system_type=matrix_type)

    torch.testing.assert_close(x, x_true)


@pytest.mark.parametrize(
    "matrix, matrix_type",
    [
        ("a_with_batch", "general"),
        ("a_sym_with_batch", "symmetric"),
        ("a_spd_with_batch", "spd"),
    ],
)
def test_direct_solver_with_batch_dim(matrix, matrix_type, request, device):
    a_with_batch = request.getfixturevalue(matrix)
    a_sdt = SparseDecoupledTensor.from_tensor(a_with_batch).to(device)
    a_dense = a_sdt.to_dense()

    n_dim = a_sdt.size(-1)
    n_batch = a_sdt.size(0)

    x_true = torch.randn(n_batch, n_dim).to(device)
    b = torch.einsum("bij,bj->bi", a_dense, x_true)

    x = nvmath_direct_solver(a_sdt, b, sparse_system_type=matrix_type)

    torch.testing.assert_close(x, x_true)


@pytest.mark.parametrize(
    "matrix, matrix_type",
    [
        ("a_with_batch", "general"),
        ("a_sym_with_batch", "symmetric"),
        ("a_spd_with_batch", "spd"),
    ],
)
def test_direct_solver_with_batch_channel_dim(matrix, matrix_type, request, device):
    a_with_batch = request.getfixturevalue(matrix)
    a_sdt = SparseDecoupledTensor.from_tensor(a_with_batch).to(device)
    a_dense = a_sdt.to_dense()

    n_dim = a_sdt.size(-1)
    n_batch = a_sdt.size(0)
    n_ch = 2

    x_true = torch.randn(n_batch, n_dim, n_ch).to(device)
    b = torch.einsum("bij,bjc->bic", a_dense, x_true)

    x = nvmath_direct_solver(a_sdt, b, sparse_system_type=matrix_type)

    torch.testing.assert_close(x, x_true)


@pytest.mark.parametrize(
    "matrix, matrix_type", [("a", "general"), ("a_sym", "symmetric"), ("a_spd", "spd")]
)
def test_direct_solver_backward(matrix, matrix_type, request, device):
    """
    Test the nvmath_direct_solver() backward pass.

    Let A @ x = b and define the loss function as L = <x, v>. Check that the gradients
    dLdA and dLdb computed through the adjoint method matches the autograd gradients
    from torch.linalg.solve() (using dense A).
    """
    a = request.getfixturevalue(matrix)
    a_sdt = SparseDecoupledTensor.from_tensor(a).to(device)
    a_dense = a_sdt.to_dense()
    n_dim = a_sdt.size(0)

    # Compute b and v
    b = torch.randn(n_dim).to(device)
    v = torch.randn(n_dim).to(device)

    # Compute the dLdA and dLdb gradients via the adjoint method.
    a_sdt.requires_grad_()
    b.requires_grad_()
    x_via_sp = nvmath_direct_solver(a_sdt, b, sparse_system_type=matrix_type)
    loss = torch.sum(x_via_sp * v)
    loss.backward()

    a_sp_grad = a_sdt.grad.detach().clone()
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


@pytest.mark.parametrize(
    "matrix, matrix_type",
    [
        ("a_with_batch", "general"),
        ("a_sym_with_batch", "symmetric"),
        ("a_spd_with_batch", "spd"),
    ],
)
def test_direct_solver_backward_with_batch_and_channel_dim(
    matrix, matrix_type, request, device
):
    a_with_batch = request.getfixturevalue(matrix)
    a_sdt = SparseDecoupledTensor.from_tensor(a_with_batch).to(device)
    a_dense = a_sdt.to_dense()

    n_batch = a_sdt.size(0)
    n_dim = a_sdt.size(-1)
    n_ch = 2

    # Compute b and v with both batch and channel dimensions.
    b = torch.randn(n_batch, n_dim, n_ch).to(device)
    v = torch.randn(n_batch, n_dim, n_ch).to(device)

    # Compute the dLdA and dLdb gradients via the adjoint method.
    a_sdt.requires_grad_()
    b.requires_grad_()
    x_via_sp = nvmath_direct_solver(a_sdt, b, sparse_system_type=matrix_type)
    loss = torch.sum(x_via_sp * v)
    loss.backward()

    a_sp_grad = a_sdt.grad.detach().clone()
    b_sp_grad = b.grad.detach().clone()

    # Compute gradients via dense autograd (torch.linalg.solve supports batched A and b).
    a_dense.requires_grad_()
    b.grad = None
    x_via_dense = torch.linalg.solve(a_dense, b)
    loss = torch.sum(x_via_dense * v)
    loss.backward()

    # Extract the nonzero elements of dLdA computed using a dense A.
    # idx_coo has shape (sp, nnz) where sp = len(*b) + 2.
    b_idx = a_sdt.pattern.idx_coo[0]
    r_idx = a_sdt.pattern.idx_coo[1]
    c_idx = a_sdt.pattern.idx_coo[2]
    a_dense_grad = a_dense.grad[b_idx, r_idx, c_idx].detach().clone()

    b_dense_grad = b.grad.detach().clone()

    # Assert that the adjoint method gradients agree with autograd.
    torch.testing.assert_close(a_sp_grad, a_dense_grad)
    torch.testing.assert_close(b_sp_grad, b_dense_grad)

import pytest
import torch

from cochain.sparse.decoupled_tensor import SparseDecoupledTensor
from cochain.sparse.linalg.solvers import NVMathDirectSolver

pytest.importorskip("nvmath")


@pytest.mark.parametrize(
    "matrix, matrix_type",
    [
        ("a", "general"),
        ("a_sym", "symmetric"),
        ("a_spd", "spd"),
    ],
)
@pytest.mark.parametrize("trans", ["N", "T"])
def test_persistent_direct_solver_forward(matrix, matrix_type, trans, request, device):
    a = request.getfixturevalue(matrix)
    a_sdt = SparseDecoupledTensor.from_tensor(a).to(device)
    a_dense = a_sdt.to_dense()

    n_dim = a_sdt.size(0)
    x_true = torch.randn(n_dim).to(device)

    match trans:
        case "N":
            b = a_dense @ x_true
        case "T":
            b = a_dense.T @ x_true

    solver = NVMathDirectSolver(a_sdt, b, sparse_system_type=matrix_type)
    x = solver(b, trans=trans)

    torch.testing.assert_close(x, x_true)


@pytest.mark.parametrize(
    "matrix, matrix_type",
    [
        ("a", "general"),
        ("a_sym", "symmetric"),
        ("a_spd", "spd"),
    ],
)
@pytest.mark.parametrize("trans", ["N", "T"])
def test_persistent_direct_solver_with_channel_dim(
    matrix, matrix_type, trans, request, device
):
    a = request.getfixturevalue(matrix)
    a_sdt = SparseDecoupledTensor.from_tensor(a).to(device)
    a_dense = a_sdt.to_dense()

    n_dim = a_sdt.size(0)
    n_ch = 2

    x_true = torch.randn(n_dim, n_ch).to(device)

    match trans:
        case "N":
            b = a_dense @ x_true
        case "T":
            b = a_dense.T @ x_true

    solver = NVMathDirectSolver(a_sdt, b, sparse_system_type=matrix_type)
    x = solver(b, trans=trans)

    torch.testing.assert_close(x, x_true)


@pytest.mark.parametrize(
    "n_ch1, n_ch2",
    [(2, 3), (2, 1), (1, 2)],
)
@pytest.mark.parametrize(
    "matrix, matrix_type",
    [
        ("a", "general"),
        ("a_sym", "symmetric"),
        ("a_spd", "spd"),
    ],
)
@pytest.mark.parametrize("trans", ["N", "T"])
def test_persistent_direct_solver_with_complex_channel_dim(
    matrix, matrix_type, trans, n_ch1, n_ch2, request, device
):
    a = request.getfixturevalue(matrix)
    a_sdt = SparseDecoupledTensor.from_tensor(a).to(device)
    a_dense = a_sdt.to_dense()

    n_dim = a_sdt.size(0)
    x_true = torch.randn(n_dim, n_ch1, n_ch2).to(device)

    match trans:
        case "N":
            b = torch.einsum("ij,jkl->ikl", a_dense, x_true)
        case "T":
            b = torch.einsum("ji,jkl->ikl", a_dense, x_true)

    solver = NVMathDirectSolver(a_sdt, b, sparse_system_type=matrix_type)
    x = solver(b, trans=trans)

    torch.testing.assert_close(x, x_true)


@pytest.mark.parametrize(
    "matrix, matrix_type",
    [
        ("a_with_batch", "general"),
        ("a_sym_with_batch", "symmetric"),
        ("a_spd_with_batch", "spd"),
    ],
)
@pytest.mark.parametrize("trans", ["N", "T"])
def test_direct_solver_with_batch_dim(matrix, matrix_type, trans, request, device):
    a_with_batch = request.getfixturevalue(matrix)
    a_sdt = SparseDecoupledTensor.from_tensor(a_with_batch).to(device)
    a_dense = a_sdt.to_dense()

    n_dim = a_sdt.size(-1)
    n_batch = a_sdt.size(0)

    x_true = torch.randn(n_batch, n_dim).to(device)

    match trans:
        case "N":
            b = torch.einsum("bij,bj->bi", a_dense, x_true)
        case "T":
            b = torch.einsum("bji,bj->bi", a_dense, x_true)

    solver = NVMathDirectSolver(a_sdt, b, sparse_system_type=matrix_type)
    x = solver(b, trans=trans)

    torch.testing.assert_close(x, x_true)


@pytest.mark.parametrize(
    "matrix, matrix_type",
    [
        ("a_with_batch", "general"),
        ("a_sym_with_batch", "symmetric"),
        ("a_spd_with_batch", "spd"),
    ],
)
@pytest.mark.parametrize("trans", ["N", "T"])
def test_direct_solver_with_batch_channel_dim(
    matrix, matrix_type, trans, request, device
):
    a_with_batch = request.getfixturevalue(matrix)
    a_sdt = SparseDecoupledTensor.from_tensor(a_with_batch).to(device)
    a_dense = a_sdt.to_dense()

    n_dim = a_sdt.size(-1)
    n_batch = a_sdt.size(0)
    n_ch = 2

    x_true = torch.randn(n_batch, n_dim, n_ch).to(device)

    match trans:
        case "N":
            b = torch.einsum("bij,bjc->bic", a_dense, x_true)
        case "T":
            b = torch.einsum("bji,bjc->bic", a_dense, x_true)

    solver = NVMathDirectSolver(a_sdt, b, sparse_system_type=matrix_type)
    x = solver(b, trans=trans)

    torch.testing.assert_close(x, x_true)


@pytest.mark.parametrize(
    "matrix, matrix_type",
    [
        ("a", "general"),
        ("a_sym", "symmetric"),
        ("a_spd", "spd"),
    ],
)
@pytest.mark.parametrize("trans1", ["N", "T"])
@pytest.mark.parametrize("trans2", ["N", "T"])
def test_persistent_direct_solver_sequential_backward_pattern_1(
    matrix, matrix_type, trans1, trans2, request, device
):
    """
    Test persistent solver sequential backward passes.

    Test that the gradients are correctly handled when the solver object is
    applied sequentially to two RHS vectors, and the gradient is cleared in between
    the two applications.
    """
    a = request.getfixturevalue(matrix)
    a_sdt = SparseDecoupledTensor.from_tensor(a).to(device)
    n_dim = a_sdt.size(0)

    a_dense = a_sdt.to_dense()
    a_dense.requires_grad_()

    # Compute b and v
    b1 = torch.randn(n_dim).to(device)
    v1 = torch.randn(n_dim).to(device)

    # Define solver
    a_sdt.requires_grad_()
    solver = NVMathDirectSolver(a_sdt, b1, sparse_system_type=matrix_type)

    # Compute the dLdA and dLdb gradients via the adjoint method.
    b1.requires_grad_()
    x1_via_sp = solver(b1, trans=trans1)
    loss1 = torch.sum(x1_via_sp * v1)
    loss1.backward()

    a_sp_grad1 = a_sdt.grad.detach().clone()
    b1_sp_grad = b1.grad.detach().clone()

    # Repeat this process with a different b and v.
    a_sdt.grad = None  # clear the existing gradient on A_op.

    b2 = torch.randn(n_dim).to(device)
    v2 = torch.randn(n_dim).to(device)

    b2.requires_grad_()
    x2_via_sp = solver(b2, trans=trans2)
    loss2 = torch.sum(x2_via_sp * v2)
    loss2.backward()

    a_sp_grad2 = a_sdt.grad.detach().clone()
    b2_sp_grad = b2.grad.detach().clone()

    # Compute the dLdA and dLdb gradients via autograd using a dense A.
    b1.grad = None  # clear the existing gradient on b1

    match trans1:
        case "N":
            x1_via_dense = torch.linalg.solve(a_dense, b1)
        case "T":
            x1_via_dense = torch.linalg.solve(a_dense.T, b1)

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

    match trans2:
        case "N":
            x2_via_dense = torch.linalg.solve(a_dense, b2)
        case "T":
            x2_via_dense = torch.linalg.solve(a_dense.T, b2)

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


@pytest.mark.parametrize(
    "matrix, matrix_type",
    [
        ("a", "general"),
        ("a_sym", "symmetric"),
        ("a_spd", "spd"),
    ],
)
@pytest.mark.parametrize("trans1", ["N", "T"])
@pytest.mark.parametrize("trans2", ["N", "T"])
def test_persistent_direct_solver_sequential_backward_pattern_2(
    matrix, matrix_type, trans1, trans2, request, device
):
    """
    Test persistent solver sequential backward passes.

    Test that the gradients are correctly handled when the solver object is
    applied sequentially to two RHS vectors, and a single loss is computed using
    the results from both operations.
    """
    a = request.getfixturevalue(matrix)
    a_sdt = SparseDecoupledTensor.from_tensor(a).to(device)
    n_dim = a_sdt.size(0)

    a_dense = a_sdt.to_dense()
    a_dense.requires_grad_()

    # Compute two different b's.
    b1 = torch.randn(n_dim).to(device)
    b2 = torch.randn(n_dim).to(device)

    # Define solver
    a_sdt.requires_grad_()
    solver = NVMathDirectSolver(a_sdt, b1, sparse_system_type=matrix_type)

    # Compute the dLdA and dLdb gradients via the adjoint method.
    b1.requires_grad_()
    b2.requires_grad_()

    x1_via_sp = solver(b1, trans=trans1)
    x2_via_sp = solver(b2, trans=trans2)

    loss = torch.sum(x1_via_sp * x2_via_sp)
    loss.backward()

    a_sp_grad = a_sdt.grad.detach().clone()
    b1_sp_grad = b1.grad.detach().clone()
    b2_sp_grad = b2.grad.detach().clone()

    # Compute the dLdA and dLdb gradients via autograd using a dense A.
    b1.grad = None  # clear the existing gradient on b1
    b2.grad = None

    match trans1:
        case "N":
            x1_via_dense = torch.linalg.solve(a_dense, b1)
        case "T":
            x1_via_dense = torch.linalg.solve(a_dense.T, b1)

    match trans2:
        case "N":
            x2_via_dense = torch.linalg.solve(a_dense, b2)
        case "T":
            x2_via_dense = torch.linalg.solve(a_dense.T, b2)

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


@pytest.mark.parametrize(
    "matrix, matrix_type",
    [
        ("a_with_batch", "general"),
        ("a_sym_with_batch", "symmetric"),
        ("a_spd_with_batch", "spd"),
    ],
)
@pytest.mark.parametrize("trans", ["N", "T"])
def test_persistent_direct_solver_backward_with_batch_and_channel_dim(
    matrix, matrix_type, trans, request, device
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

    solver = NVMathDirectSolver(a_sdt, b, sparse_system_type=matrix_type)
    x_via_sp = solver(b, trans=trans)

    loss = torch.sum(x_via_sp * v)
    loss.backward()

    a_sp_grad = a_sdt.grad.detach().clone()
    b_sp_grad = b.grad.detach().clone()

    # Compute gradients via dense autograd (torch.linalg.solve supports batched A and b).
    a_dense.requires_grad_()
    b.grad = None

    match trans:
        case "N":
            x_via_dense = torch.linalg.solve(a_dense, b)
        case "T":
            x_via_dense = torch.linalg.solve(a_dense.transpose(-1, -2), b)

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

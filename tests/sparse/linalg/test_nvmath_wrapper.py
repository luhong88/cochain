import pytest
import torch as t

from cochain.sparse.linalg import nvmath_direct_solver


@pytest.mark.gpu_only
def test_direct_solver_forward(A, device):
    A_sp = A.to(device)
    A_dense = A_sp.to_dense()

    n_dim = A_sp.size(0)

    x_true = t.randn(n_dim).to(device)
    b = A_dense @ x_true

    x = nvmath_direct_solver(A_sp, b)

    t.testing.assert_close(x, x_true)


@pytest.mark.gpu_only
def test_direct_solver_with_channel_dim(A, device):
    A_sp = A.to(device)
    A_dense = A_sp.to_dense()

    n_dim = A_sp.size(0)
    n_ch = 2

    x_true = t.randn(n_ch, n_dim).to(device)
    b = t.einsum("ij,kj->ki", A_dense, x_true)

    x = nvmath_direct_solver(A_sp, b)

    t.testing.assert_close(x, x_true.T)


@pytest.mark.gpu_only
def test_direct_solver_with_batch_dim(A_batched, device):
    A_sp = A_batched.to(device)
    A_dense = A_sp.to_dense()

    n_dim = A_sp.size(-1)
    n_batch = A_sp.size(0)

    x_true = t.randn(n_batch, n_dim).to(device)
    b = t.einsum("bij,bj->bi", A_dense, x_true).view(n_batch, 1, n_dim)

    x = nvmath_direct_solver(A_sp, b)
    x_true_shaped = x_true.view(n_batch, n_dim, 1)

    t.testing.assert_close(x, x_true_shaped)

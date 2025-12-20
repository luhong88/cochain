import torch as t

from cochain.sparse.linalg import splu


def test_scipy_forward():
    n_dim = 4

    A_dense = t.rand(n_dim, n_dim) + t.eye(n_dim) * n_dim
    A_sp = A_dense.to_sparse_coo().coalesce()

    x_true = t.randn(n_dim)
    b = A_dense @ x_true

    x = splu(A_sp, b, backend="scipy")

    t.testing.assert_close(x, x_true)


def test_scipy_forward_with_channel_dim():
    n_dim = 4
    n_ch = 2

    A_dense = t.rand(n_dim, n_dim) + t.eye(n_dim) * n_dim
    A_sp = A_dense.to_sparse_coo().coalesce()

    x_true = t.randn(n_dim, n_ch)
    b = A_dense @ x_true

    x = splu(A_sp, b, backend="scipy")
    x_T = splu(A_sp, b.T, backend="scipy", channel_first=True)

    t.testing.assert_close(x.T, x_T)
    t.testing.assert_close(x, x_true)


def test_scipy_forward_with_complex_channel_dim():
    n_dim = 4
    n_ch1 = 2
    n_ch2 = 3

    A_dense = t.rand(n_dim, n_dim) + t.eye(n_dim) * n_dim
    A_sp = A_dense.to_sparse_coo().coalesce()

    x_true = t.randn(n_dim, n_ch1, n_ch2)
    b = t.einsum("ij,jkl->ikl", A_dense, x_true)

    x = splu(A_sp, b, backend="scipy")
    x_T = splu(A_sp, b.movedim(0, -1), backend="scipy", channel_first=True)

    t.testing.assert_close(x.movedim(0, -1), x_T)
    t.testing.assert_close(x, x_true)

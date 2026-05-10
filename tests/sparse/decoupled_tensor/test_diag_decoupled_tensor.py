import pytest
import torch

from cochain.sparse.decoupled_tensor import (
    DiagDecoupledTensor,
    SparseDecoupledTensor,
    SparsityPattern,
)


@pytest.fixture
def diag():
    n_dim = 4
    return torch.randn(n_dim)


@pytest.fixture
def diag_batched():
    n_batch = 2
    n_dim = 4
    return torch.randn(n_batch, n_dim)


def test_post_init_exceptions(device):
    # Test non-strided tensor
    with pytest.raises(TypeError):
        sp_tensor = torch.sparse_coo_tensor(
            indices=torch.tensor([[0, 1], [0, 1]]),
            values=torch.tensor([1.0, 2.0]),
            size=(2, 2),
        )
        DiagDecoupledTensor(sp_tensor)

    # Test wrong ndim
    with pytest.raises(ValueError):
        DiagDecoupledTensor(torch.randn(2, 2, 2, device=device))

    # Test NaN or Inf
    with pytest.raises(ValueError):
        DiagDecoupledTensor(torch.tensor([1.0, float("inf")], device=device))
    with pytest.raises(ValueError):
        DiagDecoupledTensor(torch.tensor([1.0, float("nan")], device=device))


def test_submatrix(diag, device):
    diag_dense = torch.diagflat(diag).to(device)
    ddt = DiagDecoupledTensor.from_tensor(diag).to(device)

    r_mask = torch.tensor([True, False, True, True], device=device)
    c_mask = torch.tensor([False, True, True, False], device=device)

    sub_ddt_1 = ddt.submatrix(r_mask).to_dense()
    sub_ddt_2 = ddt.submatrix(r_mask, r_mask).to_dense()
    sub_ddt_3 = ddt.submatrix(r_mask, c_mask).to_dense()

    sub_diag_dense_1 = diag_dense[r_mask][:, r_mask]
    sub_diag_dense_2 = diag_dense[r_mask][:, c_mask]

    torch.testing.assert_close(sub_ddt_1, sub_diag_dense_1)
    torch.testing.assert_close(sub_ddt_2, sub_diag_dense_1)
    torch.testing.assert_close(sub_ddt_3, sub_diag_dense_2)


def test_submatrix_with_batch_dim(diag_batched, device):
    diag_dense = torch.diag_embed(diag_batched).to(device)
    ddt = DiagDecoupledTensor.from_tensor(diag_batched).to(device)

    r_mask = torch.tensor([True, False, True, True], device=device)
    c_mask = torch.tensor([False, True, True, False], device=device)

    sub_ddt_1 = ddt.submatrix(r_mask).to_dense()
    sub_ddt_2 = ddt.submatrix(r_mask, r_mask).to_dense()
    sub_ddt_3 = ddt.submatrix(r_mask, c_mask).to_dense()

    sub_diag_dense_1 = diag_dense[:, r_mask][:, :, r_mask]
    sub_diag_dense_2 = diag_dense[:, r_mask][:, :, c_mask]

    torch.testing.assert_close(sub_ddt_1, sub_diag_dense_1)
    torch.testing.assert_close(sub_ddt_2, sub_diag_dense_1)
    torch.testing.assert_close(sub_ddt_3, sub_diag_dense_2)


def test_dense_conversion(diag, device):
    diag_val = diag.to(device)
    diag_dense = torch.diagflat(diag_val)

    ddt = DiagDecoupledTensor.from_tensor(diag_val)
    ddt_dense = ddt.to_dense()

    assert diag_dense.dtype == ddt.dtype

    torch.testing.assert_close(diag_dense, ddt_dense)


def test_coo_conversion(diag, device):
    diag_val = diag.to(device)
    diag_dense = torch.diagflat(diag_val)

    ddt = DiagDecoupledTensor.from_tensor(diag).to(device)
    ddt_coo = ddt.to_sparse_coo().coalesce()
    ddt_dense = ddt_coo.to_dense()

    assert ddt_coo.shape == diag_dense.shape
    assert ddt_coo.indices().dtype == torch.int64
    assert ddt_coo.dtype == diag_dense.dtype

    torch.testing.assert_close(diag_dense, ddt_dense)


def test_coo_conversion_with_batch_dim(diag_batched, device):
    diag_val = diag_batched.to(device)
    diag_dense = torch.diag_embed(diag_val)

    ddt = DiagDecoupledTensor.from_tensor(diag_val)
    ddt_coo = ddt.to_sparse_coo().coalesce()
    ddt_dense = ddt_coo.to_dense()

    assert ddt_coo.shape == diag_dense.shape
    assert ddt_coo.indices().dtype == torch.int64
    assert ddt_coo.dtype == diag_dense.dtype

    torch.testing.assert_close(diag_dense, ddt_dense)


def test_csr_conversion(diag, device):
    diag_dense = torch.diagflat(diag).to(device)
    ddt = DiagDecoupledTensor.from_tensor(diag).to(device)

    ddt_csr_true = diag_dense.to_sparse_csr()
    ddt_csr = ddt.to_sparse_csr()

    assert ddt_csr.shape == ddt_csr_true.shape
    assert ddt_csr.crow_indices().dtype == torch.int32
    assert ddt_csr.col_indices().dtype == torch.int32
    assert ddt_csr.dtype == ddt_csr_true.dtype

    torch.testing.assert_close(
        ddt_csr.crow_indices().to(torch.int64), ddt_csr_true.crow_indices()
    )
    torch.testing.assert_close(
        ddt_csr.col_indices().to(torch.int64), ddt_csr_true.col_indices()
    )
    torch.testing.assert_close(ddt_csr.values(), ddt_csr_true.values())


def test_csr_conversion_with_batch_dim(diag_batched, device):
    diag_dense = torch.diag_embed(diag_batched).to(device)
    ddt = DiagDecoupledTensor.from_tensor(diag_batched).to(device)

    ddt_csr = ddt.to_sparse_csr()

    assert ddt_csr.shape == ddt.shape
    assert ddt_csr.crow_indices().dtype == torch.int32
    assert ddt_csr.col_indices().dtype == torch.int32
    assert ddt_csr.dtype == ddt.dtype

    # Since it is not possible to directly convert a batched sparse coo tensor
    # to a batched sparse csr tensor, we directly check for value agreement in
    # dense format.
    torch.testing.assert_close(diag_dense.to_dense(), ddt_csr.to_dense())


def test_csc_conversion(diag, device):
    diag_dense = torch.diagflat(diag).to(device)
    ddt = DiagDecoupledTensor.from_tensor(diag).to(device)

    ddt_csc_true = diag_dense.to_sparse_csc()
    ddt_csc = ddt.to_sparse_csc()

    assert ddt_csc.shape == ddt_csc_true.shape
    assert ddt_csc.ccol_indices().dtype == torch.int32
    assert ddt_csc.row_indices().dtype == torch.int32
    assert ddt_csc.dtype == ddt_csc_true.dtype

    torch.testing.assert_close(
        ddt_csc.ccol_indices().to(torch.int64), ddt_csc_true.ccol_indices()
    )
    torch.testing.assert_close(
        ddt_csc.row_indices().to(torch.int64), ddt_csc_true.row_indices()
    )
    torch.testing.assert_close(ddt_csc.values(), ddt_csc_true.values())


def test_csc_conversion_with_batch_dim(diag_batched, device):
    diag_dense = torch.diag_embed(diag_batched).to(device)
    ddt = DiagDecoupledTensor.from_tensor(diag_batched).to(device)

    ddt_csc = ddt.to_sparse_csc()

    assert ddt_csc.shape == ddt.shape
    assert ddt_csc.ccol_indices().dtype == torch.int32
    assert ddt_csc.row_indices().dtype == torch.int32
    assert ddt_csc.dtype == ddt.dtype

    # Since it is not possible to directly convert a batched sparse coo tensor
    # to a batched sparse csc tensor, we directly check for value agreement in
    # dense format.
    torch.testing.assert_close(diag_dense.to_dense(), ddt_csc.to_dense())


def test_matmul_with_batch_dim(diag, diag_batched, a_with_batch, device):
    ddt = DiagDecoupledTensor.from_tensor(diag).to(device)
    diag_batched_operator = DiagDecoupledTensor.from_tensor(diag_batched).to(device)

    sp_batched_operator = SparseDecoupledTensor.from_tensor(a_with_batch).to(device)

    b_dense = torch.randn(
        diag_batched_operator.shape[-1],
        dtype=diag_batched_operator.dtype,
        device=device,
    )

    with pytest.raises(NotImplementedError):
        diag_batched_operator @ b_dense

    with pytest.raises(NotImplementedError):
        b_dense @ diag_batched_operator

    with pytest.raises(NotImplementedError):
        diag_batched_operator @ ddt

    with pytest.raises(NotImplementedError):
        ddt @ diag_batched_operator

    with pytest.raises(NotImplementedError):
        ddt @ sp_batched_operator

    with pytest.raises(NotImplementedError):
        sp_batched_operator @ ddt


def test_matmul_with_dense_dim(diag, device):
    ddt = DiagDecoupledTensor.from_tensor(diag).to(device)

    val = torch.randn(4, 2)
    idx_coo = torch.tensor([[0, 1, 2, 2], [1, 0, 1, 2]])
    shape = (4, 4)

    hybrid_sdt = SparseDecoupledTensor(SparsityPattern(idx_coo, shape), val).to(device)

    with pytest.raises(NotImplementedError):
        ddt @ hybrid_sdt

    with pytest.raises(NotImplementedError):
        hybrid_sdt @ ddt


def test_matmul_with_wrong_tensor_ndim(diag, device):
    diag_val = diag.to(device)
    ddt = DiagDecoupledTensor.from_tensor(diag_val)

    b_dense = torch.randn(
        (diag_val.shape[-1],) * 3, dtype=diag_val.dtype, device=device
    )

    with pytest.raises(NotImplementedError):
        ddt @ b_dense

    with pytest.raises(NotImplementedError):
        b_dense @ ddt


def test_dim(diag, device):
    ddt = DiagDecoupledTensor.from_tensor(diag).to(device)

    assert len(ddt.shape) == 2

    assert ddt.n_dense_dim == 0
    assert ddt.n_sp_dim == 2
    assert ddt.n_batch_dim == 0
    assert ddt.n_dim == 2


def test_dim_with_batch(diag_batched, device):
    ddt = DiagDecoupledTensor.from_tensor(diag_batched).to(device)

    assert len(ddt.shape) == 3

    assert ddt.n_dense_dim == 0
    assert ddt.n_sp_dim == 2
    assert ddt.n_batch_dim == 1
    assert ddt.n_dim == 3


def test_transpose(diag, device):
    diag_val = diag.to(device)
    ddt = DiagDecoupledTensor.from_tensor(diag_val)

    diag_dense_T = torch.diagflat(diag_val).T
    ddt_T = ddt.T.to_dense()

    torch.testing.assert_close(ddt_T, diag_dense_T)


def test_transpose_with_batch_dim(diag_batched, device):
    diag_val = diag_batched.to(device)
    ddt = DiagDecoupledTensor.from_tensor(diag_val)

    diag_dense_T = torch.diag_embed(diag_val).transpose(-1, -2)
    ddt_T = ddt.T.to_dense()

    torch.testing.assert_close(ddt_T, diag_dense_T)


def test_requires_grad_is_false(diag, device):
    diag_val = diag.to(device)
    ddt = DiagDecoupledTensor.from_tensor(diag_val)

    assert diag_val.requires_grad == ddt.requires_grad


def test_requires_grad_is_true(diag, device):
    diag_val = diag.to(device)
    diag_val.requires_grad_()

    ddt = DiagDecoupledTensor.from_tensor(diag_val)

    assert diag_val.requires_grad == ddt.requires_grad


def test_requires_grad_(diag, device):
    diag_val = diag.to(device)
    ddt = DiagDecoupledTensor.from_tensor(diag_val)

    ddt.requires_grad_()
    assert ddt.values.requires_grad


def test_nnz(diag, device):
    diag_val = diag.to(device)
    ddt = DiagDecoupledTensor.from_tensor(diag_val)

    assert ddt._nnz() == diag_val.numel()


def test_nnz_with_batch_dim(diag_batched, device):
    diag_val = diag_batched.to(device)
    ddt = DiagDecoupledTensor.from_tensor(diag_val)

    assert ddt._nnz() == diag_val.numel()


def test_size(diag_batched, device):
    ddt = DiagDecoupledTensor.from_tensor(diag_batched).to(device)
    shape = (2, 4, 4)

    assert ddt.size() == ddt.shape

    for idx, val in enumerate(shape):
        assert ddt.size(idx) == val


def test_eye(device):
    n = 4
    ddt = DiagDecoupledTensor.eye(n, dtype=torch.float32, device=device)

    assert ddt.shape == (n, n)
    assert ddt.dtype == torch.float32
    assert ddt.device.type == device.type
    torch.testing.assert_close(
        ddt.to_dense(), torch.eye(n, dtype=torch.float32, device=device)
    )


def test_to_float64(diag, device):
    diag_val = diag.to(device)
    ddt = DiagDecoupledTensor.from_tensor(diag_val).to(torch.float64)

    assert ddt.values.dtype == torch.float64


def test_to_device(diag, device):
    diag_val = diag
    ddt = DiagDecoupledTensor.from_tensor(diag_val).to(device)

    assert ddt.values.device.type == device.type


def test_clone(diag, device):
    ddt = DiagDecoupledTensor.from_tensor(diag).to(device)
    ddt_clone = ddt.clone()

    assert ddt is not ddt_clone
    assert ddt.values is not ddt_clone.values
    torch.testing.assert_close(ddt_clone.to_dense(), ddt.to_dense())


def test_detach(diag, device):
    diag_req_grad = diag.clone().to(device).requires_grad_()
    ddt = DiagDecoupledTensor.from_tensor(diag_req_grad)
    ddt_detached = ddt.detach()

    assert ddt.requires_grad
    assert not ddt_detached.requires_grad
    torch.testing.assert_close(ddt_detached.to_dense(), ddt.to_dense())


def test_to_sdt(diag, device):
    ddt = DiagDecoupledTensor.from_tensor(diag).to(device)
    sdt = ddt.to_sdt()

    assert isinstance(sdt, SparseDecoupledTensor)
    assert sdt.shape == ddt.shape
    torch.testing.assert_close(sdt.to_dense(), ddt.to_dense())


def test_apply(diag, device):
    diag_dense = torch.diagflat(diag).to(device)
    ddt = DiagDecoupledTensor.from_tensor(diag).clone().to(device)

    diag_dense_applied = torch.relu(diag_dense)
    ddt_applied = ddt.apply(torch.relu)

    torch.testing.assert_close(ddt_applied.to_dense(), diag_dense_applied)


def test_neg(diag, device):
    diag_dense = torch.diagflat(diag).to(device)
    ddt = DiagDecoupledTensor.from_tensor(diag).to(device)

    neg_diag_dense = -diag_dense
    neg_ddt = -ddt

    torch.testing.assert_close(neg_ddt.to_dense(), neg_diag_dense)


def test_abs(diag, device):
    diag_dense = torch.diagflat(diag).to(device)
    ddt = DiagDecoupledTensor.from_tensor(diag).to(device)

    torch.testing.assert_close(ddt.abs().to_dense(), diag_dense.abs())


def test_diagonal(diag, device):
    ddt = DiagDecoupledTensor.from_tensor(diag).to(device)

    torch.testing.assert_close(ddt.diagonal(), diag.to(device))


def test_pow(diag, device):
    diag_dense = torch.diagflat(diag).to(device)
    ddt = DiagDecoupledTensor.from_tensor(diag).to(device)

    diag_dense_pow = diag_dense**2
    ddt_pow = ddt**2

    torch.testing.assert_close(ddt_pow.to_dense(), diag_dense_pow)

    diag_dense_pow = diag_dense.pow(2)
    ddt_pow = ddt.pow(2)

    torch.testing.assert_close(ddt_pow.to_dense(), diag_dense_pow)


def test_inv(diag, device):
    diag_dense = torch.diagflat(diag).to(device)
    ddt = DiagDecoupledTensor.from_tensor(diag).to(device)

    diag_dense_inv = torch.linalg.inv(diag_dense)
    ddt_inv = ddt.inv

    torch.testing.assert_close(ddt_inv.to_dense(), diag_dense_inv)


def test_tr(diag, device):
    diag_dense = torch.diagflat(diag).to(device)
    ddt = DiagDecoupledTensor.from_tensor(diag).to(device)

    diag_dense_tr = diag_dense.trace()
    ddt_tr = ddt.tr

    torch.testing.assert_close(ddt_tr, diag_dense_tr)


def test_tr_with_batch(diag_batched, device):
    diag_dense = torch.diag_embed(diag_batched).to(device)
    ddt = DiagDecoupledTensor.from_tensor(diag_batched).to(device)

    diag_dense_tr = torch.einsum("ijj->i", diag_dense)
    ddt_tr = ddt.tr

    torch.testing.assert_close(ddt_tr, diag_dense_tr)


def test_add(diag, device):
    diag_dense = torch.diagflat(diag).to(device)
    ddt = DiagDecoupledTensor.from_tensor(diag).to(device)

    diag_dense_sum = diag_dense + diag_dense
    ddt_sum = ddt + ddt

    torch.testing.assert_close(ddt_sum.to_dense(), diag_dense_sum)


def test_sub(diag, device):
    diag_dense = torch.diagflat(diag).to(device)
    ddt = DiagDecoupledTensor.from_tensor(diag).to(device)

    diag_dense_sub = diag_dense - diag_dense
    ddt_sub = ddt - ddt

    torch.testing.assert_close(ddt_sub.to_dense(), diag_dense_sub)


def test_mul(diag, device):
    diag_dense = torch.diagflat(diag).to(device)
    ddt = DiagDecoupledTensor.from_tensor(diag).to(device)

    for scalar in [2, 3.0, torch.tensor(-9.0).to(device)]:
        diag_dense_scaled = scalar * diag_dense
        ddt_scaled = scalar * ddt
        ddt_rscaled = ddt * scalar

        torch.testing.assert_close(ddt_scaled.to_dense(), diag_dense_scaled.to_dense())
        torch.testing.assert_close(ddt_rscaled.to_dense(), diag_dense_scaled.to_dense())


def test_trudiv(diag, device):
    diag_dense = torch.diagflat(diag).to(device)
    ddt = DiagDecoupledTensor.from_tensor(diag).to(device)

    for scalar in [2, 3.0, torch.tensor(-9.0).to(device)]:
        diag_dense_scaled = diag_dense / scalar
        ddt_scaled = ddt / scalar

        torch.testing.assert_close(ddt_scaled.to_dense(), diag_dense_scaled.to_dense())


def test_truediv_with_diag(diag, device):
    ddt_1 = DiagDecoupledTensor.from_tensor(diag).to(device)

    diag_2 = torch.randn_like(diag)
    ddt_2 = DiagDecoupledTensor.from_tensor(diag_2).to(device)

    ddt_div = ddt_1 / ddt_2
    torch.testing.assert_close(ddt_div.diagonal(), diag.to(device) / diag_2.to(device))


def test_arithmetic_exceptions(diag, device):
    ddt = DiagDecoupledTensor.from_tensor(diag).to(device)

    with pytest.raises(TypeError):
        ddt + 1.0

    with pytest.raises(TypeError):
        ddt - 1.0

    bad_tensor = torch.randn(2, 2, device=device)

    with pytest.raises(TypeError):
        ddt * bad_tensor

    with pytest.raises(TypeError):
        ddt / bad_tensor

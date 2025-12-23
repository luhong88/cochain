import pytest
import torch as t

from cochain.sparse._sp_topo import SparseTopology
from cochain.sparse.diag_operator import DiagOperator
from cochain.sparse.sp_operator import SparseOperator


@pytest.fixture
def diag():
    n_dim = 4
    return t.randn(n_dim)


@pytest.fixture
def diag_batched():
    n_batch = 2
    n_dim = 4
    return t.randn(n_batch, n_dim)


def test_dense_conversion(diag, device):
    A_tensor = diag.to(device)
    A_tensor_expanded = t.diagflat(A_tensor)

    A_operator = DiagOperator.from_tensor(A_tensor)
    A_operator_dense = A_operator.to_dense()

    assert A_tensor_expanded.dtype == A_operator.dtype

    t.testing.assert_close(A_tensor_expanded, A_operator_dense)


def test_coo_conversion(diag, device):
    A_tensor = diag.to(device)
    A_tensor_expanded = t.diagflat(A_tensor)

    A_operator = DiagOperator.from_tensor(diag).to(device)
    A_coo = A_operator.to_sparse_coo().coalesce()
    A_dense = A_coo.to_dense()

    assert A_coo.shape == A_tensor_expanded.shape
    assert A_coo.indices().dtype == t.int64
    assert A_coo.dtype == A_tensor_expanded.dtype

    t.testing.assert_close(A_tensor_expanded, A_dense)


def test_coo_conversion_with_batch_dim(diag_batched, device):
    A_tensor = diag_batched.to(device)
    A_tensor_expanded = t.diag_embed(A_tensor)

    A_operator = DiagOperator.from_tensor(A_tensor)
    A_coo = A_operator.to_sparse_coo().coalesce()
    A_dense = A_coo.to_dense()

    assert A_coo.shape == A_tensor_expanded.shape
    assert A_coo.indices().dtype == t.int64
    assert A_coo.dtype == A_tensor_expanded.dtype

    t.testing.assert_close(A_tensor_expanded, A_dense)


def test_csr_conversion(diag, device):
    A_tensor = t.diagflat(diag).to(device)
    A_operator = DiagOperator.from_tensor(diag).to(device)

    A_csr_true = A_tensor.to_sparse_csr()
    A_csr = A_operator.to_sparse_csr()

    assert A_csr.shape == A_csr_true.shape
    assert A_csr.crow_indices().dtype == t.int64
    assert A_csr.col_indices().dtype == t.int64
    assert A_csr.dtype == A_csr_true.dtype

    t.testing.assert_close(A_csr.crow_indices(), A_csr_true.crow_indices())
    t.testing.assert_close(A_csr.col_indices(), A_csr_true.col_indices())
    t.testing.assert_close(A_csr.values(), A_csr_true.values())

    A_csr_int32 = A_operator.to_sparse_csr(int32=True)

    assert A_csr_int32.crow_indices().dtype == t.int32
    assert A_csr_int32.col_indices().dtype == t.int32

    t.testing.assert_close(A_csr.crow_indices(), A_csr_int32.crow_indices().to(t.int64))
    t.testing.assert_close(A_csr.col_indices(), A_csr_int32.col_indices().to(t.int64))


def test_csr_conversion_with_batch_dim(diag_batched, device):
    A_tensor = t.diag_embed(diag_batched).to(device)
    A_operator = DiagOperator.from_tensor(diag_batched).to(device)

    A_csr = A_operator.to_sparse_csr()

    assert A_csr.shape == A_operator.shape
    assert A_csr.crow_indices().dtype == t.int64
    assert A_csr.col_indices().dtype == t.int64
    assert A_csr.dtype == A_operator.dtype

    A_csr_int32 = A_operator.to_sparse_csr(int32=True)

    assert A_csr_int32.crow_indices().dtype == t.int32
    assert A_csr_int32.col_indices().dtype == t.int32

    t.testing.assert_close(A_csr.crow_indices(), A_csr_int32.crow_indices().to(t.int64))
    t.testing.assert_close(A_csr.col_indices(), A_csr_int32.col_indices().to(t.int64))

    # Since it is not possible to directly convert a batched sparse coo tensor
    # to a batched sparse csr tensor, we directly check for value agreement in
    # dense format.
    t.testing.assert_close(A_tensor.to_dense(), A_csr.to_dense())


def test_csc_conversion(diag, device):
    A_tensor = t.diagflat(diag).to(device)
    A_operator = DiagOperator.from_tensor(diag).to(device)

    A_csc_true = A_tensor.to_sparse_csc()
    A_csc = A_operator.to_sparse_csc()

    assert A_csc.shape == A_csc_true.shape
    assert A_csc.ccol_indices().dtype == t.int64
    assert A_csc.row_indices().dtype == t.int64
    assert A_csc.dtype == A_csc_true.dtype

    t.testing.assert_close(A_csc.ccol_indices(), A_csc_true.ccol_indices())
    t.testing.assert_close(A_csc.row_indices(), A_csc_true.row_indices())
    t.testing.assert_close(A_csc.values(), A_csc_true.values())

    A_csc_int32 = A_operator.to_sparse_csc(int32=True)

    assert A_csc_int32.ccol_indices().dtype == t.int32
    assert A_csc_int32.row_indices().dtype == t.int32

    t.testing.assert_close(A_csc.ccol_indices(), A_csc_int32.ccol_indices().to(t.int64))
    t.testing.assert_close(A_csc.row_indices(), A_csc_int32.row_indices().to(t.int64))


def test_csc_conversion_with_batch_dim(diag_batched, device):
    A_tensor = t.diag_embed(diag_batched).to(device)
    A_operator = DiagOperator.from_tensor(diag_batched).to(device)

    A_csc = A_operator.to_sparse_csc()

    assert A_csc.shape == A_operator.shape
    assert A_csc.ccol_indices().dtype == t.int64
    assert A_csc.row_indices().dtype == t.int64
    assert A_csc.dtype == A_operator.dtype

    A_csc_int32 = A_operator.to_sparse_csc(int32=True)

    assert A_csc_int32.ccol_indices().dtype == t.int32
    assert A_csc_int32.row_indices().dtype == t.int32

    t.testing.assert_close(A_csc.ccol_indices(), A_csc_int32.ccol_indices().to(t.int64))
    t.testing.assert_close(A_csc.row_indices(), A_csc_int32.row_indices().to(t.int64))

    # Since it is not possible to directly convert a batched sparse coo tensor
    # to a batched sparse csc tensor, we directly check for value agreement in
    # dense format.
    t.testing.assert_close(A_tensor.to_dense(), A_csc.to_dense())


def test_diag_dense_mm(diag, device):
    A_tensor = t.diagflat(diag).to(device)
    A_operator = DiagOperator.from_tensor(diag).to(device)

    B_dense = t.randn(A_tensor.shape[::-1], dtype=A_tensor.dtype, device=device)

    C_dense_true = A_tensor @ B_dense
    C_dense = A_operator @ B_dense

    t.testing.assert_close(C_dense, C_dense_true)


def test_dense_diag_mm(diag, device):
    A_tensor = t.diagflat(diag).to(device)
    A_operator = DiagOperator.from_tensor(diag).to(device)

    B_dense = t.randn(A_tensor.shape[::-1], dtype=A_tensor.dtype, device=device)

    C_dense_true = B_dense @ A_tensor
    C_dense = B_dense @ A_operator

    t.testing.assert_close(C_dense, C_dense_true)


def test_diag_sp_mm(A, diag, device):
    sp_tensor = A.to(device)
    sp_operator = SparseOperator.from_tensor(sp_tensor)

    diag_tensor = t.diagflat(diag).to(device)
    diag_operator = DiagOperator.from_tensor(diag).to(device)

    diag_sp_true = diag_tensor @ sp_tensor
    diag_sp = diag_operator @ sp_operator

    t.testing.assert_close(diag_sp.to_dense(), diag_sp_true.to_dense())


def test_sp_diag_mm(A, diag, device):
    sp_tensor = A.to(device)
    sp_operator = SparseOperator.from_tensor(sp_tensor)

    diag_tensor = t.diagflat(diag).to(device)
    diag_operator = DiagOperator.from_tensor(diag).to(device)

    sp_diag_true = sp_tensor @ diag_tensor
    sp_diag = sp_operator @ diag_operator

    t.testing.assert_close(sp_diag.to_dense(), sp_diag_true.to_dense())


def test_diag_diag_mm(diag, device):
    diag_tensor = t.diagflat(diag).to(device)
    diag_operator = DiagOperator.from_tensor(diag).to(device)

    diag_diag_true = diag_tensor @ diag_tensor.T
    diag_diag = diag_operator @ diag_operator.T

    t.testing.assert_close(diag_diag.to_dense(), diag_diag_true.to_dense())


def test_sp_mv(diag, device):
    A_tensor = t.diagflat(diag).to(device)
    A_operator = DiagOperator.from_tensor(diag).to(device)

    b_dense = t.randn(A_tensor.shape[-1], dtype=A_tensor.dtype, device=device)

    mv_true = A_tensor @ b_dense
    mv = A_operator @ b_dense

    t.testing.assert_close(mv, mv_true)


def test_sp_vm(diag, device):
    A_tensor = t.diagflat(diag).to(device)
    A_operator = DiagOperator.from_tensor(diag).to(device)

    b_dense = t.randn(A_tensor.shape[0], dtype=A_tensor.dtype, device=device)

    vm_true = b_dense @ A_tensor
    vm = b_dense @ A_operator

    t.testing.assert_close(vm, vm_true)


def test_matmul_with_batch_dim(diag, diag_batched, A_batched, device):
    diag_operator = DiagOperator.from_tensor(diag).to(device)
    diag_batched_operator = DiagOperator.from_tensor(diag_batched).to(device)

    sp_batched_operator = SparseOperator.from_tensor(A_batched).to(device)

    b_dense = t.randn(
        diag_batched_operator.shape[-1],
        dtype=diag_batched_operator.dtype,
        device=device,
    )

    with pytest.raises(NotImplementedError):
        diag_batched_operator @ b_dense

    with pytest.raises(NotImplementedError):
        b_dense @ diag_batched_operator

    with pytest.raises(NotImplementedError):
        diag_batched_operator @ diag_operator

    with pytest.raises(NotImplementedError):
        diag_operator @ diag_batched_operator

    with pytest.raises(NotImplementedError):
        diag_operator @ sp_batched_operator

    with pytest.raises(NotImplementedError):
        sp_batched_operator @ diag_operator


def test_matmul_with_dense_dim(diag, device):
    A_operator = DiagOperator.from_tensor(diag).to(device)

    val = t.randn(4, 2)
    idx_coo = t.tensor([[0, 1, 2, 2], [1, 0, 1, 2]])
    shape = (4, 4)

    hybrid_operator = SparseOperator(SparseTopology(idx_coo, shape), val).to(device)

    with pytest.raises(NotImplementedError):
        A_operator @ hybrid_operator

    with pytest.raises(NotImplementedError):
        hybrid_operator @ A_operator


def test_matmul_with_wrong_tensor_ndim(diag, device):
    A_tensor = diag.to(device)
    A_operator = DiagOperator.from_tensor(A_tensor)

    b_dense = t.randn((A_tensor.shape[-1],) * 3, dtype=A_tensor.dtype, device=device)

    with pytest.raises(NotImplementedError):
        A_operator @ b_dense

    with pytest.raises(NotImplementedError):
        b_dense @ A_operator


def test_dim(diag, device):
    A_operator = DiagOperator.from_tensor(diag).to(device)

    assert len(A_operator.shape) == 2

    assert A_operator.n_dense_dim == 0
    assert A_operator.n_sp_dim == 2
    assert A_operator.n_batch_dim == 0
    assert A_operator.n_dim == 2


def test_dim_with_batch(diag_batched, device):
    A_operator = DiagOperator.from_tensor(diag_batched).to(device)

    assert len(A_operator.shape) == 3

    assert A_operator.n_dense_dim == 0
    assert A_operator.n_sp_dim == 2
    assert A_operator.n_batch_dim == 1
    assert A_operator.n_dim == 3


def test_transpose(diag, device):
    A_tensor = diag.to(device)
    A_operator = DiagOperator.from_tensor(A_tensor)

    A_tensor_T = t.diagflat(A_tensor).T
    A_operator_T = A_operator.T.to_dense()

    t.testing.assert_close(A_operator_T, A_tensor_T)


def test_transpose_with_batch_dim(diag_batched, device):
    A_tensor = diag_batched.to(device)
    A_operator = DiagOperator.from_tensor(A_tensor)

    A_tensor_T = t.diag_embed(A_tensor).transpose(-1, -2)
    A_operator_T = A_operator.T.to_dense()

    t.testing.assert_close(A_operator_T, A_tensor_T)


def test_requires_grad_is_false(diag, device):
    A_tensor = diag.to(device)
    A_operator = DiagOperator.from_tensor(A_tensor)

    assert A_tensor.requires_grad == A_operator.requires_grad


def test_requires_grad_is_true(diag, device):
    A_tensor = diag.to(device)
    A_tensor.requires_grad_()

    A_operator = DiagOperator.from_tensor(A_tensor)

    assert A_tensor.requires_grad == A_operator.requires_grad


def test_requires_grad_(diag, device):
    A_tensor = diag.to(device)
    A_operator = DiagOperator.from_tensor(A_tensor)

    A_operator.requires_grad_()
    assert A_operator.val.requires_grad


def test_nnz(diag, device):
    A_tensor = diag.to(device)
    A_operator = DiagOperator.from_tensor(A_tensor)

    assert A_operator._nnz() == A_tensor.numel()


def test_nnz_with_batch_dim(diag_batched, device):
    A_tensor = diag_batched.to(device)
    A_operator = DiagOperator.from_tensor(A_tensor)

    assert A_operator._nnz() == A_tensor.numel()


def test_size(diag_batched, device):
    A_operator = DiagOperator.from_tensor(diag_batched).to(device)
    shape = (2, 4, 4)

    assert A_operator.size() == A_operator.shape

    for idx, val in enumerate(shape):
        assert A_operator.size(idx) == val


def test_to_float64(diag, device):
    A_tensor = diag.to(device)
    A_operator = DiagOperator.from_tensor(A_tensor).to(t.float64)

    assert A_operator.val.dtype == t.float64


def test_to_device(diag, device):
    A_tensor = diag
    A_operator = DiagOperator.from_tensor(A_tensor).to(device)

    assert A_operator.val.device.type == device.type

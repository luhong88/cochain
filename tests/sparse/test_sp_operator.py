import pytest
import torch as t

from cochain.sparse._sp_topo import SparseTopology
from cochain.sparse.sp_operator import SparseOperator


@pytest.mark.gpu_only
def test_device_mismatch(A, device):
    val = A.values()

    idx_coo = A.indices()
    shape = A.shape
    sp_topo = SparseTopology(idx_coo, shape)

    with pytest.raises(RuntimeError):
        SparseOperator(val, sp_topo.to(device))

    with pytest.raises(RuntimeError):
        SparseOperator(val.to(device), sp_topo)


def test_nnz_mismatch(device):
    idx_coo = t.tensor([[0, 0, 1, 2], [0, 1, 0, 3]])
    shape = (4, 4)
    sp_topo = SparseTopology(idx_coo, shape).to(device)

    val = t.randn(3).to(device)

    with pytest.raises(ValueError):
        SparseOperator(val, sp_topo)


def test_dense_conversion(A, device):
    A_tensor = A.to(device)
    A_tensor_dense = A_tensor.to_dense()

    A_operator = SparseOperator.from_tensor(A_tensor)
    A_operator_dense = A_operator.to_dense()

    assert A_tensor_dense.dtype == A_operator.dtype

    t.testing.assert_close(A_tensor_dense, A_operator_dense)


def test_coo_conversion(A, device):
    A_coo_true = A.to(device)
    A_operator = SparseOperator.from_tensor(A_coo_true)

    A_coo = A_operator.to_sparse_coo().coalesce()

    assert A_coo.shape == A_coo_true.shape
    assert A_coo.indices().dtype == t.int64
    assert A_coo.dtype == A_coo_true.dtype

    t.testing.assert_close(A_coo.indices(), A_coo_true.indices())
    t.testing.assert_close(A_coo.values(), A_coo_true.values())


def test_coo_conversion_with_batch_dim(A_batched, device):
    A_coo_true = A_batched.to(device)
    A_operator = SparseOperator.from_tensor(A_coo_true)

    A_coo = A_operator.to_sparse_coo().coalesce()

    assert A_coo.shape == A_coo_true.shape
    assert A_coo.indices().dtype == t.int64
    assert A_coo.dtype == A_coo_true.dtype

    t.testing.assert_close(A_coo.indices(), A_coo_true.indices())
    t.testing.assert_close(A_coo.values(), A_coo_true.values())


def test_csr_conversion(A, device):
    A_tensor = A.to(device)
    A_operator = SparseOperator.from_tensor(A_tensor)

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


def test_csr_conversion_with_batch_dim(A_batched, device):
    A_tensor = A_batched.to(device)
    A_operator = SparseOperator.from_tensor(A_tensor)

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


def test_csc_conversion(A, device):
    A_tensor = A.to(device)
    A_operator = SparseOperator.from_tensor(A_tensor)

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


def test_csc_conversion_with_batch_dim(A_batched, device):
    A_tensor = A_batched.to(device)
    A_operator = SparseOperator.from_tensor(A_tensor)

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


def test_sp_dense_mm(A, device):
    A_tensor = A.to(device)
    A_operator = SparseOperator.from_tensor(A_tensor)

    B_dense = t.randn(A_tensor.shape[::-1], dtype=A_tensor.dtype, device=device)

    C_dense_true = A_tensor @ B_dense
    C_dense = A_operator @ B_dense

    t.testing.assert_close(C_dense, C_dense_true)


def test_dense_sp_mm(A, device):
    A_tensor = A.to(device)
    A_operator = SparseOperator.from_tensor(A_tensor)

    B_dense = t.randn(A_tensor.shape[::-1], dtype=A_tensor.dtype, device=device)

    C_dense_true = B_dense @ A_tensor
    C_dense = B_dense @ A_operator

    t.testing.assert_close(C_dense, C_dense_true)


def test_sp_sp_mm(A, device):
    A_tensor = A.to(device)
    A_operator = SparseOperator.from_tensor(A_tensor)

    sp_sp_true = A_tensor @ A_tensor.T
    sp_sp = A_operator @ A_operator.T

    t.testing.assert_close(sp_sp.to_dense(), sp_sp_true.to_dense())


def test_sp_mv(A, device):
    A_tensor = A.to(device)
    A_operator = SparseOperator.from_tensor(A_tensor)

    b_dense = t.randn(A_tensor.shape[-1], dtype=A_tensor.dtype, device=device)

    mv_true = A_tensor @ b_dense
    mv = A_operator @ b_dense

    t.testing.assert_close(mv, mv_true)


def test_sp_vm(A, device):
    A_tensor = A.to(device)
    A_operator = SparseOperator.from_tensor(A_tensor)

    b_dense = t.randn(A_tensor.shape[0], dtype=A_tensor.dtype, device=device)

    vm_true = b_dense @ A_tensor
    vm = b_dense @ A_operator

    t.testing.assert_close(vm, vm_true)


def test_matmul_with_batch_dim(A, A_batched, device):
    A_operator = SparseOperator.from_tensor(A).to(device)
    A_batched_operator = SparseOperator.from_tensor(A_batched).to(device)

    b_dense = t.randn(
        A_batched_operator.shape[-1], dtype=A_batched_operator.dtype, device=device
    )

    with pytest.raises(NotImplementedError):
        A_batched_operator @ b_dense

    with pytest.raises(NotImplementedError):
        A_batched_operator @ A_operator

    with pytest.raises(NotImplementedError):
        b_dense @ A_batched_operator

    with pytest.raises(NotImplementedError):
        A_operator @ A_batched_operator


def test_matmul_with_dense_dim(A, device):
    A_operator = SparseOperator.from_tensor(A).to(device)

    val = t.randn(4, 2)
    idx_coo = t.tensor([[0, 1, 2, 2], [1, 0, 1, 2]])
    shape = (4, 4)

    hybrid_operator = SparseOperator(val, SparseTopology(idx_coo, shape)).to(device)

    b_dense = t.randn(shape[-1], dtype=hybrid_operator.dtype, device=device)

    with pytest.raises(NotImplementedError):
        hybrid_operator @ b_dense

    with pytest.raises(NotImplementedError):
        hybrid_operator @ A_operator

    with pytest.raises(NotImplementedError):
        b_dense @ hybrid_operator

    with pytest.raises(NotImplementedError):
        A_operator @ hybrid_operator


def test_matmul_with_wrong_tensor_ndim(A, device):
    A_tensor = A.to(device)
    A_operator = SparseOperator.from_tensor(A_tensor)

    b_dense = t.randn((A_tensor.shape[-1],) * 3, dtype=A_tensor.dtype, device=device)

    with pytest.raises(NotImplementedError):
        A_operator @ b_dense

    with pytest.raises(NotImplementedError):
        b_dense @ A_operator


def test_dim(A, device):
    A_operator = SparseOperator.from_tensor(A).to(device)

    assert len(A_operator.shape) == 2

    assert A_operator.n_dense_dim == 0
    assert A_operator.n_sp_dim == 2
    assert A_operator.n_batch_dim == 0
    assert A_operator.n_dim == 2


def test_dim_with_batch(A_batched, device):
    A_operator = SparseOperator.from_tensor(A_batched).to(device)

    assert len(A_operator.shape) == 3

    assert A_operator.n_dense_dim == 0
    assert A_operator.n_sp_dim == 2
    assert A_operator.n_batch_dim == 1
    assert A_operator.n_dim == 3


def test_dim_with_batch_dense(device):
    val = t.randn(4, 2)
    idx_coo = t.tensor([[0, 0, 1, 1], [0, 1, 2, 2], [1, 0, 1, 2]])
    shape = (2, 4, 4)

    A_operator = SparseOperator(val, SparseTopology(idx_coo, shape)).to(device)

    assert len(A_operator.shape) == 4

    assert A_operator.n_dense_dim == 1
    assert A_operator.n_sp_dim == 2
    assert A_operator.n_batch_dim == 1
    assert A_operator.n_dim == 4


def test_transpose(A, device):
    A_tensor = A.to(device)
    A_operator = SparseOperator.from_tensor(A_tensor)

    A_tensor_T = A_tensor.T.to_dense()
    A_operator_T = A_operator.T.to_dense()

    t.testing.assert_close(A_operator_T, A_tensor_T)


def test_transpose_with_batch_dim(A_batched, device):
    A_tensor = A_batched.to(device)
    A_operator = SparseOperator.from_tensor(A_tensor)

    A_tensor_T = A_tensor.to_dense().transpose(-1, -2)
    A_operator_T = A_operator.T.to_dense()

    t.testing.assert_close(A_operator_T, A_tensor_T)


def test_transpose_with_batch_dense_dim(device):
    val = t.randn(4, 2)
    idx_coo = t.tensor([[0, 0, 1, 1], [0, 1, 2, 2], [1, 0, 1, 2]])
    shape = (2, 4, 4)

    sp_op = SparseOperator(val, SparseTopology(idx_coo, shape)).to(device)

    sp_op_T = sp_op.T.to_dense()
    sp_tensor_T = sp_op.to_dense().transpose(1, 2)

    t.testing.assert_close(sp_op_T, sp_tensor_T)


def test_requires_grad_is_false(A, device):
    A_tensor = A.to(device)
    A_operator = SparseOperator.from_tensor(A_tensor)

    assert A_tensor.requires_grad == A_operator.requires_grad


def test_requires_grad_is_true(A, device):
    A_tensor = A.to(device)
    A_tensor.requires_grad_()

    A_operator = SparseOperator.from_tensor(A_tensor)

    assert A_tensor.requires_grad == A_operator.requires_grad


def test_requires_grad_(A, device):
    A_tensor = A.to(device)
    A_operator = SparseOperator.from_tensor(A_tensor)

    A_operator.requires_grad_()
    assert A_operator.val.requires_grad


def test_nnz(A, device):
    A_tensor = A.to(device)
    A_operator = SparseOperator.from_tensor(A_tensor)

    assert A_operator._nnz() == A_tensor._nnz()


def test_nnz_with_batch_dim(A_batched, device):
    A_tensor = A_batched.to(device)
    A_operator = SparseOperator.from_tensor(A_tensor)

    assert A_operator._nnz() == A_tensor._nnz()


def test_size(device):
    val = t.randn(4, 2)
    idx_coo = t.tensor([[0, 0, 1, 1], [0, 1, 2, 2], [1, 0, 1, 2]])
    shape = (2, 4, 4)

    sp_op = SparseOperator(val, SparseTopology(idx_coo, shape)).to(device)

    assert sp_op.size() == sp_op.shape

    for idx, val in enumerate(shape + (val.shape[-1],)):
        assert sp_op.size(idx) == val


def test_to_float64(A, device):
    A_tensor = A.to(device)
    A_operator = SparseOperator.from_tensor(A_tensor).to(t.float64)

    assert A_operator.val.dtype == t.float64


def test_to_device(A, device):
    A_tensor = A
    A_operator = SparseOperator.from_tensor(A_tensor).to(device)

    assert A_operator.val.device.type == device.type
    assert A_operator.sp_topo.device.type == device.type

import pytest
import torch

from cochain.sparse.decoupled_tensor import (
    DiagDecoupledTensor,
    SparseDecoupledTensor,
    SparsityPattern,
)


@pytest.mark.gpu_only
def test_device_mismatch(A, device):
    val = A.values()

    idx_coo = A.indices()
    shape = A.shape
    pattern = SparsityPattern(idx_coo, shape)

    with pytest.raises(RuntimeError):
        SparseDecoupledTensor(pattern.to(device), val)

    with pytest.raises(RuntimeError):
        SparseDecoupledTensor(pattern, val.to(device))


def test_nnz_mismatch(device):
    idx_coo = torch.tensor([[0, 0, 1, 2], [0, 1, 0, 3]])
    shape = (4, 4)
    pattern = SparsityPattern(idx_coo, shape).to(device)

    val = torch.randn(3).to(device)

    with pytest.raises(ValueError):
        SparseDecoupledTensor(pattern, val)


def test_dense_conversion(A, device):
    A_tensor = A.to(device)
    A_tensor_dense = A_tensor.to_dense()

    A_operator = SparseDecoupledTensor.from_tensor(A_tensor)
    A_operator_dense = A_operator.to_dense()

    assert A_tensor_dense.dtype == A_operator.dtype

    torch.testing.assert_close(A_tensor_dense, A_operator_dense)


def test_coo_conversion(A, device):
    A_coo_true = A.to(device)
    A_operator = SparseDecoupledTensor.from_tensor(A_coo_true)

    A_coo = A_operator.to_sparse_coo().coalesce()

    assert A_coo.shape == A_coo_true.shape
    assert A_coo.indices().dtype == torch.int64
    assert A_coo.dtype == A_coo_true.dtype

    torch.testing.assert_close(A_coo.indices(), A_coo_true.indices())
    torch.testing.assert_close(A_coo.values(), A_coo_true.values())


def test_coo_conversion_with_batch_dim(A_batched, device):
    A_coo_true = A_batched.to(device)
    A_operator = SparseDecoupledTensor.from_tensor(A_coo_true)

    A_coo = A_operator.to_sparse_coo().coalesce()

    assert A_coo.shape == A_coo_true.shape
    assert A_coo.indices().dtype == torch.int64
    assert A_coo.dtype == A_coo_true.dtype

    torch.testing.assert_close(A_coo.indices(), A_coo_true.indices())
    torch.testing.assert_close(A_coo.values(), A_coo_true.values())


def test_csr_conversion(A, device):
    A_tensor = A.to(device)
    A_operator = SparseDecoupledTensor.from_tensor(A_tensor)

    A_csr_true = A_tensor.to_sparse_csr()
    A_csr = A_operator.to_sparse_csr()

    assert A_csr.shape == A_csr_true.shape
    assert A_csr.crow_indices().dtype == torch.int64
    assert A_csr.col_indices().dtype == torch.int64
    assert A_csr.dtype == A_csr_true.dtype

    torch.testing.assert_close(A_csr.crow_indices(), A_csr_true.crow_indices())
    torch.testing.assert_close(A_csr.col_indices(), A_csr_true.col_indices())
    torch.testing.assert_close(A_csr.values(), A_csr_true.values())

    A_csr_int32 = A_operator.to_sparse_csr(int32=True)

    assert A_csr_int32.crow_indices().dtype == torch.int32
    assert A_csr_int32.col_indices().dtype == torch.int32

    torch.testing.assert_close(
        A_csr.crow_indices(), A_csr_int32.crow_indices().to(torch.int64)
    )
    torch.testing.assert_close(
        A_csr.col_indices(), A_csr_int32.col_indices().to(torch.int64)
    )


def test_csr_conversion_with_batch_dim(A_batched, device):
    A_tensor = A_batched.to(device)
    A_operator = SparseDecoupledTensor.from_tensor(A_tensor)

    A_csr = A_operator.to_sparse_csr()

    assert A_csr.shape == A_operator.shape
    assert A_csr.crow_indices().dtype == torch.int64
    assert A_csr.col_indices().dtype == torch.int64
    assert A_csr.dtype == A_operator.dtype

    A_csr_int32 = A_operator.to_sparse_csr(int32=True)

    assert A_csr_int32.crow_indices().dtype == torch.int32
    assert A_csr_int32.col_indices().dtype == torch.int32

    torch.testing.assert_close(
        A_csr.crow_indices(), A_csr_int32.crow_indices().to(torch.int64)
    )
    torch.testing.assert_close(
        A_csr.col_indices(), A_csr_int32.col_indices().to(torch.int64)
    )

    # Since it is not possible to directly convert a batched sparse coo tensor
    # to a batched sparse csr tensor, we directly check for value agreement in
    # dense format.
    torch.testing.assert_close(A_tensor.to_dense(), A_csr.to_dense())


def test_csr_transposed_conversion(A, device):
    A_op = SparseDecoupledTensor.from_tensor(A).to(device)
    A_T_direct = A_op.to_sparse_csr_transposed()
    A_T_indirect = A_op.T.to_sparse_csr()

    assert A_T_direct.shape == A_T_indirect.shape

    torch.testing.assert_close(A_T_direct.values(), A_T_indirect.values())
    torch.testing.assert_close(A_T_direct.crow_indices(), A_T_indirect.crow_indices())
    torch.testing.assert_close(A_T_direct.col_indices(), A_T_indirect.col_indices())


def test_csr_transposed_conversion_with_batch_dim(A_batched, device):
    A_op = SparseDecoupledTensor.from_tensor(A_batched).to(device)
    A_T_direct = A_op.to_sparse_csr_transposed()
    A_T_indirect = A_op.T.to_sparse_csr()

    assert A_T_direct.shape == A_T_indirect.shape

    torch.testing.assert_close(A_T_direct.values(), A_T_indirect.values())
    torch.testing.assert_close(A_T_direct.crow_indices(), A_T_indirect.crow_indices())
    torch.testing.assert_close(A_T_direct.col_indices(), A_T_indirect.col_indices())


def test_csc_conversion(A, device):
    A_tensor = A.to(device)
    A_operator = SparseDecoupledTensor.from_tensor(A_tensor)

    A_csc_true = A_tensor.to_sparse_csc()
    A_csc = A_operator.to_sparse_csc()

    assert A_csc.shape == A_csc_true.shape
    assert A_csc.ccol_indices().dtype == torch.int64
    assert A_csc.row_indices().dtype == torch.int64
    assert A_csc.dtype == A_csc_true.dtype

    torch.testing.assert_close(A_csc.ccol_indices(), A_csc_true.ccol_indices())
    torch.testing.assert_close(A_csc.row_indices(), A_csc_true.row_indices())
    torch.testing.assert_close(A_csc.values(), A_csc_true.values())

    A_csc_int32 = A_operator.to_sparse_csc(int32=True)

    assert A_csc_int32.ccol_indices().dtype == torch.int32
    assert A_csc_int32.row_indices().dtype == torch.int32

    torch.testing.assert_close(
        A_csc.ccol_indices(), A_csc_int32.ccol_indices().to(torch.int64)
    )
    torch.testing.assert_close(
        A_csc.row_indices(), A_csc_int32.row_indices().to(torch.int64)
    )


def test_csc_conversion_with_batch_dim(A_batched, device):
    A_tensor = A_batched.to(device)
    A_operator = SparseDecoupledTensor.from_tensor(A_tensor)

    A_csc = A_operator.to_sparse_csc()

    assert A_csc.shape == A_operator.shape
    assert A_csc.ccol_indices().dtype == torch.int64
    assert A_csc.row_indices().dtype == torch.int64
    assert A_csc.dtype == A_operator.dtype

    A_csc_int32 = A_operator.to_sparse_csc(int32=True)

    assert A_csc_int32.ccol_indices().dtype == torch.int32
    assert A_csc_int32.row_indices().dtype == torch.int32

    torch.testing.assert_close(
        A_csc.ccol_indices(), A_csc_int32.ccol_indices().to(torch.int64)
    )
    torch.testing.assert_close(
        A_csc.row_indices(), A_csc_int32.row_indices().to(torch.int64)
    )

    # Since it is not possible to directly convert a batched sparse coo tensor
    # to a batched sparse csc tensor, we directly check for value agreement in
    # dense format.
    torch.testing.assert_close(A_tensor.to_dense(), A_csc.to_dense())


def test_sp_dense_mm(A, device):
    A_tensor = A.to(device)
    A_operator = SparseDecoupledTensor.from_tensor(A_tensor)

    B_dense = torch.randn(A_tensor.shape[::-1], dtype=A_tensor.dtype, device=device)

    C_dense_true = A_tensor @ B_dense
    C_dense = A_operator @ B_dense

    torch.testing.assert_close(C_dense, C_dense_true)


def test_dense_sp_mm(A, device):
    A_tensor = A.to(device)
    A_operator = SparseDecoupledTensor.from_tensor(A_tensor)

    B_dense = torch.randn(A_tensor.shape[::-1], dtype=A_tensor.dtype, device=device)

    C_dense_true = B_dense @ A_tensor
    C_dense = B_dense @ A_operator

    torch.testing.assert_close(C_dense, C_dense_true)


def test_sp_sp_mm(A, device):
    A_tensor = A.to(device)
    A_operator = SparseDecoupledTensor.from_tensor(A_tensor)

    sp_sp_true = A_tensor @ A_tensor.T
    sp_sp = A_operator @ A_operator.T

    torch.testing.assert_close(sp_sp.to_dense(), sp_sp_true.to_dense())


def test_sp_mv(A, device):
    A_tensor = A.to(device)
    A_operator = SparseDecoupledTensor.from_tensor(A_tensor)

    b_dense = torch.randn(A_tensor.shape[-1], dtype=A_tensor.dtype, device=device)

    mv_true = A_tensor @ b_dense
    mv = A_operator @ b_dense

    torch.testing.assert_close(mv, mv_true)


def test_sp_vm(A, device):
    A_tensor = A.to(device)
    A_operator = SparseDecoupledTensor.from_tensor(A_tensor)

    b_dense = torch.randn(A_tensor.shape[0], dtype=A_tensor.dtype, device=device)

    vm_true = b_dense @ A_tensor
    vm = b_dense @ A_operator

    torch.testing.assert_close(vm, vm_true)


def test_matmul_with_batch_dim(A, A_batched, device):
    A_operator = SparseDecoupledTensor.from_tensor(A).to(device)
    A_batched_operator = SparseDecoupledTensor.from_tensor(A_batched).to(device)

    b_dense = torch.randn(
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
    A_operator = SparseDecoupledTensor.from_tensor(A).to(device)

    val = torch.randn(4, 2)
    idx_coo = torch.tensor([[0, 1, 2, 2], [1, 0, 1, 2]])
    shape = (4, 4)

    hybrid_operator = SparseDecoupledTensor(SparsityPattern(idx_coo, shape), val).to(
        device
    )

    b_dense = torch.randn(shape[-1], dtype=hybrid_operator.dtype, device=device)

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
    A_operator = SparseDecoupledTensor.from_tensor(A_tensor)

    b_dense = torch.randn(
        (A_tensor.shape[-1],) * 3, dtype=A_tensor.dtype, device=device
    )

    with pytest.raises(NotImplementedError):
        A_operator @ b_dense

    with pytest.raises(NotImplementedError):
        b_dense @ A_operator


def test_dim(A, device):
    A_operator = SparseDecoupledTensor.from_tensor(A).to(device)

    assert len(A_operator.shape) == 2

    assert A_operator.n_dense_dim == 0
    assert A_operator.n_sp_dim == 2
    assert A_operator.n_batch_dim == 0
    assert A_operator.n_dim == 2


def test_dim_with_batch(A_batched, device):
    A_operator = SparseDecoupledTensor.from_tensor(A_batched).to(device)

    assert len(A_operator.shape) == 3

    assert A_operator.n_dense_dim == 0
    assert A_operator.n_sp_dim == 2
    assert A_operator.n_batch_dim == 1
    assert A_operator.n_dim == 3


def test_dim_with_batch_dense(device):
    val = torch.randn(4, 2)
    idx_coo = torch.tensor([[0, 0, 1, 1], [0, 1, 2, 2], [1, 0, 1, 2]])
    shape = (2, 4, 4)

    A_operator = SparseDecoupledTensor(SparsityPattern(idx_coo, shape), val).to(device)

    assert len(A_operator.shape) == 4

    assert A_operator.n_dense_dim == 1
    assert A_operator.n_sp_dim == 2
    assert A_operator.n_batch_dim == 1
    assert A_operator.n_dim == 4


def test_transpose(A, device):
    A_tensor = A.to(device)
    A_operator = SparseDecoupledTensor.from_tensor(A_tensor)

    A_tensor_T = A_tensor.T.to_dense()
    A_operator_T = A_operator.T.to_dense()

    torch.testing.assert_close(A_operator_T, A_tensor_T)


def test_transpose_with_batch_dim(A_batched, device):
    A_tensor = A_batched.to(device)
    A_operator = SparseDecoupledTensor.from_tensor(A_tensor)

    A_tensor_T = A_tensor.to_dense().transpose(-1, -2)
    A_operator_T = A_operator.T.to_dense()

    torch.testing.assert_close(A_operator_T, A_tensor_T)


def test_transpose_with_batch_dense_dim(device):
    val = torch.randn(4, 2)
    idx_coo = torch.tensor([[0, 0, 1, 1], [0, 1, 2, 2], [1, 0, 1, 2]])
    shape = (2, 4, 4)

    sdt = SparseDecoupledTensor(SparsityPattern(idx_coo, shape), val).to(device)

    sp_op_T = sdt.T.to_dense()
    sp_tensor_T = sdt.to_dense().transpose(1, 2)

    torch.testing.assert_close(sp_op_T, sp_tensor_T)


def test_requires_grad_is_false(A, device):
    A_tensor = A.to(device)
    A_operator = SparseDecoupledTensor.from_tensor(A_tensor)

    assert A_tensor.requires_grad == A_operator.requires_grad


def test_requires_grad_is_true(A, device):
    A_tensor = A.to(device)
    A_tensor.requires_grad_()

    A_operator = SparseDecoupledTensor.from_tensor(A_tensor)

    assert A_tensor.requires_grad == A_operator.requires_grad


def test_requires_grad_(A, device):
    A_tensor = A.to(device)
    A_operator = SparseDecoupledTensor.from_tensor(A_tensor)

    A_operator.requires_grad_()
    assert A_operator.val.requires_grad


def test_nnz(A, device):
    A_tensor = A.to(device)
    A_operator = SparseDecoupledTensor.from_tensor(A_tensor)

    assert A_operator._nnz() == A_tensor._nnz()


def test_nnz_with_batch_dim(A_batched, device):
    A_tensor = A_batched.to(device)
    A_operator = SparseDecoupledTensor.from_tensor(A_tensor)

    assert A_operator._nnz() == A_tensor._nnz()


def test_size(device):
    val = torch.randn(4, 2)
    idx_coo = torch.tensor([[0, 0, 1, 1], [0, 1, 2, 2], [1, 0, 1, 2]])
    shape = (2, 4, 4)

    sdt = SparseDecoupledTensor(SparsityPattern(idx_coo, shape), val).to(device)

    assert sdt.size() == sdt.shape

    for idx, val in enumerate(shape + (val.shape[-1],)):
        assert sdt.size(idx) == val


def test_to_float64(A, device):
    A_tensor = A.to(device)
    A_operator = SparseDecoupledTensor.from_tensor(A_tensor).to(torch.float64)

    assert A_operator.val.dtype == torch.float64


def test_to_device(A, device):
    A_tensor = A
    A_operator = SparseDecoupledTensor.from_tensor(A_tensor).to(device)

    assert A_operator.val.device.type == device.type
    assert A_operator.pattern.device.type == device.type


def test_apply(A, device):
    A_tensor = A.to(device)
    A_op = SparseDecoupledTensor.from_tensor(A).clone().to(device)

    tensor_applied = torch.relu(A_tensor.to_dense())
    op_applied = A_op.apply(torch.relu).to_dense()

    torch.testing.assert_close(op_applied, tensor_applied)


def test_neg(A, device):
    A_tensor = A.to(device)
    neg_A_tensor = -A_tensor
    A_op = SparseDecoupledTensor.from_tensor(A).to(device)
    neg_A_op = -A_op

    torch.testing.assert_close(neg_A_op.to_dense(), neg_A_tensor.to_dense())


def test_add(A, device):
    A_tensor = A.to(device)
    A_op = SparseDecoupledTensor.from_tensor(A).to(device)

    tensor_sum = A_tensor + A_tensor
    op_sum = A_op + A_op

    torch.testing.assert_close(op_sum.to_dense(), tensor_sum.to_dense())


def test_sub(A, device):
    A_tensor = A.to(device)
    A_op = SparseDecoupledTensor.from_tensor(A).to(device)

    tensor_sub = A_tensor - A_tensor
    op_sub = A_op - A_op

    torch.testing.assert_close(op_sub.to_dense(), tensor_sub.to_dense())


def test_assemble(A, device):
    A_tensor = A.to(device)
    A_op = SparseDecoupledTensor.from_tensor(A).to(device)

    tensor_sum = A_tensor + A_tensor.T
    op_sum = SparseDecoupledTensor.assemble(A_op, A_op.T)

    torch.testing.assert_close(op_sum.to_dense(), tensor_sum.to_dense())


def test_assemble_with_diag_operator(A, device):
    A_tensor = A.to(device)
    A_op = SparseDecoupledTensor.from_tensor(A).to(device)

    diag = torch.randn(A_tensor.size(0))
    diag_tensor = torch.diagflat(diag).to(device)
    ddt = DiagDecoupledTensor.from_tensor(diag).to(device)

    tensor_sum = diag_tensor + A_tensor
    op_sum = SparseDecoupledTensor.assemble(A_op, ddt)

    torch.testing.assert_close(op_sum.to_dense(), tensor_sum)


def test_mul(A, device):
    A_tensor = A.to(device)
    A_op = SparseDecoupledTensor.from_tensor(A).to(device)

    for scalar in [2, 3.0, torch.tensor(-9.0).to(device)]:
        tensor_scaled = scalar * A_tensor
        op_scaled = scalar * A_op
        op_rscaled = A_op * scalar

        torch.testing.assert_close(op_scaled.to_dense(), tensor_scaled.to_dense())
        torch.testing.assert_close(op_rscaled.to_dense(), tensor_scaled.to_dense())


def test_trudiv(A, device):
    A_tensor = A.to(device)
    A_op = SparseDecoupledTensor.from_tensor(A).to(device)

    for scalar in [2, 3.0, torch.tensor(-9.0).to(device)]:
        tensor_scaled = A_tensor / scalar
        op_scaled = A_op / scalar

        torch.testing.assert_close(op_scaled.to_dense(), tensor_scaled.to_dense())


def test_pack_block_diag(device):
    a = (
        torch.randint(0, 3, (3, 3))
        .to_sparse_coo()
        .to(dtype=torch.float32, device=device)
    )
    b = SparseDecoupledTensor.from_tensor(torch.randint(0, 3, (2, 2))).to(
        dtype=torch.float32, device=device
    )
    c = DiagDecoupledTensor.from_tensor(torch.randint(0, 3, (4,))).to(
        dtype=torch.float32, device=device
    )

    true_block_diag = torch.block_diag(a.to_dense(), b.to_dense(), c.to_dense())
    block_diag = SparseDecoupledTensor.pack_block_diag((a, b, c))

    torch.testing.assert_close(block_diag.to_dense(), true_block_diag)

    sp_ops = block_diag.unpack_block_diag()

    for sdt, ori_op in zip(sp_ops, [a, b, c]):
        torch.testing.assert_close(sdt.to_dense(), ori_op.to_dense())


def test_unpack_block_diag_via_ptrs(device):
    a = (
        torch.randint(0, 3, (3, 2))
        .to_sparse_coo()
        .to(dtype=torch.float32, device=device)
    )
    b = SparseDecoupledTensor.from_tensor(torch.randint(0, 3, (2, 4))).to(
        dtype=torch.float32, device=device
    )
    c = DiagDecoupledTensor.from_tensor(torch.randint(0, 3, (4,))).to(
        dtype=torch.float32, device=device
    )

    block_diag = SparseDecoupledTensor.pack_block_diag((a, b, c))

    row_ptrs = torch.tensor(
        [0, 0, 0, 1, 1, 2, 2, 2, 2], dtype=torch.long, device=device
    )
    col_ptrs = torch.tensor(
        [0, 0, 1, 1, 1, 1, 2, 2, 2, 2], dtype=torch.long, device=device
    )

    sp_ops = block_diag.unpack_by_ptrs(n_blocks=3, row_ptrs=row_ptrs, col_ptrs=col_ptrs)

    for sdt, ori_op in zip(sp_ops, [a, b, c]):
        torch.testing.assert_close(sdt.to_dense(), ori_op.to_dense())


def test_pack_block_diag_with_batch_dim(A_batched, device):
    block_diag = SparseDecoupledTensor.pack_block_diag((A_batched, -A_batched)).to(
        device
    )

    A_batched_dense = A_batched.to_dense().to(device)

    block_diag_true = torch.stack(
        (
            torch.block_diag(A_batched_dense[0], -A_batched_dense[0]),
            torch.block_diag(A_batched_dense[1], -A_batched_dense[1]),
        )
    )

    torch.testing.assert_close(block_diag.to_dense(), block_diag_true)

    sp_ops = block_diag.unpack_block_diag()

    torch.testing.assert_close(sp_ops[0].to_dense(), A_batched_dense)
    torch.testing.assert_close(sp_ops[1].to_dense(), -A_batched_dense)


def test_pack_block_diag_with_dense_dim(device):
    a = SparseDecoupledTensor.from_tensor(
        torch.sparse_coo_tensor(
            indices=torch.randint(0, 4, (2, 5)),
            values=torch.randn(5, 2),
            size=(4, 4, 2),
        ).coalesce()
    ).to(device)

    b = SparseDecoupledTensor.from_tensor(
        torch.sparse_coo_tensor(
            indices=torch.randint(0, 3, (2, 3)),
            values=torch.randn(3, 2),
            size=(3, 3, 2),
        ).coalesce()
    ).to(device)

    block_diag = SparseDecoupledTensor.pack_block_diag((a, b))

    block_diag_true = torch.stack(
        (
            torch.block_diag(a.to_dense()[:, :, 0], b.to_dense()[:, :, 0]),
            torch.block_diag(a.to_dense()[:, :, 1], b.to_dense()[:, :, 1]),
        ),
        dim=-1,
    )

    torch.testing.assert_close(block_diag.to_dense(), block_diag_true)


def test_bmat(device):
    a = torch.randn(2, 3).to(device)
    b = torch.randn(2, 4).to(device)
    c = torch.randn(4, 3).to(device)
    d = torch.zeros(4, 4).to(device)

    bmat = SparseDecoupledTensor.bmat([[a, b], [c, None]])
    bmat_true = torch.cat((torch.cat((a, b), dim=-1), torch.cat((c, d), dim=-1)), dim=0)

    torch.testing.assert_close(bmat.to_dense(), bmat_true)


def test_bmat_with_batch_dim(device):
    a = torch.randn(2, 2, 3).to(device)
    b = torch.randn(2, 2, 4).to(device)
    c = torch.randn(2, 4, 3).to(device)
    d = torch.zeros(2, 4, 4).to(device)

    bmat = SparseDecoupledTensor.bmat([[a, b], [c, None]])
    bmat_true = torch.stack(
        (
            torch.cat(
                (torch.cat((a[0], b[0]), dim=-1), torch.cat((c[0], d[0]), dim=-1)),
                dim=0,
            ),
            torch.cat(
                (torch.cat((a[1], b[1]), dim=-1), torch.cat((c[1], d[1]), dim=-1)),
                dim=0,
            ),
        )
    )

    torch.testing.assert_close(bmat.to_dense(), bmat_true)


def test_bmat_with_dense_dim(device):
    a = SparseDecoupledTensor.from_tensor(
        torch.sparse_coo_tensor(
            indices=torch.randint(0, 4, (2, 5)),
            values=torch.randn(5, 2),
            size=(4, 4, 2),
        ).coalesce()
    ).to(device)

    b = SparseDecoupledTensor.from_tensor(
        torch.sparse_coo_tensor(
            indices=torch.randint(0, 3, (2, 3)),
            values=torch.randn(3, 2),
            size=(3, 3, 2),
        ).coalesce()
    ).to(device)

    bmat = SparseDecoupledTensor.bmat([[a, None], [None, b]])

    bmat_true = torch.stack(
        (
            torch.block_diag(a.to_dense()[:, :, 0], b.to_dense()[:, :, 0]),
            torch.block_diag(a.to_dense()[:, :, 1], b.to_dense()[:, :, 1]),
        ),
        dim=-1,
    )

    torch.testing.assert_close(bmat.to_dense(), bmat_true)


def test_bmat_with_invalid_row_col(A, device):
    a = A.to(device)

    # Test degenerate column
    bmat_1 = SparseDecoupledTensor.bmat([[a, None], [a, None]])
    bmat_2 = SparseDecoupledTensor.bmat([[a], [a]])

    torch.testing.assert_close(bmat_1.to_dense(), bmat_2.to_dense())

    # Test degenerate row
    bmat_1 = SparseDecoupledTensor.bmat([[None, None], [a, a]])
    bmat_2 = SparseDecoupledTensor.bmat([[a, a]])

    torch.testing.assert_close(bmat_1.to_dense(), bmat_2.to_dense())

    # Test full degenerate bmat
    with pytest.raises(ValueError):
        SparseDecoupledTensor.bmat([[None, None], [None, None]])

    # Test invalid dtype
    with pytest.raises(TypeError):
        SparseDecoupledTensor.bmat([a, 3], [None, a])

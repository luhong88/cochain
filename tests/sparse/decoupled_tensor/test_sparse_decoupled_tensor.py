import pytest
import torch

from cochain.sparse.decoupled_tensor import (
    DiagDecoupledTensor,
    SparseDecoupledTensor,
    SparsityPattern,
)


@pytest.mark.gpu_only
def test_device_mismatch(a, device):
    val = a.values()

    idx_coo = a.indices()
    shape = a.shape
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


def test_dense_conversion(a, device):
    a_coo = a.to(device)
    a_coo_to_dense = a_coo.to_dense()

    a_sdt = SparseDecoupledTensor.from_tensor(a_coo)
    a_sdt_to_dense = a_sdt.to_dense()

    assert a_coo_to_dense.dtype == a_sdt.dtype

    torch.testing.assert_close(a_coo_to_dense, a_sdt_to_dense)


def test_coo_conversion(a, device):
    a_coo = a.to(device)
    a_sdt = SparseDecoupledTensor.from_tensor(a_coo)

    a_sdt_to_coo = a_sdt.to_sparse_coo().coalesce()

    assert a_sdt_to_coo.shape == a_coo.shape
    assert a_sdt_to_coo.indices().dtype == torch.int64
    assert a_sdt_to_coo.dtype == a_coo.dtype

    torch.testing.assert_close(a_sdt_to_coo.indices(), a_coo.indices())
    torch.testing.assert_close(a_sdt_to_coo.values(), a_coo.values())


def test_coo_conversion_with_batch_dim(a_with_batch, device):
    a_coo = a_with_batch.to(device)
    a_sdt = SparseDecoupledTensor.from_tensor(a_coo)

    a_sdt_to_coo = a_sdt.to_sparse_coo().coalesce()

    assert a_sdt_to_coo.shape == a_coo.shape
    assert a_sdt_to_coo.indices().dtype == torch.int64
    assert a_sdt_to_coo.dtype == a_coo.dtype

    torch.testing.assert_close(a_sdt_to_coo.indices(), a_coo.indices())
    torch.testing.assert_close(a_sdt_to_coo.values(), a_coo.values())


def test_csr_conversion(a, device):
    a_coo = a.to(device)
    a_sdt = SparseDecoupledTensor.from_tensor(a_coo)

    a_coo_to_csr = a_coo.to_sparse_csr()
    a_sdt_to_csr = a_sdt.to_sparse_csr()

    assert a_sdt_to_csr.shape == a_coo_to_csr.shape
    assert a_sdt_to_csr.crow_indices().dtype == torch.int32
    assert a_sdt_to_csr.col_indices().dtype == torch.int32
    assert a_sdt_to_csr.dtype == a_coo_to_csr.dtype

    torch.testing.assert_close(
        a_sdt_to_csr.crow_indices().to(torch.int64), a_coo_to_csr.crow_indices()
    )
    torch.testing.assert_close(
        a_sdt_to_csr.col_indices().to(torch.int64), a_coo_to_csr.col_indices()
    )
    torch.testing.assert_close(a_sdt_to_csr.values(), a_coo_to_csr.values())


def test_csr_conversion_with_batch_dim(a_with_batch, device):
    a_coo = a_with_batch.to(device)
    a_sdt = SparseDecoupledTensor.from_tensor(a_coo)

    a_sdt_to_csr = a_sdt.to_sparse_csr()

    assert a_sdt_to_csr.shape == a_sdt.shape
    assert a_sdt_to_csr.crow_indices().dtype == torch.int32
    assert a_sdt_to_csr.col_indices().dtype == torch.int32
    assert a_sdt_to_csr.dtype == a_sdt.dtype

    # Since it is not possible to directly convert a batched sparse coo tensor
    # to a batched sparse csr tensor, we directly check for value agreement in
    # dense format.
    torch.testing.assert_close(a_coo.to_dense(), a_sdt_to_csr.to_dense())


def test_csr_transposed_conversion(a, device):
    a_sdt = SparseDecoupledTensor.from_tensor(a).to(device)
    a_sdt_T_direct = a_sdt.to_sparse_csr_transposed()
    a_sdt_T_indirect = a_sdt.T.to_sparse_csr()

    assert a_sdt_T_direct.shape == a_sdt_T_indirect.shape

    torch.testing.assert_close(a_sdt_T_direct.values(), a_sdt_T_indirect.values())
    torch.testing.assert_close(
        a_sdt_T_direct.crow_indices(), a_sdt_T_indirect.crow_indices()
    )
    torch.testing.assert_close(
        a_sdt_T_direct.col_indices(), a_sdt_T_indirect.col_indices()
    )


def test_csr_transposed_conversion_with_batch_dim(a_with_batch, device):
    a_sdt = SparseDecoupledTensor.from_tensor(a_with_batch).to(device)
    a_sdt_T_direct = a_sdt.to_sparse_csr_transposed()
    a_sdt_T_indirect = a_sdt.T.to_sparse_csr()

    assert a_sdt_T_direct.shape == a_sdt_T_indirect.shape

    torch.testing.assert_close(a_sdt_T_direct.values(), a_sdt_T_indirect.values())
    torch.testing.assert_close(
        a_sdt_T_direct.crow_indices(), a_sdt_T_indirect.crow_indices()
    )
    torch.testing.assert_close(
        a_sdt_T_direct.col_indices(), a_sdt_T_indirect.col_indices()
    )


def test_csc_conversion(a, device):
    a_coo = a.to(device)
    a_sdt = SparseDecoupledTensor.from_tensor(a_coo)

    a_coo_to_csc = a_coo.to_sparse_csc()
    a_sdt_to_csc = a_sdt.to_sparse_csc()

    assert a_sdt_to_csc.shape == a_coo_to_csc.shape
    assert a_sdt_to_csc.ccol_indices().dtype == torch.int32
    assert a_sdt_to_csc.row_indices().dtype == torch.int32
    assert a_sdt_to_csc.dtype == a_coo_to_csc.dtype

    torch.testing.assert_close(
        a_sdt_to_csc.ccol_indices().to(torch.int64), a_coo_to_csc.ccol_indices()
    )
    torch.testing.assert_close(
        a_sdt_to_csc.row_indices().to(torch.int64), a_coo_to_csc.row_indices()
    )
    torch.testing.assert_close(a_sdt_to_csc.values(), a_coo_to_csc.values())


def test_csc_conversion_with_batch_dim(a_with_batch, device):
    a_coo = a_with_batch.to(device)
    a_sdt = SparseDecoupledTensor.from_tensor(a_coo)

    a_sdt_to_csc = a_sdt.to_sparse_csc()

    assert a_sdt_to_csc.shape == a_sdt.shape
    assert a_sdt_to_csc.ccol_indices().dtype == torch.int32
    assert a_sdt_to_csc.row_indices().dtype == torch.int32
    assert a_sdt_to_csc.dtype == a_sdt.dtype

    # Since it is not possible to directly convert a batched sparse coo tensor
    # to a batched sparse csc tensor, we directly check for value agreement in
    # dense format.
    torch.testing.assert_close(a_coo.to_dense(), a_sdt_to_csc.to_dense())


def test_matmul_with_batch_dim(a, a_with_batch, device):
    a_sdt = SparseDecoupledTensor.from_tensor(a).to(device)
    a_sdt_batched = SparseDecoupledTensor.from_tensor(a_with_batch).to(device)

    b_dense = torch.randn(
        a_sdt_batched.shape[-1], dtype=a_sdt_batched.dtype, device=device
    )

    with pytest.raises(NotImplementedError):
        a_sdt_batched @ b_dense

    with pytest.raises(NotImplementedError):
        a_sdt_batched @ a_sdt

    with pytest.raises(NotImplementedError):
        b_dense @ a_sdt_batched

    with pytest.raises(NotImplementedError):
        a_sdt @ a_sdt_batched


def test_matmul_with_dense_dim(a, device):
    a_sdt = SparseDecoupledTensor.from_tensor(a).to(device)

    val = torch.randn(4, 2)
    idx_coo = torch.tensor([[0, 1, 2, 2], [1, 0, 1, 2]])
    shape = (4, 4)

    hybrid_sdt = SparseDecoupledTensor(SparsityPattern(idx_coo, shape), val).to(device)

    b_dense = torch.randn(shape[-1], dtype=hybrid_sdt.dtype, device=device)

    with pytest.raises(NotImplementedError):
        hybrid_sdt @ b_dense

    with pytest.raises(NotImplementedError):
        hybrid_sdt @ a_sdt

    with pytest.raises(NotImplementedError):
        b_dense @ hybrid_sdt

    with pytest.raises(NotImplementedError):
        a_sdt @ hybrid_sdt


def test_matmul_with_wrong_tensor_ndim(a, device):
    a_coo = a.to(device)
    a_sdt = SparseDecoupledTensor.from_tensor(a_coo)

    b_dense = torch.randn((a_coo.shape[-1],) * 3, dtype=a_coo.dtype, device=device)

    with pytest.raises(NotImplementedError):
        a_sdt @ b_dense

    with pytest.raises(NotImplementedError):
        b_dense @ a_sdt


def test_dim(a, device):
    a_sdt = SparseDecoupledTensor.from_tensor(a).to(device)

    assert len(a_sdt.shape) == 2

    assert a_sdt.n_dense_dim == 0
    assert a_sdt.n_sp_dim == 2
    assert a_sdt.n_batch_dim == 0
    assert a_sdt.n_dim == 2


def test_dim_with_batch(a_with_batch, device):
    a_sdt = SparseDecoupledTensor.from_tensor(a_with_batch).to(device)

    assert len(a_sdt.shape) == 3

    assert a_sdt.n_dense_dim == 0
    assert a_sdt.n_sp_dim == 2
    assert a_sdt.n_batch_dim == 1
    assert a_sdt.n_dim == 3


def test_dim_with_batch_dense(device):
    val = torch.randn(4, 2)
    idx_coo = torch.tensor([[0, 0, 1, 1], [0, 1, 2, 2], [1, 0, 1, 2]])
    shape = (2, 4, 4)

    a_sdt = SparseDecoupledTensor(SparsityPattern(idx_coo, shape), val).to(device)

    assert len(a_sdt.shape) == 4

    assert a_sdt.n_dense_dim == 1
    assert a_sdt.n_sp_dim == 2
    assert a_sdt.n_batch_dim == 1
    assert a_sdt.n_dim == 4


def test_transpose(a, device):
    a_coo = a.to(device)
    a_sdt = SparseDecoupledTensor.from_tensor(a_coo)

    a_coo_T = a_coo.T.to_dense()
    a_sdt_T = a_sdt.T.to_dense()

    torch.testing.assert_close(a_sdt_T, a_coo_T)


def test_transpose_with_batch_dim(a_with_batch, device):
    a_coo = a_with_batch.to(device)
    a_sdt = SparseDecoupledTensor.from_tensor(a_coo)

    a_coo_T = a_coo.to_dense().transpose(-1, -2)
    a_sdt_T = a_sdt.T.to_dense()

    torch.testing.assert_close(a_sdt_T, a_coo_T)


def test_transpose_with_batch_dense_dim(device):
    val = torch.randn(4, 2)
    idx_coo = torch.tensor([[0, 0, 1, 1], [0, 1, 2, 2], [1, 0, 1, 2]])
    shape = (2, 4, 4)

    sdt = SparseDecoupledTensor(SparsityPattern(idx_coo, shape), val).to(device)

    sp_op_T = sdt.T.to_dense()
    sp_tensor_T = sdt.to_dense().transpose(1, 2)

    torch.testing.assert_close(sp_op_T, sp_tensor_T)


def test_requires_grad_is_false(a, device):
    a_coo = a.to(device)
    a_sdt = SparseDecoupledTensor.from_tensor(a_coo)

    assert a_coo.requires_grad == a_sdt.requires_grad


def test_requires_grad_is_true(a, device):
    a_coo = a.to(device)
    a_coo.requires_grad_()

    a_sdt = SparseDecoupledTensor.from_tensor(a_coo)

    assert a_coo.requires_grad == a_sdt.requires_grad


def test_requires_grad_(a, device):
    a_coo = a.to(device)
    a_sdt = SparseDecoupledTensor.from_tensor(a_coo)

    a_sdt.requires_grad_()
    assert a_sdt.values.requires_grad


def test_nnz(a, device):
    a_coo = a.to(device)
    a_sdt = SparseDecoupledTensor.from_tensor(a_coo)

    assert a_sdt._nnz() == a_coo._nnz()


def test_nnz_with_batch_dim(a_with_batch, device):
    a_coo = a_with_batch.to(device)
    a_sdt = SparseDecoupledTensor.from_tensor(a_coo)

    assert a_sdt._nnz() == a_coo._nnz()


def test_size(device):
    val = torch.randn(4, 2)
    idx_coo = torch.tensor([[0, 0, 1, 1], [0, 1, 2, 2], [1, 0, 1, 2]])
    shape = (2, 4, 4)

    sdt = SparseDecoupledTensor(SparsityPattern(idx_coo, shape), val).to(device)

    assert sdt.size() == sdt.shape

    for idx, val in enumerate(shape + (val.shape[-1],)):
        assert sdt.size(idx) == val


def test_to_float64(a, device):
    a_coo = a.to(device)
    a_sdt = SparseDecoupledTensor.from_tensor(a_coo).to(torch.float64)

    assert a_sdt.values.dtype == torch.float64


def test_to_device(a, device):
    a_coo = a
    a_sdt = SparseDecoupledTensor.from_tensor(a_coo).to(device)

    assert a_sdt.values.device.type == device.type
    assert a_sdt.pattern.device.type == device.type


def test_apply(a, device):
    a_coo = a.to(device)
    a_sdt = SparseDecoupledTensor.from_tensor(a).clone().to(device)

    a_coo_applied = torch.relu(a_coo.to_dense())
    a_sdt_applied = a_sdt.apply(torch.relu).to_dense()

    torch.testing.assert_close(a_sdt_applied, a_coo_applied)


def test_neg(a, device):
    a_coo = a.to(device)
    neg_a_coo = -a_coo

    a_sdt = SparseDecoupledTensor.from_tensor(a).to(device)
    neg_a_sdt = -a_sdt

    torch.testing.assert_close(neg_a_sdt.to_dense(), neg_a_coo.to_dense())


def test_add(a, device):
    a_coo = a.to(device)
    a_sdt = SparseDecoupledTensor.from_tensor(a).to(device)

    a_coo_sum = a_coo + a_coo
    a_sdt_sum = a_sdt + a_sdt

    torch.testing.assert_close(a_sdt_sum.to_dense(), a_coo_sum.to_dense())


def test_sub(a, device):
    a_coo = a.to(device)
    a_sdt = SparseDecoupledTensor.from_tensor(a).to(device)

    a_coo_sub = a_coo - a_coo
    a_sdt_sub = a_sdt - a_sdt

    torch.testing.assert_close(a_sdt_sub.to_dense(), a_coo_sub.to_dense())


def test_assemble(a, device):
    a_coo = a.to(device)
    a_sdt = SparseDecoupledTensor.from_tensor(a).to(device)

    a_coo_sum = a_coo + a_coo.T
    a_sdt_sum = SparseDecoupledTensor.assemble(a_sdt, a_sdt.T)

    torch.testing.assert_close(a_sdt_sum.to_dense(), a_coo_sum.to_dense())


def test_assemble_with_diag_operator(a, device):
    a_coo = a.to(device)
    a_sdt = SparseDecoupledTensor.from_tensor(a).to(device)

    diag = torch.randn(a_coo.size(0))
    diag_dense = torch.diagflat(diag).to(device)
    ddt = DiagDecoupledTensor.from_tensor(diag).to(device)

    dense_sum = diag_dense + a_coo
    sparse_sum = SparseDecoupledTensor.assemble(a_sdt, ddt)

    torch.testing.assert_close(sparse_sum.to_dense(), dense_sum)


def test_mul(a, device):
    a_coo = a.to(device)
    a_sdt = SparseDecoupledTensor.from_tensor(a).to(device)

    for scalar in [2, 3.0, torch.tensor(-9.0).to(device)]:
        a_coo_scaled = scalar * a_coo
        a_sdt_scaled = scalar * a_sdt
        a_sdt_rscaled = a_sdt * scalar

        torch.testing.assert_close(a_sdt_scaled.to_dense(), a_coo_scaled.to_dense())
        torch.testing.assert_close(a_sdt_rscaled.to_dense(), a_coo_scaled.to_dense())


def test_trudiv(a, device):
    a_coo = a.to(device)
    a_sdt = SparseDecoupledTensor.from_tensor(a).to(device)

    for scalar in [2, 3.0, torch.tensor(-9.0).to(device)]:
        a_coo_scaled = a_coo / scalar
        a_sdt_scaled = a_sdt / scalar

        torch.testing.assert_close(a_sdt_scaled.to_dense(), a_coo_scaled.to_dense())


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
        [0, 0, 0, 1, 1, 2, 2, 2, 2], dtype=torch.int64, device=device
    )
    col_ptrs = torch.tensor(
        [0, 0, 1, 1, 1, 1, 2, 2, 2, 2], dtype=torch.int64, device=device
    )

    sp_ops = block_diag.unpack_by_ptrs(n_blocks=3, row_ptrs=row_ptrs, col_ptrs=col_ptrs)

    for sdt, ori_op in zip(sp_ops, [a, b, c]):
        torch.testing.assert_close(sdt.to_dense(), ori_op.to_dense())


def test_pack_block_diag_with_batch_dim(a_with_batch, device):
    block_diag_sdt = SparseDecoupledTensor.pack_block_diag(
        (a_with_batch, -a_with_batch)
    ).to(device)

    block_diag_dense = a_with_batch.to_dense().to(device)

    block_diag_true = torch.stack(
        (
            torch.block_diag(block_diag_dense[0], -block_diag_dense[0]),
            torch.block_diag(block_diag_dense[1], -block_diag_dense[1]),
        )
    )

    torch.testing.assert_close(block_diag_sdt.to_dense(), block_diag_true)

    sdt_list = block_diag_sdt.unpack_block_diag()

    torch.testing.assert_close(sdt_list[0].to_dense(), block_diag_dense)
    torch.testing.assert_close(sdt_list[1].to_dense(), -block_diag_dense)


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


def test_bmat_with_invalid_row_col(a, device):
    a_coo = a.to(device)

    # Test degenerate column
    bmat_1 = SparseDecoupledTensor.bmat([[a_coo, None], [a_coo, None]])
    bmat_2 = SparseDecoupledTensor.bmat([[a_coo], [a_coo]])

    torch.testing.assert_close(bmat_1.to_dense(), bmat_2.to_dense())

    # Test degenerate row
    bmat_1 = SparseDecoupledTensor.bmat([[None, None], [a_coo, a_coo]])
    bmat_2 = SparseDecoupledTensor.bmat([[a_coo, a_coo]])

    torch.testing.assert_close(bmat_1.to_dense(), bmat_2.to_dense())

    # Test full degenerate bmat
    with pytest.raises(ValueError):
        SparseDecoupledTensor.bmat([[None, None], [None, None]])

    # Test invalid dtype
    with pytest.raises(TypeError):
        SparseDecoupledTensor.bmat([a_coo, 3], [None, a_coo])


def test_submatrix(a, device):
    a_dense = a.to(device).to_dense()
    a_sdt = SparseDecoupledTensor.from_tensor(a).to(device)

    r_mask = torch.tensor([True, False, True, True], device=device)
    c_mask = torch.tensor([False, True, True, False], device=device)

    sub_sdt_1 = a_sdt.submatrix(r_mask).to_dense()
    sub_sdt_2 = a_sdt.submatrix(r_mask, r_mask).to_dense()
    sub_sdt_3 = a_sdt.submatrix(r_mask, c_mask).to_dense()

    sub_dense_1 = a_dense[r_mask][:, r_mask]
    sub_dense_2 = a_dense[r_mask][:, c_mask]

    torch.testing.assert_close(sub_sdt_1, sub_dense_1)
    torch.testing.assert_close(sub_sdt_2, sub_dense_1)
    torch.testing.assert_close(sub_sdt_3, sub_dense_2)


def test_submatrix_with_batch_dim(a_with_batch, device):
    a_dense = a_with_batch.to(device).to_dense()
    a_sdt = SparseDecoupledTensor.from_tensor(a_with_batch).to(device)

    r_mask = torch.tensor([True, False, True, True], device=device)
    c_mask = torch.tensor([False, True, True, False], device=device)

    sub_sdt_1 = a_sdt.submatrix(r_mask).to_dense()
    sub_sdt_2 = a_sdt.submatrix(r_mask, r_mask).to_dense()
    sub_sdt_3 = a_sdt.submatrix(r_mask, c_mask).to_dense()

    sub_dense_1 = a_dense[:, r_mask][:, :, r_mask]
    sub_dense_2 = a_dense[:, r_mask][:, :, c_mask]

    torch.testing.assert_close(sub_sdt_1, sub_dense_1)
    torch.testing.assert_close(sub_sdt_2, sub_dense_1)
    torch.testing.assert_close(sub_sdt_3, sub_dense_2)


def test_submatrix_with_block_diag_config(a, device):
    a_sdt = SparseDecoupledTensor.from_tensor(a).to(device)
    block_diag_sdt = SparseDecoupledTensor.pack_block_diag((a_sdt, a_sdt))

    mask_1 = torch.tensor([True, False, True, True], device=device)
    mask_2 = torch.tensor([False, True, True, False], device=device)
    mask_3 = torch.tensor([False, False, False, False], device=device)

    r_mask = torch.cat((mask_1, mask_2))
    c_mask = torch.cat((mask_2, mask_3))

    sub_block_1, sub_block_2 = block_diag_sdt.submatrix(r_mask).unpack_block_diag()
    sub_block_1_true = a_sdt.submatrix(mask_1)
    sub_block_2_true = a_sdt.submatrix(mask_2)

    torch.testing.assert_close(sub_block_1.to_dense(), sub_block_1_true.to_dense())
    torch.testing.assert_close(sub_block_2.to_dense(), sub_block_2_true.to_dense())

    # Test an edge case where one block is completely degenerate after masking.
    sub_block_1, sub_block_2 = block_diag_sdt.submatrix(
        r_mask, c_mask
    ).unpack_block_diag()
    sub_block_1_true = a_sdt.submatrix(mask_1, mask_2)
    sub_block_2_true = a_sdt.submatrix(mask_2, mask_3)

    torch.testing.assert_close(sub_block_1.to_dense(), sub_block_1_true.to_dense())
    torch.testing.assert_close(sub_block_2.to_dense(), sub_block_2_true.to_dense())
    assert sub_block_2.shape == sub_block_2_true.shape


def test_constrain(device):
    a_dense = torch.randn(5, 5, device=device)
    a_sym = a_dense + a_dense.T
    a_diag_dom = a_sym + 2.0 * torch.eye(5, device=device)

    a_sdt = SparseDecoupledTensor.from_tensor(a_diag_dom)

    mask = torch.tensor([True, True, False, True, False], device=device)

    a_sdt_constrained = a_sdt.constrain(mask).to_dense()

    a_sdt_to_dense = a_sdt.to_dense()
    a_sdt_to_dense[~mask] = 0.0
    a_sdt_to_dense[:, ~mask] = 0.0
    a_sdt_to_dense[~mask, ~mask] = 1.0

    torch.testing.assert_close(a_sdt_constrained, a_sdt_to_dense)


def test_constrain_exceptions(device):
    with pytest.raises(ValueError) as excinfo:
        a_dense = torch.randn(5, 5, device=device)
        a_sdt = SparseDecoupledTensor.from_tensor(a_dense)
        mask = torch.tensor([True, True, False, True, False], device=device)
        a_sdt.constrain(mask)

    assert "symmetric" in str(excinfo.value)

    with pytest.raises(ValueError) as excinfo:
        a_dense = torch.randn(6, 5, device=device)
        a_sdt = SparseDecoupledTensor.from_tensor(a_dense)
        mask = torch.tensor([True, True, False, True, False], device=device)
        a_sdt.constrain(mask)

    assert "square" in str(excinfo.value)

    with pytest.raises(ValueError) as excinfo:
        a_dense = torch.sparse_coo_tensor(
            indices=torch.tensor([[0, 1, 2, 3], [0, 1, 2, 2]], device=device),
            values=torch.randn(4, device=device),
            size=torch.Size([4, 4]),
        )

        a_sdt = SparseDecoupledTensor.from_tensor(a_dense)

        mask = torch.tensor([True, True, False, False], device=device)

        a_sdt.constrain(mask)

    assert "nonzero" in str(excinfo.value)

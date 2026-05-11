import gc

import pytest
import torch

from cochain.sparse.decoupled_tensor import (
    DiagDecoupledTensor,
    SparseDecoupledTensor,
    SparsityPattern,
)


@pytest.fixture(
    params=[
        "a",
        "a_with_batch",
        "a_with_dense",
        "a_with_batch_dense",
    ]
)
def any_a(request):
    return request.getfixturevalue(request.param)


@pytest.fixture(params=["a", "a_with_batch"])
def a_or_batched_a(request):
    return request.getfixturevalue(request.param)


@pytest.fixture(params=["a", "a_with_dense"])
def unbatched_a(request):
    return request.getfixturevalue(request.param)


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


def test_nan_inf_val_exceptions(device):
    shape = (2, 2)
    idx_coo = torch.tensor([[0, 1], [0, 1]], device=device)
    pattern = SparsityPattern(idx_coo, shape)

    # Test NaN.
    with pytest.raises(ValueError, match="contain NaN or Inf"):
        val = torch.tensor([1.0, float("nan")], device=device)
        SparseDecoupledTensor(pattern, val)

    # Test Inf.
    with pytest.raises(ValueError, match="contain NaN or Inf"):
        val = torch.tensor([float("inf"), 1.0], device=device)
        SparseDecoupledTensor(pattern, val)


def test_dense_conversion(any_a, device):
    a_coo = any_a.to(device)
    a_coo_to_dense = a_coo.to_dense()

    a_sdt = SparseDecoupledTensor.from_tensor(a_coo)
    a_sdt_to_dense = a_sdt.to_dense()

    assert a_coo_to_dense.dtype == a_sdt.dtype

    torch.testing.assert_close(a_coo_to_dense, a_sdt_to_dense)


def test_coo_conversion(any_a, device):
    a_coo = any_a.to(device)
    a_sdt = SparseDecoupledTensor.from_tensor(a_coo)
    a_sdt_to_coo = a_sdt.to_sparse_coo().coalesce()

    assert a_sdt_to_coo.shape == a_coo.shape
    assert a_sdt_to_coo.indices().dtype == torch.int64
    assert a_sdt_to_coo.dtype == a_coo.dtype

    torch.testing.assert_close(a_sdt_to_coo.indices(), a_coo.indices())
    torch.testing.assert_close(a_sdt_to_coo.values(), a_coo.values())


def test_csr_conversion(any_a, device):
    a_coo = any_a.to(device)
    a_sdt = SparseDecoupledTensor.from_tensor(a_coo)
    a_sdt_to_csr = a_sdt.to_sparse_csr()

    assert a_sdt_to_csr.shape == a_sdt.shape
    assert a_sdt_to_csr.crow_indices().dtype == torch.int32
    assert a_sdt_to_csr.col_indices().dtype == torch.int32
    assert a_sdt_to_csr.dtype == a_sdt.dtype

    torch.testing.assert_close(a_coo.to_dense(), a_sdt_to_csr.to_dense())


def test_csr_transposed_conversion(any_a, device):
    a_sdt = SparseDecoupledTensor.from_tensor(any_a).to(device)
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


def test_csc_conversion(any_a, device):
    a_coo = any_a.to(device)
    a_sdt = SparseDecoupledTensor.from_tensor(a_coo)
    a_sdt_to_csc = a_sdt.to_sparse_csc()

    assert a_sdt_to_csc.shape == a_sdt.shape
    assert a_sdt_to_csc.ccol_indices().dtype == torch.int32
    assert a_sdt_to_csc.row_indices().dtype == torch.int32
    assert a_sdt_to_csc.dtype == a_sdt.dtype

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


def test_spsp_matmul_caching_fwd_plan(a, b, device):
    a_sdt = SparseDecoupledTensor.from_tensor(a).to(device)
    b_sdt = SparseDecoupledTensor.from_tensor(b).to(device)

    assert len(a_sdt.pattern._spsp_matmul_plans) == 0

    c_sdt = a_sdt @ b_sdt

    assert len(a_sdt.pattern._spsp_matmul_plans) == 1
    assert b_sdt.pattern in a_sdt.pattern._spsp_matmul_plans

    plan = a_sdt.pattern._spsp_matmul_plans[b_sdt.pattern]
    assert plan.fwd_plan is not None
    assert plan.bwd_plan_A is None
    assert plan.bwd_plan_B is None


def test_spsp_matmul_caching_reuse_plan(a, b, device):
    a_sdt = SparseDecoupledTensor.from_tensor(a).to(device)
    b_sdt = SparseDecoupledTensor.from_tensor(b).to(device)

    c_sdt_1 = a_sdt @ b_sdt
    plan_1 = a_sdt.pattern._spsp_matmul_plans[b_sdt.pattern]

    c_sdt_2 = a_sdt @ b_sdt
    assert len(a_sdt.pattern._spsp_matmul_plans) == 1
    assert a_sdt.pattern._spsp_matmul_plans[b_sdt.pattern] is plan_1


def test_spsp_matmul_caching_bwd_plan(a, b, device):
    a_sdt = SparseDecoupledTensor.from_tensor(a).to(device)
    b_sdt = SparseDecoupledTensor.from_tensor(b).to(device)

    c_sdt_1 = a_sdt @ b_sdt
    plan = a_sdt.pattern._spsp_matmul_plans[b_sdt.pattern]

    a_sdt.requires_grad_(True)
    c_sdt_2 = a_sdt @ b_sdt

    assert plan.bwd_plan_A is not None
    assert plan.bwd_plan_B is None

    b_sdt.requires_grad_(True)
    c_sdt_3 = a_sdt @ b_sdt

    assert plan.bwd_plan_B is not None


def test_spsp_matmul_caching_different_values(a, b, device):
    a_sdt = SparseDecoupledTensor.from_tensor(a).to(device)
    b_sdt = SparseDecoupledTensor.from_tensor(b).to(device)

    c_dst_1 = a_sdt @ b_sdt
    plan = a_sdt.pattern._spsp_matmul_plans[b_sdt.pattern]

    a_sdt_new_vals = 2.0 * a_sdt
    b_sdt_new_vals = 2.0 * b_sdt

    c_sdt_2 = a_sdt_new_vals @ b_sdt_new_vals

    assert len(a_sdt_new_vals.pattern._spsp_matmul_plans) == 1
    assert a_sdt_new_vals.pattern._spsp_matmul_plans[b_sdt_new_vals.pattern] is plan


def test_spsp_matmul_caching_eviction(a, b, device):
    a_sdt = SparseDecoupledTensor.from_tensor(a).to(device)
    b_sdt = SparseDecoupledTensor.from_tensor(b).to(device)

    c_sdt = a_sdt @ b_sdt

    del c_sdt
    del b_sdt

    gc.collect()

    assert len(a_sdt.pattern._spsp_matmul_plans) == 0


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


def test_dims_from_tensor(a_with_dense, a_with_batch_dense, device):
    sdt_dense = SparseDecoupledTensor.from_tensor(a_with_dense.to(device))
    assert sdt_dense.n_batch_dim == 0
    assert sdt_dense.n_dense_dim == 1
    assert sdt_dense.n_sp_dim == 2

    sdt_batch_dense = SparseDecoupledTensor.from_tensor(a_with_batch_dense.to(device))
    assert sdt_batch_dense.n_batch_dim == 1
    assert sdt_batch_dense.n_dense_dim == 1
    assert sdt_batch_dense.n_sp_dim == 2


def test_transpose(any_a, device):
    a_coo = any_a.to(device)
    a_sdt = SparseDecoupledTensor.from_tensor(a_coo)

    sp_dim_1 = a_sdt.n_batch_dim
    sp_dim_2 = a_sdt.n_batch_dim + 1
    a_coo_T = a_coo.to_dense().transpose(sp_dim_1, sp_dim_2)

    a_sdt_T = a_sdt.T.to_dense()

    torch.testing.assert_close(a_sdt_T, a_coo_T)


def test_requires_grad_is_false(any_a, device):
    a_coo = any_a.to(device)
    a_sdt = SparseDecoupledTensor.from_tensor(a_coo)

    assert a_coo.requires_grad == a_sdt.requires_grad


def test_requires_grad_is_true(any_a, device):
    a_coo = any_a.to(device)
    a_coo.requires_grad_()

    a_sdt = SparseDecoupledTensor.from_tensor(a_coo)

    assert a_coo.requires_grad == a_sdt.requires_grad


def test_requires_grad_(any_a, device):
    a_coo = any_a.to(device)
    a_sdt = SparseDecoupledTensor.from_tensor(a_coo)

    a_sdt.requires_grad_()
    assert a_sdt.values.requires_grad


def test_nnz(any_a, device):
    a_coo = any_a.to(device)
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


def test_to_float64(any_a, device):
    a_coo = any_a.to(device)
    a_sdt = SparseDecoupledTensor.from_tensor(a_coo).to(torch.float64)

    assert a_sdt.values.dtype == torch.float64
    # The index tensors should not be affected by float dtype conversion.
    assert a_sdt.pattern.dtype == torch.int64


def test_to_device(any_a, device):
    a_coo = any_a
    a_sdt = SparseDecoupledTensor.from_tensor(a_coo).to(device)

    assert a_sdt.values.device.type == device.type
    assert a_sdt.pattern.device.type == device.type


def test_clone(any_a, device):
    a_sdt = SparseDecoupledTensor.from_tensor(any_a.to(device))
    a_sdt_clone = a_sdt.clone()

    assert a_sdt is not a_sdt_clone
    assert a_sdt.values is not a_sdt_clone.values
    assert a_sdt.pattern is a_sdt_clone.pattern
    torch.testing.assert_close(a_sdt_clone.to_dense(), a_sdt.to_dense())


def test_detach(any_a, device):
    a_req_grad = any_a.clone().to(device).requires_grad_()
    a_sdt = SparseDecoupledTensor.from_tensor(a_req_grad)
    a_sdt_detached = a_sdt.detach()

    assert a_sdt.requires_grad
    assert not a_sdt_detached.requires_grad
    torch.testing.assert_close(a_sdt_detached.to_dense(), a_sdt.to_dense())


def test_to_sdt(any_a, device):
    a_sdt = SparseDecoupledTensor.from_tensor(any_a.to(device))
    a_sdt_returned = a_sdt.to_sdt()

    assert a_sdt_returned is a_sdt


def test_apply(any_a, device):
    a_coo = any_a.to(device)
    a_sdt = SparseDecoupledTensor.from_tensor(any_a).clone().to(device)

    a_coo_applied = torch.relu(a_coo.to_dense())
    a_sdt_applied = a_sdt.apply(torch.relu).to_dense()

    torch.testing.assert_close(a_sdt_applied, a_coo_applied)


def test_apply_shape_exception(any_a, device):
    a_sdt = SparseDecoupledTensor.from_tensor(any_a.to(device))

    # Passing a function that removes the last nonzero element
    with pytest.raises(RuntimeError, match="changed the nnz dim"):
        a_sdt.apply(lambda x: x[:-1])


def test_neg(any_a, device):
    a_coo = any_a.to(device)
    neg_a_coo = -a_coo

    a_sdt = SparseDecoupledTensor.from_tensor(a_coo)
    neg_a_sdt = -a_sdt

    torch.testing.assert_close(neg_a_sdt.to_dense(), neg_a_coo.to_dense())


def test_abs(any_a, device):
    a_coo = any_a.to(device)
    a_sdt = SparseDecoupledTensor.from_tensor(a_coo)

    sdt_abs = a_sdt.abs()
    coo_dense_abs = a_coo.to_dense().abs()

    torch.testing.assert_close(sdt_abs.to_dense(), coo_dense_abs)


def test_diagonal(any_a, device):
    a_coo = any_a.to(device)
    a_sdt = SparseDecoupledTensor.from_tensor(a_coo)

    diag_sdt = a_sdt.diagonal().to_dense()

    a_dense = a_coo.to_dense()
    n_diag = min(a_sdt.pattern.shape[-2:])

    idx = torch.arange(n_diag)
    if a_sdt.n_batch_dim > 0:
        diag_true = a_dense[:, idx, idx]
    else:
        diag_true = a_dense[idx, idx]

    torch.testing.assert_close(diag_sdt, diag_true)


def test_off_diagonal(any_a, device):
    a_coo = any_a.to(device)
    a_sdt = SparseDecoupledTensor.from_tensor(a_coo)

    off_diag_sdt = a_sdt.off_diagonal()

    a_dense = a_coo.to_dense()
    n_diag = min(a_sdt.pattern.shape[-2:])
    idx = torch.arange(n_diag)

    off_diag_true = a_dense.clone()
    if a_sdt.n_batch_dim > 0:
        off_diag_true[:, idx, idx] = 0.0
    else:
        off_diag_true[idx, idx] = 0.0

    torch.testing.assert_close(off_diag_sdt.to_dense(), off_diag_true)


def test_batched_off_diagonal_equal_nnz_exception(device):
    idx_coo = torch.tensor([[0, 0, 1, 1], [0, 0, 0, 1], [0, 1, 0, 1]], device=device)
    val = torch.randn(4, device=device)
    pattern = SparsityPattern(idx_coo, (2, 2, 2))
    a_sdt = SparseDecoupledTensor(pattern, val)

    with pytest.raises(ValueError):
        a_sdt.off_diagonal()


def test_tr(any_a, device):
    a_coo = any_a.to(device)
    a_sdt = SparseDecoupledTensor.from_tensor(a_coo)

    tr_sdt = a_sdt.tr

    a_dense = a_coo.to_dense()
    n_diag = min(a_sdt.pattern.shape[-2:])
    idx = torch.arange(n_diag)

    if a_sdt.n_batch_dim > 0:
        tr_true = a_dense[:, idx, idx].sum(dim=1)
    else:
        tr_true = a_dense[idx, idx].sum(dim=0)

    torch.testing.assert_close(tr_sdt, tr_true)


def test_triu(unbatched_a, device):
    a_coo = unbatched_a.to(device)
    a_sdt = SparseDecoupledTensor.from_tensor(a_coo)

    for diag in [0, 1, -1]:
        triu_sdt = a_sdt.triu(diagonal=diag)

        a_dense = a_coo.to_dense()
        r, c = a_dense.shape[:2]
        row_idx = torch.arange(r, device=device).view(-1, 1)
        col_idx = torch.arange(c, device=device).view(1, -1)
        mask = row_idx <= col_idx - diag

        triu_true = torch.zeros_like(a_dense)
        triu_true[mask] = a_dense[mask]

        torch.testing.assert_close(triu_sdt.to_dense(), triu_true)


def test_batched_triu_equal_nnz_exception(device):
    idx_coo_2 = torch.tensor([[0, 0, 1, 1], [0, 1, 0, 0], [1, 0, 1, 2]], device=device)
    val_2 = torch.randn(4, device=device)
    pattern_2 = SparsityPattern(idx_coo_2, (2, 3, 3))
    a_sdt_2 = SparseDecoupledTensor(pattern_2, val_2)

    with pytest.raises(ValueError):
        a_sdt_2.triu()


def test_add(any_a, device):
    a_coo = any_a.to(device)
    a_sdt = SparseDecoupledTensor.from_tensor(a_coo)

    a_coo_sum = a_coo + a_coo
    a_sdt_sum = a_sdt + a_sdt

    torch.testing.assert_close(a_sdt_sum.to_dense(), a_coo_sum.to_dense())


def test_sub(any_a, device):
    a_coo = any_a.to(device)
    a_sdt = SparseDecoupledTensor.from_tensor(a_coo)

    a_coo_sub = a_coo - a_coo
    a_sdt_sub = a_sdt - a_sdt

    torch.testing.assert_close(a_sdt_sub.to_dense(), a_coo_sub.to_dense())


def test_topology_mismatch_exceptions(a, b, device):
    a_sdt = SparseDecoupledTensor.from_tensor(a.to(device))
    b_sdt = SparseDecoupledTensor.from_tensor(b.to(device))

    with pytest.raises(ValueError, match="identical topologies"):
        a_sdt + b_sdt

    with pytest.raises(ValueError, match="identical topologies"):
        a_sdt - b_sdt


def test_assemble(unbatched_a, device):
    a_coo = unbatched_a.to(device)
    a_sdt = SparseDecoupledTensor.from_tensor(a_coo).to(device)

    sp_dim_1 = a_sdt.n_batch_dim
    sp_dim_2 = a_sdt.n_batch_dim + 1
    a_coo_sum = a_coo.to_dense() + a_coo.to_dense().transpose(sp_dim_1, sp_dim_2)
    a_sdt_sum = SparseDecoupledTensor.assemble(a_sdt, a_sdt.T)

    torch.testing.assert_close(a_sdt_sum.to_dense(), a_coo_sum)


def test_assemble_with_diag_operator(a_or_batched_a, device):
    a_coo = a_or_batched_a.to(device)
    a_sdt = SparseDecoupledTensor.from_tensor(a_or_batched_a).to(device)

    if a_sdt.n_batch_dim == 0:
        diag = torch.randn(a_coo.size(0), device=device)
        diag_dense = torch.diagflat(diag)
    else:
        diag = torch.randn(a_coo.size(0), a_coo.size(1), device=device)
        diag_dense = torch.diag_embed(diag)

    ddt = DiagDecoupledTensor.from_tensor(diag)

    dense_sum = diag_dense + a_coo
    sparse_sum = SparseDecoupledTensor.assemble(a_sdt, ddt)

    torch.testing.assert_close(sparse_sum.to_dense(), dense_sum)


def test_empty_assemble_exception():
    with pytest.raises(ValueError, match="No operators to assemble"):
        SparseDecoupledTensor.assemble()


def test_mul(any_a, device):
    a_coo = any_a.to(device)
    a_sdt = SparseDecoupledTensor.from_tensor(any_a).to(device)

    for scalar in [2, 3.0, torch.tensor(-9.0).to(device)]:
        a_coo_scaled = scalar * a_coo
        a_sdt_scaled = scalar * a_sdt
        a_sdt_rscaled = a_sdt * scalar

        torch.testing.assert_close(a_sdt_scaled.to_dense(), a_coo_scaled.to_dense())
        torch.testing.assert_close(a_sdt_rscaled.to_dense(), a_coo_scaled.to_dense())


def test_trudiv(any_a, device):
    a_coo = any_a.to(device)
    a_sdt = SparseDecoupledTensor.from_tensor(any_a).to(device)

    for scalar in [2, 3.0, torch.tensor(-9.0).to(device)]:
        a_coo_scaled = a_coo / scalar
        a_sdt_scaled = a_sdt / scalar

        torch.testing.assert_close(a_sdt_scaled.to_dense(), a_coo_scaled.to_dense())


def test_scalar_arithmetic_exceptions(a, device):
    a_sdt = SparseDecoupledTensor.from_tensor(a.to(device))
    bad_other = torch.randn(2, 2, device=device)

    with pytest.raises(TypeError):
        a_sdt * bad_other

    with pytest.raises(TypeError):
        a_sdt / bad_other


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


def test_pack_block_diag_invalid_input_exception(a, device):
    a_sdt = SparseDecoupledTensor.from_tensor(a.to(device))

    with pytest.raises(TypeError):
        SparseDecoupledTensor.pack_block_diag((a_sdt, "invalid_block"))


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


def test_submatrix(any_a, device):
    a_dense = any_a.to(device).to_dense()
    a_sdt = SparseDecoupledTensor.from_tensor(any_a).to(device)

    r_mask = torch.tensor([True, False, True, True], device=device)
    c_mask = torch.tensor([False, True, True, False], device=device)

    sub_sdt_1 = a_sdt.submatrix(r_mask).to_dense()
    sub_sdt_2 = a_sdt.submatrix(r_mask, r_mask).to_dense()
    sub_sdt_3 = a_sdt.submatrix(r_mask, c_mask).to_dense()

    if a_sdt.n_batch_dim == 0:
        sub_dense_1 = a_dense[r_mask][:, r_mask]
        sub_dense_2 = a_dense[r_mask][:, c_mask]
    else:
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


def test_batched_submatrix_equal_nnz_exception(device):
    idx_coo = torch.tensor([[0, 0, 1, 1], [0, 1, 0, 1], [0, 1, 1, 0]], device=device)
    val = torch.randn(4, device=device)
    pattern = SparsityPattern(idx_coo, (2, 2, 2))
    a_sdt = SparseDecoupledTensor(pattern, val)

    r_mask = torch.tensor([True, False], device=device)
    with pytest.raises(ValueError):
        a_sdt.submatrix(r_mask)


def test_constrain(unbatched_a, device):
    a_sdt = SparseDecoupledTensor.from_tensor(unbatched_a).to(device)
    a_sym_sdt = SparseDecoupledTensor.assemble(a_sdt, a_sdt.T)

    mask_len = a_sym_sdt.pattern.size(-1)
    mask = torch.randint(0, 2, (mask_len,), dtype=torch.bool, device=device)
    # Ensure at least one True and one False to test properly
    mask[0] = True
    mask[-1] = False

    a_sdt_constrained = a_sym_sdt.constrain(mask).to_dense()

    a_sdt_to_dense = a_sym_sdt.to_dense()

    a_sdt_to_dense[~mask] = 0.0
    a_sdt_to_dense[:, ~mask] = 0.0
    a_sdt_to_dense[(~mask, ~mask)] = 1.0

    torch.testing.assert_close(a_sdt_constrained, a_sdt_to_dense)


def test_constrain_exceptions(device):
    # Asymmetric matrix.
    with pytest.raises(ValueError) as excinfo:
        a_dense = torch.randn(5, 5, device=device)
        a_sdt = SparseDecoupledTensor.from_tensor(a_dense)
        mask = torch.tensor([True, True, False, True, False], device=device)
        a_sdt.constrain(mask)

    assert "symmetric" in str(excinfo.value)

    # Nonsquare matrix.
    with pytest.raises(ValueError) as excinfo:
        a_dense = torch.randn(6, 5, device=device)
        a_sdt = SparseDecoupledTensor.from_tensor(a_dense)
        mask = torch.tensor([True, True, False, True, False], device=device)
        a_sdt.constrain(mask)

    assert "square" in str(excinfo.value)

    # Zero on masked diagonals.
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

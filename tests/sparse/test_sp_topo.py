from dataclasses import FrozenInstanceError

import pytest
import torch as t

from cochain.sparse._sp_topo import SparseTopology


@pytest.fixture
def sp_with_empty_row():
    # | 0 0 0 0 |
    # | 0 1 0 1 |
    # | 1 0 1 1 |

    idx_coo = t.tensor([[1, 1, 2, 2, 2], [1, 3, 0, 2, 3]])
    shape = (3, 4)

    idx_crow = t.tensor([0, 0, 2, 5])
    idx_col = t.tensor([1, 3, 0, 2, 3])

    idx_ccol = t.tensor([0, 1, 2, 3, 5])
    idx_row = t.tensor([2, 1, 2, 1, 2])

    coo_to_csc_perm = t.tensor([2, 0, 3, 1, 4])

    return idx_coo, shape, idx_crow, idx_col, idx_ccol, idx_row, coo_to_csc_perm


@pytest.fixture
def sp_with_empty_col():
    # | 0 0 1 |
    # | 0 1 0 |
    # | 0 0 1 |
    # | 0 1 1 |

    idx_coo = t.tensor([[0, 1, 2, 3, 3], [2, 1, 2, 1, 2]])
    shape = (4, 3)

    idx_crow = t.tensor([0, 1, 2, 3, 5])
    idx_col = t.tensor([2, 1, 2, 1, 2])

    idx_ccol = t.tensor([0, 0, 2, 5])
    idx_row = t.tensor([1, 3, 0, 2, 3])

    coo_to_csc_perm = t.tensor([1, 3, 0, 2, 4])

    return idx_coo, shape, idx_crow, idx_col, idx_ccol, idx_row, coo_to_csc_perm


@pytest.fixture
def sp_with_batch_dim():
    # | 1 0 0 |
    # | 0 0 0 |
    # | 0 1 1 |
    #
    # | 0 1 0 |
    # | 0 1 0 |
    # | 0 0 1 |

    idx_coo = t.tensor([[0, 0, 0, 1, 1, 1], [0, 2, 2, 0, 1, 2], [0, 1, 2, 1, 1, 2]])
    shape = (2, 3, 3)

    idx_crow = t.tensor([[0, 1, 1, 3], [0, 1, 2, 3]])
    idx_col = t.tensor([[0, 1, 2], [1, 1, 2]])

    idx_ccol = t.tensor([[0, 1, 2, 3], [0, 0, 2, 3]])
    idx_row = t.tensor([[0, 2, 2], [0, 1, 2]])

    coo_to_csc_perm = t.tensor([0, 1, 2, 3, 4, 5])

    return idx_coo, shape, idx_crow, idx_col, idx_ccol, idx_row, coo_to_csc_perm


@pytest.fixture
def sp_with_batch_dim_T():
    # | 1 0 0 |
    # | 0 0 1 |
    # | 0 0 1 |
    #
    # | 0 0 0 |
    # | 1 1 0 |
    # | 0 0 1 |

    idx_coo = t.tensor([[0, 0, 0, 1, 1, 1], [0, 1, 2, 1, 1, 2], [0, 2, 2, 0, 1, 2]])
    shape = (2, 3, 3)

    idx_crow = t.tensor([[0, 1, 2, 3], [0, 0, 2, 3]])
    idx_col = t.tensor([[0, 2, 2], [0, 1, 2]])

    idx_ccol = t.tensor([[0, 1, 1, 3], [0, 1, 2, 3]])
    idx_row = t.tensor([[0, 1, 2], [1, 1, 2]])

    coo_to_csc_perm = t.tensor([0, 1, 2, 3, 4, 5])

    return idx_coo, shape, idx_crow, idx_col, idx_ccol, idx_row, coo_to_csc_perm


def test_immutability(device):
    idx_coo = t.tensor([[0, 0, 1, 2], [0, 1, 0, 3]]).to(device)
    shape = (4, 4)
    sp_topo = SparseTopology(idx_coo, shape)

    with pytest.raises(FrozenInstanceError):
        sp_topo._idx_coo = t.tensor([[0, 1, 1, 2], [0, 1, 0, 3]])

    with pytest.raises(FrozenInstanceError):
        sp_topo.idx_coo = t.tensor([[0, 1, 1, 2], [0, 1, 0, 3]])

    with pytest.raises(FrozenInstanceError):
        sp_topo.shape = (5, 5)


def test_shape_validation(device):
    # sparse coo index with only one dimension.
    with pytest.raises(ValueError):
        idx_coo = t.tensor([0, 1, 3, 5]).to(device)
        shape = (4, 4)
        SparseTopology(idx_coo, shape)

    # coo tensor with 2 batch dimensions.
    with pytest.raises(NotImplementedError):
        idx_coo = t.tensor([[0, 0, 1, 1], [0, 1, 0, 1], [0, 0, 1, 2], [0, 1, 0, 3]]).to(
            device
        )
        shape = (2, 2, 4, 4)
        SparseTopology(idx_coo, shape)

    # coo index/shape mismatch
    with pytest.raises(ValueError):
        idx_coo = t.tensor([[0, 0, 1, 2], [0, 1, 0, 3]]).to(device)
        shape = (2, 4, 4)
        SparseTopology(idx_coo, shape)

    with pytest.raises(ValueError):
        idx_coo = t.tensor([[0, 0, 1, 1], [0, 0, 1, 2], [0, 1, 0, 3]]).to(device)
        shape = (4, 4)
        SparseTopology(idx_coo, shape)

    # batched coo tensor with uneven nnz per batch item.
    with pytest.raises(ValueError):
        idx_coo = t.tensor([[0, 0, 1, 1, 1], [0, 0, 1, 2, 3], [0, 1, 0, 3, 0]]).to(
            device
        )
        shape = (4, 4)
        SparseTopology(idx_coo, shape)

    with pytest.raises(ValueError):
        idx_coo = t.tensor([[0, 1, 1, 1], [0, 0, 1, 2], [0, 1, 0, 3]]).to(device)
        shape = (4, 4)
        SparseTopology(idx_coo, shape)


def test_oob_validation(device):
    with pytest.raises(ValueError):
        idx_coo = t.tensor([[0, 0, 1, 2], [0, 1, 0, 4]]).to(device)
        shape = (4, 4)
        SparseTopology(idx_coo, shape)

    with pytest.raises(ValueError):
        idx_coo = t.tensor([[-1, 0, 1, 2], [0, 1, 0, 3]]).to(device)
        shape = (4, 4)
        SparseTopology(idx_coo, shape)


def test_coo_to_csr_conversion(sp_with_empty_row, device):
    (
        idx_coo,
        shape,
        true_idx_crow,
        true_idx_col,
        true_idx_ccol,
        true_idx_row,
        true_idx_coo_to_csc_perm,
    ) = sp_with_empty_row

    sp_topo = SparseTopology(idx_coo, shape).to(device)

    t.testing.assert_close(sp_topo.idx_crow, true_idx_crow.to(device))
    t.testing.assert_close(sp_topo.idx_col, true_idx_col.to(device))


def test_coo_to_csc_conversion(sp_with_empty_row, device):
    (
        idx_coo,
        shape,
        true_idx_crow,
        true_idx_col,
        true_idx_ccol,
        true_idx_row,
        true_coo_to_csc_perm,
    ) = sp_with_empty_row

    sp_topo = SparseTopology(idx_coo, shape).to(device)

    t.testing.assert_close(sp_topo.idx_ccol, true_idx_ccol.to(device))
    t.testing.assert_close(sp_topo.idx_row, true_idx_row.to(device))
    t.testing.assert_close(sp_topo.coo_to_csc_perm, true_coo_to_csc_perm.to(device))


def test_coo_to_csr_conversion_with_batch_dim(sp_with_batch_dim, device):
    (
        idx_coo,
        shape,
        true_idx_crow,
        true_idx_col,
        true_idx_ccol,
        true_idx_row,
        true_idx_coo_to_csc_perm,
    ) = sp_with_batch_dim

    sp_topo = SparseTopology(idx_coo, shape).to(device)

    t.testing.assert_close(sp_topo.idx_crow, true_idx_crow.to(device))
    t.testing.assert_close(sp_topo.idx_col, true_idx_col.to(device))


def test_coo_to_csc_conversion_with_batch_dim(sp_with_batch_dim, device):
    (
        idx_coo,
        shape,
        true_idx_crow,
        true_idx_col,
        true_idx_ccol,
        true_idx_row,
        true_coo_to_csc_perm,
    ) = sp_with_batch_dim

    sp_topo = SparseTopology(idx_coo, shape).to(device)

    t.testing.assert_close(sp_topo.idx_ccol, true_idx_ccol.to(device))
    t.testing.assert_close(sp_topo.idx_row, true_idx_row.to(device))
    t.testing.assert_close(sp_topo.coo_to_csc_perm, true_coo_to_csc_perm.to(device))


def test_idx_dtype(device):
    idx_coo = t.tensor([[0, 0, 1, 2], [0, 1, 0, 3]]).to(device)
    shape = (4, 4)
    sp_topo = SparseTopology(idx_coo, shape)

    target_dtype = sp_topo.idx_coo.dtype

    assert sp_topo.idx_crow.dtype == target_dtype
    assert sp_topo.idx_col.dtype == target_dtype
    assert sp_topo.idx_ccol.dtype == target_dtype
    assert sp_topo.idx_row.dtype == target_dtype
    assert sp_topo.coo_to_csc_perm.dtype == target_dtype

    assert sp_topo.idx_crow_int32.dtype == t.int32
    assert sp_topo.idx_col_int32.dtype == t.int32
    assert sp_topo.idx_ccol_int32.dtype == t.int32
    assert sp_topo.idx_row_int32.dtype == t.int32

    t.testing.assert_close(sp_topo.idx_crow, sp_topo.idx_crow_int32.to(t.int64))
    t.testing.assert_close(sp_topo.idx_col, sp_topo.idx_col_int32.to(t.int64))
    t.testing.assert_close(sp_topo.idx_ccol, sp_topo.idx_ccol_int32.to(t.int64))
    t.testing.assert_close(sp_topo.idx_row, sp_topo.idx_row_int32.to(t.int64))


def test_sp_dim(sp_with_empty_row, device):
    (
        idx_coo,
        shape,
        true_idx_crow,
        true_idx_col,
        true_idx_ccol,
        true_idx_row,
        true_coo_to_csc_perm,
    ) = sp_with_empty_row

    sp_topo = SparseTopology(idx_coo, shape).to(device)

    assert sp_topo.n_batch_dim == 0
    assert sp_topo.n_sp_dim == 2


def test_batch_dim(sp_with_batch_dim, device):
    (
        idx_coo,
        shape,
        true_idx_crow,
        true_idx_col,
        true_idx_ccol,
        true_idx_row,
        true_coo_to_csc_perm,
    ) = sp_with_batch_dim

    sp_topo = SparseTopology(idx_coo, shape).to(device)

    assert sp_topo.n_batch_dim == 1
    assert sp_topo.n_sp_dim == 2


def test_transpose(sp_with_empty_row, sp_with_empty_col, device):
    (
        idx_coo,
        shape,
        true_idx_crow,
        true_idx_col,
        true_idx_ccol,
        true_idx_row,
        true_coo_to_csc_perm,
    ) = sp_with_empty_row

    (
        idx_coo_T,
        shape_T,
        true_idx_crow_T,
        true_idx_col_T,
        true_idx_ccol_T,
        true_idx_row_T,
        true_coo_to_csc_perm_T,
    ) = sp_with_empty_col

    sp_topo = SparseTopology(idx_coo, shape).to(device)
    sp_topo_T = sp_topo.T

    assert sp_topo.shape[::-1] == sp_topo_T.shape

    t.testing.assert_close(sp_topo_T.idx_crow, true_idx_crow_T.to(device))
    t.testing.assert_close(sp_topo_T.idx_col, true_idx_col_T.to(device))
    t.testing.assert_close(sp_topo_T.idx_ccol, true_idx_ccol_T.to(device))
    t.testing.assert_close(sp_topo_T.idx_row, true_idx_row_T.to(device))
    t.testing.assert_close(sp_topo_T.coo_to_csc_perm, true_coo_to_csc_perm_T.to(device))


def test_transpose_with_batch_dim(sp_with_batch_dim, sp_with_batch_dim_T, device):
    (
        idx_coo,
        shape,
        true_idx_crow,
        true_idx_col,
        true_idx_ccol,
        true_idx_row,
        true_coo_to_csc_perm,
    ) = sp_with_batch_dim

    (
        idx_coo_T,
        shape_T,
        true_idx_crow_T,
        true_idx_col_T,
        true_idx_ccol_T,
        true_idx_row_T,
        true_coo_to_csc_perm_T,
    ) = sp_with_batch_dim_T

    sp_topo = SparseTopology(idx_coo, shape).to(device)
    sp_topo_T = sp_topo.T

    assert (sp_topo.shape[0], sp_topo.shape[2], sp_topo.shape[1]) == tuple(
        sp_topo_T.shape
    )

    t.testing.assert_close(sp_topo_T.idx_crow, true_idx_crow_T.to(device))
    t.testing.assert_close(sp_topo_T.idx_col, true_idx_col_T.to(device))
    t.testing.assert_close(sp_topo_T.idx_ccol, true_idx_ccol_T.to(device))
    t.testing.assert_close(sp_topo_T.idx_row, true_idx_row_T.to(device))
    t.testing.assert_close(sp_topo_T.coo_to_csc_perm, true_coo_to_csc_perm_T.to(device))


# TODO: also test non_blocking and copy in to()
def test_to_float(device):
    idx_coo = t.tensor([[0, 0, 1, 2], [0, 1, 0, 3]]).to(device)
    shape = (4, 4)
    sp_topo = SparseTopology(idx_coo, shape)

    with pytest.raises(ValueError):
        sp_topo.to(t.float32)


def test_to_int32(device):
    idx_coo = t.tensor([[0, 0, 1, 2], [0, 1, 0, 3]]).to(device)
    shape = (4, 4)
    sp_topo = SparseTopology(idx_coo, shape).to(t.int32)

    assert sp_topo.idx_coo.dtype == t.int32


def test_to_device(device):
    idx_coo = t.tensor([[0, 0, 1, 2], [0, 1, 0, 3]])
    shape = (4, 4)
    sp_topo = SparseTopology(idx_coo, shape).to(device)

    assert sp_topo.idx_coo.device.type == device.type


def test_nnz(sp_with_empty_row, device):
    (
        idx_coo,
        shape,
        true_idx_crow,
        true_idx_col,
        true_idx_ccol,
        true_idx_row,
        true_coo_to_csc_perm,
    ) = sp_with_empty_row

    sp_topo = SparseTopology(idx_coo, shape).to(device)

    assert sp_topo._nnz() == 5


def test_nnz_with_batch_dim(sp_with_batch_dim, device):
    (
        idx_coo,
        shape,
        true_idx_crow,
        true_idx_col,
        true_idx_ccol,
        true_idx_row,
        true_coo_to_csc_perm,
    ) = sp_with_batch_dim

    sp_topo = SparseTopology(idx_coo, shape).to(device)

    assert sp_topo._nnz() == 6


def test_size(device):
    idx_coo = t.tensor([[0, 0, 1, 2], [0, 1, 0, 3]]).to(device)
    shape = (4, 4)
    sp_topo = SparseTopology(idx_coo, shape)

    assert sp_topo.size() == sp_topo.shape

    for idx, val in enumerate(shape):
        assert sp_topo.size(idx) == val


# TODO: test empty edge case?

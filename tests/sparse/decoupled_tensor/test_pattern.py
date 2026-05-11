from dataclasses import FrozenInstanceError

import pytest
import torch

from cochain.sparse.decoupled_tensor import SparsityPattern
from cochain.sparse.decoupled_tensor.pattern import (
    BlockDiagConfig,
    check_pattern_equality,
)


@pytest.fixture
def sp_with_empty_row():
    # | 0 0 0 0 |
    # | 0 1 0 1 |
    # | 1 0 1 1 |

    idx_coo = torch.tensor([[1, 1, 2, 2, 2], [1, 3, 0, 2, 3]])
    shape = (3, 4)

    idx_crow = torch.tensor([0, 0, 2, 5], dtype=torch.int32)
    idx_col = torch.tensor([1, 3, 0, 2, 3], dtype=torch.int32)

    idx_ccol = torch.tensor([0, 1, 2, 3, 5], dtype=torch.int32)
    idx_row_csc = torch.tensor([2, 1, 2, 1, 2], dtype=torch.int32)

    csc_to_coo_map = torch.tensor([2, 0, 3, 1, 4])

    return idx_coo, shape, idx_crow, idx_col, idx_ccol, idx_row_csc, csc_to_coo_map


@pytest.fixture
def sp_with_empty_col():
    # | 0 0 1 |
    # | 0 1 0 |
    # | 0 0 1 |
    # | 0 1 1 |

    idx_coo = torch.tensor([[0, 1, 2, 3, 3], [2, 1, 2, 1, 2]])
    shape = (4, 3)

    idx_crow = torch.tensor([0, 1, 2, 3, 5], dtype=torch.int32)
    idx_col = torch.tensor([2, 1, 2, 1, 2], dtype=torch.int32)

    idx_ccol = torch.tensor([0, 0, 2, 5], dtype=torch.int32)
    idx_row_csc = torch.tensor([1, 3, 0, 2, 3], dtype=torch.int32)

    csc_to_coo_map = torch.tensor([1, 3, 0, 2, 4])

    return idx_coo, shape, idx_crow, idx_col, idx_ccol, idx_row_csc, csc_to_coo_map


@pytest.fixture
def sp_with_batch_dim():
    # | 1 0 0 |
    # | 0 0 0 |
    # | 0 1 1 |
    #
    # | 0 1 0 |
    # | 0 1 0 |
    # | 0 0 1 |

    idx_coo = torch.tensor([[0, 0, 0, 1, 1, 1], [0, 2, 2, 0, 1, 2], [0, 1, 2, 1, 1, 2]])
    shape = (2, 3, 3)

    idx_crow = torch.tensor([[0, 1, 1, 3], [0, 1, 2, 3]], dtype=torch.int32)
    idx_col = torch.tensor([[0, 1, 2], [1, 1, 2]], dtype=torch.int32)

    idx_ccol = torch.tensor([[0, 1, 2, 3], [0, 0, 2, 3]], dtype=torch.int32)
    idx_row_csc = torch.tensor([[0, 2, 2], [0, 1, 2]], dtype=torch.int32)

    csc_to_coo_map = torch.tensor([0, 1, 2, 3, 4, 5])

    return idx_coo, shape, idx_crow, idx_col, idx_ccol, idx_row_csc, csc_to_coo_map


@pytest.fixture
def sp_with_batch_dim_T():
    # | 1 0 0 |
    # | 0 0 1 |
    # | 0 0 1 |
    #
    # | 0 0 0 |
    # | 1 1 0 |
    # | 0 0 1 |

    idx_coo = torch.tensor([[0, 0, 0, 1, 1, 1], [0, 1, 2, 1, 1, 2], [0, 2, 2, 0, 1, 2]])
    shape = (2, 3, 3)

    idx_crow = torch.tensor([[0, 1, 2, 3], [0, 0, 2, 3]], dtype=torch.int32)
    idx_col = torch.tensor([[0, 2, 2], [0, 1, 2]], dtype=torch.int32)

    idx_ccol = torch.tensor([[0, 1, 1, 3], [0, 1, 2, 3]], dtype=torch.int32)
    idx_row_csc = torch.tensor([[0, 1, 2], [1, 1, 2]], dtype=torch.int32)

    csc_to_coo_map = torch.tensor([0, 1, 2, 3, 4, 5])

    return idx_coo, shape, idx_crow, idx_col, idx_ccol, idx_row_csc, csc_to_coo_map


def test_check_pattern_equality(device):
    idx_coo = torch.tensor([[0, 1], [1, 0]]).to(device)
    shape = (2, 2)

    p1 = SparsityPattern(idx_coo, shape)
    p2 = SparsityPattern(idx_coo.clone(), shape)

    # Same shape and indices.
    check_pattern_equality(p1, p2, "Patterns should be equal")

    # Same object.
    check_pattern_equality(p1, p1, "Patterns should be equal")

    # Different shape.
    p3 = SparsityPattern(idx_coo, (3, 3))
    with pytest.raises(ValueError, match="Patterns not equal"):
        check_pattern_equality(p1, p3, "Patterns not equal")

    # Different indices.
    idx_coo_diff = torch.tensor([[0, 1], [1, 1]]).to(device)
    p4 = SparsityPattern(idx_coo_diff, shape)
    with pytest.raises(ValueError, match="Patterns not equal"):
        check_pattern_equality(p1, p4, "Patterns not equal")


def test_immutability(device):
    idx_coo = torch.tensor([[0, 0, 1, 2], [0, 1, 0, 3]]).to(device)
    shape = (4, 4)
    pattern = SparsityPattern(idx_coo, shape)

    with pytest.raises(FrozenInstanceError):
        pattern._idx_coo = torch.tensor([[0, 1, 1, 2], [0, 1, 0, 3]])

    with pytest.raises(FrozenInstanceError):
        pattern.idx_coo = torch.tensor([[0, 1, 1, 2], [0, 1, 0, 3]])

    with pytest.raises(FrozenInstanceError):
        pattern.shape = (5, 5)


def test_shape_validation(device):
    # sparse coo index with only one dimension.
    with pytest.raises(ValueError):
        idx_coo = torch.tensor([0, 1, 3, 5]).to(device)
        shape = (4, 4)
        SparsityPattern(idx_coo, shape)

    # coo tensor with 2 batch dimensions.
    with pytest.raises(NotImplementedError):
        idx_coo = torch.tensor(
            [[0, 0, 1, 1], [0, 1, 0, 1], [0, 0, 1, 2], [0, 1, 0, 3]]
        ).to(device)
        shape = (2, 2, 4, 4)
        SparsityPattern(idx_coo, shape)

    # coo index/shape mismatch
    with pytest.raises(ValueError):
        idx_coo = torch.tensor([[0, 0, 1, 2], [0, 1, 0, 3]]).to(device)
        shape = (2, 4, 4)
        SparsityPattern(idx_coo, shape)

    with pytest.raises(ValueError):
        idx_coo = torch.tensor([[0, 0, 1, 1], [0, 0, 1, 2], [0, 1, 0, 3]]).to(device)
        shape = (4, 4)
        SparsityPattern(idx_coo, shape)

    # batched coo tensor with uneven nnz per batch item.
    with pytest.raises(ValueError):
        idx_coo = torch.tensor([[0, 0, 1, 1, 1], [0, 0, 1, 2, 3], [0, 1, 0, 3, 0]]).to(
            device
        )
        shape = (4, 4)
        SparsityPattern(idx_coo, shape)

    with pytest.raises(ValueError):
        idx_coo = torch.tensor([[0, 1, 1, 1], [0, 0, 1, 2], [0, 1, 0, 3]]).to(device)
        shape = (4, 4)
        SparsityPattern(idx_coo, shape)


def test_oob_validation(device):
    with pytest.raises(ValueError):
        idx_coo = torch.tensor([[0, 0, 1, 2], [0, 1, 0, 4]]).to(device)
        shape = (4, 4)
        SparsityPattern(idx_coo, shape)

    with pytest.raises(ValueError):
        idx_coo = torch.tensor([[-1, 0, 1, 2], [0, 1, 0, 3]]).to(device)
        shape = (4, 4)
        SparsityPattern(idx_coo, shape)


def test_idx_coo_contiguous(device):
    idx_coo_T = torch.tensor([[0, 0, 1, 2], [0, 1, 0, 3]]).T.contiguous().to(device)
    shape = (4, 4)

    pattern = SparsityPattern(idx_coo_T.T, shape)

    assert not idx_coo_T.T.is_contiguous()
    assert pattern.idx_coo.is_contiguous()


def test_coo_to_csr_conversion(sp_with_empty_row, device):
    (
        idx_coo,
        shape,
        true_idx_crow,
        true_idx_col,
        true_idx_ccol,
        true_idx_row_csc,
        true_idx_csc_to_coo_map,
    ) = sp_with_empty_row

    pattern = SparsityPattern(idx_coo, shape).to(device)

    torch.testing.assert_close(pattern.idx_crow, true_idx_crow.to(device))
    torch.testing.assert_close(pattern.idx_col, true_idx_col.to(device))


def test_coo_to_csc_conversion(sp_with_empty_row, device):
    (
        idx_coo,
        shape,
        true_idx_crow,
        true_idx_col,
        true_idx_ccol,
        true_idx_row_csc,
        true_csc_to_coo_map,
    ) = sp_with_empty_row

    pattern = SparsityPattern(idx_coo, shape).to(device)

    torch.testing.assert_close(pattern.idx_ccol, true_idx_ccol.to(device))
    torch.testing.assert_close(pattern.idx_row_csc, true_idx_row_csc.to(device))
    torch.testing.assert_close(pattern.csc_to_coo_map, true_csc_to_coo_map.to(device))


def test_coo_to_csr_conversion_with_batch_dim(sp_with_batch_dim, device):
    (
        idx_coo,
        shape,
        true_idx_crow,
        true_idx_col,
        true_idx_ccol,
        true_idx_row_csc,
        true_idx_csc_to_coo_map,
    ) = sp_with_batch_dim

    pattern = SparsityPattern(idx_coo, shape).to(device)

    torch.testing.assert_close(pattern.idx_crow, true_idx_crow.to(device))
    torch.testing.assert_close(pattern.idx_col, true_idx_col.to(device))


def test_coo_to_csc_conversion_with_batch_dim(sp_with_batch_dim, device):
    (
        idx_coo,
        shape,
        true_idx_crow,
        true_idx_col,
        true_idx_ccol,
        true_idx_row_csc,
        true_csc_to_coo_map,
    ) = sp_with_batch_dim

    pattern = SparsityPattern(idx_coo, shape).to(device)

    torch.testing.assert_close(pattern.idx_ccol, true_idx_ccol.to(device))
    torch.testing.assert_close(pattern.idx_row_csc, true_idx_row_csc.to(device))
    torch.testing.assert_close(pattern.csc_to_coo_map, true_csc_to_coo_map.to(device))


def test_idx_dtype(device):
    idx_coo = torch.tensor([[0, 0, 1, 2], [0, 1, 0, 3]]).to(device)
    shape = (4, 4)
    pattern = SparsityPattern(idx_coo, shape)

    assert pattern.idx_coo.dtype == torch.int64
    assert pattern.csc_to_coo_map.dtype == torch.int64
    assert pattern.idx_crow.dtype == torch.int32
    assert pattern.idx_col.dtype == torch.int32
    assert pattern.idx_ccol.dtype == torch.int32
    assert pattern.idx_row_csc.dtype == torch.int32


def test_size(device):
    idx_coo = torch.tensor([[0, 0, 1, 2], [0, 1, 0, 3]]).to(device)
    shape = (4, 4)
    pattern = SparsityPattern(idx_coo, shape)

    assert pattern.size() == pattern.shape

    for idx, val in enumerate(shape):
        assert pattern.size(idx) == val


def test_nnz(sp_with_empty_row, device):
    (
        idx_coo,
        shape,
        true_idx_crow,
        true_idx_col,
        true_idx_ccol,
        true_idx_row_csc,
        true_csc_to_coo_map,
    ) = sp_with_empty_row

    pattern = SparsityPattern(idx_coo, shape).to(device)

    assert pattern._nnz() == 5


def test_nnz_with_batch_dim(sp_with_batch_dim, device):
    (
        idx_coo,
        shape,
        true_idx_crow,
        true_idx_col,
        true_idx_ccol,
        true_idx_row_csc,
        true_csc_to_coo_map,
    ) = sp_with_batch_dim

    pattern = SparsityPattern(idx_coo, shape).to(device)

    assert pattern._nnz() == 6


def test_sp_dim(sp_with_empty_row, device):
    (
        idx_coo,
        shape,
        true_idx_crow,
        true_idx_col,
        true_idx_ccol,
        true_idx_row_csc,
        true_csc_to_coo_map,
    ) = sp_with_empty_row

    pattern = SparsityPattern(idx_coo, shape).to(device)

    assert pattern.n_batch_dim == 0
    assert pattern.n_sp_dim == 2


def test_batch_dim(sp_with_batch_dim, device):
    (
        idx_coo,
        shape,
        true_idx_crow,
        true_idx_col,
        true_idx_ccol,
        true_idx_row_csc,
        true_csc_to_coo_map,
    ) = sp_with_batch_dim

    pattern = SparsityPattern(idx_coo, shape).to(device)

    assert pattern.n_batch_dim == 1
    assert pattern.n_sp_dim == 2


def test_transpose(sp_with_empty_row, sp_with_empty_col, device):
    (
        idx_coo,
        shape,
        true_idx_crow,
        true_idx_col,
        true_idx_ccol,
        true_idx_row_csc,
        true_csc_to_coo_map,
    ) = sp_with_empty_row

    (
        idx_coo_T,
        shape_T,
        true_idx_crow_T,
        true_idx_col_T,
        true_idx_ccol_T,
        true_idx_row_csc_T,
        true_csc_to_coo_map_T,
    ) = sp_with_empty_col

    pattern = SparsityPattern(idx_coo, shape).to(device)
    pattern_T = pattern.T

    assert pattern.shape[::-1] == pattern_T.shape

    torch.testing.assert_close(pattern_T.idx_crow, true_idx_crow_T.to(device))
    torch.testing.assert_close(pattern_T.idx_col, true_idx_col_T.to(device))
    torch.testing.assert_close(pattern_T.idx_ccol, true_idx_ccol_T.to(device))
    torch.testing.assert_close(pattern_T.idx_row_csc, true_idx_row_csc_T.to(device))
    torch.testing.assert_close(
        pattern_T.csc_to_coo_map, true_csc_to_coo_map_T.to(device)
    )


def test_transpose_with_batch_dim(sp_with_batch_dim, sp_with_batch_dim_T, device):
    (
        idx_coo,
        shape,
        true_idx_crow,
        true_idx_col,
        true_idx_ccol,
        true_idx_row_csc,
        true_csc_to_coo_map,
    ) = sp_with_batch_dim

    (
        idx_coo_T,
        shape_T,
        true_idx_crow_T,
        true_idx_col_T,
        true_idx_ccol_T,
        true_idx_row_csc_T,
        true_csc_to_coo_map_T,
    ) = sp_with_batch_dim_T

    pattern = SparsityPattern(idx_coo, shape).to(device)
    pattern_T = pattern.T

    assert (pattern.shape[0], pattern.shape[2], pattern.shape[1]) == tuple(
        pattern_T.shape
    )

    torch.testing.assert_close(pattern_T.idx_crow, true_idx_crow_T.to(device))
    torch.testing.assert_close(pattern_T.idx_col, true_idx_col_T.to(device))
    torch.testing.assert_close(pattern_T.idx_ccol, true_idx_ccol_T.to(device))
    torch.testing.assert_close(pattern_T.idx_row_csc, true_idx_row_csc_T.to(device))
    torch.testing.assert_close(
        pattern_T.csc_to_coo_map, true_csc_to_coo_map_T.to(device)
    )


def test_to_device(device):
    idx_coo = torch.tensor([[0, 0, 1, 2], [0, 1, 0, 3]])
    shape = (4, 4)
    pattern = SparsityPattern(idx_coo, shape).to(device)

    assert pattern.idx_coo.device.type == device.type


def test_to_with_cached_properties(sp_with_empty_row, device):
    (idx_coo, shape, _, _, _, _, _) = sp_with_empty_row

    pattern = SparsityPattern(idx_coo, shape).to(device)

    # Access properties to cache them.
    _ = pattern.idx_crow
    _ = pattern.idx_col
    _ = pattern.idx_ccol
    _ = pattern.idx_row_csc
    _ = pattern.csc_to_coo_map

    # Call `to` to test if cached attributes are properly handled.
    new_pattern = pattern.to(torch.int64)
    assert new_pattern.idx_crow.dtype == torch.int32
    assert new_pattern.idx_col.dtype == torch.int32
    assert new_pattern.idx_ccol.dtype == torch.int32
    assert new_pattern.idx_row_csc.dtype == torch.int32
    assert new_pattern.csc_to_coo_map.dtype == torch.int64


def test_block_diag_config_inv(device):
    batch_perm = torch.tensor([2, 0, 1]).to(device)
    nnzs = [1, 2]
    pattern_shapes = [torch.Size([2, 2]), torch.Size([3, 3])]

    config = BlockDiagConfig(batch_perm, nnzs, pattern_shapes)

    inv = config.batch_perm_inv
    expected_inv = torch.tensor([1, 2, 0]).to(device)
    torch.testing.assert_close(inv, expected_inv)


def test_block_diag_config_to(device):
    batch_perm = torch.tensor([2, 0, 1]).to(device)
    nnzs = [1, 2]
    pattern_shapes = [torch.Size([2, 2]), torch.Size([3, 3])]

    config = BlockDiagConfig(batch_perm, nnzs, pattern_shapes)

    config_cast = config.to(torch.int32)
    # BlockDiagConfig.to ignores dtype conversions, so it should still be the original dtype
    assert config_cast.batch_perm.dtype == batch_perm.dtype


# TODO: test empty edge case?

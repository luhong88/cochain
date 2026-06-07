import pytest
import torch
from jaxtyping import Float
from torch import Tensor


@pytest.fixture
def a() -> Float[Tensor, "4 4"]:
    """Construct a 4x4 diagonally-dominant sparse COO tensor."""
    n_row = 4
    nnz = int(n_row * n_row * 0.4)

    idx = torch.hstack(
        (torch.randint(0, n_row, (2, nnz)), torch.tile(torch.arange(n_row), (2, 1)))
    )
    val = torch.hstack((torch.randn(nnz), n_row * torch.ones(n_row))).to(
        dtype=torch.float32
    )

    tensor = torch.sparse_coo_tensor(idx, val, (n_row, n_row)).coalesce()

    return tensor


@pytest.fixture
def b() -> Float[Tensor, "4 5"]:
    n_row = 4
    n_col = 5
    nnz = int(n_row * n_col * 0.4)

    idx_off_diag = torch.vstack(
        (torch.randint(0, n_row, (nnz,)), torch.randint(0, n_col, (nnz,)))
    )
    diag_len = min(n_row, n_col)
    idx_diag = torch.tile(torch.arange(diag_len), (2, 1))

    idx = torch.hstack((idx_off_diag, idx_diag))
    val = torch.hstack((torch.randn(nnz), diag_len * torch.ones(diag_len))).to(
        dtype=torch.float32
    )

    tensor = torch.sparse_coo_tensor(idx, val, (n_row, n_col)).coalesce()

    return tensor


@pytest.fixture
def a_with_batch() -> Float[Tensor, "2 4 4"]:
    n_batch = 2
    n_row = 4
    target_nnz = int(n_row * n_row * 0.2)

    idx_list = []
    idx_size_list = []
    for _ in range(n_batch):
        # Use modular offset to generate random off-diagonal nonzero indices. Use
        # 10x oversampling to guarantee nonduplicate off-diagonal indices.
        r0 = torch.randint(0, n_row, (10 * target_nnz,))
        offset = torch.randint(1, n_row, (10 * target_nnz,))
        r1 = (r0 + offset) % (n_row)

        unique_idx = torch.vstack((r0, r1)).unique(dim=1)
        # the random permutation is required since unique() will sort the idx.
        perm = torch.randperm(unique_idx.size(1))

        idx_list.append(unique_idx[:, perm])
        idx_size_list.append(unique_idx.size(1))

    # Since stacked CSR requires that each tensor has the same nnz, find the minimum
    # number of unique off-diagonal indices that will work for the whole batch.
    nnz = min(*idx_size_list, target_nnz)

    tensor_list = []
    for unique_idx in idx_list:
        idx_off_diag = unique_idx[:, :nnz]

        idx_diag = torch.tile(torch.arange(n_row), (2, 1))
        idx = torch.hstack((idx_off_diag, idx_diag))

        val = torch.hstack((torch.randn(nnz), n_row * torch.ones(n_row))).to(
            dtype=torch.float32
        )

        tensor_list.append(torch.sparse_coo_tensor(idx, val, (n_row, n_row)))

    tensor = torch.stack(tensor_list).to_sparse_coo().coalesce()

    return tensor


@pytest.fixture
def a_with_dense() -> Float[Tensor, "4 4 3"]:
    n_row = 4
    n_dense = 3
    nnz = int(n_row * n_row * 0.4)

    idx = torch.hstack(
        (torch.randint(0, n_row, (2, nnz)), torch.tile(torch.arange(n_row), (2, 1)))
    )
    val = torch.vstack(
        (torch.randn(nnz, n_dense), n_row * torch.ones(n_row, n_dense))
    ).to(dtype=torch.float32)

    tensor = torch.sparse_coo_tensor(idx, val, (n_row, n_row, n_dense)).coalesce()

    return tensor


@pytest.fixture
def a_with_batch_dense() -> Float[Tensor, "2 4 4 3"]:
    n_batch = 2
    n_row = 4
    n_dense = 3
    target_nnz = int(n_row * n_row * 0.2)

    idx_list = []
    idx_size_list = []
    for _ in range(n_batch):
        # Use modular offset to generate random off-diagonal nonzero indices. Use
        # 10x oversampling to guarantee nonduplicate off-diagonal indices.
        r0 = torch.randint(0, n_row, (10 * target_nnz,))
        offset = torch.randint(1, n_row, (10 * target_nnz,))
        r1 = (r0 + offset) % (n_row)

        unique_idx = torch.vstack((r0, r1)).unique(dim=1)
        # the random permutation is required since unique() will sort the idx.
        perm = torch.randperm(unique_idx.size(1))

        idx_list.append(unique_idx[:, perm])
        idx_size_list.append(unique_idx.size(1))

    # Since stacked CSR requires that each tensor has the same nnz, find the minimum
    # number of unique off-diagonal indices that will work for the whole batch.
    nnz = min(*idx_size_list, target_nnz)

    tensor_list = []
    for unique_idx in idx_list:
        idx_off_diag = unique_idx[:, :nnz]

        idx_diag = torch.tile(torch.arange(n_row), (2, 1))
        idx = torch.hstack((idx_off_diag, idx_diag))

        val = torch.vstack(
            (torch.randn(nnz, n_dense), n_row * torch.ones(n_row, n_dense))
        ).to(dtype=torch.float32)

        tensor_list.append(torch.sparse_coo_tensor(idx, val, (n_row, n_row, n_dense)))

    tensor = torch.stack(tensor_list).to_sparse_coo().coalesce()

    return tensor

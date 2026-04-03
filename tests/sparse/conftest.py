import pytest
import torch
from jaxtyping import Float
from torch import Tensor


@pytest.fixture
def A() -> Float[Tensor, "4 4"]:
    n_dim = 4
    nnz = int(n_dim * n_dim * 0.4)

    idx = torch.hstack(
        (torch.randint(0, n_dim, (2, nnz)), torch.tile(torch.arange(n_dim), (2, 1)))
    )
    val = torch.hstack((torch.randn(nnz), n_dim * torch.ones(n_dim))).to(
        dtype=torch.float
    )

    A = torch.sparse_coo_tensor(idx, val, (n_dim, n_dim)).coalesce()

    return A


@pytest.fixture
def A_batched() -> Float[Tensor, "2 4 4"]:
    n_batch = 2
    n_dim = 4
    target_nnz = int(n_dim * n_dim * 0.2)

    idx_list = []
    idx_size_list = []
    for _ in range(n_batch):
        # Use modular offset to generate random off-diagonal nonzero indices. Use
        # 10x oversampling to guarantee nonduplicate off-diagonal indices.
        r0 = torch.randint(0, n_dim, (10 * target_nnz,))
        offset = torch.randint(1, n_dim, (10 * target_nnz,))
        r1 = (r0 + offset) % (n_dim)

        unique_idx = torch.vstack((r0, r1)).unique(dim=1)
        # the random permutation is required since unique() will sort the idx.
        perm = torch.randperm(unique_idx.size(1))

        idx_list.append(unique_idx[:, perm])
        idx_size_list.append(unique_idx.size(1))

    # Since stacked CSR requires that each tensor has the same nnz, find the minimum
    # number of unique off-diagonal indices that will work for the whole batch.
    nnz = min(*idx_size_list, target_nnz)

    A_list = []
    for unique_idx in idx_list:
        idx_off_diag = unique_idx[:, :nnz]

        idx_diag = torch.tile(torch.arange(n_dim), (2, 1))
        idx = torch.hstack((idx_off_diag, idx_diag))

        val = torch.hstack((torch.randn(nnz), n_dim * torch.ones(n_dim))).to(
            dtype=torch.float
        )

        A_list.append(torch.sparse_coo_tensor(idx, val, (n_dim, n_dim)))

    A = torch.stack(A_list).to_sparse_coo().coalesce()

    return A

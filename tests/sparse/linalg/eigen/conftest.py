import math
import random

import pytest
import torch
from jaxtyping import Float
from torch import Tensor


def givens_rotation_matrix(
    i: int, j: int, theta: float, n: int
) -> Float[Tensor, "n n"]:
    cos = math.cos(theta)
    sin = math.sin(theta)

    idx_coo_diag = torch.tile(torch.arange(n), (2, 1))
    val_diag = torch.ones(n, dtype=torch.float64)
    val_diag[i] = cos
    val_diag[j] = cos

    idx_coo_off_diag = torch.tensor([[i, j], [j, i]])
    val_off_diag = torch.tensor([-sin, sin], dtype=torch.float64)

    mat = torch.sparse_coo_tensor(
        indices=torch.hstack((idx_coo_diag, idx_coo_off_diag)),
        values=torch.hstack((val_diag, val_off_diag)),
        size=(n, n),
    ).coalesce()

    return mat


def rand_sp_sym_matrix(lambdas: Float[Tensor, " n"], k: int) -> Float[Tensor, "n n"]:
    n = lambdas.size(0)

    mat = torch.sparse_coo_tensor(
        indices=torch.tile(torch.arange(n), (2, 1)),
        values=lambdas.to(dtype=torch.float64),
        size=(n, n),
    )

    for _ in range(k):
        # It is important to sample without replacement to avoid selecting i = j.
        i, j = random.sample(range(n), 2)
        theta = 2 * math.pi * random.random()
        givens = givens_rotation_matrix(i, j, theta, n)
        mat = givens @ mat @ givens.T

    return mat.coalesce()


def rand_sym_gep_matrices(
    lambdas: Float[Tensor, " n"], rho: float, k: int
) -> tuple[Float[Tensor, "n n"], Float[Tensor, "n n"]]:
    n = lambdas.size(0)

    # Construct the diagonal eigenvalue matrix.
    diag = torch.sparse_coo_tensor(
        indices=torch.tile(torch.arange(n), (2, 1)),
        values=lambdas.to(dtype=torch.float64),
        size=(n, n),
    ).coalesce()

    # Construct the stencil matrix.
    nnz = int(n * rho)

    idx_off_diag = torch.randint(0, n, (2, nnz))
    idx_diag = torch.tile(torch.arange(n), (2, 1))
    idx = torch.hstack((idx_off_diag, idx_diag))

    val = torch.hstack(
        (
            torch.randn(nnz, dtype=torch.float64),
            2.0 * torch.ones(n, dtype=torch.float64),
        )
    )

    stencil = torch.sparse_coo_tensor(idx, val, (n, n)).coalesce()

    # Construct A, B, such that they satisfy the GEP with the given eigenvalues.
    b = stencil.T @ stencil
    a = stencil.T @ diag @ stencil

    for _ in range(k):
        i, j = random.sample(range(n), 2)
        theta = 2 * math.pi * random.random()
        givens = givens_rotation_matrix(i, j, theta, n)

        a = givens @ a @ givens.T
        b = givens @ b @ givens.T

    return a.coalesce(), b.coalesce()


@pytest.fixture
def rand_sp_spd_6x6() -> Float[Tensor, "5 5"]:
    lambdas = torch.tensor([0.5, 0.9, 3.2, 20.0, 35.2, 36.0])
    mat = rand_sp_sym_matrix(lambdas, k=10)
    return mat


@pytest.fixture
def rand_sp_spd_9x9() -> Float[Tensor, "9 9"]:
    lambdas = torch.tensor([0.1, 0.15, 0.2, 0.25, 0.28, 0.3, 41.9, 44.0, 45.0])
    mat = rand_sp_sym_matrix(lambdas, k=10)
    return mat


@pytest.fixture
def rand_sp_gep_6x6() -> tuple[Float[Tensor, "5 5"], Float[Tensor, "6 6"]]:
    lambdas = torch.tensor([0.5, 0.9, 3.2, 20.0, 35.2, 36.0])
    a, b = rand_sym_gep_matrices(lambdas, rho=0.4, k=10)
    return a, b


@pytest.fixture
def rand_sp_gep_9x9() -> tuple[Float[Tensor, "9 9"], Float[Tensor, "9 9"]]:
    lambdas = torch.tensor([0.1, 0.15, 0.2, 0.25, 0.28, 0.3, 41.9, 44.0, 45.0])
    a, b = rand_sym_gep_matrices(lambdas, rho=0.4, k=10)
    return a, b

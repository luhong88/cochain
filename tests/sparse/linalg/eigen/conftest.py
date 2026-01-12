import math
import random

import pytest
import torch as t
from jaxtyping import Float


def givens_rotation_matrix(
    i: int, j: int, theta: float, n: int
) -> Float[t.Tensor, "n n"]:
    cos = math.cos(theta)
    sin = math.sin(theta)

    idx_coo_diag = t.tile(t.arange(n), (2, 1))
    val_diag = t.ones(n, dtype=t.float64)
    val_diag[i] = cos
    val_diag[j] = cos

    idx_coo_off_diag = t.tensor([[i, j], [j, i]])
    val_off_diag = t.tensor([-sin, sin], dtype=t.float64)

    mat = t.sparse_coo_tensor(
        indices=t.hstack((idx_coo_diag, idx_coo_off_diag)),
        values=t.hstack((val_diag, val_off_diag)),
        size=(n, n),
    ).coalesce()

    return mat


def rand_sp_sym_matrix(
    lambdas: Float[t.Tensor, " n"], k: int
) -> Float[t.Tensor, "n n"]:
    n = lambdas.size(0)

    mat = t.sparse_coo_tensor(
        indices=t.tile(t.arange(n), (2, 1)),
        values=lambdas.to(dtype=t.float64),
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
    lambdas: Float[t.Tensor, " n"], rho: float, k: int
) -> tuple[Float[t.Tensor, "n n"], Float[t.Tensor, "n n"]]:
    n = lambdas.size(0)

    # Construct the diagonal eigenvalue matrix.
    diag = t.sparse_coo_tensor(
        indices=t.tile(t.arange(n), (2, 1)),
        values=lambdas.to(dtype=t.float64),
        size=(n, n),
    ).coalesce()

    # Construct the stencil matrix.
    nnz = int(n * rho)

    idx_off_diag = t.randint(0, n, (2, nnz))
    idx_diag = t.tile(t.arange(n), (2, 1))
    idx = t.hstack((idx_off_diag, idx_diag))

    val = t.hstack((t.randn(nnz, dtype=t.float64), 2.0 * t.ones(n, dtype=t.float64)))

    stencil = t.sparse_coo_tensor(idx, val, (n, n)).coalesce()

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
def rand_sp_spd_6x6() -> Float[t.Tensor, "5 5"]:
    lambdas = t.tensor([0.5, 0.9, 3.2, 20.0, 35.2, 36.0])
    mat = rand_sp_sym_matrix(lambdas, k=10)
    return mat


@pytest.fixture
def rand_sp_spd_9x9() -> Float[t.Tensor, "9 9"]:
    lambdas = t.tensor([0.1, 0.15, 0.2, 0.25, 0.28, 0.3, 41.9, 44.0, 45.0])
    mat = rand_sp_sym_matrix(lambdas, k=10)
    return mat


@pytest.fixture
def rand_sp_gep_6x6() -> tuple[Float[t.Tensor, "5 5"], Float[t.Tensor, "6 6"]]:
    lambdas = t.tensor([0.5, 0.9, 3.2, 20.0, 35.2, 36.0])
    a, b = rand_sym_gep_matrices(lambdas, rho=0.4, k=10)
    return a, b


@pytest.fixture
def rand_sp_gep_9x9() -> tuple[Float[t.Tensor, "9 9"], Float[t.Tensor, "9 9"]]:
    lambdas = t.tensor([0.1, 0.15, 0.2, 0.25, 0.28, 0.3, 41.9, 44.0, 45.0])
    a, b = rand_sym_gep_matrices(lambdas, rho=0.4, k=10)
    return a, b

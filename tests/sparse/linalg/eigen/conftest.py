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
    val_diag = t.ones(n)
    val_diag[i] = cos
    val_diag[j] = cos

    idx_coo_off_diag = t.tensor([[i, j], [j, i]])
    val_off_diag = t.tensor([-sin, sin])

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
        indices=t.tile(t.arange(n), (2, 1)), values=lambdas, size=(n, n)
    )

    for _ in range(k):
        i = random.randint(0, n - 1)
        j = random.randint(0, n - 1)
        theta = 2 * math.pi * random.random()
        givens = givens_rotation_matrix(i, j, theta, n)
        mat = givens @ mat @ givens.T

    return mat.coalesce()


def rand_sym_gep_matrices(
    lambdas: Float[t.Tensor, " n"], rho: float
) -> tuple[Float[t.Tensor, "n n"], Float[t.Tensor, "n n"]]:
    n = lambdas.size(0)

    # Construct the diagonal eigenvalue matrix.
    diag = t.sparse_coo_tensor(
        indices=t.tile(t.arange(n), (2, 1)), values=lambdas, size=(n, n)
    ).coalesce()

    # Construct the stencil matrix.
    nnz = int(n * rho)

    idx_off_diag = t.randint(0, n, (2, nnz))
    idx_diag = t.tile(t.arange(n), (2, 1))
    idx = t.hstack((idx_off_diag, idx_diag))

    val = t.hstack((t.randn(nnz), n * t.ones(n)))

    stencil = t.sparse_coo_tensor(idx, val, (n, n)).coalesce()

    # Construct A, B, such that they satisfy the GEP with the given eigenvalues.
    b = stencil.T @ stencil
    a = stencil.T @ diag @ stencil

    return a, b


@pytest.fixture
def rand_sp_spd_5x5() -> Float[t.Tensor, "5 5"]:
    lambdas = t.tensor([0.5, 3.2, 20.0, 35.2])
    mat = rand_sp_sym_matrix(lambdas, 4)
    return mat

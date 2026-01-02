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

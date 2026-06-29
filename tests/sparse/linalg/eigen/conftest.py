import math
import random

import pytest
import torch
from jaxtyping import Float
from torch import Tensor


def givens_rotation_matrix(
    i: int, j: int, theta: float, n: int
) -> Float[Tensor, "n n"]:
    """
    Compute a Givens rotation matrix.

    For a plane defined by two coordinate axes i and j, a Givens rotation
    matrix is an orthogonal matrix that represents rotations within the plane
    by an angle of θ. More specifically, the Givens rotation is an (n, n)
    identity matrix with the (i, i) and (j, j) elements replaced by cos(θ),
    the (i, j) element replaced by -sin(θ), and the (j, i) element replaced by
    sin(θ).
    """
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
    """
    Generate a random, sparse, symmetric matrix with a predefined set of eigenvalues.

    Let A be a diagonal matrix whose elements are the λ's. This function generates
    a Givens matrix G for a random coordinate plane and angle, and performs a similarity
    transformation A <- G @ A @ G.T, which preserves the eigenvalues of the original
    A, and this process is repeated k times. A larger k results in a more "scrambled",
    denser output matrix.
    """
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
    """
    Generate two matrices that satisfy a GEP with a predefined set of eigenvalues.

    Let Λ be a diagonal matrix whose diagonal values are the predefined
    eigenvalues. Define an (n, n) sparse and diagonally dominant "stencil" matrix
    S with off-diagonal values drawn from the standard normal distribution with
    fill density ρ. Then, define A = S.T @ Λ @ S and B = S.T @ S (note that B is
    symmetric positive definite). It follows that A and B satisfy the desired GEP;
    to see why, note that (S.T @ Λ @ S) y = λ (S.T @ S) y can be simplified to
    Λ x = λ x if we set x = S y and multiply both sides with inv(S.T) on the left.
    Lastly, we apply random Givens rotations to A and B simultaneously k times,
    which preserves the original eigenvalue spectrum but further scrambles the
    matrices and the eigenvectors. More specifically, the Givens rotations weaken
    the diagonal dominant structure of B (which is inherited from the construction
    of S). In addition, since Λ x = λ x, the eigenvector x's are aligned with the
    coordinate axes, and y = inv(S) x are also closely aligned with the coordinate
    axes (since inv(S) will also be diagonally dominant); the Givens rotations
    weaken this alignment.
    """
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


@pytest.fixture
def rand_sp_spd_degenerate_9x9() -> Float[Tensor, "9 9"]:
    # Includes a perfectly degenerate subspace (3 eigenvalues equal to 2.0)
    # and a near-degenerate subspace (41.9 and 41.95).
    lambdas = torch.tensor([0.1, 0.2, 2.0, 2.0, 2.0, 10.0, 41.9, 41.95, 45.0])
    mat = rand_sp_sym_matrix(lambdas, k=15)
    return mat

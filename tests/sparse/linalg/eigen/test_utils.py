import pytest
import torch

from cochain.sparse.decoupled_tensor import SparseDecoupledTensor
from cochain.sparse.linalg.eigen import (
    canonicalize_eig_vec_signs,
    grassmann_proj_dists,
    m_orthonormalize,
)


@pytest.fixture
def m():
    """Define the ambient dimension."""
    return 50


@pytest.fixture
def n():
    """Define the number of vectors."""
    return 10


@pytest.fixture
def spd_matrix_m(m):
    """Define a symmetric positive definite (SPD) metric matrix."""
    # First generate an m x m sparse, diagonally dominant matrix.
    nnz = int(m * m * 0.4)

    idx = torch.hstack(
        (torch.randint(0, m, (2, nnz)), torch.tile(torch.arange(m), (2, 1)))
    )
    val = torch.hstack((torch.randn(nnz), m * torch.ones(m))).to(dtype=torch.float64)

    a_coo = torch.sparse_coo_tensor(idx, val, (m, m)).coalesce()

    # A @ A^T + eps * I ensures strict positive definiteness.
    m_coo = a_coo @ a_coo.T + 1e-3 * torch.eye(m, dtype=a_coo.dtype).to_sparse_coo()
    m_sdt = SparseDecoupledTensor.from_tensor(m_coo)

    return m_sdt


@pytest.fixture
def v_dense(m, n):
    """Define a standard random dense basis."""
    return torch.randn(m, n, dtype=torch.float64)


@pytest.fixture
def v_rank_deficient(v_dense):
    """Define a matrix with explicitly linearly dependent columns."""
    v = v_dense.clone()
    # Make column 2 a scalar multiple of column 0.
    v[:, 2] = 3.5 * v[:, 0]
    # Make column 4 identically zero.
    v[:, 4] = 0.0
    return v


@pytest.fixture
def v_ill_conditioned(v_dense):
    """Define a matrix where column norms span wildly different magnitudes."""
    v = v_dense.clone()
    v[:, 0] *= 1e6
    v[:, 1] *= 1e-6
    return v


def test_m_orthonormalize_strict_orthogonality(v_dense, spd_matrix_m, n):
    """Check that V^T@M@V = I for well-conditioned inputs."""
    v_ortho = m_orthonormalize(v_dense, spd_matrix_m)

    # Check shape.
    assert v_ortho.shape == v_dense.shape

    # Check M-orthonormality.
    identity_approx = v_ortho.T @ (spd_matrix_m @ v_ortho)
    identity_exact = torch.eye(n, dtype=v_ortho.dtype, device=v_ortho.device)

    torch.testing.assert_close(identity_approx, identity_exact)


def test_m_orthonormalize_rank_adaptivity(v_rank_deficient, spd_matrix_m, n):
    """Check that redundant basis vectors are dropped."""
    v_ortho = m_orthonormalize(v_rank_deficient, spd_matrix_m)

    # We introduced 2 linear dependencies, so rank should drop by 2.
    expected_cols = n - 2
    assert v_ortho.shape[1] == expected_cols

    # The remaining columns must still be exactly M-orthonormal.
    identity_approx = v_ortho.T @ (spd_matrix_m @ v_ortho)
    identity_exact = torch.eye(expected_cols, dtype=v_ortho.dtype)
    torch.testing.assert_close(identity_approx, identity_exact)


def test_m_orthonormalize_soft_restart(v_rank_deficient, spd_matrix_m, n):
    """Check that random vectors are padded correctly when falling below n_min."""
    n_min = n  # We want the original number of vectors back.
    generator = torch.Generator().manual_seed(123)

    v_ortho = m_orthonormalize(
        v_rank_deficient, spd_matrix_m, n_min=n_min, generator=generator
    )

    # Output must strictly meet n_min.
    assert v_ortho.shape[1] == n_min

    # The entire padded subspace must be M-orthonormal.
    identity_approx = v_ortho.T @ (spd_matrix_m @ v_ortho)
    identity_exact = torch.eye(n_min, dtype=v_ortho.dtype)
    torch.testing.assert_close(identity_approx, identity_exact)


def test_m_orthonormalize_ill_conditioned(v_ill_conditioned, spd_matrix_m, n):
    """Check Jacobi preconditioning handles extreme magnitude differences."""
    v_ortho = m_orthonormalize(v_ill_conditioned, spd_matrix_m)

    identity_approx = v_ortho.T @ (spd_matrix_m @ v_ortho)
    identity_exact = torch.eye(n, dtype=v_ortho.dtype)

    torch.testing.assert_close(identity_approx, identity_exact)


def test_canonicalize_eig_vec_signs_deterministic():
    """Ensure the maximum absolute element in every column is strictly positive."""
    # Create vectors with known negative maximum absolute values.
    v = torch.tensor([[-5.0, 1.0], [2.0, -8.0], [1.0, 3.0]])

    v_canon = canonicalize_eig_vec_signs(v)

    # The max absolute values are at index 0 for col 0, and index 1 for col 1.
    # They should have been flipped to positive.
    assert v_canon[0, 0] == 5.0
    assert v_canon[1, 1] == 8.0


def test_canonicalize_eig_vec_signs_idempotent():
    """Ensure canonicalizing an already canonicalized matrix changes nothing."""
    v = torch.randn(10, 5)
    v_canon_first = canonicalize_eig_vec_signs(v)
    v_canon_second = canonicalize_eig_vec_signs(v_canon_first)

    torch.testing.assert_close(v_canon_first, v_canon_second)


def test_grassmann_proj_dists_identity(v_dense, spd_matrix_m):
    """Distance between a subspace and itself must be zero."""
    # Must use M-orthonormal inputs for Grassmann distance to be valid.
    v_ortho = m_orthonormalize(v_dense, spd_matrix_m)

    dist_pairwise = grassmann_proj_dists(
        v_ortho, v_ortho, spd_matrix_m, mode="pairwise"
    )
    dist_subspace = grassmann_proj_dists(
        v_ortho, v_ortho, spd_matrix_m, mode="subspace"
    )

    torch.testing.assert_close(dist_pairwise, torch.zeros_like(dist_pairwise))
    torch.testing.assert_close(dist_subspace, torch.zeros_like(dist_subspace))


def test_grassmann_proj_dists_subspace_invariance(v_dense, spd_matrix_m, n):
    """Subspace mode must be invariant to intra-subspace rotations."""
    v_true = m_orthonormalize(v_dense, spd_matrix_m)

    # Deterministically scramble the internal basis by shifting columns by 1.
    # The spanned subspace is identical, but v_pred[:, i] is now orthogonal to v_true[:, i].
    v_pred = torch.roll(v_true, shifts=1, dims=1)

    # Subspace distance should remain 0.
    dist_subspace = grassmann_proj_dists(v_pred, v_true, spd_matrix_m, mode="subspace")
    torch.testing.assert_close(dist_subspace, torch.zeros_like(dist_subspace))

    # Pairwise distance should be exactly 1.0 for all pairs, because orthogonal
    # vectors have a chordal distance of 1.
    dist_pairwise = grassmann_proj_dists(v_pred, v_true, spd_matrix_m, mode="pairwise")
    expected_pairwise = torch.ones_like(dist_pairwise)
    torch.testing.assert_close(dist_pairwise, expected_pairwise)


def test_grassmann_proj_dists_maximal_orthogonal(m, n, spd_matrix_m):
    """Mutually M-orthogonal subspaces should yield a maximal distance of k."""
    # Generate an overcomplete basis and orthonormalize to get 2n orthogonal vectors.
    v_large = torch.randn(m, n * 2, dtype=torch.float64)
    v_ortho_large = m_orthonormalize(v_large, spd_matrix_m)

    # Split into two completely disjoint M-orthogonal subspaces.
    v_true = v_ortho_large[:, :n]
    v_pred = v_ortho_large[:, n:]

    dist_subspace = grassmann_proj_dists(v_pred, v_true, spd_matrix_m, mode="subspace")

    # Distance should be exactly k (dim_n)
    expected_dist = torch.tensor(n, dtype=torch.float64)
    torch.testing.assert_close(dist_subspace, expected_dist)

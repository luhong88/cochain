import pytest
import torch

from cochain.sparse.decoupled_tensor import DiagDecoupledTensor, SparseDecoupledTensor
from cochain.sparse.linalg.eigen import (
    canonicalize_eig_vec_signs,
    grassmann_proj_dists,
    m_orthonormalize,
)
from cochain.sparse.linalg.eigen.base.utils import _m_orthonormalize_one_iter


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


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_m_orthonormalize_extreme_orthogonal_column_scales(dtype, device):
    """Preserve tiny and huge directions without squaring their original scales."""
    scales = torch.tensor(
        [1e-30, -1e-20, 1e-8, -1e20, 1e30]
        if dtype == torch.float32
        else [1e-310, -1e-200, 1e-8, -1e200, 1e300],
        dtype=dtype,
        device=device,
    )
    basis = torch.eye(8, dtype=dtype, device=device)[:, :5]
    v = basis * scales
    metric = DiagDecoupledTensor(torch.ones(8, dtype=dtype, device=device)).to_sdt()

    # Exercise both the helper in its input precision and the public refinement.
    one_iter, cond = _m_orthonormalize_one_iter(v, metric)
    refined = m_orthonormalize(v, metric)
    torch.testing.assert_close(cond, cond.new_tensor(1.0))
    for u in (one_iter, refined):
        assert u.shape == v.shape
        assert u.dtype == dtype
        assert u.device == v.device
        assert torch.isfinite(u).all()
        torch.testing.assert_close(
            u.T @ (metric @ u), torch.eye(5, device=device, dtype=dtype)
        )
        torch.testing.assert_close(u @ u.T, basis @ basis.T)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("rank_deficient", [False, True])
def test_m_orthonormalize_column_rescaling_invariance(
    v_dense, v_rank_deficient, spd_matrix_m, dtype, rank_deficient, device
):
    """Independent signed rescaling preserves rank, span, and metric orthonormality."""
    v = (v_rank_deficient if rank_deficient else v_dense).to(device=device, dtype=dtype)
    metric = spd_matrix_m.to(device=device, dtype=dtype)
    exponent = 30 if dtype == torch.float32 else 300
    scales = torch.logspace(-exponent, exponent, v.size(1), dtype=dtype, device=device)
    scales[::2] *= -1

    reference = m_orthonormalize(v, metric)
    u = m_orthonormalize(v * scales, metric)
    expected_rank = v.size(1) - (2 if rank_deficient else 0)
    assert u.shape == (v.size(0), expected_rank)
    identity = torch.eye(expected_rank, dtype=dtype, device=device)
    torch.testing.assert_close(u.T @ (metric @ u), identity)

    # Unit singular values of the metric overlap establish identical subspaces.
    overlap = reference.T @ (metric @ u)
    torch.testing.assert_close(overlap @ overlap.T, identity)


@pytest.mark.parametrize("scales", [(1.0, 1.0), (1e-200, -1e200)])
def test_m_orthonormalize_relative_rank_tolerance(scales, device):
    """Near dependence is detected after normalization, independently of amplitude."""
    v = torch.tensor(
        [[1.0, 1.0], [0.0, 1e-4], [0.0, 0.0]], dtype=torch.float64, device=device
    )
    metric = DiagDecoupledTensor(torch.ones(3, dtype=v.dtype, device=device)).to_sdt()
    v *= v.new_tensor(scales)

    dependent = m_orthonormalize(v, metric, rtol=1e-6)
    independent = m_orthonormalize(v, metric, rtol=1e-10)
    assert dependent.shape == (3, 1)
    assert independent.shape == (3, 2)
    torch.testing.assert_close(dependent.T @ (metric @ dependent), v.new_ones((1, 1)))
    torch.testing.assert_close(
        independent.T @ (metric @ independent),
        torch.eye(2, dtype=v.dtype, device=device),
    )


def test_m_orthonormalize_no_retained_eigenvalues(device):
    """An explicit rank tolerance can discard all otherwise valid directions."""
    v = torch.eye(3, dtype=torch.float64, device=device)
    metric = DiagDecoupledTensor(torch.ones(3, dtype=v.dtype, device=device)).to_sdt()
    u, cond = _m_orthonormalize_one_iter(v, metric, rtol=1.0)
    assert u.shape == (3, 0)
    torch.testing.assert_close(cond, cond.new_tensor(0.0))


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), -float("inf")])
def test_m_orthonormalize_nonfinite_columns(invalid, device):
    """Nonfinite columns are rejected rather than hidden by rank truncation."""
    v = torch.eye(3, dtype=torch.float64, device=device)
    v[0, 1] = invalid
    metric = DiagDecoupledTensor(torch.ones(3, dtype=v.dtype, device=device)).to_sdt()
    with pytest.raises(ValueError, match="finite entries"):
        m_orthonormalize(v, metric)


@pytest.mark.parametrize("metric_value", [0.0, -1.0, float("nan"), float("inf"), 1e308])
def test_m_orthonormalize_invalid_metric_norms(metric_value, device):
    """Reject zero, negative, nonfinite, or overflowing computed metric norms."""
    v = torch.ones((2, 1), dtype=torch.float64, device=device)
    metric = DiagDecoupledTensor(v.new_ones(2)).to_sdt()
    # Simulate invalid values introduced after the tensor constructor's checks.
    metric.values.fill_(metric_value)
    with pytest.raises(ValueError, match="strictly positive squared metric norms"):
        _m_orthonormalize_one_iter(v, metric)
    # The public API's precision conversion can reject nonfinite metric values
    # in the sparse tensor constructor before invoking the helper.
    with pytest.raises(
        ValueError,
        match="strictly positive squared metric norms|values contain NaN or Inf",
    ):
        m_orthonormalize(v, metric)


def test_m_orthonormalize_nonfinite_normalized_gram(device):
    """Report a nonfinite normalized Gram even when individual norms are valid."""
    v = torch.eye(2, dtype=torch.float64, device=device)
    # This indefinite metric has positive diagonal entries, but normalization
    # overflows the off-diagonal Gram entries. Positive column norms alone do
    # not establish that the supplied metric is positive definite.
    metric_dense = v.new_tensor([[1e-200, 1e200], [1e200, 1e-200]])
    metric = SparseDecoupledTensor.from_tensor(metric_dense.to_sparse_coo())
    with pytest.raises(ValueError, match="normalized Gram matrix must be finite"):
        m_orthonormalize(v, metric)


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

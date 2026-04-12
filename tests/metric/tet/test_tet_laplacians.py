from functools import partial

import pytest
import torch

from cochain.complex import SimplicialMesh
from cochain.metric.tet import tet_hodge_stars, tet_laplacians, tet_masses
from cochain.metric.tet.tet_laplacians import MixedWeakLaplacianBlocks
from cochain.sparse.linalg.solvers import SuperLU


@pytest.mark.parametrize(
    "weak_laplacian, betti",
    [
        (partial(tet_laplacians.weak_laplacian_0, method="cotan"), 1),
        (partial(tet_laplacians.weak_laplacian_0, method="consistent"), 1),
        (tet_laplacians.weak_laplacian_1, 0),
        (partial(tet_laplacians.weak_laplacian_2, method="dense"), 0),
        (partial(tet_laplacians.weak_laplacian_2, method="inv_star"), 0),
        (partial(tet_laplacians.weak_laplacian_3, method="dense"), 0),
        (partial(tet_laplacians.weak_laplacian_3, method="inv_star"), 0),
    ],
)
def test_sphere_homology_group_dims(
    weak_laplacian, betti, two_tets_mesh: SimplicialMesh, device
):
    mesh = two_tets_mesh.to(device)

    operator = weak_laplacian(mesh).to_dense()
    dim_ker = operator.shape[0] - torch.linalg.matrix_rank(operator)

    torch.testing.assert_close(dim_ker, torch.tensor(betti, device=device))


@pytest.mark.parametrize(
    "weak_laplacian, betti",
    [
        (partial(tet_laplacians.weak_laplacian_0, method="cotan"), 1),
        (partial(tet_laplacians.weak_laplacian_0, method="consistent"), 1),
        (tet_laplacians.weak_laplacian_1, 1),
        (partial(tet_laplacians.weak_laplacian_2, method="dense"), 0),
        (partial(tet_laplacians.weak_laplacian_2, method="inv_star"), 0),
        (partial(tet_laplacians.weak_laplacian_3, method="dense"), 0),
        (partial(tet_laplacians.weak_laplacian_3, method="inv_star"), 0),
    ],
)
def test_torus_homology_group_dims(
    weak_laplacian, betti, solid_torus_mesh: SimplicialMesh, device
):
    mesh = solid_torus_mesh.to(device)

    operator = weak_laplacian(mesh).to_dense()
    dim_ker = operator.shape[0] - torch.linalg.matrix_rank(operator)

    torch.testing.assert_close(dim_ker, torch.tensor(betti, device=device))


def test_laplacian_0_equivalence(two_tets_mesh: SimplicialMesh, device):
    """
    Check consistency in weak 0-Laplacian construction methods.

    Check that the weak 0-Laplacians constructed using the cotan formula and
    the mass-1 matrix are equivalent (for a well-centered and Delaunay mesh)
    """
    mesh = two_tets_mesh.to(device)

    l0_cotan = tet_laplacians.weak_laplacian_0(mesh, method="cotan").to_dense()
    l0_consistent = tet_laplacians.weak_laplacian_0(
        mesh, method="consistent"
    ).to_dense()

    torch.testing.assert_close(l0_cotan, l0_consistent)


@pytest.mark.parametrize(
    "weak_laplacian",
    [
        partial(tet_laplacians.weak_laplacian_0, method="cotan"),
        partial(tet_laplacians.weak_laplacian_0, method="consistent"),
        tet_laplacians.weak_laplacian_1,
        partial(tet_laplacians.weak_laplacian_2, method="dense"),
        partial(tet_laplacians.weak_laplacian_2, method="inv_star"),
        partial(tet_laplacians.weak_laplacian_3, method="dense"),
        partial(tet_laplacians.weak_laplacian_3, method="inv_star"),
    ],
)
def test_laplacian_symmetry(weak_laplacian, two_tets_mesh: SimplicialMesh, device):
    """Test that the weak Laplacians are symmetric."""
    mesh = two_tets_mesh.to(device)

    weak_laplacian_i = weak_laplacian(mesh)
    weak_laplacian_i_T = weak_laplacian_i.T
    torch.testing.assert_close(
        weak_laplacian_i.to_dense(), weak_laplacian_i_T.to_dense()
    )


@pytest.mark.parametrize(
    "weak_laplacian",
    [
        partial(tet_laplacians.weak_laplacian_0, method="cotan"),
        partial(tet_laplacians.weak_laplacian_0, method="consistent"),
        tet_laplacians.weak_laplacian_1,
        partial(tet_laplacians.weak_laplacian_2, method="dense"),
        partial(tet_laplacians.weak_laplacian_2, method="inv_star"),
        partial(tet_laplacians.weak_laplacian_3, method="dense"),
        partial(tet_laplacians.weak_laplacian_3, method="inv_star"),
    ],
)
def test_laplacian_PSD(weak_laplacian, two_tets_mesh: SimplicialMesh, device):
    """Test that the stiffness matrices are positive semi-definite."""
    mesh = two_tets_mesh.to(device)

    laplacian_i = weak_laplacian(mesh).to_dense()
    eigs = torch.linalg.eigvalsh(laplacian_i)

    assert eigs.min() >= -1e-6


@pytest.mark.parametrize(
    "weak_laplacian",
    [
        partial(tet_laplacians.weak_laplacian_0, method="cotan"),
        partial(tet_laplacians.weak_laplacian_0, method="consistent"),
    ],
)
def test_laplacian_0_kernel(weak_laplacian, two_tets_mesh: SimplicialMesh, device):
    mesh = two_tets_mesh.to(device)

    l0 = weak_laplacian(mesh)
    row_sum = l0.to_dense().sum(dim=-1)

    torch.testing.assert_close(row_sum, torch.zeros_like(row_sum))


@pytest.mark.parametrize(
    "weak_laplacian",
    [
        tet_laplacians.weak_laplacian_2_curl_curl,
        tet_laplacians.weak_laplacian_2,
        tet_laplacians.weak_laplacian_3,
    ],
)
def test_mixed_formulation_forward_pass(
    weak_laplacian, solid_torus_mesh: SimplicialMesh, device
):
    # Double precision is required for this test to pass.
    mesh = solid_torus_mesh.to(dtype=torch.float64, device=device)

    l_dense = weak_laplacian(mesh, method="dense")
    l_block: MixedWeakLaplacianBlocks = weak_laplacian(mesh, method="mixed")

    x = torch.randn(l_dense.size(0), dtype=mesh.dtype, device=device)

    b_dense = l_dense @ x

    lhs, rhs = l_block.get_codiff_system(x)
    solver = SuperLU(lhs, backend="scipy")
    y = solver(rhs)
    b_mix = l_block.get_forward_pass(x, y)

    torch.testing.assert_close(b_mix, b_dense)


@pytest.mark.parametrize(
    "weak_laplacian",
    [
        tet_laplacians.weak_laplacian_2,
        tet_laplacians.weak_laplacian_3,
    ],
)
def test_mixed_formulation_linear_solve(
    weak_laplacian, solid_torus_mesh: SimplicialMesh, device
):
    mesh = solid_torus_mesh.to(device)

    l_dense = weak_laplacian(mesh, method="dense")
    l_block: MixedWeakLaplacianBlocks = weak_laplacian(mesh, method="mixed")

    b = torch.randn(l_dense.size(0), dtype=mesh.dtype, device=device)

    x_dense = torch.linalg.solve(l_dense, b)

    l_mix, b_mix = l_block.get_full_system(b)
    solver = SuperLU(l_mix, backend="scipy")
    x_full = solver(b_mix)
    x_mix, _ = l_block.unpack_mixed_cochain(x_full)

    torch.testing.assert_close(x_mix, x_dense)


# TODO: improve mixed formulation GEP test coverage.
@pytest.mark.parametrize(
    "weak_laplacian",
    [
        tet_laplacians.weak_laplacian_2,
        tet_laplacians.weak_laplacian_3,
    ],
)
def test_mixed_formulation_gep_smoke(
    weak_laplacian, two_tets_mesh: SimplicialMesh, device
):
    mesh = two_tets_mesh.to(device)
    l_block: MixedWeakLaplacianBlocks = weak_laplacian(mesh, method="mixed")
    l_block.get_gep()


@pytest.mark.parametrize(
    "down, up, mass",
    [
        (
            tet_laplacians.weak_laplacian_1_grad_div,
            tet_laplacians.weak_laplacian_1_curl_curl,
            tet_masses.mass_1,
        ),
        (
            partial(tet_laplacians.weak_laplacian_2_curl_curl, method="dense"),
            tet_laplacians.weak_laplacian_2_grad_div,
            tet_masses.mass_2,
        ),
    ],
)
def test_laplacian_orthogonality(down, up, mass, two_tets_mesh: SimplicialMesh, device):
    """Test that composing the up and down k-Laplacian gives zero."""
    mesh = two_tets_mesh.to(dtype=torch.float64, device=device)

    l_down = down(mesh).to_dense()
    l_up = up(mesh).to_dense()

    m = mass(mesh).to_dense()

    composition_1 = l_down @ torch.linalg.solve(m, l_up)  # dg @ inv_m @ cc
    composition_2 = l_up @ torch.linalg.solve(m, l_down)  # cc @ inv_m @ dg

    # For these tests, the numerical tolerance needs to be more lenient, since
    # the calculation involves a long chain of matrix multiplications and effectively
    # two matrix inverses.
    torch.testing.assert_close(composition_1, torch.zeros_like(composition_1))
    torch.testing.assert_close(composition_2, torch.zeros_like(composition_2))


def test_laplacian_1_curl_free(two_tets_mesh: SimplicialMesh, device):
    """The curl-curl 1-Laplacian annihilates a curl-free 1-cochain."""
    mesh = two_tets_mesh.to(device)

    l1_curl_curl = tet_laplacians.weak_laplacian_1_curl_curl(mesh)

    d0 = mesh.cbd[0]
    x0 = torch.randn(mesh.n_verts, dtype=mesh.dtype, device=mesh.device)
    # A gradient field is irrotational.
    x1_irrotational = d0 @ x0

    x1_zero = (l1_curl_curl @ x1_irrotational).to_dense()

    torch.testing.assert_close(x1_zero, torch.zeros_like(x1_zero))


def test_laplacian_1_div_free(two_tets_mesh: SimplicialMesh, device):
    """The grad-div 1-Laplacian annihilates a div-free 1-cochain."""
    mesh = two_tets_mesh.to(device)

    l1_grad_div = tet_laplacians.weak_laplacian_1_grad_div(mesh)

    d1_T = mesh.cbd[1].T.to_dense()
    m1 = tet_masses.mass_1(mesh).to_dense()

    x2 = torch.randn(mesh.n_tris, dtype=mesh.dtype, device=mesh.device)
    # Generate a solenoidal 1-cochain as δ_2 @ x2 = inv_m1 @ d1_T @ m2 @ x2
    # ignore m2 since it gets absorbed by the random x2.
    x1_solenoidal = torch.linalg.solve(m1, d1_T @ x2)

    # The codifferentials satisfy the exactness relation δ_1 ∘ δ_2 = 0.
    x1_zero = (l1_grad_div @ x1_solenoidal).to_dense()

    torch.testing.assert_close(x1_zero, torch.zeros_like(x1_zero))


def test_laplacian_2_curl_free(two_tets_mesh: SimplicialMesh, device):
    """The curl-curl 2-Laplacian annihilates a curl-free 2-cochain."""
    mesh = two_tets_mesh.to(device)

    l2_curl_curl = tet_laplacians.weak_laplacian_2_curl_curl(mesh, method="dense")

    d2_T = mesh.cbd[2].T.to_dense()
    m2 = tet_masses.mass_2(mesh).to_dense()

    x3 = torch.randn(mesh.n_tets, dtype=mesh.dtype, device=mesh.device)
    # Generate a solenoidal 2-cochain as δ_3 @ x3 = inv_m2 @ d2_T @ m3 @ x3
    # ignore m3 since it gets absorbed by the random x3.
    x2_irrotational = torch.linalg.solve(m2, d2_T @ x3)

    # The codifferentials satisfy the exactness relation δ_2 ∘ δ_3 = 0.
    x2_zero = (l2_curl_curl @ x2_irrotational).to_dense()

    torch.testing.assert_close(x2_zero, torch.zeros_like(x2_zero))


def test_laplacian_2_div_free(two_tets_mesh: SimplicialMesh, device):
    """The grad-div component of the 2-Laplacian annihilates a div-free 2-cochain."""
    mesh = two_tets_mesh.to(device)

    l2_grad_div = tet_laplacians.weak_laplacian_2_grad_div(mesh)

    d1 = mesh.cbd[1]
    x1 = torch.randn(mesh.n_edges, dtype=mesh.dtype, device=mesh.device)

    # The curl of a vector field is divergence-free.
    x2_solenoidal = d1 @ x1
    x2_zero = (l2_grad_div @ x2_solenoidal).to_dense()

    torch.testing.assert_close(x2_zero, torch.zeros_like(x2_zero))


def test_codiff_1_adjoint_relation(two_tets_mesh: SimplicialMesh, device):
    """Check that the 1-codiff and the 0-cbd are adjoints."""
    mesh = two_tets_mesh.to(device)

    m0 = tet_hodge_stars.star_0(mesh)
    inv_m0 = m0.inv
    m1 = tet_masses.mass_1(mesh)

    d0 = mesh.cbd[0]
    d0_T = d0.T

    codiff_1 = inv_m0 @ d0_T @ m1

    x0 = torch.randn(mesh.n_verts, dtype=mesh.dtype, device=mesh.device)
    x1 = torch.randn(mesh.n_edges, dtype=mesh.dtype, device=mesh.device)

    dot_1 = torch.dot(d0 @ x0, m1 @ x1)
    dot_2 = torch.dot(x0, m0 @ (codiff_1 @ x1))

    torch.testing.assert_close(dot_1, dot_2)


def test_codiff_2_adjoint_relation(two_tets_mesh: SimplicialMesh, device):
    """Check that the 2-codiff and the 1-cbd are adjoints."""
    mesh = two_tets_mesh.to(device)

    m1 = tet_masses.mass_1(mesh).to_dense()
    m2 = tet_masses.mass_2(mesh).to_dense()

    d1 = mesh.cbd[1].to_dense()
    d1_T = d1.T

    codiff_2 = torch.linalg.solve(m1, d1_T @ m2)  # inv_m1 @ d1_T @ m2

    x1 = torch.randn(mesh.n_edges, dtype=mesh.dtype, device=mesh.device)
    x2 = torch.randn(mesh.n_tris, dtype=mesh.dtype, device=mesh.device)

    dot_1 = torch.dot(d1 @ x1, m2 @ x2)
    dot_2 = torch.dot(x1, m1 @ (codiff_2 @ x2))

    torch.testing.assert_close(dot_1, dot_2)


def test_codiff_3_adjoint_relation(two_tets_mesh: SimplicialMesh, device):
    """Check that the 3-codiff and the 2-cbd operators are adjoints."""
    mesh = two_tets_mesh.to(device)

    m2 = tet_masses.mass_2(mesh).to_dense()
    m3 = tet_masses.mass_3(mesh).to_dense()

    d2 = mesh.cbd[2].to_dense()
    d2_T = d2.T

    codiff_3 = torch.linalg.solve(m2, d2_T @ m3)  # inv_m2 @ d2_T @ m3

    x2 = torch.randn(mesh.n_tris, dtype=mesh.dtype, device=mesh.device)
    x3 = torch.randn(mesh.n_tets, dtype=mesh.dtype, device=mesh.device)

    dot_1 = torch.dot(d2 @ x2, m3 @ x3)
    dot_2 = torch.dot(x2, m2 @ (codiff_3 @ x3))

    torch.testing.assert_close(dot_1, dot_2)


# TODO: add rotation/translation/scaling invariance tests

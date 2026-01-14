from functools import partial

import pytest
import torch as t

from cochain.complex import SimplicialComplex
from cochain.geometry.tet import tet_laplacians, tet_masses


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
    weak_laplacian, betti, two_tets_mesh: SimplicialComplex
):
    operator = weak_laplacian(two_tets_mesh).to_dense()
    dim_ker = operator.shape[0] - t.linalg.matrix_rank(operator)
    t.testing.assert_close(dim_ker, t.tensor(betti))


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
    weak_laplacian, betti, solid_torus_mesh: SimplicialComplex
):
    operator = weak_laplacian(solid_torus_mesh).to_dense()
    dim_ker = operator.shape[0] - t.linalg.matrix_rank(operator)
    t.testing.assert_close(dim_ker, t.tensor(betti))


def test_laplacian_0_equivalence(two_tets_mesh: SimplicialComplex):
    """
    Check that the weak 0-Laplacians constructed using the cotan formula and
    the mass-1 matrix are equivalent (for a well-centered and Delaunay mesh)
    """
    l0_cotan = tet_laplacians.weak_laplacian_0(two_tets_mesh, method="cotan").to_dense()
    l0_consistent = tet_laplacians.weak_laplacian_0(
        two_tets_mesh, method="consistent"
    ).to_dense()

    t.testing.assert_close(l0_cotan, l0_consistent)


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
def test_laplacian_symmetry(weak_laplacian, two_tets_mesh: SimplicialComplex):
    """
    Test that the weak Laplacians are symmetric.
    """
    weak_laplacian_i = weak_laplacian(two_tets_mesh)
    weak_laplacian_i_T = weak_laplacian_i.T
    t.testing.assert_close(weak_laplacian_i.to_dense(), weak_laplacian_i_T.to_dense())


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
def test_laplacian_PSD(weak_laplacian, two_tets_mesh: SimplicialComplex):
    """
    Test that the stiffness matrices are positive semi-definite.
    """
    laplacian_i = weak_laplacian(two_tets_mesh).to_dense()

    eigs = t.linalg.eigvalsh(laplacian_i)
    assert eigs.min() >= -1e-6


@pytest.mark.parametrize(
    "weak_laplacian",
    [
        partial(tet_laplacians.weak_laplacian_0, method="cotan"),
        partial(tet_laplacians.weak_laplacian_0, method="consistent"),
    ],
)
def test_laplacian_0_kernel(weak_laplacian, two_tets_mesh: SimplicialComplex):
    l0 = weak_laplacian(two_tets_mesh)
    row_sum = l0.to_dense().sum(dim=-1)
    t.testing.assert_close(row_sum, t.zeros_like(row_sum))


# TODO: update to use custom solver wrapper
@pytest.mark.parametrize(
    "div_grad, curl_curl, mass",
    [
        (
            tet_laplacians.weak_laplacian_1_div_grad,
            tet_laplacians.weak_laplacian_1_curl_curl,
            tet_masses.mass_1,
        ),
        (
            tet_laplacians.weak_laplacian_2_div_grad,
            partial(tet_laplacians.weak_laplacian_2_curl_curl, method="dense"),
            tet_masses.mass_2,
        ),
    ],
)
def test_laplacian_orthogonality(
    div_grad, curl_curl, mass, two_tets_mesh: SimplicialComplex
):
    dg = div_grad(two_tets_mesh).to_dense()
    cc = curl_curl(two_tets_mesh).to_dense()

    m = mass(two_tets_mesh).to_dense()

    composition_1 = dg @ t.linalg.solve(m, cc)  # dg @ inv_m @ cc
    composition_2 = cc @ t.linalg.solve(m, dg)  # cc @ inv_m @ dg

    # For these tests, the numerical tolerance needs to be more lenient, since
    # the calculation involves a long chain of matrix multiplications and effectively
    # two matrix inverses.
    t.testing.assert_close(
        composition_1, t.zeros_like(composition_1), atol=1e-4, rtol=0
    )
    t.testing.assert_close(
        composition_2, t.zeros_like(composition_2), atol=1e-4, rtol=0
    )


def test_laplacian_1_curl_free(two_tets_mesh: SimplicialComplex):
    """
    The curl curl component of the 1-Laplacian acting on a curl-free 1-cochain/
    1-form produces 0.
    """
    l1_curl_curl = tet_laplacians.weak_laplacian_1_curl_curl(two_tets_mesh)

    d0 = two_tets_mesh.coboundary[0]
    x0 = t.arange(two_tets_mesh.n_verts).to(
        dtype=t.float, device=two_tets_mesh.vert_coords.device
    )
    x1_curl_free = d0 @ x0

    x1_zero = (l1_curl_curl @ x1_curl_free).to_dense()

    t.testing.assert_close(x1_zero, t.zeros_like(x1_zero))


# TODO: update to use custom solver wrapper
def test_laplacian_1_div_free(two_tets_mesh: SimplicialComplex):
    """
    The div grad component of the 1-Laplacian acting on a div-free 1-cochain/
    1-form produces 0.
    """
    l1_div_grad = tet_laplacians.weak_laplacian_1_div_grad(two_tets_mesh)

    d1_T = two_tets_mesh.coboundary[1].T.to_dense()

    m1 = tet_masses.mass_1(two_tets_mesh).to_dense()

    x2 = t.arange(two_tets_mesh.n_tris).to(
        dtype=t.float, device=two_tets_mesh.vert_coords.device
    )
    x1_div_free = t.linalg.solve(m1, d1_T) @ x2  # inv_m1 @ d1_T @ x2

    x1_zero = (l1_div_grad @ x1_div_free).to_dense()

    t.testing.assert_close(x1_zero, t.zeros_like(x1_zero))


def test_laplacian_2_curl_free(two_tets_mesh: SimplicialComplex):
    """
    The curl curl component of the 2-Laplacian acting on a curl-free 2-cochain/
    2-form produces 0.
    """
    l2_curl_curl = tet_laplacians.weak_laplacian_2_curl_curl(
        two_tets_mesh, method="dense"
    )

    d2_T = two_tets_mesh.coboundary[2].T.to_dense()

    m2 = tet_masses.mass_2(two_tets_mesh).to_dense()

    x3 = t.arange(two_tets_mesh.n_tets).to(
        dtype=t.float, device=two_tets_mesh.vert_coords.device
    )
    x2_curl_free = t.linalg.solve(m2, d2_T) @ x3  # inv_m2 @ d2_T @ x3

    x2_zero = (l2_curl_curl @ x2_curl_free).to_dense()

    t.testing.assert_close(x2_zero, t.zeros_like(x2_zero))


# TODO: update to use custom solver wrapper
def test_laplacian_2_div_free(two_tets_mesh: SimplicialComplex):
    """
    The div grad component of the 2-Laplacian acting on a div-free 2-cochain/
    2-form produces 0.
    """
    l2_div_grad = tet_laplacians.weak_laplacian_2_div_grad(two_tets_mesh)

    d1 = two_tets_mesh.coboundary[1]
    x1 = t.arange(two_tets_mesh.n_edges).to(
        dtype=t.float, device=two_tets_mesh.vert_coords.device
    )
    x2_div_free = d1 @ x1

    x2_zero = (l2_div_grad @ x2_div_free).to_dense()

    t.testing.assert_close(x2_zero, t.zeros_like(x2_zero))


def test_codiff_1_adjoint_relation(two_tets_mesh: SimplicialComplex):
    """
    Check that the 1-codifferential and the 0-coboundary operators are adjoints
    with respect to the mass matrix-weighted inner product.
    """
    m0 = tet_masses.mass_0(two_tets_mesh)
    inv_m0 = m0.inv

    m1 = tet_masses.mass_1(two_tets_mesh)

    d0 = two_tets_mesh.coboundary[0]
    d0_T = d0.T

    codiff_1 = inv_m0 @ d0_T @ m1

    x0 = t.arange(two_tets_mesh.n_verts).to(
        dtype=t.float, device=two_tets_mesh.vert_coords.device
    )
    x1 = t.arange(two_tets_mesh.n_edges).to(
        dtype=t.float, device=two_tets_mesh.vert_coords.device
    )

    dot_1 = t.dot(d0 @ x0, m1 @ x1)
    dot_2 = t.dot(x0, m0 @ (codiff_1 @ x1))

    t.testing.assert_close(dot_1, dot_2)


# TODO: update to use custom solver wrapper
def test_codiff_2_adjoint_relation(two_tets_mesh: SimplicialComplex):
    """
    Check that the 2-codifferential and the 1-coboundary operators are adjoints
    with respect to the mass matrix-weighted inner product.
    """
    m1 = tet_masses.mass_1(two_tets_mesh).to_dense()
    m2 = tet_masses.mass_2(two_tets_mesh).to_dense()

    d1 = two_tets_mesh.coboundary[1].to_dense()
    d1_T = d1.transpose(0, 1)

    codiff_2 = t.linalg.solve(m1, d1_T) @ m2  # inv_m1 @ d1_T @ m2

    x1 = t.arange(two_tets_mesh.n_edges).to(
        dtype=t.float, device=two_tets_mesh.vert_coords.device
    )
    x2 = t.arange(two_tets_mesh.n_tris).to(
        dtype=t.float, device=two_tets_mesh.vert_coords.device
    )

    dot_1 = t.dot(d1 @ x1, m2 @ x2)
    dot_2 = t.dot(x1, m1 @ (codiff_2 @ x2))

    t.testing.assert_close(dot_1, dot_2)


# TODO: update to use custom solver wrapper
def test_codiff_3_adjoint_relation(two_tets_mesh: SimplicialComplex):
    """
    Check that the 3-codifferential and the 2-coboundary operators are adjoints
    with respect to the mass matrix-weighted inner product.
    """
    m2 = tet_masses.mass_2(two_tets_mesh).to_dense()
    m3 = tet_masses.mass_3(two_tets_mesh).to_dense()

    d2 = two_tets_mesh.coboundary[2].to_dense()
    d2_T = d2.transpose(0, 1)

    codiff_3 = t.linalg.solve(m2, d2_T) @ m3  # inv_m2 @ d2_T @ m3

    x2 = t.arange(two_tets_mesh.n_tris).to(
        dtype=t.float, device=two_tets_mesh.vert_coords.device
    )
    x3 = t.arange(two_tets_mesh.n_tets).to(
        dtype=t.float, device=two_tets_mesh.vert_coords.device
    )

    dot_1 = t.dot(d2 @ x2, m3 @ x3)
    dot_2 = t.dot(x2, m2 @ (codiff_3 @ x3))

    t.testing.assert_close(dot_1, dot_2)


# TODO: add rotation/translation/scaling invariance tests

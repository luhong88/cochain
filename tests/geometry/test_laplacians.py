import pytest
import torch as t

from cochain.complex import Simplicial2Complex
from cochain.geometry import hodge_stars, laplacians
from cochain.geometry.stiffness import stiffness_matrix


def test_l0_stiffness_relation(two_tris_mesh: Simplicial2Complex):
    """
    Check that the 0-Laplacian and the stiffness matrix is related through the
    Hodge 0-star.
    """
    stiffness_direct = stiffness_matrix(two_tris_mesh).to_dense()

    s0 = hodge_stars.star_0(two_tris_mesh)
    l0 = laplacians.laplacian_0(two_tris_mesh)
    stiffness_indirect = laplacians._diag_sp_mm(s0, l0).to_dense()

    t.testing.assert_close(stiffness_indirect, stiffness_direct)


def test_l0_direct_construction(two_tris_mesh: Simplicial2Complex):
    """
    Constructing 0-Laplacian through the codifferential and coboundary operators
    should give the same matrix as through the stiffness matrix.
    """
    l0_via_cotan = laplacians.laplacian_0(two_tris_mesh).to_dense()

    codiff_1 = laplacians.codifferential_1(two_tris_mesh)
    l0 = (codiff_1 @ two_tris_mesh.coboundary_0).to_dense()

    t.testing.assert_close(l0, l0_via_cotan)


@pytest.mark.parametrize(
    "laplacian, betti",
    [
        (laplacians.laplacian_0, 1),
        (laplacians.laplacian_1, 0),
        (laplacians.laplacian_2, 0),
    ],
)
def test_disk_homology_group_dims(laplacian, betti, tent_mesh: Simplicial2Complex):
    operator = laplacian(tent_mesh).to_dense()
    dim_ker = operator.shape[0] - t.linalg.matrix_rank(operator)
    t.testing.assert_close(dim_ker, t.tensor(betti))


@pytest.mark.parametrize(
    "laplacian, betti",
    [
        (laplacians.laplacian_0, 1),
        (laplacians.laplacian_1, 1),
        (laplacians.laplacian_2, 0),
    ],
)
def test_annulus_homology_group_dims(
    laplacian, betti, flat_annulus_mesh: Simplicial2Complex
):
    operator = laplacian(flat_annulus_mesh).to_dense()
    dim_ker = operator.shape[0] - t.linalg.matrix_rank(operator)
    t.testing.assert_close(dim_ker, t.tensor(betti))


@pytest.mark.parametrize(
    "laplacian, betti",
    [
        (laplacians.laplacian_0, 1),
        (laplacians.laplacian_1, 0),
        (laplacians.laplacian_2, 1),
    ],
)
def test_sphere_homology_group_dims(
    laplacian, betti, icosphere_mesh: Simplicial2Complex
):
    operator = laplacian(icosphere_mesh).to_dense()
    dim_ker = operator.shape[0] - t.linalg.matrix_rank(operator)
    t.testing.assert_close(dim_ker, t.tensor(betti))


def test_laplacian_0_kernel(tent_mesh: Simplicial2Complex):
    l0 = laplacians.laplacian_0(tent_mesh)
    row_sum = l0.to_dense().sum(dim=-1)
    t.testing.assert_close(row_sum, t.zeros_like(row_sum))


def test_laplacian_2_kernel(tet_mesh: Simplicial2Complex):
    l2 = laplacians.laplacian_2(tet_mesh)
    areas = hodge_stars._tri_area(tet_mesh.vert_coords, tet_mesh.tris)

    zeros = (l2 @ areas).to_dense()

    t.testing.assert_close(zeros, t.zeros_like(zeros))


@pytest.mark.parametrize(
    "laplacian, star",
    [
        (laplacians.laplacian_0, hodge_stars.star_0),
        (laplacians.laplacian_1, hodge_stars.star_1),
        (laplacians.laplacian_2, hodge_stars.star_2),
    ],
)
def test_laplacian_symmetry(laplacian, star, tet_mesh: Simplicial2Complex):
    """
    Test that the stiffness matrices are symmetric, but the corresponding
    Laplacians are (in general) asymmetric.
    """
    star_i = star(tet_mesh)
    laplacian_i = laplacian(tet_mesh)
    stiffness_i = laplacians._diag_sp_mm(star_i, laplacian_i)

    laplacian_i_dense = laplacian_i.to_dense()
    stiffness_i_dense = stiffness_i.to_dense()

    t.testing.assert_close(stiffness_i_dense, stiffness_i_dense.T)
    assert not t.allclose(laplacian_i_dense, laplacian_i_dense.T)


@pytest.mark.parametrize(
    "laplacian, star",
    [
        (laplacians.laplacian_0, hodge_stars.star_0),
        (laplacians.laplacian_1, hodge_stars.star_1),
        (laplacians.laplacian_2, hodge_stars.star_2),
    ],
)
def test_laplacian_PSD(laplacian, star, tet_mesh: Simplicial2Complex):
    """
    Test that the stiffness matrices are positive semi-definite.
    """
    star_i = star(tet_mesh)
    laplacian_i = laplacian(tet_mesh)
    stiffness_i = laplacians._diag_sp_mm(star_i, laplacian_i).to_dense()

    eigs = t.linalg.eigvalsh(stiffness_i)
    assert eigs.min() >= -1e-6


def test_laplacian_1_orthogonality(tet_mesh: Simplicial2Complex):
    l1_div_grad = laplacians.laplacian_1_div_grad(tet_mesh).to_dense()
    l1_curl_curl = laplacians.laplacian_1_curl_curl(tet_mesh).to_dense()

    composition_1 = l1_div_grad @ l1_curl_curl
    composition_2 = l1_curl_curl @ l1_div_grad

    t.testing.assert_close(composition_1, t.zeros_like(composition_1))
    t.testing.assert_close(composition_2, t.zeros_like(composition_2))


def test_laplacian_1_curl_free(tet_mesh: Simplicial2Complex):
    """
    The curl curl component of the 1-Laplacian acting on a curl-free 1-cochain/
    1-form produces 0.
    """
    l1_curl_curl = laplacians.laplacian_1_curl_curl(tet_mesh)

    x0 = tet_mesh.vert_coords.sum(axis=-1, keepdim=True)
    x1_curl_free = tet_mesh.coboundary_0 @ x0

    x1_zero = (l1_curl_curl @ x1_curl_free).to_dense()

    t.testing.assert_close(x1_zero, t.zeros_like(x1_zero))


def test_laplacian_1_div_free(tet_mesh: Simplicial2Complex):
    """
    The div grad component of the 1-Laplacian acting on a div-free 1-cochain/
    1-form produces 0.
    """
    codiff_2 = laplacians.codifferential_2(tet_mesh)
    l1_div_grad = laplacians.laplacian_1_div_grad(tet_mesh)

    x2 = t.arange(tet_mesh.n_tris).to(dtype=t.float, device=tet_mesh.vert_coords.device)
    x1_div_free = codiff_2 @ x2

    x1_zero = (l1_div_grad @ x1_div_free).to_dense()

    t.testing.assert_close(x1_zero, t.zeros_like(x1_zero))


def test_codiff_1_adjoint_relation(tet_mesh: Simplicial2Complex):
    """
    Check that the 1-codifferential and the coboundary-0 operators are adjoints
    with respect to the Hodge star-weighted inner product.
    """
    s0 = t.diagflat(hodge_stars.star_0(tet_mesh).to_dense())
    s1 = t.diagflat(hodge_stars.star_1(tet_mesh).to_dense())

    d0 = tet_mesh.coboundary_0
    codiff_1 = laplacians.codifferential_1(tet_mesh)

    x0 = t.arange(tet_mesh.n_verts).to(
        dtype=t.float, device=tet_mesh.vert_coords.device
    )
    x1 = t.arange(tet_mesh.n_edges).to(
        dtype=t.float, device=tet_mesh.vert_coords.device
    )

    dot_1 = t.dot(d0 @ x0, s1 @ x1)
    dot_2 = t.dot(x0, s0 @ (codiff_1 @ x1))

    t.testing.assert_close(dot_1, dot_2)


def test_codiff_2_adjoint_relation(tet_mesh: Simplicial2Complex):
    """
    Check that the 2-codifferential and the coboundary-1 operators are adjoints
    with respect to the Hodge star-weighted inner products.
    """
    s1 = t.diagflat(hodge_stars.star_1(tet_mesh).to_dense())
    s2 = t.diagflat(hodge_stars.star_2(tet_mesh).to_dense())

    d1 = tet_mesh.coboundary_1
    codiff_2 = laplacians.codifferential_2(tet_mesh)

    x1 = t.arange(tet_mesh.n_edges).to(
        dtype=t.float, device=tet_mesh.vert_coords.device
    )
    x2 = t.arange(tet_mesh.n_tris).to(dtype=t.float, device=tet_mesh.vert_coords.device)

    dot_1 = t.dot(d1 @ x1, s2 @ x2)
    dot_2 = t.dot(x1, s1 @ (codiff_2 @ x2))

    t.testing.assert_close(dot_1, dot_2)

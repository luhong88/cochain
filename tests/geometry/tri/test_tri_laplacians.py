import pytest
import torch

from cochain.complex import SimplicialMesh
from cochain.geometry.tri import tri_hodge_stars, tri_laplacians
from cochain.geometry.tri.tri_stiffness import stiffness_matrix


def test_l0_stiffness_relation(two_tris_mesh: SimplicialMesh):
    """
    Check that the 0-Laplacian and the stiffness matrix is related through the
    Hodge 0-star.
    """
    stiffness_direct = stiffness_matrix(two_tris_mesh).to_dense()

    s0 = tri_hodge_stars.star_0(two_tris_mesh)
    l0 = tri_laplacians.laplacian_0(two_tris_mesh, dual_complex="circumcentric")
    stiffness_indirect = (s0 @ l0).to_dense()

    torch.testing.assert_close(stiffness_indirect, stiffness_direct)


def test_l0_direct_construction(two_tris_mesh: SimplicialMesh):
    """
    Constructing 0-Laplacian through the codifferential and coboundary operators
    should give the same matrix as through the stiffness matrix.
    """
    l0_via_cotan = tri_laplacians.laplacian_0(
        two_tris_mesh, dual_complex="circumcentric"
    ).to_dense()

    codiff_1 = tri_laplacians.codifferential_1(
        two_tris_mesh, dual_complex="circumcentric"
    )
    l0 = (codiff_1 @ two_tris_mesh.cbd[0]).to_dense()

    torch.testing.assert_close(l0, l0_via_cotan)


@pytest.mark.parametrize(
    "laplacian, dual_complex, betti",
    [
        (tri_laplacians.laplacian_0, "circumcentric", 1),
        (tri_laplacians.laplacian_0, "barycentric", 1),
        (tri_laplacians.laplacian_1, "circumcentric", 0),
        (tri_laplacians.laplacian_1, "barycentric", 0),
        (tri_laplacians.laplacian_2, "circumcentric", 0),
        (tri_laplacians.laplacian_2, "barycentric", 0),
    ],
)
def test_disk_homology_group_dims(
    laplacian, dual_complex, betti, tent_mesh: SimplicialMesh
):
    operator = laplacian(tent_mesh, dual_complex).to_dense()
    dim_ker = operator.shape[0] - torch.linalg.matrix_rank(operator)
    torch.testing.assert_close(dim_ker, torch.tensor(betti))


@pytest.mark.parametrize(
    "laplacian, dual_complex, betti",
    [
        (tri_laplacians.laplacian_0, "circumcentric", 1),
        (tri_laplacians.laplacian_0, "barycentric", 1),
        (tri_laplacians.laplacian_1, "circumcentric", 1),
        (tri_laplacians.laplacian_1, "barycentric", 1),
        (tri_laplacians.laplacian_2, "circumcentric", 0),
        (tri_laplacians.laplacian_2, "barycentric", 0),
    ],
)
def test_annulus_homology_group_dims(
    laplacian, dual_complex, betti, flat_annulus_mesh: SimplicialMesh
):
    operator = laplacian(flat_annulus_mesh, dual_complex).to_dense()
    dim_ker = operator.shape[0] - torch.linalg.matrix_rank(operator)
    torch.testing.assert_close(dim_ker, torch.tensor(betti))


@pytest.mark.parametrize(
    "laplacian, dual_complex, betti",
    [
        (tri_laplacians.laplacian_0, "circumcentric", 1),
        (tri_laplacians.laplacian_0, "barycentric", 1),
        (tri_laplacians.laplacian_1, "circumcentric", 0),
        (tri_laplacians.laplacian_1, "barycentric", 0),
        (tri_laplacians.laplacian_2, "circumcentric", 1),
        (tri_laplacians.laplacian_2, "barycentric", 1),
    ],
)
def test_sphere_homology_group_dims(
    laplacian, dual_complex, betti, icosphere_mesh: SimplicialMesh
):
    operator = laplacian(icosphere_mesh, dual_complex).to_dense()
    dim_ker = operator.shape[0] - torch.linalg.matrix_rank(operator)
    torch.testing.assert_close(dim_ker, torch.tensor(betti))


@pytest.mark.parametrize(
    "dual_complex",
    ["circumcentric", "barycentric"],
)
def test_laplacian_0_kernel(dual_complex, tent_mesh: SimplicialMesh):
    l0 = tri_laplacians.laplacian_0(tent_mesh, dual_complex)
    row_sum = l0.to_dense().sum(dim=-1)
    torch.testing.assert_close(row_sum, torch.zeros_like(row_sum))


@pytest.mark.parametrize(
    "dual_complex",
    ["circumcentric", "barycentric"],
)
def test_laplacian_2_kernel(dual_complex, hollow_tet_mesh: SimplicialMesh):
    """
    Check that the triangle area vector is in the kernel of the 2-Laplacian for
    a closed mesh.
    """
    l2 = tri_laplacians.laplacian_2(hollow_tet_mesh, dual_complex)
    areas = tri_hodge_stars.compute_tri_areas(
        hollow_tet_mesh.vert_coords, hollow_tet_mesh.tris
    )

    zeros = (l2 @ areas).to_dense()

    torch.testing.assert_close(zeros, torch.zeros_like(zeros))


@pytest.mark.parametrize(
    "laplacian, dual_complex, star",
    [
        (tri_laplacians.laplacian_0, "circumcentric", tri_hodge_stars.star_0),
        (tri_laplacians.laplacian_0, "barycentric", tri_hodge_stars.star_0),
        (
            tri_laplacians.laplacian_1,
            "circumcentric",
            tri_hodge_stars._star_1_circumcentric,
        ),
        (
            tri_laplacians.laplacian_1,
            "barycentric",
            tri_hodge_stars._star_1_barycentric,
        ),
        (tri_laplacians.laplacian_2, "circumcentric", tri_hodge_stars.star_2),
        (tri_laplacians.laplacian_2, "barycentric", tri_hodge_stars.star_2),
    ],
)
def test_laplacian_symmetry(
    laplacian, dual_complex, star, hollow_tet_mesh: SimplicialMesh
):
    """
    Test that the stiffness matrices are symmetric, but the corresponding
    Laplacians are (in general) asymmetric.
    """
    star_i = star(hollow_tet_mesh)
    laplacian_i = laplacian(hollow_tet_mesh, dual_complex)
    stiffness_i = star_i @ laplacian_i

    laplacian_i_T = laplacian_i.T
    stiffness_i_T = stiffness_i.T

    torch.testing.assert_close(stiffness_i.to_dense(), stiffness_i_T.to_dense())
    assert not torch.allclose(laplacian_i.to_dense(), laplacian_i_T.to_dense())


@pytest.mark.parametrize(
    "laplacian, dual_complex, star",
    [
        (tri_laplacians.laplacian_0, "circumcentric", tri_hodge_stars.star_0),
        (tri_laplacians.laplacian_0, "barycentric", tri_hodge_stars.star_0),
        (
            tri_laplacians.laplacian_1,
            "circumcentric",
            tri_hodge_stars._star_1_circumcentric,
        ),
        (
            tri_laplacians.laplacian_1,
            "barycentric",
            tri_hodge_stars._star_1_barycentric,
        ),
        (tri_laplacians.laplacian_2, "circumcentric", tri_hodge_stars.star_2),
        (tri_laplacians.laplacian_2, "barycentric", tri_hodge_stars.star_2),
    ],
)
def test_laplacian_PSD(laplacian, dual_complex, star, hollow_tet_mesh: SimplicialMesh):
    """
    Test that the stiffness matrices are positive semi-definite.
    """
    star_i = star(hollow_tet_mesh)
    laplacian_i = laplacian(hollow_tet_mesh, dual_complex)
    stiffness_i = (star_i @ laplacian_i).to_dense()

    eigs = torch.linalg.eigvalsh(stiffness_i)
    assert eigs.min() >= -1e-6


@pytest.mark.parametrize(
    "dual_complex",
    ["circumcentric", "barycentric"],
)
def test_laplacian_1_orthogonality(dual_complex, hollow_tet_mesh: SimplicialMesh):
    l1_div_grad = tri_laplacians.laplacian_1_grad_div(hollow_tet_mesh, dual_complex)
    l1_curl_curl = tri_laplacians.laplacian_1_curl_curl(hollow_tet_mesh, dual_complex)

    composition_1 = (l1_div_grad @ l1_curl_curl).to_dense()
    composition_2 = (l1_curl_curl @ l1_div_grad).to_dense()

    torch.testing.assert_close(composition_1, torch.zeros_like(composition_1))
    torch.testing.assert_close(composition_2, torch.zeros_like(composition_2))


@pytest.mark.parametrize(
    "dual_complex",
    ["circumcentric", "barycentric"],
)
def test_laplacian_1_curl_free(dual_complex, hollow_tet_mesh: SimplicialMesh):
    """
    The curl curl component of the 1-Laplacian acting on a curl-free 1-cochain/
    1-form produces 0.
    """
    l1_curl_curl = tri_laplacians.laplacian_1_curl_curl(hollow_tet_mesh, dual_complex)

    x0 = hollow_tet_mesh.vert_coords.sum(axis=-1, keepdim=True)
    x1_curl_free = hollow_tet_mesh.cbd[0] @ x0

    x1_zero = (l1_curl_curl @ x1_curl_free).to_dense()

    torch.testing.assert_close(x1_zero, torch.zeros_like(x1_zero))


@pytest.mark.parametrize(
    "dual_complex",
    ["circumcentric", "barycentric"],
)
def test_laplacian_1_div_free(dual_complex, hollow_tet_mesh: SimplicialMesh):
    """
    The div grad component of the 1-Laplacian acting on a div-free 1-cochain/
    1-form produces 0.
    """
    codiff_2 = tri_laplacians.codifferential_2(hollow_tet_mesh, dual_complex)
    l1_div_grad = tri_laplacians.laplacian_1_grad_div(hollow_tet_mesh, dual_complex)

    x2 = torch.arange(hollow_tet_mesh.n_tris).to(
        dtype=torch.float, device=hollow_tet_mesh.vert_coords.device
    )
    x1_div_free = codiff_2 @ x2

    x1_zero = (l1_div_grad @ x1_div_free).to_dense()

    torch.testing.assert_close(x1_zero, torch.zeros_like(x1_zero))


@pytest.mark.parametrize(
    "dual_complex",
    ["circumcentric", "barycentric"],
)
def test_codiff_1_adjoint_relation(dual_complex, hollow_tet_mesh: SimplicialMesh):
    """
    Check that the 1-codifferential and the coboundary-0 operators are adjoints
    with respect to the Hodge star-weighted inner product.
    """
    s0 = tri_hodge_stars.star_0(hollow_tet_mesh)
    s1 = tri_hodge_stars.star_1(hollow_tet_mesh, dual_complex)

    d0 = hollow_tet_mesh.cbd[0]
    codiff_1 = tri_laplacians.codifferential_1(hollow_tet_mesh, dual_complex)

    x0 = torch.arange(hollow_tet_mesh.n_verts).to(
        dtype=torch.float, device=hollow_tet_mesh.vert_coords.device
    )
    x1 = torch.arange(hollow_tet_mesh.n_edges).to(
        dtype=torch.float, device=hollow_tet_mesh.vert_coords.device
    )

    dot_1 = torch.dot(d0 @ x0, s1 @ x1)
    dot_2 = torch.dot(x0, s0 @ (codiff_1 @ x1))

    torch.testing.assert_close(dot_1, dot_2)


@pytest.mark.parametrize(
    "dual_complex",
    ["circumcentric", "barycentric"],
)
def test_codiff_2_adjoint_relation(dual_complex, hollow_tet_mesh: SimplicialMesh):
    """
    Check that the 2-codifferential and the coboundary-1 operators are adjoints
    with respect to the Hodge star-weighted inner products.
    """
    s1 = tri_hodge_stars.star_1(hollow_tet_mesh, dual_complex)
    s2 = tri_hodge_stars.star_2(hollow_tet_mesh)

    d1 = hollow_tet_mesh.cbd[1]
    codiff_2 = tri_laplacians.codifferential_2(hollow_tet_mesh, dual_complex)

    x1 = torch.arange(hollow_tet_mesh.n_edges).to(
        dtype=torch.float, device=hollow_tet_mesh.vert_coords.device
    )
    x2 = torch.arange(hollow_tet_mesh.n_tris).to(
        dtype=torch.float, device=hollow_tet_mesh.vert_coords.device
    )

    dot_1 = torch.dot(d1 @ x1, s2 @ x2)
    dot_2 = torch.dot(x1, s1 @ (codiff_2 @ x2))

    torch.testing.assert_close(dot_1, dot_2)

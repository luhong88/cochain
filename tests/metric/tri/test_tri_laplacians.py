import pytest
import torch

from cochain.complex import SimplicialMesh
from cochain.metric.tri import tri_hodge_stars, tri_laplacians
from cochain.metric.tri.tri_stiffness import stiffness_matrix


def test_l0_stiffness_relation(two_tris_mesh: SimplicialMesh, device):
    """Check that the 0-Laplacian and the stiffness matrix is related through the 0-star."""
    mesh = two_tris_mesh.to(device)

    stiffness_direct = stiffness_matrix(mesh).to_dense()

    s0 = tri_hodge_stars.star_0(mesh)
    l0 = tri_laplacians.laplacian_0(mesh, dual_complex="circumcentric")
    stiffness_indirect = (s0 @ l0).to_dense()

    torch.testing.assert_close(stiffness_indirect, stiffness_direct)


def test_l0_direct_construction(two_tris_mesh: SimplicialMesh, device):
    """
    Test alternative 0-Laplacian construction routes.

    Constructing 0-Laplacian through the codifferential and coboundary operators
    should give the same matrix as through the stiffness matrix.
    """
    mesh = two_tris_mesh.to(device)

    l0_via_cotan = tri_laplacians.laplacian_0(
        mesh, dual_complex="circumcentric"
    ).to_dense()

    codiff_1 = tri_laplacians.codifferential_1(mesh, dual_complex="circumcentric")
    l0 = (codiff_1 @ mesh.cbd[0]).to_dense()

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
    laplacian, dual_complex, betti, tent_mesh: SimplicialMesh, device
):
    mesh = tent_mesh.to(device)
    operator = laplacian(mesh, dual_complex).to_dense()
    dim_ker = operator.shape[0] - torch.linalg.matrix_rank(operator)
    torch.testing.assert_close(dim_ker, torch.tensor(betti, device=device))


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
    laplacian, dual_complex, betti, flat_annulus_mesh: SimplicialMesh, device
):
    mesh = flat_annulus_mesh.to(device)
    operator = laplacian(mesh, dual_complex).to_dense()
    dim_ker = operator.shape[0] - torch.linalg.matrix_rank(operator)
    torch.testing.assert_close(dim_ker, torch.tensor(betti, device=device))


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
    laplacian, dual_complex, betti, icosphere_mesh: SimplicialMesh, device
):
    mesh = icosphere_mesh.to(device)
    operator = laplacian(mesh, dual_complex).to_dense()
    dim_ker = operator.shape[0] - torch.linalg.matrix_rank(operator)
    torch.testing.assert_close(dim_ker, torch.tensor(betti, device=device))


@pytest.mark.parametrize(
    "dual_complex",
    ["circumcentric", "barycentric"],
)
def test_laplacian_0_kernel(dual_complex, tent_mesh: SimplicialMesh, device):
    mesh = tent_mesh.to(device)
    l0 = tri_laplacians.laplacian_0(mesh, dual_complex)
    row_sum = l0.to_dense().sum(dim=-1)
    torch.testing.assert_close(row_sum, torch.zeros_like(row_sum))


@pytest.mark.parametrize(
    "dual_complex",
    ["circumcentric", "barycentric"],
)
def test_laplacian_2_kernel(dual_complex, hollow_tet_mesh: SimplicialMesh, device):
    """Check that the tri area vector is in the kernel of the 2-Laplacian for a closed mesh."""
    mesh = hollow_tet_mesh.to(device)

    l2 = tri_laplacians.laplacian_2(mesh, dual_complex)
    areas = tri_hodge_stars.compute_tri_areas(mesh.vert_coords, mesh.tris)

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
    laplacian, dual_complex, star, hollow_tet_mesh: SimplicialMesh, device
):
    """Test that the stiffness matrices are symmetric, but the corresponding Laplacians are not."""
    mesh = hollow_tet_mesh.to(device)

    star_i = star(mesh)
    laplacian_i = laplacian(mesh, dual_complex)
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
def test_laplacian_PSD(
    laplacian, dual_complex, star, hollow_tet_mesh: SimplicialMesh, device
):
    """Test that the stiffness matrices are positive semi-definite."""
    mesh = hollow_tet_mesh.to(device)

    star_i = star(mesh)
    laplacian_i = laplacian(mesh, dual_complex)
    stiffness_i = (star_i @ laplacian_i).to_dense()

    eigs = torch.linalg.eigvalsh(stiffness_i)
    assert eigs.min() >= -1e-6


@pytest.mark.parametrize(
    "dual_complex",
    ["circumcentric", "barycentric"],
)
def test_laplacian_1_orthogonality(
    dual_complex, hollow_tet_mesh: SimplicialMesh, device
):
    """Test that composing the up and down 1-Laplacian gives zero."""
    mesh = hollow_tet_mesh.to(device)

    l1_grad_div = tri_laplacians.laplacian_1_grad_div(mesh, dual_complex)
    l1_curl_curl = tri_laplacians.laplacian_1_curl_curl(mesh, dual_complex)

    composition_1 = (l1_grad_div @ l1_curl_curl).to_dense()
    composition_2 = (l1_curl_curl @ l1_grad_div).to_dense()

    torch.testing.assert_close(composition_1, torch.zeros_like(composition_1))
    torch.testing.assert_close(composition_2, torch.zeros_like(composition_2))


@pytest.mark.parametrize(
    "dual_complex",
    ["circumcentric", "barycentric"],
)
def test_laplacian_1_curl_free(dual_complex, hollow_tet_mesh: SimplicialMesh, device):
    """The curl-curl 1-Laplacian annihilates a curl-free 1-cochain."""
    mesh = hollow_tet_mesh.to(device)

    l1_curl_curl = tri_laplacians.laplacian_1_curl_curl(mesh, dual_complex)

    x0 = mesh.vert_coords.sum(axis=-1, keepdim=True)
    # A gradient field is irrotational.
    x1_irrotational = mesh.cbd[0] @ x0

    x1_zero = (l1_curl_curl @ x1_irrotational).to_dense()

    torch.testing.assert_close(x1_zero, torch.zeros_like(x1_zero))


@pytest.mark.parametrize(
    "dual_complex",
    ["circumcentric", "barycentric"],
)
def test_laplacian_1_div_free(dual_complex, hollow_tet_mesh: SimplicialMesh, device):
    """The grad-div 1-Laplacian annihilates a div-free 1-cochain."""
    mesh = hollow_tet_mesh.to(device)

    codiff_2 = tri_laplacians.codifferential_2(mesh, dual_complex)
    l1_grad_div = tri_laplacians.laplacian_1_grad_div(mesh, dual_complex)

    x2 = torch.randn(mesh.n_tris, dtype=torch.float32, device=mesh.device)

    # The curl of a vector field is divergence-free.
    x1_solenoidal = codiff_2 @ x2
    x1_zero = (l1_grad_div @ x1_solenoidal).to_dense()

    torch.testing.assert_close(x1_zero, torch.zeros_like(x1_zero))


@pytest.mark.parametrize(
    "dual_complex",
    ["circumcentric", "barycentric"],
)
def test_codiff_1_adjoint_relation(
    dual_complex, hollow_tet_mesh: SimplicialMesh, device
):
    """Check that the 1-codiff and the 0-cbd are adjoints."""
    mesh = hollow_tet_mesh.to(device)

    s0 = tri_hodge_stars.star_0(mesh)
    s1 = tri_hodge_stars.star_1(mesh, dual_complex)

    d0 = mesh.cbd[0]
    codiff_1 = tri_laplacians.codifferential_1(mesh, dual_complex)

    x0 = torch.randn(mesh.n_verts, dtype=mesh.dtype, device=mesh.device)
    x1 = torch.randn(mesh.n_edges, dtype=mesh.dtype, device=mesh.device)

    dot_1 = torch.dot(d0 @ x0, s1 @ x1)
    dot_2 = torch.dot(x0, s0 @ (codiff_1 @ x1))

    torch.testing.assert_close(dot_1, dot_2)


@pytest.mark.parametrize(
    "dual_complex",
    ["circumcentric", "barycentric"],
)
def test_codiff_2_adjoint_relation(
    dual_complex, hollow_tet_mesh: SimplicialMesh, device
):
    """Check that the 2-codiff and the 1-cbd are adjoints."""
    mesh = hollow_tet_mesh.to(device)

    s1 = tri_hodge_stars.star_1(mesh, dual_complex)
    s2 = tri_hodge_stars.star_2(mesh)

    d1 = mesh.cbd[1]
    codiff_2 = tri_laplacians.codifferential_2(mesh, dual_complex)

    x1 = torch.randn(mesh.n_edges, dtype=mesh.dtype, device=mesh.device)
    x2 = torch.randn(mesh.n_tris, dtype=mesh.dtype, device=mesh.device)

    dot_1 = torch.dot(d1 @ x1, s2 @ x2)
    dot_2 = torch.dot(x1, s1 @ (codiff_2 @ x2))

    torch.testing.assert_close(dot_1, dot_2)


@pytest.mark.parametrize(
    "laplacian, dual_complex",
    [
        (tri_laplacians.laplacian_0, "circumcentric"),
        (tri_laplacians.laplacian_0, "barycentric"),
        (
            tri_laplacians.laplacian_1,
            "circumcentric",
        ),
        (
            tri_laplacians.laplacian_1,
            "barycentric",
        ),
        (tri_laplacians.laplacian_2, "circumcentric"),
        (tri_laplacians.laplacian_2, "barycentric"),
    ],
)
def test_laplacian_backward(
    laplacian, dual_complex, hollow_tet_mesh: SimplicialMesh, device
):
    mesh = hollow_tet_mesh.to(device)
    mesh.requires_grad_()

    l = laplacian(mesh, dual_complex)
    output = l.values.sum()
    output.backward()

    assert mesh.grad is not None
    assert torch.isfinite(mesh.grad).all()


@pytest.mark.parametrize(
    "laplacian, dual_complex",
    [
        (tri_laplacians.laplacian_0, "circumcentric"),
        (tri_laplacians.laplacian_0, "barycentric"),
        (
            tri_laplacians.laplacian_1,
            "circumcentric",
        ),
        (
            tri_laplacians.laplacian_1,
            "barycentric",
        ),
        (tri_laplacians.laplacian_2, "circumcentric"),
        (tri_laplacians.laplacian_2, "barycentric"),
    ],
)
def test_laplacian_gradcheck(
    laplacian, dual_complex, hollow_tet_mesh: SimplicialMesh, device
):
    # Scale the vertex coordinates by a factor of 100 to improve numerical
    # precision for gradcheck (esp. for the grad-div component of L_1).
    vert_coords = 100.0 * hollow_tet_mesh.vert_coords.clone().to(
        dtype=torch.float64, device=device
    )
    vert_coords.requires_grad_()

    def laplacian_fxn(test_vert_coords):
        mesh = hollow_tet_mesh.to(device=device, dtype=torch.float64)
        mesh.vert_coords = test_vert_coords
        l = laplacian(mesh, dual_complex)
        return l.values.sum()

    assert torch.autograd.gradcheck(laplacian_fxn, (vert_coords,), fast_mode=True)

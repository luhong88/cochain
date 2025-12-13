import pytest
import torch as t

from cochain.complex import SimplicialComplex
from cochain.experimental import tri_jacobians
from cochain.geometry.tri import tri_hodge_stars, tri_masses
from cochain.geometry.tri.tri_stiffness import stiffness_matrix


@pytest.mark.parametrize(
    "star, d_star_d_vert_coords",
    [
        (tri_hodge_stars.star_0, tri_jacobians.d_star_0_d_vert_coords),
        (
            tri_hodge_stars.star_1_circumcentric,
            tri_jacobians.d_star_1_circumcentric_d_vert_coords,
        ),
        (tri_hodge_stars.star_2, tri_jacobians.d_star_2_d_vert_coords),
    ],
)
def test_star_jacobian(star, d_star_d_vert_coords, hollow_tet_mesh: SimplicialComplex):
    vert_coords = hollow_tet_mesh.vert_coords.clone()
    tris = hollow_tet_mesh.tris.clone()

    autograd_jacobian = t.autograd.functional.jacobian(
        lambda vert_coords: star(SimplicialComplex.from_tri_mesh(vert_coords, tris)),
        vert_coords,
    )

    analytical_jacobian = d_star_d_vert_coords(hollow_tet_mesh).to_dense()

    t.testing.assert_close(autograd_jacobian, analytical_jacobian)


@pytest.mark.parametrize(
    "star, d_inv_star_d_vert_coords",
    [
        (tri_hodge_stars.star_0, tri_jacobians.d_inv_star_0_d_vert_coords),
        (
            tri_hodge_stars.star_1_circumcentric,
            tri_jacobians.d_inv_star_1_circumcentric_d_vert_coords,
        ),
        (tri_hodge_stars.star_2, tri_jacobians.d_inv_star_2_d_vert_coords),
    ],
)
def test_inv_star_jacobian(
    star, d_inv_star_d_vert_coords, hollow_tet_mesh: SimplicialComplex
):
    vert_coords = hollow_tet_mesh.vert_coords.clone()
    tris = hollow_tet_mesh.tris.clone()

    autograd_jacobian = t.autograd.functional.jacobian(
        lambda vert_coords: 1.0
        / star(SimplicialComplex.from_tri_mesh(vert_coords, tris)),
        vert_coords,
    )

    analytical_jacobian = d_inv_star_d_vert_coords(hollow_tet_mesh).to_dense()

    t.testing.assert_close(autograd_jacobian, analytical_jacobian)


def tet_mass_1_aurograd(two_tris_mesh: SimplicialComplex):
    two_tris_mesh.vert_coords.requires_grad = True
    mass_1 = tri_masses.mass_1(two_tris_mesh).to_dense()
    y = (mass_1**2).sum()

    dMdV = tri_jacobians.d_mass_1_d_vert_coords(two_tris_mesh).to_dense()
    dydM = t.autograd.grad(y, mass_1, retain_graph=True)[0]
    custom_grad = t.einsum("ij,ijkl->kl", dydM, dMdV)

    auto_grad = t.autograd.grad(y, two_tris_mesh.vert_coords)[0]

    assert t.allclose(custom_grad, auto_grad, atol=1e-4)


def test_stiffness_autograd(two_tris_mesh: SimplicialComplex):
    """
    Check that the custom gradient matches the automatic gradient for the stiffness
    matrix.
    """
    two_tris_mesh.vert_coords.requires_grad = True
    two_tris_S = stiffness_matrix(two_tris_mesh).to_dense()
    y = (two_tris_S**2).sum()

    dLdV = tri_jacobians.d_stiffness_d_vert_coords(two_tris_mesh).to_dense()
    dydL = t.autograd.grad(y, two_tris_S, retain_graph=True)[0]
    custom_grad = t.einsum("ij,ijkl->kl", dydL, dLdV)

    auto_grad = t.autograd.grad(y, two_tris_mesh.vert_coords)[0]

    assert t.allclose(custom_grad, auto_grad, atol=1e-4)

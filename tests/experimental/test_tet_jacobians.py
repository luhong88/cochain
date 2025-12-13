import torch as t

from cochain.complex import SimplicialComplex
from cochain.experimental import tet_jacobians
from cochain.geometry.tet import tet_masses
from cochain.geometry.tet.tet_stiffness import stiffness_matrix


def tet_mass_1_aurograd(two_tets_mesh: SimplicialComplex):
    two_tets_mesh.vert_coords.requires_grad = True
    mass_1 = tet_masses.mass_1(two_tets_mesh).to_dense()
    y = (mass_1**2).sum()

    dMdV = tet_jacobians.d_mass_1_d_vert_coords(two_tets_mesh).to_dense()
    dydM = t.autograd.grad(y, mass_1, retain_graph=True)[0]
    custom_grad = t.einsum("ij,ijkl->kl", dydM, dMdV)

    auto_grad = t.autograd.grad(y, two_tets_mesh.vert_coords)[0]

    assert t.allclose(custom_grad, auto_grad, atol=1e-4)


def tet_mass_2_aurograd(two_tets_mesh: SimplicialComplex):
    two_tets_mesh.vert_coords.requires_grad = True
    mass_2 = tet_masses.mass_2(two_tets_mesh).to_dense()
    y = (mass_2**2).sum()

    dMdV = tet_jacobians.d_mass_2_d_vert_coords(two_tets_mesh).to_dense()
    dydM = t.autograd.grad(y, mass_2, retain_graph=True)[0]
    custom_grad = t.einsum("ij,ijkl->kl", dydM, dMdV)

    auto_grad = t.autograd.grad(y, two_tets_mesh.vert_coords)[0]

    assert t.allclose(custom_grad, auto_grad, atol=1e-4)


def test_stiffness_autograd(two_tets_mesh: SimplicialComplex):
    """
    Check that the custom gradient matches the automatic gradient for the stiffness
    matrix.
    """
    two_tets_mesh.vert_coords.requires_grad = True
    two_tets_S = stiffness_matrix(two_tets_mesh).to_dense()
    y = (two_tets_S**2).sum()

    dLdV = tet_jacobians.d_stiffness_d_vert_coords(two_tets_mesh).to_dense()
    dydL = t.autograd.grad(y, two_tets_S, retain_graph=True)[0]
    custom_grad = t.einsum("ij,ijkl->kl", dydL, dLdV)

    auto_grad = t.autograd.grad(y, two_tets_mesh.vert_coords)[0]

    assert t.allclose(custom_grad, auto_grad, atol=1e-4)

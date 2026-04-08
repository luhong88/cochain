import itertools

import pytest
import torch

from cochain.cochain.ext_prod.cup import AntisymmetricCupProduct
from cochain.cochain.ext_prod.whitney import WhitneyWedgeL2Projector
from cochain.complex import SimplicialMesh
from cochain.geometry.tet import tet_masses
from cochain.geometry.tri import tri_hodge_stars, tri_masses


def _compute_mass_matrix(mesh: SimplicialMesh, k: int):
    match mesh.dim:
        case 2:
            match k:
                case 0:
                    mass = tri_masses.mass_0(mesh)
                case 1:
                    mass = tri_masses.mass_1(mesh)
                case 2:
                    mass = tri_hodge_stars.star_2(mesh)
                case _:
                    raise ValueError()
        case 3:
            match k:
                case 0:
                    mass = tet_masses.mass_0(mesh)
                case 1:
                    mass = tet_masses.mass_1(mesh)
                case 2:
                    mass = tet_masses.mass_2(mesh)
                case 3:
                    mass = tet_masses.mass_3(mesh)
                case _:
                    raise ValueError()
        case _:
            raise NotImplementedError()

    return mass


@pytest.mark.parametrize("mesh_name", ["two_tris_mesh", "two_tets_mesh"])
def test_const_0_form_galerkin_wedge_product(mesh_name, request, device):
    """
    The wedge product between a constant 0-form and an arbitrary k-form should
    be identical to the k-form.
    """
    mesh: SimplicialMesh = request.getfixturevalue(mesh_name).to(device)

    n_splx_map = mesh.n_splx

    const_0_form = torch.ones(n_splx_map[0], dtype=mesh.dtype, device=device)
    for k in range(mesh.dim + 1):
        k_form = torch.randn(n_splx_map[k]).to(device)

        proj = WhitneyWedgeL2Projector(0, k, mesh)

        b = proj(const_0_form, k_form)

        mass = _compute_mass_matrix(mesh, k).to_dense().to(device)

        w = torch.linalg.solve(mass, b)

        torch.testing.assert_close(w, k_form)


@pytest.mark.parametrize("mesh_name", ["two_tris_mesh", "two_tets_mesh"])
def test_galerkin_wedge_product_graded_commutativity(mesh_name, request, device):
    mesh: SimplicialMesh = request.getfixturevalue(mesh_name)

    n_splx_map = {
        dim: n_splx
        for dim, n_splx in enumerate(
            [mesh.n_verts, mesh.n_edges, mesh.n_tris, mesh.n_tets]
        )
    }

    for k, l in itertools.product(range(mesh.dim + 1), repeat=2):
        if k + l <= mesh.dim:
            k_cochain = torch.randn(n_splx_map[k]).to(device)
            l_cochain = torch.randn(n_splx_map[l]).to(device)

            proj_kl = WhitneyWedgeL2Projector(k, l, mesh).to(device)
            proj_lk = WhitneyWedgeL2Projector(l, k, mesh).to(device)

            mass = _compute_mass_matrix(mesh, k + l).to_dense().to(device)

            w_kl = torch.linalg.solve(mass.to_dense(), proj_kl(k_cochain, l_cochain))
            w_lk = torch.linalg.solve(mass.to_dense(), proj_lk(l_cochain, k_cochain))

            sign = (-1.0) ** (k * l)

            torch.testing.assert_close(w_kl, sign * w_lk)


@pytest.mark.parametrize("mesh_name", ["two_tris_mesh", "two_tets_mesh"])
def test_galerkin_wedge_product_bilinearity(mesh_name, request, device):
    mesh: SimplicialMesh = request.getfixturevalue(mesh_name)

    n_splx_map = mesh.n_splx

    for k, l in itertools.product(range(mesh.dim + 1), repeat=2):
        if k + l <= mesh.dim:
            mass = _compute_mass_matrix(mesh, k + l).to_dense().to(device)

            k1_cochain = torch.randn(n_splx_map[k]).to(device)
            k2_cochain = torch.randn(n_splx_map[k]).to(device)
            l_cochain = torch.randn(n_splx_map[l]).to(device)

            c1, c2 = torch.randn(2)

            proj_kl = WhitneyWedgeL2Projector(k, l, mesh).to(device)

            lhs = torch.linalg.solve(
                mass, proj_kl(c1 * k1_cochain + c2 * k2_cochain, l_cochain)
            )
            rhs = c1 * torch.linalg.solve(
                mass, proj_kl(k1_cochain, l_cochain)
            ) + c2 * torch.linalg.solve(mass, proj_kl(k2_cochain, l_cochain))

            torch.testing.assert_close(lhs, rhs)

            k_cochain = torch.randn(n_splx_map[k]).to(device)
            l1_cochain = torch.randn(n_splx_map[l]).to(device)
            l2_cochain = torch.randn(n_splx_map[l]).to(device)

            c1, c2 = torch.randn(2)

            lhs = torch.linalg.solve(
                mass, proj_kl(k_cochain, c1 * l1_cochain + c2 * l2_cochain)
            )
            rhs = c1 * torch.linalg.solve(
                mass, proj_kl(k_cochain, l1_cochain)
            ) + c2 * torch.linalg.solve(mass, proj_kl(k_cochain, l2_cochain))

            torch.testing.assert_close(lhs, rhs)


def test_galerkin_wedge_product_cohomology_class(hollow_tet_mesh, device):
    """
    The Galerkin wedge product and the antisymmetric cup product should belong to
    the same cohomology class (i.e., differ by a coboundary). Therefore, on a
    closed mesh, the surface integral of the two products of exact forms should match.
    """
    n_splx_map = hollow_tet_mesh.n_splx

    for k in range(hollow_tet_mesh.dim + 1):
        l = hollow_tet_mesh.dim - k

        if k == 0:
            k_cochain = torch.randn(1).expand(n_splx_map[k]).to(device)
        if k > 0:
            d_k_1 = hollow_tet_mesh.cbd[k - 1].to(device)
            k_1_cochain = torch.randn(n_splx_map[k - 1]).to(device)
            k_cochain = d_k_1 @ k_1_cochain

        if l == 0:
            l_cochain = torch.randn(1).expand(n_splx_map[l]).to(device)
        if l > 0:
            d_l_1 = hollow_tet_mesh.cbd[l - 1].to(device)
            l_1_cochain = torch.randn(n_splx_map[l - 1]).to(device)
            l_cochain = d_l_1 @ l_1_cochain

        anti_cup_kl = AntisymmetricCupProduct(k, l, hollow_tet_mesh).to(device)

        proj_kl = WhitneyWedgeL2Projector(k, l, hollow_tet_mesh).to(device)
        mass = _compute_mass_matrix(hollow_tet_mesh, k + l).to_dense().to(device)
        wedge_kl = torch.linalg.solve(mass, proj_kl(k_cochain, l_cochain))

        torch.testing.assert_close(
            wedge_kl.sum(),
            anti_cup_kl(k_cochain, l_cochain).sum(),
        )


# TODO: test other pairing methods

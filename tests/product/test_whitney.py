import itertools

import pytest
import torch as t

from cochain.complex import SimplicialComplex
from cochain.geometry.tet import tet_masses
from cochain.geometry.tri import tri_hodge_stars, tri_masses
from cochain.product.cup import AntisymmetricCupProduct
from cochain.product.whitney import WhitneyWedgeProjection


def _compute_mass_matrix(mesh: SimplicialComplex, k: int):
    match mesh.dim:
        case 2:
            match k:
                case 0:
                    mass = tri_masses.mass_0_consistent(mesh)
                case 1:
                    mass = tri_masses.mass_1(mesh)
                case 2:
                    mass = tri_hodge_stars.star_2(mesh)
                case _:
                    raise ValueError()
        case 3:
            match k:
                case 0:
                    mass = tet_masses.mass_0_consistent(mesh)
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
def test_const_0_form_whitney_wedge_product(mesh_name, request, device):
    """
    The wedge product between a constant 0-form and an arbitrary k-form should
    be identical to the k-form.
    """
    mesh: SimplicialComplex = request.getfixturevalue(mesh_name).to(device)

    n_simp_map = {
        dim: n_simp
        for dim, n_simp in enumerate(
            [mesh.n_verts, mesh.n_edges, mesh.n_tris, mesh.n_tets]
        )
    }

    const_0_form = t.ones(n_simp_map[0], dtype=mesh.vert_coords.dtype, device=device)
    for k in range(mesh.dim + 1):
        k_form = t.randn(n_simp_map[k]).to(device)

        proj = WhitneyWedgeProjection(0, k, mesh)

        b = proj(const_0_form, k_form)

        mass = _compute_mass_matrix(mesh, k).to_dense().to(device)

        w = t.linalg.solve(mass, b)

        t.testing.assert_close(w, k_form)


@pytest.mark.parametrize("mesh_name", ["two_tris_mesh", "two_tets_mesh"])
def test_whitney_wedge_product_graded_commutativity(mesh_name, request, device):
    mesh: SimplicialComplex = request.getfixturevalue(mesh_name)

    n_simp_map = {
        dim: n_simp
        for dim, n_simp in enumerate(
            [mesh.n_verts, mesh.n_edges, mesh.n_tris, mesh.n_tets]
        )
    }

    for k, l in itertools.product(range(mesh.dim + 1), repeat=2):
        if k + l <= mesh.dim:
            k_cochain = t.randn(n_simp_map[k]).to(device)
            l_cochain = t.randn(n_simp_map[l]).to(device)

            proj_kl = WhitneyWedgeProjection(k, l, mesh).to(device)
            proj_lk = WhitneyWedgeProjection(l, k, mesh).to(device)

            mass = _compute_mass_matrix(mesh, k + l).to_dense().to(device)

            w_kl = t.linalg.solve(mass.to_dense(), proj_kl(k_cochain, l_cochain))
            w_lk = t.linalg.solve(mass.to_dense(), proj_lk(l_cochain, k_cochain))

            sign = (-1.0) ** (k * l)

            t.testing.assert_close(w_kl, sign * w_lk)


@pytest.mark.parametrize("mesh_name", ["two_tris_mesh", "two_tets_mesh"])
def test_whitney_wedge_product_bilinearity(mesh_name, request, device):
    mesh: SimplicialComplex = request.getfixturevalue(mesh_name)

    n_simp_map = {
        dim: n_simp
        for dim, n_simp in enumerate(
            [mesh.n_verts, mesh.n_edges, mesh.n_tris, mesh.n_tets]
        )
    }

    for k, l in itertools.product(range(mesh.dim + 1), repeat=2):
        if k + l <= mesh.dim:
            mass = _compute_mass_matrix(mesh, k + l).to_dense().to(device)

            k1_cochain = t.randn(n_simp_map[k]).to(device)
            k2_cochain = t.randn(n_simp_map[k]).to(device)
            l_cochain = t.randn(n_simp_map[l]).to(device)

            c1, c2 = t.randn(2)

            proj_kl = WhitneyWedgeProjection(k, l, mesh).to(device)

            lhs = t.linalg.solve(
                mass, proj_kl(c1 * k1_cochain + c2 * k2_cochain, l_cochain)
            )
            rhs = c1 * t.linalg.solve(
                mass, proj_kl(k1_cochain, l_cochain)
            ) + c2 * t.linalg.solve(mass, proj_kl(k2_cochain, l_cochain))

            t.testing.assert_close(lhs, rhs)

            k_cochain = t.randn(n_simp_map[k]).to(device)
            l1_cochain = t.randn(n_simp_map[l]).to(device)
            l2_cochain = t.randn(n_simp_map[l]).to(device)

            c1, c2 = t.randn(2)

            lhs = t.linalg.solve(
                mass, proj_kl(k_cochain, c1 * l1_cochain + c2 * l2_cochain)
            )
            rhs = c1 * t.linalg.solve(
                mass, proj_kl(k_cochain, l1_cochain)
            ) + c2 * t.linalg.solve(mass, proj_kl(k_cochain, l2_cochain))

            t.testing.assert_close(lhs, rhs)


def test_whitney_wedge_product_cohomology_class(hollow_tet_mesh, device):
    """
    The Whitney wedge product and the antisymmetric cup product should belong to
    the same cohomology class (i.e., differ by a coboundary). Therefore, on a
    closed mesh, the surface integral of the two products of exact forms should match.
    """
    n_simp_map = {
        dim: n_simp
        for dim, n_simp in enumerate(
            [
                hollow_tet_mesh.n_verts,
                hollow_tet_mesh.n_edges,
                hollow_tet_mesh.n_tris,
                hollow_tet_mesh.n_tets,
            ]
        )
    }

    for k in range(hollow_tet_mesh.dim + 1):
        l = hollow_tet_mesh.dim - k

        if k == 0:
            k_cochain = t.randn(1).expand(n_simp_map[k]).to(device)
        if k > 0:
            d_k_1 = getattr(hollow_tet_mesh, f"coboundary_{k - 1}").to(device)
            k_1_cochain = t.randn(n_simp_map[k - 1]).to(device)
            k_cochain = d_k_1 @ k_1_cochain

        if l == 0:
            l_cochain = t.randn(1).expand(n_simp_map[l]).to(device)
        if l > 0:
            d_l_1 = getattr(hollow_tet_mesh, f"coboundary_{l - 1}").to(device)
            l_1_cochain = t.randn(n_simp_map[l - 1]).to(device)
            l_cochain = d_l_1 @ l_1_cochain

        anti_cup_kl = AntisymmetricCupProduct(k, l, hollow_tet_mesh).to(device)

        proj_kl = WhitneyWedgeProjection(k, l, hollow_tet_mesh).to(device)
        mass = _compute_mass_matrix(hollow_tet_mesh, k + l).to_dense().to(device)
        wedge_kl = t.linalg.solve(mass, proj_kl(k_cochain, l_cochain))

        t.testing.assert_close(
            wedge_kl.sum(),
            anti_cup_kl(k_cochain, l_cochain).sum(),
        )


# TODO: test other pairing methods

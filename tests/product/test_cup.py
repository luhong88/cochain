import itertools

import pytest
import torch as t

from cochain.complex import SimplicialComplex
from cochain.geometry.tri.tri_geometry import compute_tri_areas
from cochain.product.cup import AntisymmetricCupProduct, CupProduct


def test_cup_product_patch(square_mesh: SimplicialComplex, device):
    """
    For a tri mesh on the z = 0 plane, the cup product between the constant
    1-forms dx and dy is not expected to exactly match the area 2-form dx ⋀ dy.
    But the absolute sum of the cup product 2-form over all 2-simplices should
    match the sum of the area form (i.e., the total area is invariant).
    """
    d_0 = square_mesh.coboundary_0.to(device)

    x = square_mesh.vert_coords[:, 0].to(device)
    y = square_mesh.vert_coords[:, 1].to(device)

    dx = d_0 @ x
    dy = d_0 @ y

    wedge = AntisymmetricCupProduct(1, 1, square_mesh).to(device)

    dxdy = wedge(dx, dy)

    tri_areas = compute_tri_areas(square_mesh.vert_coords, square_mesh.tris).to(device)

    t.testing.assert_close(dxdy.abs().sum(), tri_areas.sum())


def test_antisymmetric_cup_product_patch(square_mesh: SimplicialComplex, device):
    """
    For a tri mesh on the z = 0 plane, the antisymmetric cup product between the
    constant 1-forms dx and dy should exactly match the area 2-form dx ⋀ dy, up
    to a sign flip.
    """
    d_0 = square_mesh.coboundary_0.to(device)

    x = square_mesh.vert_coords[:, 0].to(device)
    y = square_mesh.vert_coords[:, 1].to(device)

    dx = d_0 @ x
    dy = d_0 @ y

    wedge = AntisymmetricCupProduct(1, 1, square_mesh).to(device)

    dxdy = wedge(dx, dy)

    tri_areas = compute_tri_areas(square_mesh.vert_coords, square_mesh.tris).to(device)

    t.testing.assert_close(dxdy.abs(), tri_areas)


@pytest.mark.parametrize("mesh_name", ["two_tris_mesh", "two_tets_mesh"])
def test_cup_product_bilinearity(mesh_name, request, device):
    mesh: SimplicialComplex = request.getfixturevalue(mesh_name)

    n_simp_map = {
        dim: n_simp
        for dim, n_simp in enumerate(
            [mesh.n_verts, mesh.n_edges, mesh.n_tris, mesh.n_tets]
        )
    }

    for k, l in itertools.product(range(mesh.dim), repeat=2):
        if k + l <= mesh.dim:
            k1_cochain = t.randn(n_simp_map[k]).to(device)
            k2_cochain = t.randn(n_simp_map[k]).to(device)
            l_cochain = t.randn(n_simp_map[l]).to(device)

            c1, c2 = t.randn(2)

            wedge_kl = CupProduct(k, l, mesh).to(device)

            lhs = wedge_kl(c1 * k1_cochain + c2 * k2_cochain, l_cochain)
            rhs = c1 * wedge_kl(k1_cochain, l_cochain) + c2 * wedge_kl(
                k2_cochain, l_cochain
            )

            t.testing.assert_close(lhs, rhs)

            k_cochain = t.randn(n_simp_map[k]).to(device)
            l1_cochain = t.randn(n_simp_map[l]).to(device)
            l2_cochain = t.randn(n_simp_map[l]).to(device)

            c1, c2 = t.randn(2)

            lhs = wedge_kl(k_cochain, c1 * l1_cochain + c2 * l2_cochain)
            rhs = c1 * wedge_kl(k_cochain, l1_cochain) + c2 * wedge_kl(
                k_cochain, l2_cochain
            )

            t.testing.assert_close(lhs, rhs)


@pytest.mark.parametrize("mesh_name", ["two_tris_mesh", "two_tets_mesh"])
def test_antisymmetric_cup_product_bilinearity(mesh_name, request, device):
    mesh: SimplicialComplex = request.getfixturevalue(mesh_name)

    n_simp_map = {
        dim: n_simp
        for dim, n_simp in enumerate(
            [mesh.n_verts, mesh.n_edges, mesh.n_tris, mesh.n_tets]
        )
    }

    for k, l in itertools.product(range(mesh.dim), repeat=2):
        if k + l <= mesh.dim:
            k1_cochain = t.randn(n_simp_map[k]).to(device)
            k2_cochain = t.randn(n_simp_map[k]).to(device)
            l_cochain = t.randn(n_simp_map[l]).to(device)

            c1, c2 = t.randn(2)

            wedge_kl = AntisymmetricCupProduct(k, l, mesh).to(device)

            lhs = wedge_kl(c1 * k1_cochain + c2 * k2_cochain, l_cochain)
            rhs = c1 * wedge_kl(k1_cochain, l_cochain) + c2 * wedge_kl(
                k2_cochain, l_cochain
            )

            t.testing.assert_close(lhs, rhs)

            k_cochain = t.randn(n_simp_map[k]).to(device)
            l1_cochain = t.randn(n_simp_map[l]).to(device)
            l2_cochain = t.randn(n_simp_map[l]).to(device)

            c1, c2 = t.randn(2)

            lhs = wedge_kl(k_cochain, c1 * l1_cochain + c2 * l2_cochain)
            rhs = c1 * wedge_kl(k_cochain, l1_cochain) + c2 * wedge_kl(
                k_cochain, l2_cochain
            )

            t.testing.assert_close(lhs, rhs)


@pytest.mark.parametrize("mesh_name", ["two_tris_mesh", "two_tets_mesh"])
def test_antisymmetric_cup_product_graded_commutativity(mesh_name, request, device):
    mesh: SimplicialComplex = request.getfixturevalue(mesh_name)

    n_simp_map = {
        dim: n_simp
        for dim, n_simp in enumerate(
            [mesh.n_verts, mesh.n_edges, mesh.n_tris, mesh.n_tets]
        )
    }

    for k, l in itertools.product(range(mesh.dim), repeat=2):
        if k + l <= mesh.dim:
            k_cochain = t.randn(n_simp_map[k]).to(device)
            l_cochain = t.randn(n_simp_map[l]).to(device)

            wedge_kl = AntisymmetricCupProduct(k, l, mesh).to(device)
            wedge_lk = AntisymmetricCupProduct(l, k, mesh).to(device)

            sign = (-1.0) ** (k * l)

            lhs = wedge_kl(k_cochain, l_cochain)
            rhs = sign * wedge_lk(l_cochain, k_cochain)

            t.testing.assert_close(lhs, rhs)


@pytest.mark.parametrize("mesh_name", ["two_tris_mesh", "two_tets_mesh"])
def test_cup_product_graded_commutativity(mesh_name, request, device):
    mesh: SimplicialComplex = request.getfixturevalue(mesh_name)

    n_simp_map = {
        dim: n_simp
        for dim, n_simp in enumerate(
            [mesh.n_verts, mesh.n_edges, mesh.n_tris, mesh.n_tets]
        )
    }

    for k, l in itertools.product(range(mesh.dim), repeat=2):
        if k + l <= mesh.dim:
            k_cochain = t.randn(n_simp_map[k]).to(device)
            l_cochain = t.randn(n_simp_map[l]).to(device)

            wedge_kl = CupProduct(k, l, mesh).to(device)
            wedge_lk = CupProduct(l, k, mesh).to(device)

            sign = (-1.0) ** (k * l)

            lhs = wedge_kl(k_cochain, l_cochain)
            rhs = sign * wedge_lk(l_cochain, k_cochain)

            if k == 0 and l == 0:
                t.testing.assert_close(lhs, rhs)
            else:
                assert not t.allclose(lhs, rhs)


@pytest.mark.parametrize("mesh_name", ["two_tris_mesh", "two_tets_mesh"])
def test_cup_product_associativity(mesh_name, request, device):
    mesh: SimplicialComplex = request.getfixturevalue(mesh_name)

    n_simp_map = {
        dim: n_simp
        for dim, n_simp in enumerate(
            [mesh.n_verts, mesh.n_edges, mesh.n_tris, mesh.n_tets]
        )
    }

    for u, v, w in itertools.product(range(mesh.dim), repeat=3):
        if u + v + w <= mesh.dim:
            u_cochain = t.randn(n_simp_map[u]).to(device)
            v_cochain = t.randn(n_simp_map[v]).to(device)
            w_cochain = t.randn(n_simp_map[w]).to(device)

            wedge_uv = CupProduct(u, v, mesh).to(device)
            wedge_vw = CupProduct(v, w, mesh).to(device)
            wedge_u_vw = CupProduct(u, v + w, mesh).to(device)
            wedge_uv_w = CupProduct(u + v, w, mesh).to(device)

            lhs = wedge_uv_w(wedge_uv(u_cochain, v_cochain), w_cochain)
            rhs = wedge_u_vw(u_cochain, wedge_vw(v_cochain, w_cochain))

            t.testing.assert_close(lhs, rhs)


@pytest.mark.parametrize("mesh_name", ["two_tris_mesh", "two_tets_mesh"])
def test_cup_product_leibniz(mesh_name, request, device):
    mesh: SimplicialComplex = request.getfixturevalue(mesh_name)

    n_simp_map = {
        dim: n_simp
        for dim, n_simp in enumerate(
            [mesh.n_verts, mesh.n_edges, mesh.n_tris, mesh.n_tets]
        )
    }

    for k, l in itertools.product(range(mesh.dim), repeat=2):
        m = k + l

        if m < mesh.dim:
            # LHS: d(ξ ⋀ η)
            d_m = getattr(mesh, f"coboundary_{m}").to(device)

            k_cochain = t.randn(n_simp_map[k]).to(device)
            l_cochain = t.randn(n_simp_map[l]).to(device)

            wedge_kl = CupProduct(k, l, mesh).to(device)
            lhs = d_m @ wedge_kl(k_cochain, l_cochain)

            # RHS: dξ ⋀ η + (-1)^k * (ξ ⋀ dη)
            d_k = getattr(mesh, f"coboundary_{k}").to(device)
            d_l = getattr(mesh, f"coboundary_{l}").to(device)

            k1_cochain = d_k @ k_cochain
            l1_cochain = d_l @ l_cochain

            wedge_k1l = CupProduct(k + 1, l, mesh).to(device)
            wedge_kl1 = CupProduct(k, l + 1, mesh).to(device)

            sign = (-1.0) ** k

            rhs = wedge_k1l(k1_cochain, l_cochain) + sign * wedge_kl1(
                k_cochain, l1_cochain
            )

            t.testing.assert_close(lhs, rhs)


def test_cup_product_cohomology_class(hollow_tet_mesh, device):
    """
    The cup product and the antisymmetric cup product should belong to the same
    cohomology class (i.e., differ by a coboundary). Therefore, on a closed mesh,
    the surface integral of the two products of exact forms should match.
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

        cup_kl = CupProduct(k, l, hollow_tet_mesh).to(device)
        anti_cup_kl = AntisymmetricCupProduct(k, l, hollow_tet_mesh).to(device)

        t.testing.assert_close(
            cup_kl(k_cochain, l_cochain).sum(),
            anti_cup_kl(k_cochain, l_cochain).sum(),
        )

import itertools

import pytest
import torch
from einops import repeat

from cochain.cochain.ext_prod.cup import AntisymmetricCupProduct, CupProduct
from cochain.complex import SimplicialMesh
from cochain.metric.tri._tri_geometry import compute_tri_areas


def test_cup_product_patch(square_mesh: SimplicialMesh, device):
    """
    Perform patch test on the cup product.

    For a tri mesh on the z = 0 plane, the cup product between the constant
    1-forms dx and dy is not expected to exactly match the area 2-form dx ⋀ dy.
    But the absolute sum of the cup product 2-form over all 2-simplices should
    match the sum of the area form (i.e., the total area is invariant).
    """
    d_0 = square_mesh.cbd[0].to(device)

    x = square_mesh.vert_coords[:, 0].to(device)
    y = square_mesh.vert_coords[:, 1].to(device)

    dx = d_0 @ x
    dy = d_0 @ y

    wedge = CupProduct(1, 1, square_mesh).to(device)

    dxdy = wedge(dx, dy)

    tri_areas = compute_tri_areas(square_mesh.vert_coords, square_mesh.tris).to(device)

    torch.testing.assert_close(dxdy.abs().sum(), tri_areas.sum())


def test_antisymmetric_cup_product_patch(square_mesh: SimplicialMesh, device):
    """
    Perform patch test on the antisymmetric cup product.

    For a tri mesh on the z = 0 plane, the antisymmetric cup product between the
    constant 1-forms dx and dy should exactly match the area 2-form dx ⋀ dy, up
    to a sign flip.
    """
    d_0 = square_mesh.cbd[0].to(device)

    x = square_mesh.vert_coords[:, 0].to(device)
    y = square_mesh.vert_coords[:, 1].to(device)

    dx = d_0 @ x
    dy = d_0 @ y

    wedge = AntisymmetricCupProduct(1, 1, square_mesh).to(device)

    dxdy = wedge(dx, dy)

    tri_areas = compute_tri_areas(square_mesh.vert_coords, square_mesh.tris).to(device)

    torch.testing.assert_close(dxdy.abs(), tri_areas)


@pytest.mark.parametrize("mesh_name", ["two_tris_mesh", "two_tets_mesh"])
def test_cup_product_bilinearity(mesh_name, request, device):
    mesh: SimplicialMesh = request.getfixturevalue(mesh_name)

    n_splx_map = mesh.n_splx

    for k, l in itertools.product(range(mesh.dim + 1), repeat=2):
        if k + l <= mesh.dim:
            k1_cochain = torch.randn(n_splx_map[k]).to(device)
            k2_cochain = torch.randn(n_splx_map[k]).to(device)
            l_cochain = torch.randn(n_splx_map[l]).to(device)

            c1, c2 = torch.randn(2)

            wedge_kl = CupProduct(k, l, mesh).to(device)

            lhs = wedge_kl(c1 * k1_cochain + c2 * k2_cochain, l_cochain)
            rhs = c1 * wedge_kl(k1_cochain, l_cochain) + c2 * wedge_kl(
                k2_cochain, l_cochain
            )

            torch.testing.assert_close(lhs, rhs)

            k_cochain = torch.randn(n_splx_map[k]).to(device)
            l1_cochain = torch.randn(n_splx_map[l]).to(device)
            l2_cochain = torch.randn(n_splx_map[l]).to(device)

            c1, c2 = torch.randn(2)

            lhs = wedge_kl(k_cochain, c1 * l1_cochain + c2 * l2_cochain)
            rhs = c1 * wedge_kl(k_cochain, l1_cochain) + c2 * wedge_kl(
                k_cochain, l2_cochain
            )

            torch.testing.assert_close(lhs, rhs)


@pytest.mark.parametrize("mesh_name", ["two_tris_mesh", "two_tets_mesh"])
def test_antisymmetric_cup_product_bilinearity(mesh_name, request, device):
    mesh: SimplicialMesh = request.getfixturevalue(mesh_name)

    n_splx_map = mesh.n_splx

    for k, l in itertools.product(range(mesh.dim + 1), repeat=2):
        if k + l <= mesh.dim:
            k1_cochain = torch.randn(n_splx_map[k]).to(device)
            k2_cochain = torch.randn(n_splx_map[k]).to(device)
            l_cochain = torch.randn(n_splx_map[l]).to(device)

            c1, c2 = torch.randn(2)

            wedge_kl = AntisymmetricCupProduct(k, l, mesh).to(device)

            lhs = wedge_kl(c1 * k1_cochain + c2 * k2_cochain, l_cochain)
            rhs = c1 * wedge_kl(k1_cochain, l_cochain) + c2 * wedge_kl(
                k2_cochain, l_cochain
            )

            torch.testing.assert_close(lhs, rhs)

            k_cochain = torch.randn(n_splx_map[k]).to(device)
            l1_cochain = torch.randn(n_splx_map[l]).to(device)
            l2_cochain = torch.randn(n_splx_map[l]).to(device)

            c1, c2 = torch.randn(2)

            lhs = wedge_kl(k_cochain, c1 * l1_cochain + c2 * l2_cochain)
            rhs = c1 * wedge_kl(k_cochain, l1_cochain) + c2 * wedge_kl(
                k_cochain, l2_cochain
            )

            torch.testing.assert_close(lhs, rhs)


@pytest.mark.parametrize("mesh_name", ["two_tris_mesh", "two_tets_mesh"])
def test_antisymmetric_cup_product_graded_commutativity(mesh_name, request, device):
    mesh: SimplicialMesh = request.getfixturevalue(mesh_name)

    n_splx_map = mesh.n_splx

    for k, l in itertools.product(range(mesh.dim + 1), repeat=2):
        if k + l <= mesh.dim:
            k_cochain = torch.randn(n_splx_map[k]).to(device)
            l_cochain = torch.randn(n_splx_map[l]).to(device)

            wedge_kl = AntisymmetricCupProduct(k, l, mesh).to(device)
            wedge_lk = AntisymmetricCupProduct(l, k, mesh).to(device)

            sign = (-1.0) ** (k * l)

            lhs = wedge_kl(k_cochain, l_cochain)
            rhs = sign * wedge_lk(l_cochain, k_cochain)

            torch.testing.assert_close(lhs, rhs)


@pytest.mark.parametrize("mesh_name", ["two_tris_mesh", "two_tets_mesh"])
def test_cup_product_graded_commutativity(mesh_name, request, device):
    mesh: SimplicialMesh = request.getfixturevalue(mesh_name)

    n_splx_map = mesh.n_splx

    for k, l in itertools.product(range(mesh.dim + 1), repeat=2):
        if k + l <= mesh.dim:
            k_cochain = torch.randn(n_splx_map[k]).to(device)
            l_cochain = torch.randn(n_splx_map[l]).to(device)

            wedge_kl = CupProduct(k, l, mesh).to(device)
            wedge_lk = CupProduct(l, k, mesh).to(device)

            sign = (-1.0) ** (k * l)

            lhs = wedge_kl(k_cochain, l_cochain)
            rhs = sign * wedge_lk(l_cochain, k_cochain)

            if k == 0 and l == 0:
                torch.testing.assert_close(lhs, rhs)
            else:
                assert not torch.allclose(lhs, rhs)


@pytest.mark.parametrize("mesh_name", ["two_tris_mesh", "two_tets_mesh"])
def test_cup_product_associativity(mesh_name, request, device):
    mesh: SimplicialMesh = request.getfixturevalue(mesh_name)

    n_splx_map = mesh.n_splx

    for u, v, w in itertools.product(range(mesh.dim + 1), repeat=3):
        if u + v + w <= mesh.dim:
            u_cochain = torch.randn(n_splx_map[u]).to(device)
            v_cochain = torch.randn(n_splx_map[v]).to(device)
            w_cochain = torch.randn(n_splx_map[w]).to(device)

            wedge_uv = CupProduct(u, v, mesh).to(device)
            wedge_vw = CupProduct(v, w, mesh).to(device)
            wedge_u_vw = CupProduct(u, v + w, mesh).to(device)
            wedge_uv_w = CupProduct(u + v, w, mesh).to(device)

            lhs = wedge_uv_w(wedge_uv(u_cochain, v_cochain), w_cochain)
            rhs = wedge_u_vw(u_cochain, wedge_vw(v_cochain, w_cochain))

            torch.testing.assert_close(lhs, rhs)


@pytest.mark.parametrize("mesh_name", ["two_tris_mesh", "two_tets_mesh"])
def test_cup_product_leibniz(mesh_name, request, device):
    mesh: SimplicialMesh = request.getfixturevalue(mesh_name)

    n_splx_map = mesh.n_splx

    for k, l in itertools.product(range(mesh.dim + 1), repeat=2):
        m = k + l

        if m < mesh.dim:
            # LHS: d(ξ ⋀ η)
            d_m = mesh.cbd[m].to(device)

            k_cochain = torch.randn(n_splx_map[k]).to(device)
            l_cochain = torch.randn(n_splx_map[l]).to(device)

            wedge_kl = CupProduct(k, l, mesh).to(device)
            lhs = d_m @ wedge_kl(k_cochain, l_cochain)

            # RHS: dξ ⋀ η + (-1)^k * (ξ ⋀ dη)
            d_k = mesh.cbd[k].to(device)
            d_l = mesh.cbd[l].to(device)

            k1_cochain = d_k @ k_cochain
            l1_cochain = d_l @ l_cochain

            wedge_k1l = CupProduct(k + 1, l, mesh).to(device)
            wedge_kl1 = CupProduct(k, l + 1, mesh).to(device)

            sign = (-1.0) ** k

            rhs = wedge_k1l(k1_cochain, l_cochain) + sign * wedge_kl1(
                k_cochain, l1_cochain
            )

            torch.testing.assert_close(lhs, rhs)


def test_cup_product_cohomology_class(hollow_tet_mesh, device):
    """
    Check that the cup and antisymmetric cup products belong to the same cohomology class.

    The cup product and the antisymmetric cup product should belong to the same
    cohomology class (i.e., differ by a coboundary). Therefore, on a closed mesh,
    the surface integral of the two products of exact forms should match.
    """
    n_splx_map = hollow_tet_mesh.n_splx

    for k in range(hollow_tet_mesh.dim + 1):
        l = hollow_tet_mesh.dim - k

        if k == 0:
            k_cochain = torch.randn(1).expand(n_splx_map[k]).to(device)
        if k > 0:
            d_km1 = hollow_tet_mesh.cbd[k - 1].to(device)
            k_1_cochain = torch.randn(n_splx_map[k - 1]).to(device)
            k_cochain = d_km1 @ k_1_cochain

        if l == 0:
            l_cochain = torch.randn(1).expand(n_splx_map[l]).to(device)
        if l > 0:
            d_lm1 = hollow_tet_mesh.cbd[l - 1].to(device)
            l_1_cochain = torch.randn(n_splx_map[l - 1]).to(device)
            l_cochain = d_lm1 @ l_1_cochain

        cup_kl = CupProduct(k, l, hollow_tet_mesh).to(device)
        anti_cup_kl = AntisymmetricCupProduct(k, l, hollow_tet_mesh).to(device)

        torch.testing.assert_close(
            cup_kl(k_cochain, l_cochain).sum(),
            anti_cup_kl(k_cochain, l_cochain).sum(),
        )


@pytest.mark.parametrize("operator", [CupProduct, AntisymmetricCupProduct])
@pytest.mark.parametrize("mesh_name", ["two_tris_mesh", "two_tets_mesh"])
def test_cup_product_pairing_methods(mesh_name, operator, request, device):
    mesh: SimplicialMesh = request.getfixturevalue(mesh_name).to(device)

    n_splx_map = mesh.n_splx

    for k, l in itertools.product(range(mesh.dim + 1), repeat=2):
        if k + l <= mesh.dim:
            # We use channel dimension = 3 so that the cross product is defined.
            k_cochain = torch.randn((n_splx_map[k], 3), device=device)
            l_cochain = torch.randn((n_splx_map[l], 3), device=device)

            wedge_kl = operator(k, l, mesh).to(device)

            # Check dot product pairing over the channel dim.
            dot_prod = wedge_kl(k_cochain, l_cochain, pairing="dot")
            scalar_prod = wedge_kl(k_cochain, l_cochain, pairing="scalar")
            torch.testing.assert_close(dot_prod, scalar_prod.sum(dim=-1, keepdim=True))

            # Outer product can be computed using the scalar pairing with expanded inputs.
            k_cochain_exp = repeat(k_cochain, "splx ch1 -> splx ch1 ch2", ch2=3)
            l_cochain_exp = repeat(l_cochain, "splx ch2 -> splx ch1 ch2", ch1=3)
            outer_prod_ref = wedge_kl(k_cochain_exp, l_cochain_exp, pairing="scalar")

            outer_prod = wedge_kl(k_cochain, l_cochain, pairing="outer")
            torch.testing.assert_close(outer_prod, outer_prod_ref)

            # Cross product can be computed using the outer product.
            # To see why, note that, for the k-cochain ξ and l-cochain η, the
            # wedge product with pairing="outer" computes the matrix, over each
            # (k+l)-simplex,
            #
            # | ξ_x η_x ξ_x η_y ξ_x η_z |
            # | ξ_y η_x ξ_y η_y ξ_y η_z |
            # | ξ_z η_x ξ_z η_y ξ_z η_z |
            #
            # where ξ_i is the i-th coordinate value of ξ at the k-front face and
            # η_j is the j-th coordinate value of η at the k-back face. The values
            # in this matrix can then be used to compute the cross product, which
            # is given by the determinant
            #
            # | e_x e_y e_z |
            # | ξ_y ξ_y ξ_y |
            # | η_x η_y η_z |
            cross_prod = wedge_kl(k_cochain, l_cochain, pairing="cross")
            cross_prod_ref = torch.stack(
                [
                    outer_prod[:, 1, 2] - outer_prod[:, 2, 1],
                    outer_prod[:, 2, 0] - outer_prod[:, 0, 2],
                    outer_prod[:, 0, 1] - outer_prod[:, 1, 0],
                ],
                dim=-1,
            )

            torch.testing.assert_close(cross_prod, cross_prod_ref)

            with pytest.raises(ValueError, match="Unknown pairing method"):
                wedge_kl(k_cochain, l_cochain, pairing="unknown")


@pytest.mark.parametrize("operator", [CupProduct, AntisymmetricCupProduct])
@pytest.mark.parametrize("mesh_name", ["two_tris_mesh", "two_tets_mesh"])
@pytest.mark.parametrize("pairing", ["scalar", "dot", "cross", "outer"])
def test_cup_product_backward(mesh_name, operator, pairing, request, device):
    mesh: SimplicialMesh = request.getfixturevalue(mesh_name).to(device)

    n_splx_map = mesh.n_splx

    for k, l in itertools.product(range(mesh.dim + 1), repeat=2):
        if k + l <= mesh.dim:
            k_cochain = torch.randn((n_splx_map[k], 3), device=device)
            k_cochain.requires_grad_()

            l_cochain = torch.randn((n_splx_map[l], 3), device=device)
            l_cochain.requires_grad_()

            wedge_kl = operator(k, l, mesh)

            m_cochain = wedge_kl(k_cochain, l_cochain, pairing=pairing)

            output = m_cochain.sum()
            output.backward()

            assert k_cochain.grad is not None
            assert torch.isfinite(k_cochain.grad).all()

            assert l_cochain.grad is not None
            assert torch.isfinite(l_cochain.grad).all()


@pytest.mark.parametrize("operator", [CupProduct, AntisymmetricCupProduct])
@pytest.mark.parametrize("mesh_name", ["two_tris_mesh", "two_tets_mesh"])
@pytest.mark.parametrize("pairing", ["scalar", "dot", "cross", "outer"])
def test_cup_product_gradcheck(mesh_name, operator, pairing, request, device):
    mesh: SimplicialMesh = request.getfixturevalue(mesh_name).to(
        dtype=torch.float64, device=device
    )

    n_splx_map = mesh.n_splx

    for k, l in itertools.product(range(mesh.dim + 1), repeat=2):
        if k + l <= mesh.dim:
            k_cochain = torch.randn(
                (n_splx_map[k], 3), dtype=torch.float64, device=device
            )
            k_cochain.requires_grad_()

            l_cochain = torch.randn(
                (n_splx_map[l], 3), dtype=torch.float64, device=device
            )
            l_cochain.requires_grad_()

            wedge_kl = operator(k, l, mesh)

            def cup_prod_fxn(test_k_cochain, test_l_cochain, wedge_op):
                m_cochain = wedge_op(test_k_cochain, test_l_cochain, pairing=pairing)
                output = m_cochain.sum()
                return output

            assert torch.autograd.gradcheck(
                cup_prod_fxn,
                (k_cochain, l_cochain, wedge_kl),
                fast_mode=True,
            )

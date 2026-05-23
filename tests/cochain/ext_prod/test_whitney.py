import itertools

import pytest
import torch
from einops import repeat

from cochain.cochain.ext_prod.cup import AntisymmetricCupProduct
from cochain.cochain.ext_prod.whitney import WhitneyWedgeL2Projector
from cochain.complex import SimplicialMesh
from cochain.metric.tet import tet_masses
from cochain.metric.tri import tri_masses


def _compute_mass_matrix(mesh: SimplicialMesh, k: int):
    match mesh.dim:
        case 2:
            mass = getattr(tri_masses, f"mass_{k}")(mesh)
        case 3:
            mass = getattr(tet_masses, f"mass_{k}")(mesh)
        case _:
            raise NotImplementedError()

    return mass


@pytest.mark.parametrize("mesh_name", ["two_tris_mesh", "two_tets_mesh"])
def test_const_0_form_galerkin_wedge_product(mesh_name, request, device):
    """
    Check the wedge product between a constant 0-form and an arbitrary k-form.

    The wedge product between a constant 0-form and an arbitrary k-form should
    be equal to a constant multiple of the original k-form.
    """
    mesh: SimplicialMesh = request.getfixturevalue(mesh_name).to(device)

    n_splx_map = mesh.n_splx

    coef = torch.randn(1).to(dtype=mesh.dtype, device=device)
    const_0_form = coef * torch.ones(n_splx_map[0], dtype=mesh.dtype, device=device)

    for k in range(mesh.dim + 1):
        k_form = torch.randn(n_splx_map[k]).to(device)

        proj = WhitneyWedgeL2Projector(0, k, mesh)

        b = proj(const_0_form, k_form)

        mass = _compute_mass_matrix(mesh, k).to_dense().to(device)

        w = torch.linalg.solve(mass, b)

        torch.testing.assert_close(w, coef * k_form)


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
    Check that the wedge and antisymmetric cup products belong to the same cohomology class.

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


@pytest.mark.parametrize("mesh_name", ["two_tris_mesh", "two_tets_mesh"])
def test_galerkin_wedge_pairing_methods(mesh_name, request, device):
    mesh: SimplicialMesh = request.getfixturevalue(mesh_name).to(device)

    n_splx_map = mesh.n_splx

    for k, l in itertools.product(range(mesh.dim + 1), repeat=2):
        if k + l <= mesh.dim:
            # We use channel dimension = 3 so that the cross product is defined.
            k_cochain = torch.randn((n_splx_map[k], 3), device=device)
            l_cochain = torch.randn((n_splx_map[l], 3), device=device)

            proj_kl = WhitneyWedgeL2Projector(k, l, mesh).to(device)

            # Check dot product pairing over the channel dim.
            dot_prod = proj_kl(k_cochain, l_cochain, pairing="dot")
            scalar_prod = proj_kl(k_cochain, l_cochain, pairing="scalar")
            torch.testing.assert_close(dot_prod, scalar_prod.sum(dim=-1, keepdim=True))

            # Outer product can be computed using the scalar pairing with expanded inputs.
            k_cochain_exp = repeat(k_cochain, "splx ch1 -> splx ch1 ch2", ch2=3)
            l_cochain_exp = repeat(l_cochain, "splx ch2 -> splx ch1 ch2", ch1=3)
            outer_prod_ref = proj_kl(k_cochain_exp, l_cochain_exp, pairing="scalar")

            outer_prod = proj_kl(k_cochain, l_cochain, pairing="outer")
            torch.testing.assert_close(outer_prod, outer_prod_ref)

            # Cross product can be computed using the outer product.
            cross_prod = proj_kl(k_cochain, l_cochain, pairing="cross")
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
                proj_kl(k_cochain, l_cochain, pairing="unknown")


@pytest.mark.parametrize("mesh_name", ["two_tris_mesh", "two_tets_mesh"])
@pytest.mark.parametrize("pairing", ["scalar", "dot", "cross", "outer"])
def test_galerkin_wedge_product_backward(mesh_name, pairing, request, device):
    mesh: SimplicialMesh = request.getfixturevalue(mesh_name).to(device)
    mesh.requires_grad_()

    n_splx_map = mesh.n_splx

    for k, l in itertools.product(range(mesh.dim + 1), repeat=2):
        if k + l <= mesh.dim:
            k_cochain = torch.randn((n_splx_map[k], 3)).to(device)
            k_cochain.requires_grad_()

            l_cochain = torch.randn((n_splx_map[l], 3)).to(device)
            l_cochain.requires_grad_()

            proj_kl = WhitneyWedgeL2Projector(k, l, mesh).to(device)

            load = proj_kl(k_cochain, l_cochain, pairing=pairing)

            output = load.sum()
            output.backward()

            assert mesh.grad is not None
            assert torch.isfinite(mesh.grad).all()

            assert k_cochain.grad is not None
            assert torch.isfinite(k_cochain.grad).all()

            assert l_cochain.grad is not None
            assert torch.isfinite(l_cochain.grad).all()


@pytest.mark.parametrize("mesh_name", ["two_tris_mesh", "two_tets_mesh"])
@pytest.mark.parametrize("pairing", ["scalar", "dot", "cross", "outer"])
def test_galerkin_wedge_product_gradcheck(mesh_name, pairing, request, device):
    mesh: SimplicialMesh = request.getfixturevalue(mesh_name)

    vert_coords = mesh.vert_coords.clone().to(dtype=torch.float64, device=device)
    vert_coords.requires_grad_()

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

            def wedge_prod_fxn(test_vert_coords, test_k_cochain, test_l_cochain):
                test_mesh = mesh.to(dtype=torch.float64, device=device)
                test_mesh.vert_coords = test_vert_coords

                proj_kl = WhitneyWedgeL2Projector(k, l, test_mesh).to(device)
                load = proj_kl(test_k_cochain, test_l_cochain, pairing=pairing)

                output = load.sum()
                return output

            assert torch.autograd.gradcheck(
                wedge_prod_fxn,
                (vert_coords, k_cochain, l_cochain),
                fast_mode=True,
            )

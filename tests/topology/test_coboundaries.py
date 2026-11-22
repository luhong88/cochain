import torch

from cochain.boundary import BoundaryRestrictionCache, BoundarySelection
from cochain.complex import SimplicialMesh


def test_exactness_on_tet(hollow_tet_mesh: SimplicialMesh, device):
    mesh = hollow_tet_mesh.to(device)
    d1_d0 = (mesh.cbd[1] @ mesh.cbd[0]).to_dense()
    torch.testing.assert_close(d1_d0, torch.zeros_like(d1_d0))


def test_exactness_on_tent(tent_mesh: SimplicialMesh, device):
    mesh = tent_mesh.to(device)
    d1_d0 = (mesh.cbd[1] @ mesh.cbd[0]).to_dense()
    torch.testing.assert_close(d1_d0, torch.zeros_like(d1_d0))


def test_exactness_on_two_tets(two_tets_mesh: SimplicialMesh, device):
    mesh = two_tets_mesh.to(device)

    d1_d0 = (mesh.cbd[1] @ mesh.cbd[0]).to_dense()
    torch.testing.assert_close(d1_d0, torch.zeros_like(d1_d0))

    d2_d1 = (mesh.cbd[2] @ mesh.cbd[1]).to_dense()
    torch.testing.assert_close(d2_d1, torch.zeros_like(d2_d1))


def test_exactness_on_annulus_with_boundary_conditions(
    flat_annulus_mesh: SimplicialMesh, device
):
    mesh = flat_annulus_mesh.to(device)

    bc_sels = [
        BoundarySelection.from_relative_bc(mesh),
        BoundarySelection.from_absolute_bc(mesh),
    ]

    for comp in [torch.le, torch.ge]:
        inner_edge_mask = mesh.bd_edge_mask & comp(
            torch.linalg.vector_norm(mesh.vert_coords[mesh.edges], dim=-1), 0.75
        ).all(dim=-1)
        bc_sel = BoundarySelection.from_tangential_bd_mask(mesh, inner_edge_mask)
        bc_sels.append(bc_sel)

    for bc_sel in bc_sels:
        bc_cache = BoundaryRestrictionCache.from_mesh(mesh, bc_sel)
        d1_d0 = (bc_cache.restricted_cbd[1] @ bc_cache.restricted_cbd[0]).to_dense()
        torch.testing.assert_close(d1_d0, torch.zeros_like(d1_d0))


def test_exactness_on_solid_spherical_shell_with_boundary_conditions(
    solid_spherical_shell_mesh: SimplicialMesh, device
):
    mesh = solid_spherical_shell_mesh.to(device)

    bc_sels = [
        BoundarySelection.from_relative_bc(mesh),
        BoundarySelection.from_absolute_bc(mesh),
    ]

    for comp in [torch.le, torch.ge]:
        inner_surface_mask = mesh.bd_tri_mask & comp(
            torch.linalg.vector_norm(mesh.vert_coords[mesh.tris], dim=-1), 1.5
        ).all(dim=-1)
        bc_sel = BoundarySelection.from_tangential_bd_mask(mesh, inner_surface_mask)
        bc_sels.append(bc_sel)

    for bc_sel in bc_sels:
        bc_cache = BoundaryRestrictionCache.from_mesh(mesh, bc_sel)

        d1_d0 = (bc_cache.restricted_cbd[1] @ bc_cache.restricted_cbd[0]).to_dense()
        torch.testing.assert_close(d1_d0, torch.zeros_like(d1_d0))

        d2_d1 = (bc_cache.restricted_cbd[2] @ bc_cache.restricted_cbd[1]).to_dense()
        torch.testing.assert_close(d2_d1, torch.zeros_like(d2_d1))

import pytest
import torch

from cochain.boundary.selection import BoundarySelection
from cochain.complex import SimplicialMesh
from cochain.topology.betti import (
    _betti_via_morse,
    _tri_manifold_betti_via_trees,
    compute_betti_numbers,
)


@pytest.mark.parametrize(
    "mesh,betti_true",
    [
        ("icosphere_mesh", [1, 0, 1, 0]),
        ("two_tris_mesh", [1, 0, 0, 0]),
        ("two_disjoint_tris_mesh", [2, 0, 0, 0]),
        ("finer_flat_annulus_mesh", [1, 1, 0, 0]),
    ],
)
def test_tri_manifold_betti_via_trees(mesh, betti_true, request, device):
    mesh = request.getfixturevalue(mesh).to(device)

    betti = _tri_manifold_betti_via_trees(mesh)

    for b, b_true in zip(betti, betti_true):
        assert b == b_true


@pytest.mark.parametrize(
    "mesh,betti_true",
    [
        ("icosphere_mesh", [1, 0, 1, 0]),
        ("two_tris_mesh", [0, 0, 1, 0]),
        ("two_disjoint_tris_mesh", [0, 0, 2, 0]),
        ("finer_flat_annulus_mesh", [0, 1, 1, 0]),
    ],
)
def test_tri_manifold_relative_betti_via_trees(mesh, betti_true, request, device):
    mesh = request.getfixturevalue(mesh).to(device)
    bc_sel = BoundarySelection.from_relative_bc(mesh)

    betti = _tri_manifold_betti_via_trees(mesh, bc_sel)

    for b, b_true in zip(betti, betti_true):
        assert b == b_true


def test_two_tris_mesh_relative_betti_with_mixed_bc_via_trees(two_tris_mesh, device):
    mesh = two_tris_mesh.to(device)

    disjoint_edges_mask = torch.tensor(
        [True, False, False, False, True], device=mesh.device
    )
    bc_sel = BoundarySelection.from_tangential_bd_mask(mesh, disjoint_edges_mask)

    betti = _tri_manifold_betti_via_trees(mesh, bc_sel)

    for b, b_true in zip(betti, [0, 1, 0, 0]):
        assert b == b_true


def test_two_disjoint_tri_mesh_relative_betti_with_mixed_bc_via_trees(
    two_disjoint_tris_mesh, device
):
    mesh = two_disjoint_tris_mesh.to(device)

    one_tri_edge_mask = torch.tensor(
        [True, True, True, False, False, False], device=mesh.device
    )
    bc_sel = BoundarySelection.from_tangential_bd_mask(mesh, one_tri_edge_mask)

    betti = _tri_manifold_betti_via_trees(mesh, bc_sel)

    for b, b_true in zip(betti, [1, 0, 1, 0]):
        assert b == b_true


def test_annulus_relative_betti_with_mixed_bc_via_trees(
    finer_flat_annulus_mesh, device
):
    mesh = finer_flat_annulus_mesh.to(device)

    # Put the relative BC on either the inner or outer edge.
    for comp in [torch.le, torch.ge]:
        inner_edge_mask = mesh.bd_edge_mask & comp(
            torch.linalg.norm(mesh.vert_coords[mesh.edges], dim=-1), 0.75
        ).all(dim=-1)
        bc_sel = BoundarySelection.from_tangential_bd_mask(mesh, inner_edge_mask)

        betti = _tri_manifold_betti_via_trees(mesh, bc_sel)

        for b, b_true in zip(betti, [0, 0, 0, 0]):
            assert b == b_true


@pytest.mark.parametrize(
    "mesh,betti_true",
    [
        ("icosphere_mesh", [1, 0, 1, 0]),
        ("two_tris_mesh", [1, 0, 0, 0]),
        ("two_disjoint_tris_mesh", [2, 0, 0, 0]),
        ("finer_flat_annulus_mesh", [1, 1, 0, 0]),
        ("two_tets_mesh", [1, 0, 0, 0]),
        ("solid_torus_mesh", [1, 1, 0, 0]),
        ("solid_spherical_shell_mesh", [1, 0, 1, 0]),
    ],
)
def test_betti_via_morse(mesh, betti_true, request, device):
    mesh = request.getfixturevalue(mesh).to(device)

    betti = _betti_via_morse(mesh)

    for b, b_true in zip(betti, betti_true):
        assert b == b_true


@pytest.mark.parametrize(
    "mesh,betti_true",
    [
        ("icosphere_mesh", [1, 0, 1, 0]),
        ("two_tris_mesh", [0, 0, 1, 0]),
        ("two_disjoint_tris_mesh", [0, 0, 2, 0]),
        ("finer_flat_annulus_mesh", [0, 1, 1, 0]),
        ("two_tets_mesh", [0, 0, 0, 1]),
        ("solid_torus_mesh", [0, 0, 1, 1]),
        ("solid_spherical_shell_mesh", [0, 1, 0, 1]),
    ],
)
def test_relative_betti_via_morse(mesh, betti_true, request, device):
    mesh = request.getfixturevalue(mesh).to(device)
    bc_sel = BoundarySelection.from_relative_bc(mesh)

    betti = _betti_via_morse(mesh, bc_sel)

    for b, b_true in zip(betti, betti_true):
        assert b == b_true


def test_solid_spherical_shell_mesh_relative_betti_with_mixed_bc_via_morse(
    solid_spherical_shell_mesh, device
):
    mesh = solid_spherical_shell_mesh.to(device)

    for comp in [torch.le, torch.ge]:
        inner_surface_mask = mesh.bd_tri_mask & comp(
            torch.linalg.norm(mesh.vert_coords[mesh.tris], dim=-1), 1.5
        ).all(dim=-1)
        bc_sel = BoundarySelection.from_tangential_bd_mask(mesh, inner_surface_mask)

        betti = _betti_via_morse(mesh, bc_sel)

        for b, b_true in zip(betti, [0, 0, 0, 0]):
            assert b == b_true


def test_simple_sc_mesh_relative_betti_with_mixed_bc_via_trees(simple_sc_mesh, device):
    mesh = simple_sc_mesh.to(device)

    disjoint_tris_mask = torch.zeros_like(mesh.bd_tri_mask, dtype=torch.bool)
    disjoint_tris_mask[1] = True
    disjoint_tris_mask[27] = True

    bc_sel = BoundarySelection.from_tangential_bd_mask(mesh, disjoint_tris_mask)

    betti = _betti_via_morse(mesh, bc_sel)

    for b, b_true in zip(betti, [0, 1, 0, 0]):
        assert b == b_true


def test_betti_on_nonmanifold_crease_tri_mesh(device):
    vert_coords = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.5],
            [-1.0, 0.0, 0.5],
            [0.0, 1.0, 0.5],
        ],
        device=device,
    )
    tris = torch.tensor([[0, 1, 2], [0, 1, 3], [0, 1, 4]], device=device)
    nonmanifold_mesh = SimplicialMesh.from_tri_mesh(vert_coords, tris)

    betti = compute_betti_numbers(nonmanifold_mesh)

    for b, b_true in zip(betti, [1, 0, 0, 0]):
        assert b == b_true


def test_betti_on_nonorientable_projective_plane_tri_mesh(device):
    # Generate an abstract six-vertex nonorientable manifold projective plane surface
    x = torch.arange(6, dtype=torch.float32)
    projective_plane = SimplicialMesh.from_tri_mesh(
        vert_coords=torch.stack((x, x**2, x**3), dim=1),
        tris=torch.tensor(
            [
                [0, 1, 2],
                [0, 1, 3],
                [0, 2, 4],
                [0, 3, 5],
                [0, 4, 5],
                [1, 2, 5],
                [1, 3, 4],
                [1, 4, 5],
                [2, 3, 4],
                [2, 3, 5],
            ]
        ),
    ).to(device)

    real_betti = compute_betti_numbers(projective_plane)
    for b, b_true in zip(real_betti, [1, 0, 0, 0]):
        assert b == b_true

    mod_2_betti = compute_betti_numbers(projective_plane, orientable_manifold=True)
    for b, b_true in zip(mod_2_betti, [1, 1, 1, 0]):
        assert b == b_true

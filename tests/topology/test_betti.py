import pytest

from cochain.topology.betti import _betti_via_morse, _tri_manifold_betti_via_trees


@pytest.mark.parametrize(
    "mesh,betti_true",
    [
        ("icosphere_mesh", [1, 0, 1]),
        ("two_tris_mesh", [1, 0, 0]),
        ("two_disjoint_tris_mesh", [2, 0, 0]),
        ("finer_flat_annulus_mesh", [1, 1, 0]),
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
        ("icosphere_mesh", [1, 0, 1]),
        ("two_tris_mesh", [1, 0, 0]),
        ("two_disjoint_tris_mesh", [2, 0, 0]),
        ("finer_flat_annulus_mesh", [1, 1, 0]),
        ("two_tets_mesh", [1, 0, 0]),
        ("solid_torus_mesh", [1, 1, 0]),
        ("solid_spherical_shell_mesh", [1, 0, 1]),
    ],
)
def test_betti_via_morse(mesh, betti_true, request, device):
    mesh = request.getfixturevalue(mesh).to(device)

    betti = _betti_via_morse(mesh)

    for b, b_true in zip(betti, betti_true):
        assert b == b_true

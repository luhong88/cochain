from cochain.complex import SimplicialMesh
from cochain.topology.betti import compute_tri_mesh_betti_numbers


def test_icosphere_betti(icosphere_mesh: SimplicialMesh, device):
    b0, b1, b2 = compute_tri_mesh_betti_numbers(icosphere_mesh.to(device))

    assert b0 == 1
    assert b1 == 0
    assert b2 == 1


def test_two_tris_betti(two_tris_mesh: SimplicialMesh, device):
    b0, b1, b2 = compute_tri_mesh_betti_numbers(two_tris_mesh.to(device))

    assert b0 == 1
    assert b1 == 0
    assert b2 == 0


def test_two_disjoint_tris_betti(two_disjoint_tris_mesh: SimplicialMesh, device):
    b0, b1, b2 = compute_tri_mesh_betti_numbers(two_disjoint_tris_mesh.to(device))

    assert b0 == 2
    assert b1 == 0
    assert b2 == 0


def test_annulus_betti(finer_flat_annulus_mesh: SimplicialMesh, device):
    b0, b1, b2 = compute_tri_mesh_betti_numbers(finer_flat_annulus_mesh.to(device))

    assert b0 == 1
    assert b1 == 1
    assert b2 == 0

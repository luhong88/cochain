from cochain.complex import SimplicialComplex
from cochain.topology.betti import compute_tri_mesh_betti_numbers


def test_hollow_tet_betti(hollow_tet_mesh: SimplicialComplex, device):
    b0, b1, b2 = compute_tri_mesh_betti_numbers(hollow_tet_mesh.to(device))

    assert b0 == 1
    assert b1 == 0
    assert b2 == 1


def test_two_tris_betti(two_tris_mesh: SimplicialComplex, device):
    b0, b1, b2 = compute_tri_mesh_betti_numbers(two_tris_mesh.to(device))

    assert b0 == 1
    assert b1 == 0
    assert b2 == 0


def test_two_disjoint_tris_betti(two_disjoint_tris_mesh: SimplicialComplex, device):
    b0, b1, b2 = compute_tri_mesh_betti_numbers(two_disjoint_tris_mesh.to(device))

    assert b0 == 2
    assert b1 == 0
    assert b2 == 0


def test_annulus_betti(flat_annulus_mesh: SimplicialComplex, device):
    b0, b1, b2 = compute_tri_mesh_betti_numbers(flat_annulus_mesh.to(device))

    assert b0 == 1
    assert b1 == 1
    assert b2 == 0

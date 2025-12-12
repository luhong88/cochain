import torch as t

from cochain.complex import SimplicialComplex


def test_exactness_on_tet(hollow_tet_mesh: SimplicialComplex):
    d1_d0 = (hollow_tet_mesh.coboundary_1 @ hollow_tet_mesh.coboundary_0).to_dense()
    t.testing.assert_close(d1_d0, t.zeros_like(d1_d0))


def test_exactness_on_tent(tent_mesh: SimplicialComplex):
    d1_d0 = (tent_mesh.coboundary_1 @ tent_mesh.coboundary_0).to_dense()
    t.testing.assert_close(d1_d0, t.zeros_like(d1_d0))


def test_exactness_on_two_tets(two_tris_mesh: SimplicialComplex):
    d1_d0 = (two_tris_mesh.coboundary_1 @ two_tris_mesh.coboundary_0).to_dense()
    t.testing.assert_close(d1_d0, t.zeros_like(d1_d0))

    d2_d1 = (two_tris_mesh.coboundary_2 @ two_tris_mesh.coboundary_1).to_dense()
    t.testing.assert_close(d2_d1, t.zeros_like(d2_d1))

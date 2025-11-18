import torch as t

from cochain.complex import Simplicial2Complex


def test_exactness_on_tet(tet_mesh: Simplicial2Complex):
    d1_d0 = (tet_mesh.coboundary_1 @ tet_mesh.coboundary_0).to_dense()
    t.testing.assert_close(d1_d0, t.zeros_like(d1_d0))


def test_exactness_on_tent(tent_mesh: Simplicial2Complex):
    d1_d0 = (tent_mesh.coboundary_1 @ tent_mesh.coboundary_0).to_dense()
    t.testing.assert_close(d1_d0, t.zeros_like(d1_d0))

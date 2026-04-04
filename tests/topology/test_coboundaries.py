import torch

from cochain.complex import SimplicialMesh


def test_exactness_on_tet(hollow_tet_mesh: SimplicialMesh):
    d1_d0 = (hollow_tet_mesh.cbd[1] @ hollow_tet_mesh.cbd[0]).to_dense()
    torch.testing.assert_close(d1_d0, torch.zeros_like(d1_d0))


def test_exactness_on_tent(tent_mesh: SimplicialMesh):
    d1_d0 = (tent_mesh.cbd[1] @ tent_mesh.cbd[0]).to_dense()
    torch.testing.assert_close(d1_d0, torch.zeros_like(d1_d0))


def test_exactness_on_two_tets(two_tris_mesh: SimplicialMesh):
    d1_d0 = (two_tris_mesh.cbd[1] @ two_tris_mesh.cbd[0]).to_dense()
    torch.testing.assert_close(d1_d0, torch.zeros_like(d1_d0))

    d2_d1 = (two_tris_mesh.cbd[2] @ two_tris_mesh.cbd[1]).to_dense()
    torch.testing.assert_close(d2_d1, torch.zeros_like(d2_d1))

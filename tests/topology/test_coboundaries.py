import torch

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

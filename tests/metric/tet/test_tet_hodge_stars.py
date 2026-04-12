import igl
import numpy as np
import pytest
import torch

from cochain.complex import SimplicialMesh
from cochain.metric.tet import tet_hodge_stars


def test_star_3_on_two_tets(two_tets_mesh: SimplicialMesh, device):
    mesh = two_tets_mesh.to(device)

    s3 = tet_hodge_stars.star_3(mesh).val.cpu().detach().numpy()

    true_s3 = 1 / igl.volume(
        mesh.vert_coords.cpu().detach().numpy(),
        mesh.tets.cpu().detach().numpy(),
    )

    np.testing.assert_allclose(s3, true_s3)


def test_star_2_on_reg_tet(reg_tet_mesh: SimplicialMesh, device):
    mesh = reg_tet_mesh.to(device)

    s2 = tet_hodge_stars.star_2(mesh).val

    true_s2 = torch.ones_like(s2) * (1.0 / 6.0)

    torch.testing.assert_close(s2, true_s2)


def test_star_1_on_reg_tet(reg_tet_mesh: SimplicialMesh, device):
    mesh = reg_tet_mesh.to(device)

    s1 = tet_hodge_stars.star_1(mesh).val

    true_s1 = torch.ones_like(s1) * (1.0 / 6.0)

    torch.testing.assert_close(s1, true_s1)


def test_star_0_on_two_tets(two_tets_mesh: SimplicialMesh, device):
    mesh = two_tets_mesh.to(device)

    s0 = tet_hodge_stars.star_0(mesh).val.cpu().detach().numpy()

    true_s0 = igl.massmatrix(
        mesh.vert_coords.cpu().detach().numpy(),
        mesh.tets.cpu().detach().numpy(),
        igl.MASSMATRIX_TYPE_BARYCENTRIC,
    ).diagonal()

    np.testing.assert_allclose(s0, true_s0)


@pytest.mark.parametrize(
    "star_op",
    [
        tet_hodge_stars.star_0,
        tet_hodge_stars.star_1,
        tet_hodge_stars.star_2,
        tet_hodge_stars.star_3,
    ],
)
def test_star_backward(star_op, two_tets_mesh: SimplicialMesh, device):
    mesh = two_tets_mesh.to(device)
    mesh.requires_grad_()

    star = star_op(mesh)
    output = star.val.sum()
    output.backward()

    assert mesh.grad is not None
    assert torch.isfinite(mesh.grad).all()


@pytest.mark.parametrize(
    "star_op",
    [
        tet_hodge_stars.star_0,
        tet_hodge_stars.star_1,
        tet_hodge_stars.star_2,
        tet_hodge_stars.star_3,
    ],
)
def test_star_gradcheck(star_op, two_tets_mesh: SimplicialMesh, device):
    vert_coords = two_tets_mesh.vert_coords.clone().to(
        dtype=torch.float64, device=device
    )
    vert_coords.requires_grad_()

    def star_fxn(test_vert_coords):
        mesh = two_tets_mesh.to(device=device, dtype=torch.float64)
        mesh.vert_coords = test_vert_coords
        s = star_op(mesh)
        return s.val.sum()

    assert torch.autograd.gradcheck(star_fxn, (vert_coords,), fast_mode=True)

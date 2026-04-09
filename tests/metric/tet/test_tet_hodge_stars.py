import pytest
import torch

from cochain.complex import SimplicialMesh
from cochain.metric.tet import tet_hodge_stars


def test_star_2_on_reg_tet(reg_tet_mesh: SimplicialMesh):
    s2 = tet_hodge_stars.star_2(reg_tet_mesh).val

    true_s2 = torch.ones_like(s2) * (1.0 / 6.0)

    torch.testing.assert_close(s2, true_s2)


def test_star_1_on_reg_tet(reg_tet_mesh: SimplicialMesh):
    s1 = tet_hodge_stars.star_1(reg_tet_mesh).val

    true_s1 = torch.ones_like(s1) * (1.0 / 6.0)

    torch.testing.assert_close(s1, true_s1)


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

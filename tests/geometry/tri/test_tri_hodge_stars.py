import math

import igl
import numpy as np
import pytest
import skfem as skfem
import torch

from cochain.complex import SimplicialMesh
from cochain.geometry.tri import tri_hodge_stars

# Test 0-, 1-, and 2-star operators on a watertight mesh and a mesh with boundaries.


def test_star_0_on_tent(tent_mesh: SimplicialMesh, device):
    mesh = tent_mesh.to(device)

    s0 = tri_hodge_stars.star_0(mesh).val

    # All triangles in this mesh have the same area
    tri_area = math.sqrt(1.25) / 2.0
    true_s0 = (
        tri_area
        * torch.tensor([4.0, 2.0, 2.0, 2.0, 2.0], dtype=mesh.dtype, device=device)
        / 3.0
    )

    torch.testing.assert_close(s0, true_s0)


def test_star_0_on_tet(hollow_tet_mesh: SimplicialMesh, device):
    mesh = hollow_tet_mesh.to(device)

    s0 = tri_hodge_stars.star_0(mesh).val.cpu().detach().numpy()

    true_s0 = igl.massmatrix(
        mesh.vert_coords.cpu().detach().numpy(),
        mesh.tris.cpu().detach().numpy(),
        igl.MASSMATRIX_TYPE_BARYCENTRIC,
    ).diagonal()

    np.testing.assert_allclose(s0, true_s0)


def test_star_1_circumcentric_on_tent(tent_mesh: SimplicialMesh, device):
    mesh = tent_mesh.to(device)

    s1 = tri_hodge_stars.star_1(mesh, dual_complex="circumcentric").val

    # Find the tangent of the angle between a base edge and side edge
    tan_ang = 2 * math.sqrt(1.25)

    # Find the dual/primal edge ratio for the side and base edges
    dual_side_edge_ratio = 1.0 / tan_ang
    dual_base_edge_ratio = (tan_ang**2 - 1) / (4 * tan_ang)

    true_s1 = torch.tensor(
        [dual_side_edge_ratio] * 4 + [dual_base_edge_ratio] * 4,
        dtype=mesh.dtype,
        device=device,
    )

    torch.testing.assert_close(s1, true_s1)


def test_star_1_barycentric_on_tent(tent_mesh: SimplicialMesh, device):
    mesh = tent_mesh.to(device)

    s1 = tri_hodge_stars.star_1(mesh, dual_complex="barycentric").val

    face_bary = torch.tensor([1.5, 0.5, 1.0], dtype=mesh.dtype, device=device) / 3.0
    side_edge_bary = (
        torch.tensor([0.5, 0.5, 1.0], dtype=mesh.dtype, device=device) / 2.0
    )
    dual_side_edge_len = 2.0 * torch.linalg.norm(face_bary - side_edge_bary)
    side_edge_len = torch.linalg.norm(2.0 * side_edge_bary)
    dual_side_edge_ratio = dual_side_edge_len / side_edge_len

    base_edge_barycenter = (
        torch.tensor([1.0, 0.0, 0.0], dtype=mesh.dtype, device=device) / 2.0
    )
    dual_base_edge_ratio = torch.linalg.norm(face_bary - base_edge_barycenter)

    true_s1 = torch.tensor(
        [dual_side_edge_ratio] * 4 + [dual_base_edge_ratio] * 4,
        dtype=mesh.dtype,
        device=device,
    )

    torch.testing.assert_close(s1, true_s1)


def test_star_1_circumcentric_on_tet(hollow_tet_mesh: SimplicialMesh, device):
    mesh = hollow_tet_mesh.to(device)

    s1 = tri_hodge_stars.star_1(mesh, dual_complex="circumcentric").val

    # extract the Hodge 1-star from `igl.cotmatrix()`.
    igl_cotan_laplacian = torch.from_numpy(
        igl.cotmatrix(
            mesh.vert_coords.cpu().detach().numpy(),
            mesh.tris.cpu().detach().numpy(),
        ).todense()
    ).to(dtype=mesh.dtype, device=device)

    true_s1 = igl_cotan_laplacian[
        hollow_tet_mesh.edges[:, 0], hollow_tet_mesh.edges[:, 1]
    ]

    torch.testing.assert_close(s1, true_s1)


def test_star_2_on_tent(tent_mesh: SimplicialMesh, device):
    mesh = tent_mesh.to(device)

    s2 = tri_hodge_stars.star_2(mesh).val
    # All triangles in this mesh have the same area
    tri_area = math.sqrt(1.25) / 2.0

    true_s2 = torch.tensor([1.0 / tri_area] * 4, dtype=mesh.dtype, device=device)

    torch.testing.assert_close(s2, true_s2)


def test_star_2_on_tet(hollow_tet_mesh: SimplicialMesh, device):
    mesh = hollow_tet_mesh.to(device)

    s2 = tri_hodge_stars.star_2(mesh).val.cpu().detach().numpy()

    true_s2 = 2.0 / igl.doublearea(
        mesh.vert_coords.cpu().detach().numpy(),
        mesh.tris.cpu().detach().numpy(),
    )

    np.testing.assert_allclose(s2, true_s2)


def test_star_0_backward(hollow_tet_mesh: SimplicialMesh, device):
    mesh = hollow_tet_mesh.to(device)
    mesh.requires_grad_()

    s0 = tri_hodge_stars.star_0(mesh)
    output = s0.val.sum()
    output.backward()

    assert mesh.grad is not None
    assert torch.isfinite(mesh.grad).all()


@pytest.mark.parametrize("dual_complex", ["circumcentric", "barycentric"])
def test_star_1_backward(hollow_tet_mesh: SimplicialMesh, dual_complex, device):
    mesh = hollow_tet_mesh.to(device)
    mesh.requires_grad_()

    s1 = tri_hodge_stars.star_1(mesh, dual_complex=dual_complex)
    output = s1.val.sum()
    output.backward()

    assert mesh.grad is not None
    assert torch.isfinite(mesh.grad).all()


def test_star_2_backward(hollow_tet_mesh: SimplicialMesh, device):
    mesh = hollow_tet_mesh.to(device)
    mesh.requires_grad_()

    s2 = tri_hodge_stars.star_2(mesh)
    output = s2.val.sum()
    output.backward()

    assert mesh.grad is not None
    assert torch.isfinite(mesh.grad).all()

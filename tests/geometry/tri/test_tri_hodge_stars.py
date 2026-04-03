import math

import igl
import numpy as np
import skfem as skfem
import torch
from torch import Tensor

from cochain.complex import SimplicialMesh
from cochain.geometry.tri import tri_geometry, tri_hodge_stars

# Test 0-, 1-, and 2-star operators on a watertight mesh and a mesh with boundaries.


def test_star_0_on_tent(tent_mesh: SimplicialMesh):
    s0 = tri_hodge_stars.star_0(tent_mesh).val

    # All triangles in this mesh have the same area
    tri_area = math.sqrt(1.25) / 2.0
    true_s0 = tri_area * Tensor([4.0, 2.0, 2.0, 2.0, 2.0]) / 3.0

    torch.testing.assert_close(s0, true_s0)


def test_star_0_on_tet(hollow_tet_mesh: SimplicialMesh):
    s0 = tri_hodge_stars.star_0(hollow_tet_mesh).val.cpu().detach().numpy()

    true_s0 = igl.massmatrix(
        hollow_tet_mesh.vert_coords.cpu().detach().numpy(),
        hollow_tet_mesh.tris.cpu().detach().numpy(),
        igl.MASSMATRIX_TYPE_BARYCENTRIC,
    ).diagonal()

    np.testing.assert_allclose(s0, true_s0)


def test_star_1_circumcentric_on_tent(tent_mesh: SimplicialMesh):
    s1 = tri_hodge_stars.star_1(tent_mesh, dual_complex="circumcentric").val

    # Find the tangent of the angle between a base edge and side edge
    tan_ang = 2 * math.sqrt(1.25)

    # Find the dual/primal edge ratio for the side and base edges
    dual_side_edge_ratio = 1.0 / tan_ang
    dual_base_edge_ratio = (tan_ang**2 - 1) / (4 * tan_ang)

    true_s1 = Tensor([dual_side_edge_ratio] * 4 + [dual_base_edge_ratio] * 4)

    torch.testing.assert_close(s1, true_s1)


def test_star_1_barycentric_on_tent(tent_mesh: SimplicialMesh):
    s1 = tri_hodge_stars.star_1(tent_mesh, dual_complex="barycentric").val

    face_bary = torch.tensor([1.5, 0.5, 1.0]) / 3.0
    side_edge_bary = torch.tensor([0.5, 0.5, 1.0]) / 2.0
    dual_side_edge_len = 2.0 * torch.linalg.norm(face_bary - side_edge_bary)
    side_edge_len = torch.linalg.norm(2.0 * side_edge_bary)
    dual_side_edge_ratio = dual_side_edge_len / side_edge_len

    base_edge_barycenter = torch.tensor([1.0, 0.0, 0.0]) / 2.0
    dual_base_edge_ratio = torch.linalg.norm(face_bary - base_edge_barycenter)

    true_s1 = Tensor([dual_side_edge_ratio] * 4 + [dual_base_edge_ratio] * 4)

    torch.testing.assert_close(s1, true_s1)


def test_star_1_circumcentric_on_tet(hollow_tet_mesh: SimplicialMesh):
    s1 = tri_hodge_stars.star_1(hollow_tet_mesh, dual_complex="circumcentric").val

    # extract the Hodge 1-star from `igl.cotmatrix()`.
    igl_cotan_laplacian = torch.from_numpy(
        igl.cotmatrix(
            hollow_tet_mesh.vert_coords.cpu().detach().numpy(),
            hollow_tet_mesh.tris.cpu().detach().numpy(),
        ).todense()
    ).to(dtype=torch.float)
    true_s1 = igl_cotan_laplacian[
        hollow_tet_mesh.edges[:, 0], hollow_tet_mesh.edges[:, 1]
    ]

    torch.testing.assert_close(s1, true_s1)


def test_star_2_on_tent(tent_mesh: SimplicialMesh):
    s2 = tri_hodge_stars.star_2(tent_mesh).val
    # All triangles in this mesh have the same area
    tri_area = math.sqrt(1.25) / 2.0

    true_s2 = Tensor([1.0 / tri_area] * 4)
    torch.testing.assert_close(s2, true_s2)


def test_star_2_on_tet(hollow_tet_mesh: SimplicialMesh):
    s2 = tri_hodge_stars.star_2(hollow_tet_mesh).val.cpu().detach().numpy()

    true_s2 = 2.0 / igl.doublearea(
        hollow_tet_mesh.vert_coords.cpu().detach().numpy(),
        hollow_tet_mesh.tris.cpu().detach().numpy(),
    )

    np.testing.assert_allclose(s2, true_s2)


def test_tri_areas_with_igl(flat_annulus_mesh: SimplicialMesh):
    tri_areas = tri_geometry.compute_tri_areas(
        flat_annulus_mesh.vert_coords, flat_annulus_mesh.tris
    )

    true_tri_areas = torch.from_numpy(
        igl.doublearea(
            flat_annulus_mesh.vert_coords.cpu().detach().numpy(),
            flat_annulus_mesh.tris.cpu().detach().numpy(),
        )
        / 2.0
    ).to(dtype=torch.float)

    torch.testing.assert_close(tri_areas, true_tri_areas)


def test_d_tri_areas_d_vert_coords(hollow_tet_mesh: SimplicialMesh):
    # Note that this function does not return the Jacobian; rather, for each
    # triangle, it returns the gradient of its area wrt each of its three verticies.
    dAdV = tri_geometry.compute_d_tri_areas_d_vert_coords(
        hollow_tet_mesh.vert_coords, hollow_tet_mesh.tris
    ).flatten(end_dim=1)

    jacobian = torch.autograd.functional.jacobian(
        lambda vert_coords: tri_geometry.compute_tri_areas(
            vert_coords, hollow_tet_mesh.tris
        ),
        hollow_tet_mesh.vert_coords,
    )
    # Extract the nonzero components of the Jacobian.
    dAdV_true = jacobian[
        torch.repeat_interleave(torch.arange(hollow_tet_mesh.n_tris), 3),
        hollow_tet_mesh.tris.flatten(),
    ]

    torch.testing.assert_close(dAdV, dAdV_true)

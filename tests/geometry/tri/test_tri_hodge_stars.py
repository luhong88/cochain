import math

import igl
import numpy as np
import skfem as skfem
import torch as t

from cochain.complex import SimplicialComplex
from cochain.geometry.tri import tri_geometry, tri_hodge_stars

# Test 0-, 1-, and 2-star operators on a watertight mesh and a mesh with boundaries.


def test_star_0_on_tent(tent_mesh: SimplicialComplex):
    s0 = tri_hodge_stars.star_0(tent_mesh).val

    # All triangles in this mesh have the same area
    tri_area = math.sqrt(1.25) / 2.0
    true_s0 = tri_area * t.Tensor([4.0, 2.0, 2.0, 2.0, 2.0]) / 3.0

    t.testing.assert_close(s0, true_s0)


def test_star_0_on_tet(hollow_tet_mesh: SimplicialComplex):
    s0 = tri_hodge_stars.star_0(hollow_tet_mesh).val.cpu().detach().numpy()

    true_s0 = igl.massmatrix(
        hollow_tet_mesh.vert_coords.cpu().detach().numpy(),
        hollow_tet_mesh.tris.cpu().detach().numpy(),
        igl.MASSMATRIX_TYPE_BARYCENTRIC,
    ).diagonal()

    np.testing.assert_allclose(s0, true_s0)


def test_star_1_circumcentric_on_tent(tent_mesh: SimplicialComplex):
    s1 = tri_hodge_stars.star_1(tent_mesh, dual_complex="circumcentric").val

    # Find the tangent of the angle between a base edge and side edge
    tan_ang = 2 * math.sqrt(1.25)

    # Find the dual/primal edge ratio for the side and base edges
    dual_side_edge_ratio = 1.0 / tan_ang
    dual_base_edge_ratio = (tan_ang**2 - 1) / (4 * tan_ang)

    true_s1 = t.Tensor([dual_side_edge_ratio] * 4 + [dual_base_edge_ratio] * 4)

    t.testing.assert_close(s1, true_s1)


def test_star_1_barycentric_on_tent(tent_mesh: SimplicialComplex):
    s1 = tri_hodge_stars.star_1(tent_mesh, dual_complex="barycentric").val

    face_bary = t.tensor([1.5, 0.5, 1.0]) / 3.0
    side_edge_bary = t.tensor([0.5, 0.5, 1.0]) / 2.0
    dual_side_edge_len = 2.0 * t.linalg.norm(face_bary - side_edge_bary)
    side_edge_len = t.linalg.norm(2.0 * side_edge_bary)
    dual_side_edge_ratio = dual_side_edge_len / side_edge_len

    base_edge_barycenter = t.tensor([1.0, 0.0, 0.0]) / 2.0
    dual_base_edge_ratio = t.linalg.norm(face_bary - base_edge_barycenter)

    true_s1 = t.Tensor([dual_side_edge_ratio] * 4 + [dual_base_edge_ratio] * 4)

    t.testing.assert_close(s1, true_s1)


def test_star_1_circumcentric_on_tet(hollow_tet_mesh: SimplicialComplex):
    s1 = tri_hodge_stars.star_1(hollow_tet_mesh, dual_complex="circumcentric").val

    # extract the Hodge 1-star from `igl.cotmatrix()`.
    igl_cotan_laplacian = t.from_numpy(
        igl.cotmatrix(
            hollow_tet_mesh.vert_coords.cpu().detach().numpy(),
            hollow_tet_mesh.tris.cpu().detach().numpy(),
        ).todense()
    ).to(dtype=t.float)
    true_s1 = igl_cotan_laplacian[
        hollow_tet_mesh.edges[:, 0], hollow_tet_mesh.edges[:, 1]
    ]

    t.testing.assert_close(s1, true_s1)


def test_star_2_on_tent(tent_mesh: SimplicialComplex):
    s2 = tri_hodge_stars.star_2(tent_mesh).val
    # All triangles in this mesh have the same area
    tri_area = math.sqrt(1.25) / 2.0

    true_s2 = t.Tensor([1.0 / tri_area] * 4)
    t.testing.assert_close(s2, true_s2)


def test_star_2_on_tet(hollow_tet_mesh: SimplicialComplex):
    s2 = tri_hodge_stars.star_2(hollow_tet_mesh).val.cpu().detach().numpy()

    true_s2 = 2.0 / igl.doublearea(
        hollow_tet_mesh.vert_coords.cpu().detach().numpy(),
        hollow_tet_mesh.tris.cpu().detach().numpy(),
    )

    np.testing.assert_allclose(s2, true_s2)


def test_tri_areas_with_igl(flat_annulus_mesh: SimplicialComplex):
    tri_areas = tri_geometry._tri_areas(
        flat_annulus_mesh.vert_coords, flat_annulus_mesh.tris
    )

    true_tri_areas = t.from_numpy(
        igl.doublearea(
            flat_annulus_mesh.vert_coords.cpu().detach().numpy(),
            flat_annulus_mesh.tris.cpu().detach().numpy(),
        )
        / 2.0
    ).to(dtype=t.float)

    t.testing.assert_close(tri_areas, true_tri_areas)


def test_d_tri_areas_d_vert_coords(hollow_tet_mesh: SimplicialComplex):
    # Note that this function does not return the Jacobian; rather, for each
    # triangle, it returns the gradient of its area wrt each of its three verticies.
    dAdV = tri_geometry._d_tri_areas_d_vert_coords(
        hollow_tet_mesh.vert_coords, hollow_tet_mesh.tris
    ).flatten(end_dim=1)

    jacobian = t.autograd.functional.jacobian(
        lambda vert_coords: tri_geometry._tri_areas(vert_coords, hollow_tet_mesh.tris),
        hollow_tet_mesh.vert_coords,
    )
    # Extract the nonzero components of the Jacobian.
    dAdV_true = jacobian[
        t.repeat_interleave(t.arange(hollow_tet_mesh.n_tris), 3),
        hollow_tet_mesh.tris.flatten(),
    ]

    t.testing.assert_close(dAdV, dAdV_true)

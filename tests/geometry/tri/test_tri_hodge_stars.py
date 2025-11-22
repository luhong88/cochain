import math

import igl
import numpy as np
import pytest
import torch as t

from cochain.complex import SimplicialComplex
from cochain.geometry.tri import tri_hodge_stars

# Test 0-, 1-, and 2-star operators on a watertight mesh and a mesh with boundaries.


def test_star_0_on_tent(tent_mesh: SimplicialComplex):
    s0 = tri_hodge_stars.star_0(tent_mesh)

    # All triangles in this mesh have the same area
    tri_area = math.sqrt(1.25) / 2.0
    true_s0 = tri_area * t.Tensor([4.0, 2.0, 2.0, 2.0, 2.0]) / 3.0

    t.testing.assert_close(s0, true_s0)


def test_star_0_on_tet(tet_mesh: SimplicialComplex):
    s0 = tri_hodge_stars.star_0(tet_mesh).cpu().detach().numpy()

    true_s0 = igl.massmatrix(
        tet_mesh.vert_coords.cpu().detach().numpy(),
        tet_mesh.tris.cpu().detach().numpy(),
        igl.MASSMATRIX_TYPE_BARYCENTRIC,
    ).diagonal()

    np.testing.assert_allclose(s0, true_s0)


def test_star_1_on_tent(tent_mesh: SimplicialComplex):
    s1 = tri_hodge_stars.star_1(tent_mesh)

    # Find the tangent of the angle between a base edge and side edge
    tan_ang = 2 * math.sqrt(1.25)

    # Find the dual/primal edge ratio for the side and base edges
    dual_side_edge_ratio = 1.0 / tan_ang
    dual_base_edge_ratio = (tan_ang**2 - 1) / (4 * tan_ang)

    true_s1 = t.Tensor([dual_side_edge_ratio] * 4 + [dual_base_edge_ratio] * 4)

    t.testing.assert_close(s1, true_s1)


def test_star_1_on_tet(tet_mesh: SimplicialComplex):
    s1 = tri_hodge_stars.star_1(tet_mesh)

    # extract the Hodge 1-star from `igl.cotmatrix()`.
    igl_cotan_laplacian = t.from_numpy(
        igl.cotmatrix(
            tet_mesh.vert_coords.cpu().detach().numpy(),
            tet_mesh.tris.cpu().detach().numpy(),
        ).todense()
    ).to(dtype=t.float)
    true_s1 = igl_cotan_laplacian[tet_mesh.edges[:, 0], tet_mesh.edges[:, 1]]

    t.testing.assert_close(s1, true_s1)


def test_star_2_on_tent(tent_mesh: SimplicialComplex):
    s2 = tri_hodge_stars.star_2(tent_mesh)
    # All triangles in this mesh have the same area
    tri_area = math.sqrt(1.25) / 2.0

    true_s2 = t.Tensor([1.0 / tri_area] * 4)
    t.testing.assert_close(s2, true_s2)


def test_star_2_on_tet(tet_mesh: SimplicialComplex):
    s2 = tri_hodge_stars.star_2(tet_mesh).cpu().detach().numpy()

    true_s2 = 2.0 / igl.doublearea(
        tet_mesh.vert_coords.cpu().detach().numpy(),
        tet_mesh.tris.cpu().detach().numpy(),
    )

    np.testing.assert_allclose(s2, true_s2)


# Test the analytical Jacobian of the Hodge stars and their inverses against
# autograd Jacobians.


@pytest.mark.parametrize(
    "star, d_star_d_vert_coords",
    [
        (tri_hodge_stars.star_0, tri_hodge_stars.d_star_0_d_vert_coords),
        (tri_hodge_stars.star_1, tri_hodge_stars.d_star_1_d_vert_coords),
        (tri_hodge_stars.star_2, tri_hodge_stars.d_star_2_d_vert_coords),
    ],
)
def test_star_jacobian(star, d_star_d_vert_coords, tet_mesh: SimplicialComplex):
    vert_coords = tet_mesh.vert_coords.clone()
    tris = tet_mesh.tris.clone()

    autograd_jacobian = t.autograd.functional.jacobian(
        lambda vert_coords: star(SimplicialComplex.from_tri_mesh(vert_coords, tris)),
        vert_coords,
    )

    analytical_jacobian = d_star_d_vert_coords(tet_mesh).to_dense()

    t.testing.assert_close(autograd_jacobian, analytical_jacobian)


@pytest.mark.parametrize(
    "star, d_inv_star_d_vert_coords",
    [
        (tri_hodge_stars.star_0, tri_hodge_stars.d_inv_star_0_d_vert_coords),
        (tri_hodge_stars.star_1, tri_hodge_stars.d_inv_star_1_d_vert_coords),
        (tri_hodge_stars.star_2, tri_hodge_stars.d_inv_star_2_d_vert_coords),
    ],
)
def test_inv_star_jacobian(star, d_inv_star_d_vert_coords, tet_mesh: SimplicialComplex):
    vert_coords = tet_mesh.vert_coords.clone()
    tris = tet_mesh.tris.clone()

    autograd_jacobian = t.autograd.functional.jacobian(
        lambda vert_coords: 1.0
        / star(SimplicialComplex.from_tri_mesh(vert_coords, tris)),
        vert_coords,
    )

    analytical_jacobian = d_inv_star_d_vert_coords(tet_mesh).to_dense()

    t.testing.assert_close(autograd_jacobian, analytical_jacobian)


def test_tri_areas_with_igl(flat_annulus_mesh: SimplicialComplex):
    tri_areas = tri_hodge_stars._tri_areas(
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


def test_d_tri_areas_d_vert_coords(tet_mesh: SimplicialComplex):
    # Note that this function does not return the Jacobian; rather, for each
    # triangle, it returns the gradient of its area wrt each of its three verticies.
    dAdV = tri_hodge_stars._d_tri_areas_d_vert_coords(
        tet_mesh.vert_coords, tet_mesh.tris
    ).flatten(end_dim=1)

    jacobian = t.autograd.functional.jacobian(
        lambda vert_coords: tri_hodge_stars._tri_areas(vert_coords, tet_mesh.tris),
        tet_mesh.vert_coords,
    )
    # Extract the nonzero components of the Jacobian.
    dAdV_true = jacobian[
        t.repeat_interleave(t.arange(tet_mesh.n_tris), 3), tet_mesh.tris.flatten()
    ]

    t.testing.assert_close(dAdV, dAdV_true)

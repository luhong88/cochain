import math

import igl
import numpy as np
import pytest
import skfem as skfem
import torch as t
from jaxtyping import Float
from skfem.helpers import dot

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


# Test the mass-1 matrix, mirroring the testing strategy for tet meshes


def test_mass_1_with_skfem(flat_annulus_mesh: SimplicialComplex):
    skfem_mesh = skfem.MeshTri(
        flat_annulus_mesh.vert_coords[:, [0, 1]].T.cpu().detach().numpy(),
        flat_annulus_mesh.tris.T.cpu().detach().numpy(),
    )

    elem = skfem.ElementTriN1()
    basis = skfem.InteriorBasis(skfem_mesh, elem)

    @skfem.BilinearForm
    def mass_form(u, v, w):
        return dot(u, v)

    skfem_mass_1 = mass_form.assemble(basis).todense()
    # Here, we compare the eigenvalues of the mass matrices rather than the mass
    # matrices themselves to avoid differences in index/orientation conventions.
    skfem_mass_1_eigs = np.linalg.eigvalsh(skfem_mass_1)
    skfem_mass_1_eigs.sort()
    skfem_mass_1_eigs_torch = t.from_numpy(skfem_mass_1_eigs).to(
        dtype=flat_annulus_mesh.vert_coords.dtype
    )

    cochain_mass_1 = tri_hodge_stars.mass_1(flat_annulus_mesh).to_dense()
    cochain_mass_1_eigs = t.linalg.eigvalsh(cochain_mass_1).sort().values

    t.testing.assert_close(cochain_mass_1_eigs, skfem_mass_1_eigs_torch)


def test_mass_1_symmetry(two_tris_mesh: SimplicialComplex):
    mass = tri_hodge_stars.mass_1(two_tris_mesh).to_dense()
    t.testing.assert_close(mass, mass.T)


def test_mass_matrix_positive_definite(two_tris_mesh: SimplicialComplex):
    mass = tri_hodge_stars.mass_1(two_tris_mesh).to_dense()
    eigs = t.linalg.eigvalsh(mass)
    assert eigs.min() >= 1e-6


def test_mass_1_matrix_connectivity(two_tris_mesh: SimplicialComplex):
    mass_1 = tri_hodge_stars.mass_1(two_tris_mesh)
    mass_1_mask = t.zeros_like(mass_1.to_dense(), dtype=t.long)
    mass_1_mask[*mass_1.indices()] = 1

    true_mass_1_mask = t.tensor(
        [
            [1, 1, 1, 0, 0],
            [1, 1, 1, 0, 0],
            [1, 1, 1, 1, 1],
            [0, 0, 1, 1, 1],
            [0, 0, 1, 1, 1],
        ],
        dtype=t.long,
    )

    t.testing.assert_close(mass_1_mask, true_mass_1_mask)


def test_mass_1_linear_potential(two_tris_mesh: SimplicialComplex):
    mass_1 = tri_hodge_stars.mass_1(two_tris_mesh)

    const_vec = t.tensor([[1.0, 3.0, 2.0]], dtype=two_tris_mesh.vert_coords.dtype)
    phi_verts: Float[t.Tensor, "tri"] = t.sum(
        two_tris_mesh.vert_coords * const_vec, dim=-1
    )

    cochain = (
        phi_verts[two_tris_mesh.edges[:, 1]] - phi_verts[two_tris_mesh.edges[:, 0]]
    )

    energy = cochain @ mass_1 @ cochain

    vert_s_coord: Float[t.Tensor, "tri 3 3"] = two_tris_mesh.vert_coords[
        two_tris_mesh.tris
    ]

    edge_ij = vert_s_coord[:, 1] - vert_s_coord[:, 0]
    edge_ik = vert_s_coord[:, 2] - vert_s_coord[:, 0]

    area_vec = t.cross(edge_ij, edge_ik, dim=-1)
    two_area = t.linalg.norm(area_vec, dim=-1)

    area_vec_norm = area_vec / two_area.view(-1, 1)
    area = two_area / 2.0

    phi_tangent = (
        const_vec
        - t.sum(area_vec_norm * const_vec, dim=-1, keepdim=True) * area_vec_norm
    )

    true_energy = t.sum(area * t.sum(phi_tangent**2, dim=-1))

    t.testing.assert_close(energy, true_energy)

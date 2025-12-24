import numpy as np
import pytest
import skfem as skfem
import torch as t
from skfem.helpers import dot

from cochain.complex import SimplicialComplex
from cochain.geometry.tet import tet_masses
from cochain.geometry.tet.tet_geometry import _tet_signed_vols


def test_mass_1_with_skfem(two_tets_mesh: SimplicialComplex):
    skfem_mesh = skfem.MeshTet(
        two_tets_mesh.vert_coords.T.cpu().detach().numpy(),
        two_tets_mesh.tets.T.cpu().detach().numpy(),
    )

    elem = skfem.ElementTetN0()
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
        dtype=two_tets_mesh.vert_coords.dtype
    )

    cochain_mass_1 = tet_masses.mass_1(two_tets_mesh).to_dense()
    cochain_mass_1_eigs = t.linalg.eigvalsh(cochain_mass_1).sort().values

    t.testing.assert_close(cochain_mass_1_eigs, skfem_mass_1_eigs_torch)


def test_mass_2_with_skfem(two_tets_mesh: SimplicialComplex):
    skfem_mesh = skfem.MeshTet(
        two_tets_mesh.vert_coords.T.cpu().detach().numpy(),
        two_tets_mesh.tets.T.cpu().detach().numpy(),
    )

    elem = skfem.ElementTetRT0()
    basis = skfem.InteriorBasis(skfem_mesh, elem)

    @skfem.BilinearForm
    def mass_form(u, v, w):
        return dot(u, v)

    skfem_mass_2 = mass_form.assemble(basis).todense()
    # Here, we compare the eigenvalues of the mass matrices rather than the mass
    # matrices themselves to avoid differences in index/orientation conventions.
    # In addition, the scikit-fem results need to be multiplied with 4 due to
    # difference in 2-form basis function definitions.
    skfem_mass_2_eigs = np.linalg.eigvalsh(skfem_mass_2) * 4
    skfem_mass_2_eigs.sort()
    skfem_mass_2_eigs_torch = t.from_numpy(skfem_mass_2_eigs).to(
        dtype=two_tets_mesh.vert_coords.dtype
    )

    cochain_mass_2 = tet_masses.mass_2(two_tets_mesh).to_dense()
    cochain_mass_2_eigs = t.linalg.eigvalsh(cochain_mass_2).sort().values

    t.testing.assert_close(cochain_mass_2_eigs, skfem_mass_2_eigs_torch)


@pytest.mark.parametrize(
    "mass_matrix",
    [
        tet_masses.mass_1,
        tet_masses.mass_2,
    ],
)
def test_mass_matrix_symmetry(mass_matrix, two_tets_mesh: SimplicialComplex):
    mass = mass_matrix(two_tets_mesh).to_dense()
    t.testing.assert_close(mass, mass.T)


@pytest.mark.parametrize(
    "mass_matrix",
    [
        tet_masses.mass_1,
        tet_masses.mass_2,
    ],
)
def test_mass_matrix_positive_definite(mass_matrix, two_tets_mesh: SimplicialComplex):
    mass = mass_matrix(two_tets_mesh).to_dense()
    eigs = t.linalg.eigvalsh(mass)
    assert eigs.min() >= 1e-6


@pytest.mark.parametrize(
    "mass_matrix",
    [
        tet_masses.mass_0,
        tet_masses.mass_3,
    ],
)
def test_mass_matrix_total_vol_partition(mass_matrix, two_tets_mesh: SimplicialComplex):
    """
    The sum of the diagonal 0- and 3-form mass matrices should be equal to the
    total volume of the tet.
    """
    total_mass = mass_matrix(two_tets_mesh).tr
    total_vol = t.sum(
        t.abs(_tet_signed_vols(two_tets_mesh.vert_coords, two_tets_mesh.tets))
    )
    t.testing.assert_close(total_mass, total_vol)


def test_mass_1_matrix_connectivity(two_tets_mesh: SimplicialComplex):
    mass_1 = tet_masses.mass_1(two_tets_mesh)
    mass_1_mask = t.zeros_like(mass_1.to_dense(), dtype=t.long)
    mass_1_mask[mass_1.sp_topo.idx_coo.unbind(0)] = 1

    true_mass_1_mask = t.tensor(
        [
            [1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 0, 1, 1, 0, 1, 0],
            [1, 1, 0, 1, 1, 0, 1, 0, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 0, 1, 1, 0, 1, 0],
            [1, 1, 0, 1, 1, 0, 1, 0, 1],
            [1, 1, 1, 0, 1, 1, 0, 1, 0],
            [1, 1, 0, 1, 1, 0, 1, 0, 1],
        ],
        dtype=t.long,
    )

    t.testing.assert_close(mass_1_mask, true_mass_1_mask)


def test_mass_2_matrix_connectivity(two_tets_mesh: SimplicialComplex):
    mass_2 = tet_masses.mass_2(two_tets_mesh)
    mass_2_mask = t.zeros_like(mass_2.to_dense(), dtype=t.long)
    mass_2_mask[mass_2.sp_topo.idx_coo.unbind(0)] = 1

    true_mass_2_mask = t.tensor(
        [
            [1, 1, 1, 1, 1, 1, 1],
            [1, 1, 0, 1, 0, 1, 0],
            [1, 0, 1, 0, 1, 0, 1],
            [1, 1, 0, 1, 0, 1, 0],
            [1, 0, 1, 0, 1, 0, 1],
            [1, 1, 0, 1, 0, 1, 0],
            [1, 0, 1, 0, 1, 0, 1],
        ],
        dtype=t.long,
    )

    t.testing.assert_close(mass_2_mask, true_mass_2_mask)


def test_mass_1_patch(two_tets_mesh: SimplicialComplex):
    mass_1 = tet_masses.mass_1(two_tets_mesh)

    const_field = t.tensor([[1.0, 3.0, 2.0]], dtype=two_tets_mesh.vert_coords.dtype)

    edges = (
        two_tets_mesh.vert_coords[two_tets_mesh.edges[:, 1]]
        - two_tets_mesh.vert_coords[two_tets_mesh.edges[:, 0]]
    )

    field_proj = t.sum(edges * const_field, dim=-1)
    energy = field_proj @ mass_1 @ field_proj

    true_energy = (
        t.sum(t.abs(_tet_signed_vols(two_tets_mesh.vert_coords, two_tets_mesh.tets)))
        * t.sum(const_field * const_field, dim=-1)
    ).squeeze()

    t.testing.assert_close(energy, true_energy)


def test_mass_2_patch(two_tets_mesh: SimplicialComplex):
    mass_2 = tet_masses.mass_2(two_tets_mesh)

    const_field = t.tensor([[1.0, 3.0, 2.0]], dtype=two_tets_mesh.vert_coords.dtype)

    # Compute the triangle vector areas following the right-hand rule using the
    # canonical triangle vertex orderings.
    vec_areas = (
        t.cross(
            two_tets_mesh.vert_coords[two_tets_mesh.tris[:, 1]]
            - two_tets_mesh.vert_coords[two_tets_mesh.tris[:, 0]],
            two_tets_mesh.vert_coords[two_tets_mesh.tris[:, 2]]
            - two_tets_mesh.vert_coords[two_tets_mesh.tris[:, 0]],
            dim=-1,
        )
        / 2.0
    )

    field_proj = t.sum(vec_areas * const_field, dim=-1)
    energy = field_proj @ mass_2 @ field_proj

    true_energy = (
        t.sum(t.abs(_tet_signed_vols(two_tets_mesh.vert_coords, two_tets_mesh.tets)))
        * t.sum(const_field * const_field, dim=-1)
    ).squeeze()

    t.testing.assert_close(energy, true_energy)

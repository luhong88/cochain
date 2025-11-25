import math

import numpy as np
import pytest
import torch as t

from cochain.complex import SimplicialComplex
from cochain.geometry.tet import tet_masses
from cochain.geometry.tet.tet_geometry import _tet_signed_vols


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
    total_mass = t.sum(mass_matrix(two_tets_mesh))
    total_vol = t.sum(
        t.abs(_tet_signed_vols(two_tets_mesh.vert_coords, two_tets_mesh.tets))
    )
    t.testing.assert_close(total_mass, total_vol)


def test_mass_1_matrix_connectivity(two_tets_mesh: SimplicialComplex):
    mass_1 = tet_masses.mass_1(two_tets_mesh)
    mass_1_mask = t.zeros_like(mass_1.to_dense(), dtype=t.long)
    mass_1_mask[*mass_1.indices()] = 1

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
    mass_2_mask[*mass_2.indices()] = 1

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

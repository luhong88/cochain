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

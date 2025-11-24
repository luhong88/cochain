import math

import numpy as np
import pytest
import torch as t

from cochain.complex import SimplicialComplex
from cochain.geometry.tet import tet_masses


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

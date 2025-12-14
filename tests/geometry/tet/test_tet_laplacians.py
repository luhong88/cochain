from functools import partial

import pytest
import torch as t
from jaxtyping import Float

from cochain.complex import SimplicialComplex
from cochain.geometry.tet import tet_laplacians


@pytest.mark.parametrize(
    "weak_laplacian",
    [
        partial(tet_laplacians.weak_laplacian_0, method="cotan"),
        partial(tet_laplacians.weak_laplacian_0, method="consistent"),
        tet_laplacians.weak_laplacian_1,
        partial(tet_laplacians.weak_laplacian_2, method="dense"),
        partial(tet_laplacians.weak_laplacian_2, method="inv_star"),
        partial(tet_laplacians.weak_laplacian_2, method="row_sum"),
        partial(tet_laplacians.weak_laplacian_3, method="dense"),
        partial(tet_laplacians.weak_laplacian_3, method="inv_star"),
        partial(tet_laplacians.weak_laplacian_3, method="row_sum"),
    ],
)
def test_laplacian_symmetry(weak_laplacian, two_tets_mesh: SimplicialComplex):
    """
    Test that the weak Laplacians are symmetric.
    """
    weak_laplacian_i = weak_laplacian(two_tets_mesh).to_dense()
    t.testing.assert_close(weak_laplacian_i, weak_laplacian_i.T)


@pytest.mark.parametrize(
    "weak_laplacian",
    [
        partial(tet_laplacians.weak_laplacian_0, method="cotan"),
        partial(tet_laplacians.weak_laplacian_0, method="consistent"),
        tet_laplacians.weak_laplacian_1,
        partial(tet_laplacians.weak_laplacian_2, method="dense"),
        partial(tet_laplacians.weak_laplacian_2, method="inv_star"),
        partial(tet_laplacians.weak_laplacian_2, method="row_sum"),
        partial(tet_laplacians.weak_laplacian_3, method="dense"),
        partial(tet_laplacians.weak_laplacian_3, method="inv_star"),
        partial(tet_laplacians.weak_laplacian_3, method="row_sum"),
    ],
)
def test_laplacian_PSD(weak_laplacian, two_tets_mesh: SimplicialComplex):
    """
    Test that the stiffness matrices are positive semi-definite.
    """
    laplacian_i = weak_laplacian(two_tets_mesh).to_dense()

    eigs = t.linalg.eigvalsh(laplacian_i)
    assert eigs.min() >= -1e-6

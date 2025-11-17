import pytest
import torch as t

from cochain.complex import Simplicial2Complex
from cochain.geometry import laplacians


@pytest.mark.parametrize(
    "laplacian, betti",
    [
        (laplacians.laplacian_0, 1),
        (laplacians.laplacian_1, 0),
        (laplacians.laplacian_2, 0),
    ],
)
def test_disk_homology_group_dims(laplacian, betti, tent_mesh: Simplicial2Complex):
    operator = laplacian(tent_mesh).to_dense()
    dim_ker = operator.shape[0] - t.linalg.matrix_rank(operator)
    t.testing.assert_close(dim_ker, t.tensor(betti))


@pytest.mark.parametrize(
    "laplacian, betti",
    [
        (laplacians.laplacian_0, 1),
        (laplacians.laplacian_1, 1),
        (laplacians.laplacian_2, 0),
    ],
)
def test_annulus_homology_group_dims(
    laplacian, betti, flat_annulus_mesh: Simplicial2Complex
):
    operator = laplacian(flat_annulus_mesh).to_dense()
    dim_ker = operator.shape[0] - t.linalg.matrix_rank(operator)
    t.testing.assert_close(dim_ker, t.tensor(betti))


@pytest.mark.parametrize(
    "laplacian, betti",
    [
        (laplacians.laplacian_0, 1),
        (laplacians.laplacian_1, 0),
        (laplacians.laplacian_2, 1),
    ],
)
def test_sphere_homology_group_dims(
    laplacian, betti, icosphere_mesh: Simplicial2Complex
):
    operator = laplacian(icosphere_mesh).to_dense()
    dim_ker = operator.shape[0] - t.linalg.matrix_rank(operator)
    t.testing.assert_close(dim_ker, t.tensor(betti))

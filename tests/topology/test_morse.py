import pytest
import torch

from cochain.complex import SimplicialMesh
from cochain.topology.morse import compute_morse_complex


@pytest.mark.parametrize(
    "mesh",
    [
        "two_tris_mesh",
        "hollow_tet_mesh",
        "finer_flat_annulus_mesh",
    ],
)
def test_morse_cbd_exactness_on_tri_mesh(mesh, request, device):
    mesh = request.getfixturevalue(mesh).to(device)

    cbd, _ = compute_morse_complex(mesh)

    d1_d0 = (cbd[1] @ cbd[0]).to_dense()
    torch.testing.assert_close(d1_d0, torch.zeros_like(d1_d0))

    assert cbd[2]._nnz() == 0


@pytest.mark.parametrize(
    "mesh",
    [
        "two_tets_mesh",
        "simple_bcc_mesh",
        "solid_torus_mesh",
    ],
)
def test_morse_cbd_exactness_on_tet_mesh(mesh, request, device):
    mesh = request.getfixturevalue(mesh).to(device)

    cbd, _ = compute_morse_complex(mesh)

    d1_d0 = (cbd[1] @ cbd[0]).to_dense()
    torch.testing.assert_close(d1_d0, torch.zeros_like(d1_d0))

    d2_d1 = (cbd[2] @ cbd[1]).to_dense()
    torch.testing.assert_close(d2_d1, torch.zeros_like(d2_d1))

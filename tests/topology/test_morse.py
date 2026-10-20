import pytest
import torch

from cochain.boundary.selection import BoundarySelection
from cochain.topology.betti import _betti_via_morse
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

    abs_bc = BoundarySelection.from_absolute_bc(mesh)
    rel_bc = BoundarySelection.from_relative_bc(mesh)

    for bc in [abs_bc, rel_bc]:
        cbd, _ = compute_morse_complex(mesh, bc_sel=bc)

        d1_d0 = (cbd[1] @ cbd[0]).to_dense()
        torch.testing.assert_close(d1_d0, torch.zeros_like(d1_d0))

        assert cbd[2]._nnz() == 0


@pytest.mark.parametrize(
    "mesh",
    [
        "two_tets_mesh",
        "simple_sc_mesh",
        "solid_torus_mesh",
    ],
)
def test_morse_cbd_exactness_on_tet_mesh(mesh, request, device):
    mesh = request.getfixturevalue(mesh).to(device)

    abs_bc = BoundarySelection.from_absolute_bc(mesh)
    rel_bc = BoundarySelection.from_relative_bc(mesh)

    for bc in [abs_bc, rel_bc]:
        cbd, _ = compute_morse_complex(mesh, bc_sel=bc)

        d1_d0 = (cbd[1] @ cbd[0]).to_dense()
        torch.testing.assert_close(d1_d0, torch.zeros_like(d1_d0))

        d2_d1 = (cbd[2] @ cbd[1]).to_dense()
        torch.testing.assert_close(d2_d1, torch.zeros_like(d2_d1))


@pytest.mark.parametrize(
    "mesh",
    [
        "two_tris_mesh",
        "hollow_tet_mesh",
        "finer_flat_annulus_mesh",
        "two_tets_mesh",
        "solid_torus_mesh",
        "solid_spherical_shell_mesh",
    ],
)
def test_euler_characteristics(mesh, request, device):
    """
    Check the Euler characteristic of the Morse CW complex.

    Euler characteristic computed on the original mesh should match that computed
    using the critical cells.
    """
    mesh = request.getfixturevalue(mesh).to(device)

    # First test with absolute boundary condition.
    abs_bc = BoundarySelection.from_absolute_bc(mesh)

    _, crit_splx = compute_morse_complex(mesh, bc_sel=abs_bc)
    n_crit_splx = [splx.size(0) for splx in crit_splx]

    chi = mesh.n_verts - mesh.n_edges + mesh.n_tris - mesh.n_tets
    chi_morse = n_crit_splx[0] - n_crit_splx[1] + n_crit_splx[2] - n_crit_splx[3]

    assert chi == chi_morse

    # Then test with relative boundary condition.
    rel_bc = BoundarySelection.from_relative_bc(mesh)

    _, crit_splx = compute_morse_complex(mesh, bc_sel=rel_bc)
    n_crit_splx = [splx.size(0) for splx in crit_splx]

    chi = (
        rel_bc.n_retained_verts
        - rel_bc.n_retained_edges
        + rel_bc.n_retained_tris
        - rel_bc.n_retained_tets
    )
    chi_morse = n_crit_splx[0] - n_crit_splx[1] + n_crit_splx[2] - n_crit_splx[3]

    assert chi == chi_morse


@pytest.mark.parametrize(
    "mesh,betti_true",
    [
        ("icosphere_mesh", [1, 0, 1, 0]),
        ("two_tris_mesh", [1, 0, 0, 0]),
        ("two_disjoint_tris_mesh", [2, 0, 0, 0]),
        ("finer_flat_annulus_mesh", [1, 1, 0, 0]),
        ("two_tets_mesh", [1, 0, 0, 0]),
        ("solid_torus_mesh", [1, 1, 0, 0]),
        ("solid_spherical_shell_mesh", [1, 0, 1, 0]),
    ],
)
def test_morse_betti_invariance_to_scalar_field(mesh, betti_true, request, device):
    mesh = request.getfixturevalue(mesh).to(device)
    field = mesh.vert_coords[:, -1]

    morse_cbd, crit_splx = compute_morse_complex(mesh, scalar_field=field)

    n_crit_splx = [splx.size(0) for splx in crit_splx]

    cbd_dense = [cbd.to_dense() for cbd in morse_cbd]
    cbd_rank = [torch.linalg.matrix_rank(cbd).item() for cbd in cbd_dense]

    betti = [
        n_crit_splx[0] - cbd_rank[0],
        n_crit_splx[1] - cbd_rank[0] - cbd_rank[1],
        n_crit_splx[2] - cbd_rank[1] - cbd_rank[2],
        n_crit_splx[3] - cbd_rank[2],
    ]

    for b, b_true in zip(betti, betti_true):
        assert b == b_true


@pytest.mark.parametrize(
    "mesh,betti_true",
    [
        ("icosphere_mesh", [1, 0, 1, 0]),
        ("two_tris_mesh", [0, 0, 1, 0]),
        ("two_disjoint_tris_mesh", [0, 0, 2, 0]),
        ("finer_flat_annulus_mesh", [0, 1, 1, 0]),
        ("two_tets_mesh", [0, 0, 0, 1]),
        ("solid_torus_mesh", [0, 0, 1, 1]),
        ("solid_spherical_shell_mesh", [0, 1, 0, 1]),
    ],
)
def test_morse_relative_betti_invariance_to_scalar_field(
    mesh, betti_true, request, device
):
    mesh = request.getfixturevalue(mesh).to(device)
    bc_sel = BoundarySelection.from_relative_bc(mesh)
    field = mesh.vert_coords[:, -1]

    morse_cbd, crit_splx = compute_morse_complex(
        mesh, scalar_field=field, bc_sel=bc_sel
    )

    n_crit_splx = [splx.size(0) for splx in crit_splx]

    cbd_dense = [cbd.to_dense() for cbd in morse_cbd]
    cbd_rank = [torch.linalg.matrix_rank(cbd).item() for cbd in cbd_dense]

    betti = [
        n_crit_splx[0] - cbd_rank[0],
        n_crit_splx[1] - cbd_rank[0] - cbd_rank[1],
        n_crit_splx[2] - cbd_rank[1] - cbd_rank[2],
        n_crit_splx[3] - cbd_rank[2],
    ]

    for b, b_true in zip(betti, betti_true):
        assert b == b_true

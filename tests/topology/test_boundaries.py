import torch as t

from cochain.topology.boundaries import detect_mesh_boundaries


def test_tent_mesh_boundaries(tent_mesh, device):
    boundary_masks = detect_mesh_boundaries(tent_mesh.to(device))

    assert boundary_masks[0].numel() == 0

    t.testing.assert_close(
        boundary_masks[1], t.tensor([False, False, False, False], device=device)
    )

    t.testing.assert_close(
        boundary_masks[2],
        t.tensor([False, False, False, False, True, True, True, True], device=device),
    )

    t.testing.assert_close(
        boundary_masks[3], t.tensor([False, True, True, True, True], device=device)
    )


def test_hollow_tet_mesh_boundaries(hollow_tet_mesh, device):
    boundary_masks = detect_mesh_boundaries(hollow_tet_mesh.to(device))

    assert boundary_masks[0].numel() == 0

    for idx in [1, 2, 3]:
        t.testing.assert_close(
            boundary_masks[idx],
            t.zeros_like(boundary_masks[idx], dtype=t.bool, device=device),
        )


def test_two_tets_mesh_boundaries(two_tets_mesh, device):
    boundary_masks = detect_mesh_boundaries(two_tets_mesh.to(device))

    t.testing.assert_close(boundary_masks[0], t.tensor([False, False], device=device))

    t.testing.assert_close(
        boundary_masks[1],
        t.tensor(
            [
                False,
                True,
                True,
                True,
                True,
                True,
                True,
            ],
            device=device,
        ),
    )

    t.testing.assert_close(
        boundary_masks[2],
        t.tensor([True] * 9, device=device),
    )

    t.testing.assert_close(
        boundary_masks[3],
        t.tensor([True] * 5, device=device),
    )

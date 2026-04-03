import torch

from cochain.topology.boundaries import detect_mesh_boundaries


def test_tent_mesh_boundaries(tent_mesh, device):
    boundary_masks = detect_mesh_boundaries(tent_mesh.to(device).cbd)

    assert boundary_masks[3].numel() == 0

    torch.testing.assert_close(
        boundary_masks[2], torch.tensor([False, False, False, False], device=device)
    )

    torch.testing.assert_close(
        boundary_masks[1],
        torch.tensor(
            [False, False, False, False, True, True, True, True], device=device
        ),
    )

    torch.testing.assert_close(
        boundary_masks[0], torch.tensor([False, True, True, True, True], device=device)
    )


def test_hollow_tet_mesh_boundaries(hollow_tet_mesh, device):
    boundary_masks = detect_mesh_boundaries(hollow_tet_mesh.to(device).cbd)

    assert boundary_masks[3].numel() == 0

    for idx in [2, 1, 0]:
        torch.testing.assert_close(
            boundary_masks[idx],
            torch.zeros_like(boundary_masks[idx], dtype=torch.bool, device=device),
        )


def test_two_tets_mesh_boundaries(two_tets_mesh, device):
    boundary_masks = detect_mesh_boundaries(two_tets_mesh.to(device).cbd)

    torch.testing.assert_close(
        boundary_masks[3], torch.tensor([False, False], device=device)
    )

    torch.testing.assert_close(
        boundary_masks[2],
        torch.tensor(
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

    torch.testing.assert_close(
        boundary_masks[1],
        torch.tensor([True] * 9, device=device),
    )

    torch.testing.assert_close(
        boundary_masks[0],
        torch.tensor([True] * 5, device=device),
    )

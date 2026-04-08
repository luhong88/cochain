import igl
import torch

from cochain.complex import SimplicialMesh
from cochain.metric.tri import _tri_geometry


def test_compute_tri_areas(flat_annulus_mesh: SimplicialMesh, device):
    mesh = flat_annulus_mesh.to(device)

    tri_areas = _tri_geometry.compute_tri_areas(mesh.vert_coords, mesh.tris)

    true_tri_areas = torch.from_numpy(
        igl.doublearea(
            flat_annulus_mesh.vert_coords.cpu().detach().numpy(),
            flat_annulus_mesh.tris.cpu().detach().numpy(),
        )
        / 2.0
    ).to(dtype=mesh.dtype, device=device)

    torch.testing.assert_close(tri_areas, true_tri_areas)


def test_compute_d_tri_areas_d_vert_coords(hollow_tet_mesh: SimplicialMesh, device):
    # Note that this function does not return the Jacobian; rather, for each
    # triangle, it returns the gradient of its area wrt each of its three verticies.
    mesh = hollow_tet_mesh.to(device)

    dAdV = _tri_geometry.compute_d_tri_areas_d_vert_coords(
        mesh.vert_coords, mesh.tris
    ).flatten(end_dim=1)

    jacobian = torch.func.jacrev(
        lambda vert_coords: _tri_geometry.compute_tri_areas(vert_coords, mesh.tris)
    )(mesh.vert_coords)

    # Extract the nonzero components of the Jacobian.
    dAdV_true = jacobian[
        torch.repeat_interleave(torch.arange(mesh.n_tris), 3),
        mesh.tris.flatten(),
    ]

    torch.testing.assert_close(dAdV, dAdV_true)

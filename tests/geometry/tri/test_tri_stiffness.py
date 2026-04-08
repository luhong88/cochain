import igl
import torch

from cochain.complex import SimplicialMesh
from cochain.geometry.tri.tri_stiffness import stiffness_matrix


def test_stiffness_with_igl(two_tris_mesh: SimplicialMesh, device):
    """
    Check stiffness matrix correctness with igl.

    Validate the stiffness matrix calculation using the external library `libigl`,
    which performs the same calculation with the `cotmatrix()` function.
    """
    mesh = two_tris_mesh.to(device)

    igl_cotan_laplacian = -torch.from_numpy(
        igl.cotmatrix(
            mesh.vert_coords.cpu().detach().numpy(),
            mesh.tris.cpu().detach().numpy(),
        ).todense(),
    ).to(dtype=mesh.dtype, device=device)

    cotan_laplacian = stiffness_matrix(mesh).to_dense()

    torch.testing.assert_close(igl_cotan_laplacian, cotan_laplacian)


def test_stiffness_kernel(icosphere_mesh: SimplicialMesh, device):
    """
    Check that constant vectors are in the kernel of the stiffness matrix.

    The stiffness matrix acting on a constant function over the vertices should
    return the zero vector. This can be checked by comparing the row sum of the
    matrix with the zero vector.
    """
    mesh = icosphere_mesh.to(device)

    sphere_S = stiffness_matrix(mesh)
    row_sum = sphere_S.to_dense().sum(dim=-1)

    torch.testing.assert_close(row_sum, torch.zeros_like(row_sum))


def test_stiffness_symmetry(icosphere_mesh: SimplicialMesh, device):
    """Check that the stifness matrix is symmetric."""
    mesh = icosphere_mesh.to(device)

    sphere_S = stiffness_matrix(mesh)
    sphere_S_dense = sphere_S.to_dense()

    torch.testing.assert_close(sphere_S_dense, sphere_S_dense.T)


def test_stiffness_PSD(icosphere_mesh: SimplicialMesh, device):
    """Check that the stiffness matrix is positive semi-definite."""
    mesh = icosphere_mesh.to(device)

    sphere_S = stiffness_matrix(mesh)
    sphere_S_dense = sphere_S.to_dense()
    eigs = torch.linalg.eigvalsh(sphere_S_dense)

    assert eigs.min() >= -1e-6


def test_stiffness_planar(flat_annulus_mesh: SimplicialMesh, device):
    """
    Check the kernel of the stiffness matrix of a planar mesh.

    The stiffness matrix acting on a planar mesh coordinates should result in
    zero (for interior vertices). This is because the stiffness matrix acting on
    the vertex coordinate vector gives the mean curvature normal vector.
    """
    mesh = flat_annulus_mesh.to(device)

    plane_S = stiffness_matrix(mesh)
    zero_tensor = plane_S @ mesh.vert_coords

    bd_mask = mesh.bd_vert_mask

    torch.testing.assert_close(
        zero_tensor[~bd_mask], torch.zeros_like(zero_tensor[~bd_mask])
    )


def test_stiffness_matrix_backward(two_tets_mesh: SimplicialMesh, device):
    mesh = two_tets_mesh.to(device)
    mesh.requires_grad_()

    stiff = stiffness_matrix(mesh)
    output = stiff.val.sum()
    output.backward()

    assert mesh.grad is not None
    assert torch.isfinite(mesh.grad).all()

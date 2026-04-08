import igl
import torch

from cochain.complex import SimplicialMesh
from cochain.geometry.tri.tri_stiffness import stiffness_matrix


def test_stiffness_with_igl(two_tris_mesh: SimplicialMesh):
    """
    Validate the stiffness matrix calculation using the external library `libigl`,
    which performs the same calculation with the `cotmatrix()` function.
    """
    igl_cotan_laplacian = -torch.from_numpy(
        igl.cotmatrix(
            two_tris_mesh.vert_coords.cpu().detach().numpy(),
            two_tris_mesh.tris.cpu().detach().numpy(),
        ).todense(),
    ).to(dtype=torch.float32)

    cochain_cotan_laplacian = stiffness_matrix(two_tris_mesh).to_dense()

    torch.testing.assert_close(igl_cotan_laplacian, cochain_cotan_laplacian)


def test_stiffness_kernel(icosphere_mesh: SimplicialMesh):
    """
    The stiffness matrix acting on a constant function over the vertices should
    return the zero vector. This can be checked by comparing the row sum of the
    matrix with the zero vector.
    """
    sphere_S = stiffness_matrix(icosphere_mesh)
    row_sum = sphere_S.to_dense().sum(dim=-1)
    torch.testing.assert_close(row_sum, torch.zeros_like(row_sum))


def test_stiffness_symmetry(icosphere_mesh: SimplicialMesh):
    """
    The stifness matrix should be a symmetric matrix.
    """
    sphere_S = stiffness_matrix(icosphere_mesh)
    sphere_S_dense = sphere_S.to_dense()
    torch.testing.assert_close(sphere_S_dense, sphere_S_dense.T)


def test_stiffness_PSD(icosphere_mesh: SimplicialMesh):
    """
    The stiffness matrix should be a positive semi-definite matrix.
    """
    sphere_S = stiffness_matrix(icosphere_mesh)
    sphere_S_dense = sphere_S.to_dense()
    eigs = torch.linalg.eigvalsh(sphere_S_dense)
    assert eigs.min() >= -1e-6


def test_stiffness_planar(square_mesh: SimplicialMesh):
    """
    The stiffness matrix acting on a planar mesh coordinates should result in
    zero (for interior vertices).
    """
    plane_S = stiffness_matrix(square_mesh)
    zero_tensor = plane_S @ square_mesh.vert_coords

    torch.testing.assert_close(zero_tensor[-1], torch.zeros_like(zero_tensor[-1]))

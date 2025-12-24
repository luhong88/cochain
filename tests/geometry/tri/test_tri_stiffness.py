import igl
import torch as t

from cochain.complex import SimplicialComplex
from cochain.geometry.tri.tri_stiffness import stiffness_matrix


def test_stiffness_with_igl(two_tris_mesh: SimplicialComplex):
    """
    Validate the stiffness matrix calculation using the external library `libigl`,
    which performs the same calculation with the `cotmatrix()` function.
    """
    igl_cotan_laplacian = -t.from_numpy(
        igl.cotmatrix(
            two_tris_mesh.vert_coords.cpu().detach().numpy(),
            two_tris_mesh.tris.cpu().detach().numpy(),
        ).todense(),
    ).to(dtype=t.float)

    cochain_cotan_laplacian = stiffness_matrix(two_tris_mesh).to_dense()

    t.testing.assert_close(igl_cotan_laplacian, cochain_cotan_laplacian)


def test_stiffness_kernel(icosphere_mesh: SimplicialComplex):
    """
    The stiffness matrix acting on a constant function over the vertices should
    return the zero vector. This can be checked by comparing the row sum of the
    matrix with the zero vector.
    """
    sphere_S = stiffness_matrix(icosphere_mesh)
    row_sum = sphere_S.to_dense().sum(dim=-1)
    t.testing.assert_close(row_sum, t.zeros_like(row_sum))


def test_stiffness_symmetry(icosphere_mesh: SimplicialComplex):
    """
    The stifness matrix should be a symmetric matrix.
    """
    sphere_S = stiffness_matrix(icosphere_mesh)
    sphere_S_dense = sphere_S.to_dense()
    t.testing.assert_close(sphere_S_dense, sphere_S_dense.T)


def test_stiffness_PSD(icosphere_mesh: SimplicialComplex):
    """
    The stiffness matrix should be a positive semi-definite matrix.
    """
    sphere_S = stiffness_matrix(icosphere_mesh)
    sphere_S_dense = sphere_S.to_dense()
    eigs = t.linalg.eigvalsh(sphere_S_dense)
    assert eigs.min() >= -1e-6


def test_stiffness_planar(square_mesh: SimplicialComplex):
    """
    The stiffness matrix acting on a planar mesh coordinates should result in
    zero (for interior vertices).
    """
    plane_S = stiffness_matrix(square_mesh)
    zero_tensor = plane_S @ square_mesh.vert_coords

    t.testing.assert_close(zero_tensor[-1], t.zeros_like(zero_tensor[-1]))

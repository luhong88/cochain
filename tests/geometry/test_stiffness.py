import potpourri3d as pp3d
import torch as t

from cochain.complex import SimplicialComplex
from cochain.geometry.stiffness import (
    d_stiffness_d_vert_coords,
    stiffness_matrix,
)


def test_stiffness_with_pp3d(tet_mesh):
    """
    Validate the stiffness matrix calculation using the external library
    `potpourri3d`, which performs the same calculation with the `cotan_laplacian()`
    function.
    """
    pp3d_cotan_laplacian = t.from_numpy(
        pp3d.cotan_laplacian(
            tet_mesh.vert_coords.cpu().detach().numpy(),
            tet_mesh.tris.cpu().detach().numpy(),
        ).todense()
    )

    cochain_cotan_laplacian = stiffness_matrix(tet_mesh).to_dense()

    t.testing.assert_close(pp3d_cotan_laplacian, cochain_cotan_laplacian)


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
    plane_S = stiffness_matrix(square_mesh).to_dense()
    zero_tensor = plane_S @ square_mesh.vert_coords

    t.testing.assert_close(zero_tensor[-1], t.zeros_like(zero_tensor[-1]))


def test_stiffness_autograd(two_tris_mesh: SimplicialComplex):
    """
    Check that the custom gradient matches the automatic gradient for the stiffness
    matrix.
    """
    two_tris_mesh.vert_coords.requires_grad = True
    sphere_S = stiffness_matrix(two_tris_mesh).to_dense()
    y = (sphere_S**2).sum()

    dLdV = d_stiffness_d_vert_coords(two_tris_mesh).to_dense()
    dydL = t.autograd.grad(y, sphere_S, retain_graph=True)[0]
    custom_grad = t.einsum("ij,ijkl->kl", dydL, dLdV)

    auto_grad = t.autograd.grad(y, two_tris_mesh.vert_coords)[0]

    assert t.allclose(custom_grad, auto_grad, atol=1e-4)

import torch as t

from cochain.geometry.cotan_laplacian import (
    cotan_laplacian,
    d_cotan_laplacian_d_vert_coords,
)


def test_cotan_laplacian_kernel(icosphere_mesh):
    """
    The Laplacian acting on a constant function over the vertices should return
    the zero vector. This can be checked by comparing the row sum of the matrix
    with the zero vector.
    """
    sphere_L0 = cotan_laplacian(icosphere_mesh)
    row_sum = sphere_L0.to_dense().sum(dim=-1)
    assert t.allclose(row_sum, t.tensor(0.0), atol=1e-6)


def test_cotan_laplacian_symmetry(icosphere_mesh):
    """
    The Laplacian should be a symmetric matrix.
    """
    sphere_L0 = cotan_laplacian(icosphere_mesh)
    sphere_L0_dense = sphere_L0.to_dense()
    assert t.allclose(sphere_L0_dense, sphere_L0_dense.T, atol=1e-6)


def test_cotan_laplacian_PSD(icosphere_mesh):
    """
    The Laplacian should be a positive semi-definite matrix.
    """
    sphere_L0 = cotan_laplacian(icosphere_mesh)
    sphere_L0_dense = sphere_L0.to_dense()
    eigs = t.linalg.eigvalsh(sphere_L0_dense)
    assert eigs.min() >= -1e-6


def test_cotan_laplacian_planar(square_mesh):
    """
    The Laplacian acting on a planar mesh coordinates should result in zero (
    for interior vertices).
    """
    L0 = cotan_laplacian(square_mesh).to_dense()
    zero_tensor = L0 @ square_mesh.vert_coords

    assert t.allclose(zero_tensor[-1], t.tensor(0.0), atol=1e-6)


def test_cotan_laplacian_autograd(two_tris_mesh):
    """
    Check that the custom gradient matches the automatic gradient.
    """
    two_tris_mesh.vert_coords.requires_grad = True
    sphere_L0 = cotan_laplacian(two_tris_mesh).to_dense()
    y = (sphere_L0**2).sum()

    dLdV = d_cotan_laplacian_d_vert_coords(two_tris_mesh).to_dense()
    dydL = t.autograd.grad(y, sphere_L0, retain_graph=True)[0]
    custom_grad = t.einsum("ij,ijkl->kl", dydL, dLdV)

    auto_grad = t.autograd.grad(y, two_tris_mesh.vert_coords)[0]

    assert t.allclose(custom_grad, auto_grad, atol=1e-4)

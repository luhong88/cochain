import torch as t

from cochain.geometry import (
    _cotan_laplacian,
    _DifferentiableCotanLaplacian,
    cotan_laplacian,
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


def test_cotan_laplacian_autograd(icosphere_mesh):
    """
    Check that the custom gradient matches the automatic gradient.
    """
    icosphere_mesh.vert_coords.requires_grad = True
    sphere_L0_custom = cotan_laplacian(icosphere_mesh).to_dense()
    y_custom = (sphere_L0_custom**2).sum()
    custom_grad = t.autograd.grad(y_custom, icosphere_mesh.vert_coords)

    sphere_L0_auto = _cotan_laplacian(
        icosphere_mesh.vert_coords, icosphere_mesh.tris
    ).to_dense()
    y_auto = (sphere_L0_auto**2).sum()
    auto_grad = t.autograd.grad(y_auto, icosphere_mesh.vert_coords)

    assert t.allclose(custom_grad[0], auto_grad[0], atol=1e-4)


def test_cotan_laplacian_gradcheck(two_tris_mesh):
    """
    Check that the custom gradient for the Laplacian agrees with numerical gradient.
    """

    def gradcheck_func(vert_coords, tris):
        L_sparse = _DifferentiableCotanLaplacian.apply(vert_coords, tris)
        L_dense = L_sparse.to_dense()

        return (L_dense**2).sum()

    vert_coords_double = two_tris_mesh.vert_coords.to(t.double)
    vert_coords_double.requires_grad = True

    assert t.autograd.gradcheck(
        gradcheck_func, (vert_coords_double, two_tris_mesh.tris)
    )

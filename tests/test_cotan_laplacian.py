import torch as t

from cochain.geometry import _DifferentiableCotanLaplacian, cotan_laplacian


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


def test_cotan_laplacian_gradcheck(two_tris_mesh):
    def gradcheck_func(vert_coords, tris):
        L_sparse = _DifferentiableCotanLaplacian.apply(vert_coords, tris)
        L_dense = L_sparse.to_dense()

        return (L_dense**2).sum()

    two_tris_mesh.vert_coords.requires_grad = True

    assert t.autograd.gradcheck(
        gradcheck_func, (two_tris_mesh.vert_coords, two_tris_mesh.tris)
    )

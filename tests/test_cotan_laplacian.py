import torch as t

from cochain.geometry import _DifferentiableCotanLaplacian


def test_cotan_laplacian_gradcheck(two_tris_mesh):
    def gradcheck_func(vert_coords, tris):
        L_sparse = _DifferentiableCotanLaplacian.apply(vert_coords, tris)
        L_dense = L_sparse.to_dense()

        return (L_dense**2).sum()

    two_tris_mesh.vert_coords.requires_grad = True

    assert t.autograd.gradcheck(
        gradcheck_func, (two_tris_mesh.vert_coords, two_tris_mesh.tris)
    )

import igl
import torch as t

from cochain.complex import SimplicialComplex
from cochain.geometry.tet.tet_stiffness import (
    d_stiffness_d_vert_coords,
    stiffness_matrix,
)


def test_stiffness_with_igl(two_tets_mesh: SimplicialComplex):
    """
    Validate the stiffness matrix calculation using the external library
    `libigl`, which performs the same calculation with the `cotmatrix()`
    function.
    """
    igl_cotan_laplacian = -t.from_numpy(
        igl.cotmatrix(
            two_tets_mesh.vert_coords.cpu().detach().numpy(),
            two_tets_mesh.tets.cpu().detach().numpy(),
        ).todense()
    ).to(dtype=t.float)

    cochain_cotan_laplacian = stiffness_matrix(two_tets_mesh).to_dense()

    t.testing.assert_close(igl_cotan_laplacian, cochain_cotan_laplacian)


def test_stiffness_kernel(small_bcc_mesh: SimplicialComplex):
    """
    The stiffness matrix acting on a constant function over the vertices should
    return the zero vector. This can be checked by comparing the row sum of the
    matrix with the zero vector.
    """
    bcc_S = stiffness_matrix(small_bcc_mesh)
    row_sum = bcc_S.to_dense().sum(dim=-1)
    t.testing.assert_close(row_sum, t.zeros_like(row_sum))


def test_stiffness_symmetry(small_bcc_mesh: SimplicialComplex):
    """
    The stifness matrix should be a symmetric matrix.
    """
    bcc_S = stiffness_matrix(small_bcc_mesh)
    bcc_S_dense = bcc_S.to_dense()
    t.testing.assert_close(bcc_S_dense, bcc_S_dense.T)


def test_stiffness_PSD(small_bcc_mesh: SimplicialComplex):
    """
    The stiffness matrix should be a positive semi-definite matrix.
    """
    bcc_S = stiffness_matrix(small_bcc_mesh)
    bcc_S_dense = bcc_S.to_dense()
    eigs = t.linalg.eigvalsh(bcc_S_dense)
    assert eigs.min() >= -1e-6


def test_stiffness_linear_precision(small_bcc_mesh: SimplicialComplex):
    """
    The stiffness matrix acting on the interior of a 3D mesh vertex coordinates
    should result in zero.
    """
    bcc_S = stiffness_matrix(small_bcc_mesh).to_dense()
    zero_tensor = bcc_S @ small_bcc_mesh.vert_coords
    interior_mask = t.abs(small_bcc_mesh.vert_coords).max(dim=-1).values < 1.0

    t.testing.assert_close(
        zero_tensor[interior_mask], t.zeros_like(zero_tensor[interior_mask])
    )


def test_stiffness_autograd(two_tets_mesh: SimplicialComplex):
    """
    Check that the custom gradient matches the automatic gradient for the stiffness
    matrix.
    """
    two_tets_mesh.vert_coords.requires_grad = True
    two_tets_S = stiffness_matrix(two_tets_mesh).to_dense()
    y = (two_tets_S**2).sum()

    dLdV = d_stiffness_d_vert_coords(two_tets_mesh).to_dense()
    dydL = t.autograd.grad(y, two_tets_S, retain_graph=True)[0]
    custom_grad = t.einsum("ij,ijkl->kl", dydL, dLdV)

    auto_grad = t.autograd.grad(y, two_tets_mesh.vert_coords)[0]

    assert t.allclose(custom_grad, auto_grad, atol=1e-4)

import igl
import torch as t

from cochain.complex import SimplicialComplex
from cochain.geometry.tet.tet_stiffness import (
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


def test_stiffness_kernel(simple_bcc_mesh: SimplicialComplex):
    """
    The stiffness matrix acting on a constant function over the vertices should
    return the zero vector. This can be checked by comparing the row sum of the
    matrix with the zero vector.
    """
    bcc_S = stiffness_matrix(simple_bcc_mesh)
    row_sum = bcc_S.to_dense().sum(dim=-1)
    t.testing.assert_close(row_sum, t.zeros_like(row_sum))


def test_stiffness_symmetry(simple_bcc_mesh: SimplicialComplex):
    """
    The stifness matrix should be a symmetric matrix.
    """
    bcc_S = stiffness_matrix(simple_bcc_mesh)
    bcc_S_dense = bcc_S.to_dense()
    t.testing.assert_close(bcc_S_dense, bcc_S_dense.T)


def test_stiffness_PSD(simple_bcc_mesh: SimplicialComplex):
    """
    The stiffness matrix should be a positive semi-definite matrix.
    """
    bcc_S = stiffness_matrix(simple_bcc_mesh)
    bcc_S_dense = bcc_S.to_dense()
    eigs = t.linalg.eigvalsh(bcc_S_dense)
    assert eigs.min() >= -1e-6


def test_stiffness_linear_precision(simple_bcc_mesh: SimplicialComplex):
    """
    The stiffness matrix acting on the interior of a 3D mesh vertex coordinates
    should result in zero.
    """
    bcc_S = stiffness_matrix(simple_bcc_mesh).to_dense()
    zero_tensor = bcc_S @ simple_bcc_mesh.vert_coords

    # Check whether a vertex is in the interior of the mesh, which is true if
    # none of its coordinates have -1 or 1; note that this will break if the size
    # or orientation of the BCC mesh changes.
    interior_mask = t.abs(simple_bcc_mesh.vert_coords).max(dim=-1).values < 1.0

    t.testing.assert_close(
        zero_tensor[interior_mask], t.zeros_like(zero_tensor[interior_mask])
    )

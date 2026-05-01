import igl
import torch

from cochain.complex import SimplicialMesh
from cochain.metric.tet.tet_stiffness import stiffness_matrix


def test_stiffness_with_igl(two_tets_mesh: SimplicialMesh, device):
    """
    Check stiffness matrix correctness with igl.

    Validate the stiffness matrix calculation using the external library `libigl`,
    which performs the same calculation with the `cotmatrix()` function.
    """
    mesh = two_tets_mesh.to(device)

    igl_cotan_laplacian = -torch.from_numpy(
        igl.cotmatrix(
            mesh.vert_coords.cpu().detach().numpy(),
            mesh.tets.cpu().detach().numpy(),
        ).todense()
    ).to(dtype=mesh.dtype, device=device)

    cochain_cotan_laplacian = stiffness_matrix(mesh).to_dense()

    torch.testing.assert_close(igl_cotan_laplacian, cochain_cotan_laplacian)


def test_stiffness_kernel(simple_bcc_mesh: SimplicialMesh, device):
    """
    Check that constant vectors are in the kernel of the stiffness matrix.

    The stiffness matrix acting on a constant function over the vertices should
    return the zero vector. This can be checked by comparing the row sum of the
    matrix with the zero vector.
    """
    mesh = simple_bcc_mesh.to(device)

    bcc_S = stiffness_matrix(mesh)
    row_sum = bcc_S.to_dense().sum(dim=-1)

    torch.testing.assert_close(row_sum, torch.zeros_like(row_sum))


def test_stiffness_symmetry(simple_bcc_mesh: SimplicialMesh, device):
    """Check that the stifness matrix is symmetric."""
    mesh = simple_bcc_mesh.to(device)

    bcc_S = stiffness_matrix(mesh)
    bcc_S_T = bcc_S.T

    torch.testing.assert_close(bcc_S.to_dense(), bcc_S_T.to_dense())


def test_stiffness_PSD(simple_bcc_mesh: SimplicialMesh, device):
    """Check that the stiffness matrix is positive semi-definite."""
    mesh = simple_bcc_mesh.to(device)

    bcc_S = stiffness_matrix(mesh)
    bcc_S_dense = bcc_S.to_dense()
    eigs = torch.linalg.eigvalsh(bcc_S_dense)

    assert eigs.min() >= -1e-6


def test_stiffness_linear_precision(simple_bcc_mesh: SimplicialMesh, device):
    """
    Check stiffness matrix linear precision.

    The stiffness matrix acting on the interior of a 3D mesh vertex coordinates
    should result in zero.
    """
    mesh = simple_bcc_mesh.to(device)

    bcc_S = stiffness_matrix(mesh).to_dense()
    zero_tensor = bcc_S @ simple_bcc_mesh.vert_coords

    bd_mask = mesh.bd_vert_mask

    torch.testing.assert_close(
        zero_tensor[~bd_mask], torch.zeros_like(zero_tensor[~bd_mask])
    )


def test_stiffness_matrix_backward(simple_bcc_mesh: SimplicialMesh, device):
    mesh = simple_bcc_mesh.to(device)
    mesh.requires_grad_()

    stiff = stiffness_matrix(mesh)
    output = stiff.values.sum()
    output.backward()

    assert mesh.grad is not None
    assert torch.isfinite(mesh.grad).all()


def test_stiffness_matrix_gradcheck(simple_bcc_mesh: SimplicialMesh, device):
    vert_coords = simple_bcc_mesh.vert_coords.clone().to(
        dtype=torch.float64, device=device
    )
    vert_coords.requires_grad_()

    def stiffness_fxn(test_vert_coords):
        mesh = simple_bcc_mesh.to(device=device, dtype=torch.float64)
        mesh.vert_coords = test_vert_coords
        s = stiffness_matrix(mesh)
        return s.values.sum()

    assert torch.autograd.gradcheck(stiffness_fxn, (vert_coords,), fast_mode=True)

import numpy as np
import pytest
import skfem as skfem
import torch
from einops import repeat
from jaxtyping import Float
from skfem.helpers import dot
from torch import Tensor

from cochain.cochain import music_ops
from cochain.complex import SimplicialMesh
from cochain.metric.tri import tri_masses


@pytest.mark.parametrize("mass_matrix", [tri_masses.mass_0, tri_masses.mass_1])
def test_mass_matrix_symmetry(mass_matrix, two_tris_mesh: SimplicialMesh, device):
    mesh = two_tris_mesh.to(device)

    mass = mass_matrix(mesh)
    mass_T = mass.T

    torch.testing.assert_close(mass.to_dense(), mass_T.to_dense())


@pytest.mark.parametrize("mass_matrix", [tri_masses.mass_0, tri_masses.mass_1])
def test_mass_matrix_positive_definite(
    mass_matrix, two_tris_mesh: SimplicialMesh, device
):
    mesh = two_tris_mesh.to(device)

    mass = mass_matrix(mesh).to_dense()
    eigs = torch.linalg.eigvalsh(mass)

    assert eigs.min() >= 1e-6


def test_mass_0_with_skfem(flat_annulus_mesh: SimplicialMesh, device):
    mesh = flat_annulus_mesh.to(device)

    skfem_mesh = skfem.MeshTri(
        mesh.vert_coords[:, [0, 1]].T.cpu().detach().numpy(),
        mesh.tris.T.cpu().detach().numpy(),
    )

    elem = skfem.ElementTriP1()
    basis = skfem.InteriorBasis(skfem_mesh, elem)

    @skfem.BilinearForm
    def mass_form(u, v, w):
        return u * v

    skfem_mass_0 = mass_form.assemble(basis).todense()
    # Here, we compare the eigenvalues of the mass matrices rather than the mass
    # matrices themselves to avoid differences in index/orientation conventions.
    skfem_mass_0_eigs = np.linalg.eigvalsh(skfem_mass_0)
    skfem_mass_0_eigs.sort()
    skfem_mass_0_eigs_torch = torch.from_numpy(skfem_mass_0_eigs).to(
        dtype=mesh.dtype, device=device
    )

    cochain_mass_0 = tri_masses.mass_0(mesh).to_dense()
    cochain_mass_0_eigs = torch.linalg.eigvalsh(cochain_mass_0).sort().values

    torch.testing.assert_close(cochain_mass_0_eigs, skfem_mass_0_eigs_torch)


def test_mass_1_with_skfem(flat_annulus_mesh: SimplicialMesh, device):
    mesh = flat_annulus_mesh.to(device)

    skfem_mesh = skfem.MeshTri(
        mesh.vert_coords[:, [0, 1]].T.cpu().detach().numpy(),
        mesh.tris.T.cpu().detach().numpy(),
    )

    elem = skfem.ElementTriN1()
    basis = skfem.InteriorBasis(skfem_mesh, elem)

    @skfem.BilinearForm
    def mass_form(u, v, w):
        return dot(u, v)

    skfem_mass_1 = mass_form.assemble(basis).todense()
    # Here, we compare the eigenvalues of the mass matrices rather than the mass
    # matrices themselves to avoid differences in index/orientation conventions.
    skfem_mass_1_eigs = np.linalg.eigvalsh(skfem_mass_1)
    skfem_mass_1_eigs.sort()
    skfem_mass_1_eigs_torch = torch.from_numpy(skfem_mass_1_eigs).to(
        dtype=mesh.dtype, device=device
    )

    cochain_mass_1 = tri_masses.mass_1(mesh).to_dense()
    cochain_mass_1_eigs = torch.linalg.eigvalsh(cochain_mass_1).sort().values

    torch.testing.assert_close(cochain_mass_1_eigs, skfem_mass_1_eigs_torch)


def test_mass_0_matrix_connectivity(two_tris_mesh: SimplicialMesh, device):
    mesh = two_tris_mesh.to(device)

    mass_0 = tri_masses.mass_0(mesh)
    mass_0_mask = torch.zeros_like(mass_0.to_dense(), dtype=torch.int64)
    mass_0_mask[mass_0.pattern.idx_coo.unbind(0)] = 1

    # All elements in a 3x3 block of the M_0 matrix are nonzero iff the three
    # vertices form a tri.
    true_mass_0_mask = torch.tensor(
        [
            [1, 1, 1, 0],
            [1, 1, 1, 1],
            [1, 1, 1, 1],
            [0, 1, 1, 1],
        ],
        dtype=torch.int64,
        device=device,
    )

    torch.testing.assert_close(mass_0_mask, true_mass_0_mask)


def test_mass_1_matrix_connectivity(two_tris_mesh: SimplicialMesh, device):
    mesh = two_tris_mesh.to(device)

    mass_1 = tri_masses.mass_1(mesh)
    mass_1_mask = torch.zeros_like(mass_1.to_dense(), dtype=torch.int64)
    mass_1_mask[mass_1.pattern.idx_coo.unbind(0)] = 1

    # M_ij is nonzero iff e_i and e_j are in the same triangle.
    true_mass_1_mask = torch.tensor(
        [
            [1, 1, 1, 0, 0],  # 01
            [1, 1, 1, 0, 0],  # 02
            [1, 1, 1, 1, 1],  # 12
            [0, 0, 1, 1, 1],  # 13
            [0, 0, 1, 1, 1],  # 23
        ],
        dtype=torch.int64,
        device=device,
    )

    torch.testing.assert_close(mass_1_mask, true_mass_1_mask)


def test_mass_1_linear_potential_dirichlet_energy(
    two_tris_mesh: SimplicialMesh, device
):
    """Check that M_1 computes the L^2 norm for a constant vec field exactly."""
    mesh = two_tris_mesh.to(device)

    mass_1 = tri_masses.mass_1(mesh)

    # Define the linear potential ϕ(x) via its gradient, which is a constant
    # vector field ∇ϕ, and find its flat dϕ, which is a 1-cochain.
    const_vec = repeat(
        torch.tensor([1.0, 3.0, 2.0], dtype=mesh.dtype, device=device),
        "coord -> tri coord",
        tri=mesh.n_tris,
    )
    cochain = music_ops.local_flat(vec_field=const_vec, mesh=mesh, mode="element")

    # Compute the discrete Dirichlet energy for the potential ϕ(x) as dϕ.T @ M @ dϕ.
    energy = cochain @ mass_1 @ cochain

    # Compute the Dirichlet energy analytically over the mesh.
    # First, find the component of ∇ϕ tangent to the triangles.
    vert_s_coord: Float[Tensor, "tri 3 3"] = mesh.vert_coords[mesh.tris]

    edge_ij = vert_s_coord[:, 1] - vert_s_coord[:, 0]
    edge_ik = vert_s_coord[:, 2] - vert_s_coord[:, 0]

    area_vec = torch.cross(edge_ij, edge_ik, dim=-1)
    two_area = torch.linalg.norm(area_vec, dim=-1)

    area_vec_norm = area_vec / two_area.view(-1, 1)
    area = two_area / 2.0

    phi_tangent = (
        const_vec
        - torch.sum(area_vec_norm * const_vec, dim=-1, keepdim=True) * area_vec_norm
    )

    # The dirichlet energy is defined as the surface integral of ||dϕ||^2, which
    # for a linear potential, can be computed exactly as \sum_i A_i*||∇ϕ_i⋅t||^2
    # where i sums over the triangles and t is the unit tangent vector over the mesh.
    true_energy = torch.sum(area * torch.sum(phi_tangent**2, dim=-1))

    torch.testing.assert_close(energy, true_energy)


@pytest.mark.parametrize("mass_matrix", [tri_masses.mass_0, tri_masses.mass_1])
def test_mass_matrix_backward(mass_matrix, two_tris_mesh: SimplicialMesh, device):
    mesh = two_tris_mesh.to(device)
    mesh.requires_grad_()

    mass = mass_matrix(mesh)
    output = mass.val.sum()
    output.backward()

    assert mesh.grad is not None
    assert torch.isfinite(mesh.grad).all()

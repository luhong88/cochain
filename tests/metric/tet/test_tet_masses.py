import numpy as np
import pytest
import skfem as skfem
import torch
from einops import repeat
from skfem.helpers import dot

from cochain.cochain.discretize import DeRhamMap
from cochain.complex import SimplicialMesh
from cochain.metric.tet import tet_hodge_stars, tet_masses
from cochain.metric.tet._tet_geometry import compute_tet_signed_vols


def test_mass_0_with_skfem(two_tets_mesh: SimplicialMesh, device):
    mesh = two_tets_mesh.to(device)

    skfem_mesh = skfem.MeshTet(
        mesh.vert_coords.T.cpu().detach().numpy(),
        mesh.tets.T.cpu().detach().numpy(),
    )

    elem = skfem.ElementTetP1()
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

    cochain_mass_0 = tet_masses.mass_0(mesh).to_dense()
    cochain_mass_0_eigs = torch.linalg.eigvalsh(cochain_mass_0).sort().values

    torch.testing.assert_close(cochain_mass_0_eigs, skfem_mass_0_eigs_torch)


def test_mass_1_with_skfem(two_tets_mesh: SimplicialMesh, device):
    mesh = two_tets_mesh.to(device)

    skfem_mesh = skfem.MeshTet(
        mesh.vert_coords.T.cpu().detach().numpy(),
        mesh.tets.T.cpu().detach().numpy(),
    )

    elem = skfem.ElementTetN0()
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

    cochain_mass_1 = tet_masses.mass_1(mesh).to_dense()
    cochain_mass_1_eigs = torch.linalg.eigvalsh(cochain_mass_1).sort().values

    torch.testing.assert_close(cochain_mass_1_eigs, skfem_mass_1_eigs_torch)


def test_mass_2_with_skfem(two_tets_mesh: SimplicialMesh, device):
    mesh = two_tets_mesh.to(device)

    skfem_mesh = skfem.MeshTet(
        mesh.vert_coords.T.cpu().detach().numpy(),
        mesh.tets.T.cpu().detach().numpy(),
    )

    elem = skfem.ElementTetRT0()
    basis = skfem.InteriorBasis(skfem_mesh, elem)

    @skfem.BilinearForm
    def mass_form(u, v, w):
        return dot(u, v)

    skfem_mass_2 = mass_form.assemble(basis).todense()
    # Here, we compare the eigenvalues of the mass matrices rather than the mass
    # matrices themselves to avoid differences in index/orientation conventions.
    # In addition, the scikit-fem results need to be multiplied with 4 due to
    # difference in 2-form basis function definitions.
    skfem_mass_2_eigs = np.linalg.eigvalsh(skfem_mass_2) * 4
    skfem_mass_2_eigs.sort()
    skfem_mass_2_eigs_torch = torch.from_numpy(skfem_mass_2_eigs).to(
        dtype=mesh.dtype, device=device
    )

    cochain_mass_2 = tet_masses.mass_2(mesh).to_dense()
    cochain_mass_2_eigs = torch.linalg.eigvalsh(cochain_mass_2).sort().values

    torch.testing.assert_close(cochain_mass_2_eigs, skfem_mass_2_eigs_torch)


@pytest.mark.parametrize(
    "mass_matrix",
    [tet_masses.mass_0, tet_masses.mass_1, tet_masses.mass_2, tet_masses.mass_3],
)
def test_mass_matrix_symmetry(mass_matrix, two_tets_mesh: SimplicialMesh, device):
    mesh = two_tets_mesh.to(device)

    mass = mass_matrix(mesh).to_dense()
    mass_T = mass.T

    torch.testing.assert_close(mass, mass_T)


@pytest.mark.parametrize(
    "mass_matrix",
    [tet_masses.mass_0, tet_masses.mass_1, tet_masses.mass_2, tet_masses.mass_3],
)
def test_mass_matrix_positive_definite(
    mass_matrix, two_tets_mesh: SimplicialMesh, device
):
    mesh = two_tets_mesh.to(device)

    mass = mass_matrix(mesh).to_dense()
    eigs = torch.linalg.eigvalsh(mass)

    assert eigs.min() >= 1e-6


def test_mass_0_matrix_total_vol_partition(two_tets_mesh: SimplicialMesh, device):
    """The sum of the 0-mass matrix diagonal should be equal to the total volume."""
    mesh = two_tets_mesh.to(device)

    total_mass = tet_hodge_stars.star_0(mesh).tr
    total_vol = torch.sum(
        torch.abs(compute_tet_signed_vols(mesh.vert_coords, mesh.tets))
    )

    torch.testing.assert_close(total_mass, total_vol)


def test_mass_3_matrix_total_vol_partition(two_tets_mesh: SimplicialMesh, device):
    """The sum of the inverse of the 3-mass matrix should be equal to the total volume."""
    mesh = two_tets_mesh.to(device)

    total_mass = tet_masses.mass_3(mesh).inv.tr
    total_vol = torch.sum(
        torch.abs(compute_tet_signed_vols(mesh.vert_coords, mesh.tets))
    )

    torch.testing.assert_close(total_mass, total_vol)


def test_mass_0_matrix_connectivity(two_tets_mesh: SimplicialMesh, device):
    mesh = two_tets_mesh.to(device)

    mass_0 = tet_masses.mass_0(mesh)
    mass_0_mask = torch.zeros_like(mass_0.to_dense(), dtype=torch.int64)
    mass_0_mask[mass_0.pattern.idx_coo.unbind(0)] = 1

    # All elements in a 4x4 block of the M_0 matrix are nonzero iff the four
    # vertices form a tet.
    true_mass_0_mask = torch.tensor(
        [
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 0],
            [1, 1, 1, 0, 1],
        ],
        dtype=torch.int64,
        device=device,
    )

    torch.testing.assert_close(mass_0_mask, true_mass_0_mask)


def test_mass_1_matrix_connectivity(two_tets_mesh: SimplicialMesh, device):
    mesh = two_tets_mesh.to(device)

    mass_1 = tet_masses.mass_1(mesh)
    mass_1_mask = torch.zeros_like(mass_1.to_dense(), dtype=torch.int64)
    mass_1_mask[mass_1.pattern.idx_coo.unbind(0)] = 1

    # M_ij is nonzero iff e_i and e_j are in the same triangle.
    true_mass_1_mask = torch.tensor(
        [
            [1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 0, 1, 1, 0, 1, 0],
            [1, 1, 0, 1, 1, 0, 1, 0, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 0, 1, 1, 0, 1, 0],
            [1, 1, 0, 1, 1, 0, 1, 0, 1],
            [1, 1, 1, 0, 1, 1, 0, 1, 0],
            [1, 1, 0, 1, 1, 0, 1, 0, 1],
        ],
        dtype=torch.int64,
        device=device,
    )

    torch.testing.assert_close(mass_1_mask, true_mass_1_mask)


def test_mass_2_matrix_connectivity(two_tets_mesh: SimplicialMesh, device):
    mesh = two_tets_mesh.to(device)

    mass_2 = tet_masses.mass_2(mesh)
    mass_2_mask = torch.zeros_like(mass_2.to_dense(), dtype=torch.int64)
    mass_2_mask[mass_2.pattern.idx_coo.unbind(0)] = 1

    # M_ij is nonzero iff t_i and t_j are in the same tet.
    true_mass_2_mask = torch.tensor(
        [
            [1, 1, 1, 1, 1, 1, 1],
            [1, 1, 0, 1, 0, 1, 0],
            [1, 0, 1, 0, 1, 0, 1],
            [1, 1, 0, 1, 0, 1, 0],
            [1, 0, 1, 0, 1, 0, 1],
            [1, 1, 0, 1, 0, 1, 0],
            [1, 0, 1, 0, 1, 0, 1],
        ],
        dtype=torch.int64,
        device=device,
    )

    torch.testing.assert_close(mass_2_mask, true_mass_2_mask)


def test_mass_1_patch(two_tets_mesh: SimplicialMesh, device):
    """Check that M_1 computes the L^2 norm for a const 1-form exactly."""
    mesh = two_tets_mesh.to(device)

    # Define a constant 1-form and discretize it with de Rham map.
    de_rham = DeRhamMap(k=1, quad_degree=1, mesh=mesh)
    n_splx, n_pts, _ = de_rham.sample_points().shape

    const_vec = torch.randn(3, dtype=mesh.dtype, device=device)
    const_1_form = repeat(
        const_vec,
        "coord -> splx pt coord",
        splx=n_splx,
        pt=n_pts,
    )

    # Compute the discrete energy for the 1-cochain over the mesh.
    mass_1 = tet_masses.mass_1(mesh)
    cochain = de_rham.discretize(const_1_form)
    energy = cochain @ mass_1 @ cochain

    # Compute the energy analytically as the volume integral of the constant 1-form
    # magnitude squared.
    total_vols = torch.sum(
        torch.abs(compute_tet_signed_vols(mesh.vert_coords, mesh.tets))
    )
    true_energy = total_vols * torch.sum(const_vec * const_vec)

    torch.testing.assert_close(energy, true_energy)


def test_mass_2_patch(two_tets_mesh: SimplicialMesh, device):
    """Check that M_2 computes the L^2 norm for a const 2-form exactly."""
    mesh = two_tets_mesh.to(device)

    # Define a constant 2-form and discretize it with de Rham map.
    de_rham = DeRhamMap(k=2, quad_degree=1, mesh=mesh)
    n_splx, n_pts, _ = de_rham.sample_points().shape

    const_vec = torch.randn(3, dtype=mesh.dtype, device=device)
    const_2_form = repeat(
        const_vec,
        "coord -> splx pt coord",
        splx=n_splx,
        pt=n_pts,
    )

    # Compute the discrete energy for the 2-cochain over the mesh.
    mass_2 = tet_masses.mass_2(mesh)
    cochain = de_rham.discretize(const_2_form)
    energy = cochain @ mass_2 @ cochain

    # Compute the energy analytically as the volume integral of the constant 1-form
    # magnitude squared.
    total_vols = torch.sum(
        torch.abs(compute_tet_signed_vols(mesh.vert_coords, mesh.tets))
    )
    true_energy = total_vols * torch.sum(const_vec * const_vec)

    torch.testing.assert_close(energy, true_energy)


@pytest.mark.parametrize(
    "mass_matrix",
    [tet_masses.mass_0, tet_masses.mass_1, tet_masses.mass_2, tet_masses.mass_3],
)
def test_mass_matrix_backward(mass_matrix, two_tets_mesh: SimplicialMesh, device):
    mesh = two_tets_mesh.to(device)
    mesh.requires_grad_()

    mass = mass_matrix(mesh)
    output = mass.val.sum()
    output.backward()

    assert mesh.grad is not None
    assert torch.isfinite(mesh.grad).all()


@pytest.mark.parametrize(
    "mass_matrix",
    [tet_masses.mass_0, tet_masses.mass_1, tet_masses.mass_2, tet_masses.mass_3],
)
def test_mass_matrix_gradcheck(mass_matrix, two_tets_mesh: SimplicialMesh, device):
    vert_coords = two_tets_mesh.vert_coords.clone().to(
        dtype=torch.float64, device=device
    )
    vert_coords.requires_grad_()

    def mass_fxn(test_vert_coords):
        mesh = two_tets_mesh.to(device=device, dtype=torch.float64)
        mesh.vert_coords = test_vert_coords
        m = mass_matrix(mesh)
        return m.val.sum()

    assert torch.autograd.gradcheck(mass_fxn, (vert_coords,), fast_mode=True)

import numpy as np
import skfem as skfem
import torch as t
from jaxtyping import Float
from skfem.helpers import dot

from cochain.complex import SimplicialComplex
from cochain.geometry.tri import tri_masses


def test_mass_1_with_skfem(flat_annulus_mesh: SimplicialComplex):
    skfem_mesh = skfem.MeshTri(
        flat_annulus_mesh.vert_coords[:, [0, 1]].T.cpu().detach().numpy(),
        flat_annulus_mesh.tris.T.cpu().detach().numpy(),
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
    skfem_mass_1_eigs_torch = t.from_numpy(skfem_mass_1_eigs).to(
        dtype=flat_annulus_mesh.vert_coords.dtype
    )

    cochain_mass_1 = tri_masses.mass_1(flat_annulus_mesh).to_dense()
    cochain_mass_1_eigs = t.linalg.eigvalsh(cochain_mass_1).sort().values

    t.testing.assert_close(cochain_mass_1_eigs, skfem_mass_1_eigs_torch)


def test_mass_1_symmetry(two_tris_mesh: SimplicialComplex):
    mass = tri_masses.mass_1(two_tris_mesh)
    mass_T = mass.T
    t.testing.assert_close(mass.to_dense(), mass_T.to_dense())


def test_mass_matrix_positive_definite(two_tris_mesh: SimplicialComplex):
    mass = tri_masses.mass_1(two_tris_mesh).to_dense()
    eigs = t.linalg.eigvalsh(mass)
    assert eigs.min() >= 1e-6


def test_mass_1_matrix_connectivity(two_tris_mesh: SimplicialComplex):
    mass_1 = tri_masses.mass_1(two_tris_mesh)
    mass_1_mask = t.zeros_like(mass_1.to_dense(), dtype=t.long)
    mass_1_mask[mass_1.sp_topo.idx_coo.unbind(0)] = 1

    true_mass_1_mask = t.tensor(
        [
            [1, 1, 1, 0, 0],
            [1, 1, 1, 0, 0],
            [1, 1, 1, 1, 1],
            [0, 0, 1, 1, 1],
            [0, 0, 1, 1, 1],
        ],
        dtype=t.long,
    )

    t.testing.assert_close(mass_1_mask, true_mass_1_mask)


def test_mass_1_linear_potential(two_tris_mesh: SimplicialComplex):
    mass_1 = tri_masses.mass_1(two_tris_mesh)

    const_vec = t.tensor([[1.0, 3.0, 2.0]], dtype=two_tris_mesh.vert_coords.dtype)
    phi_verts: Float[t.Tensor, " tri"] = t.sum(
        two_tris_mesh.vert_coords * const_vec, dim=-1
    )

    cochain = (
        phi_verts[two_tris_mesh.edges[:, 1]] - phi_verts[two_tris_mesh.edges[:, 0]]
    )

    energy = cochain @ mass_1 @ cochain

    vert_s_coord: Float[t.Tensor, "tri 3 3"] = two_tris_mesh.vert_coords[
        two_tris_mesh.tris
    ]

    edge_ij = vert_s_coord[:, 1] - vert_s_coord[:, 0]
    edge_ik = vert_s_coord[:, 2] - vert_s_coord[:, 0]

    area_vec = t.cross(edge_ij, edge_ik, dim=-1)
    two_area = t.linalg.norm(area_vec, dim=-1)

    area_vec_norm = area_vec / two_area.view(-1, 1)
    area = two_area / 2.0

    phi_tangent = (
        const_vec
        - t.sum(area_vec_norm * const_vec, dim=-1, keepdim=True) * area_vec_norm
    )

    true_energy = t.sum(area * t.sum(phi_tangent**2, dim=-1))

    t.testing.assert_close(energy, true_energy)

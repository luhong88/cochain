import itertools

import pytest
import torch as t

from cochain.complex import SimplicialComplex
from cochain.geometry.tet import tet_masses
from cochain.geometry.tri import tri_hodge_stars, tri_masses
from cochain.product.whitney import WhitneyWedgeProjection


def _compute_mass_matrix(mesh: SimplicialComplex, k: int):
    match mesh.dim:
        case 2:
            match k:
                case 0:
                    mass = tri_masses.mass_0_consistent(mesh)
                case 1:
                    mass = tri_masses.mass_1(mesh)
                case 2:
                    mass = tri_hodge_stars.star_2(mesh)
                case _:
                    raise ValueError()
        case 3:
            match k:
                case 0:
                    mass = tet_masses.mass_0_consistent(mesh)
                case 1:
                    mass = tet_masses.mass_1(mesh)
                case 2:
                    mass = tet_masses.mass_2(mesh)
                case 3:
                    mass = tet_masses.mass_3(mesh)
                case _:
                    raise ValueError()
        case _:
            raise NotImplementedError()

    return mass


@pytest.mark.parametrize("mesh_name", ["two_tris_mesh", "two_tets_mesh"])
def test_0_form_wedge_product(mesh_name, request, device):
    mesh: SimplicialComplex = request.getfixturevalue(mesh_name).to(device)

    n_simp_map = {
        dim: n_simp
        for dim, n_simp in enumerate(
            [mesh.n_verts, mesh.n_edges, mesh.n_tris, mesh.n_tets]
        )
    }

    const_0_form = t.ones(n_simp_map[0], dtype=mesh.vert_coords.dtype, device=device)
    for k in range(mesh.dim + 1):
        k_form = t.randn(n_simp_map[k]).to(device)

        proj = WhitneyWedgeProjection(0, k, mesh)

        b = proj(const_0_form, k_form)

        mass = _compute_mass_matrix(mesh, k).to_dense().to(device)

        w = t.linalg.solve(mass, b)

        t.testing.assert_close(w, k_form)

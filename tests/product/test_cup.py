import itertools

import pytest
import torch as t

from cochain.complex import SimplicialComplex
from cochain.product.cup import AntisymmetricCupProduct, CupProduct


@pytest.mark.parametrize("mesh_name", ["two_tris_mesh", "two_tets_mesh"])
def test_cup_product_graded_commutativity(mesh_name, request, device):
    mesh: SimplicialComplex = request.getfixturevalue(mesh_name).to(device)

    n_simp_map = {
        dim: n_simp
        for dim, n_simp in enumerate(
            [mesh.n_verts, mesh.n_edges, mesh.n_tris, mesh.n_tets]
        )
    }

    for k, l in itertools.product(range(mesh.dim), repeat=2):
        if k + l <= mesh.dim:
            k_cochain = t.randn(n_simp_map[k]).to(device)
            l_cochain = t.randn(n_simp_map[l]).to(device)

            wedge_kl = CupProduct(k, l, mesh).to(device)
            wedge_lk = CupProduct(l, k, mesh).to(device)

            sign = (-1.0) ** (k * l)

            lhs = wedge_kl(k_cochain, l_cochain)
            rhs = sign * wedge_lk(l_cochain, k_cochain)

            if k == 0 and l == 0:
                t.testing.assert_close(lhs, rhs)
            else:
                assert not t.allclose(lhs, rhs)

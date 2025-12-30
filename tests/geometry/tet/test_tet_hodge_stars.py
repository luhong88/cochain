import torch as t

from cochain.complex import SimplicialComplex
from cochain.geometry.tet import tet_hodge_stars


def test_star_2_on_reg_tet(reg_tet_mesh: SimplicialComplex):
    s2 = tet_hodge_stars.star_2(reg_tet_mesh).val

    true_s2 = t.ones_like(s2) * (1.0 / 6.0)

    t.testing.assert_close(s2, true_s2)


def test_star_1_on_reg_tet(reg_tet_mesh: SimplicialComplex):
    s1 = tet_hodge_stars.star_1(reg_tet_mesh).val

    true_s1 = t.ones_like(s1) * (1.0 / 6.0)

    t.testing.assert_close(s1, true_s1)

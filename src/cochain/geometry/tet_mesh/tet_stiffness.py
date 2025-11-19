import torch as t
from jaxtyping import Float, Integer

from ...complex import SimplicialComplex
from ...utils.constants import EPS

"""
For a given tetrahedron ijkl. We define a "local reference frame" for each edge.
For example, consider the edge ij as the "self", or s. Then
  * the opposite edge, kl, can be denoted as o.
  * The edge ik connecting the tail of ij and kl can be denoted as tt.
  * The edge jl connecting the head of ij and kl can be denoted as hh.
  * The edge il connecting the tail of ij with head of kl is th.
  * The edge jk connecting the head of ij with tail of kl is ht.

These local relations can be translated into global relations as follows:

-------------------------------
s     o    tt    hh    th    ht
-------------------------------
ij    kl   ik    jl    il    jk
jk    il   ij    kl    jl    ik
kl    ij   ik    jl    jk    il
li    jk   jl    ik    kl    ij
-------------------------------
"""


def _tet_vol(
    vert_coords: Float[t.Tensor, "vert 3"], tets: Integer[t.LongTensor, "tet 4"]
) -> Float[t.Tensor, "tet"]:
    tet_vert_coords: Float[t.Tensor, "tet 4 3"] = vert_coords[tets]

    # For each tet ijkl, compute the edge vectors ij, ik, and il. The volume of
    # the tet is given by the absolute value of the scalar triple product of these
    # three vectors, divided by 6.
    tet_edges = tet_vert_coords[:, [1, 2, 3], :] - tet_vert_coords[:, [0, 0, 0], :]

    tet_vols = (
        t.abs(
            t.sum(
                t.cross(tet_edges[:, 0], tet_edges[:, 1], dim=-1) * tet_edges[:, 2],
                dim=-1,
            )
        )
        / 6.0
    )

    return tet_vols


def _cotan_weights(
    vert_coords: Float[t.Tensor, "vert 3"],
    tets: Integer[t.LongTensor, "tri 3"],
    n_verts: int,
) -> Float[t.Tensor, "vert vert"]:
    raise NotImplementedError()

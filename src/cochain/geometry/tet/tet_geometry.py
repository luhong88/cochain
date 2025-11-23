import torch as t
from jaxtyping import Float, Integer


def _tet_signed_vols(
    vert_coords: Float[t.Tensor, "vert 3"], tets: Integer[t.LongTensor, "tet 4"]
) -> Float[t.Tensor, "tet"]:
    """
    Compute the signed volume of each tetrahedron in a 3D mesh. A tet is assigned
    a positive volume if it satisfies the right-hand rule (or, equivalently, its
    vertex indices can be reordered into ascending order with an even permutation).
    """
    tet_vert_coords: Float[t.Tensor, "tet 4 3"] = vert_coords[tets]

    # For each tet ijkl, compute the edge vectors ij, ik, and il. The volume of
    # the tet is given by the absolute value of the scalar triple product of these
    # three vectors, divided by 6.
    tet_edges = tet_vert_coords[:, [1, 2, 3], :] - tet_vert_coords[:, [0, 0, 0], :]

    tet_signed_vols = (
        t.sum(
            t.cross(tet_edges[:, 0], tet_edges[:, 1], dim=-1) * tet_edges[:, 2],
            dim=-1,
        )
        / 6.0
    )

    return tet_signed_vols


def _d_tet_signed_vols_d_vert_coords(
    vert_coords: Float[t.Tensor, "vert 3"], tets: Integer[t.LongTensor, "tet 4"]
) -> Float[t.Tensor, "tet 4 3"]:
    """
    Compute the gradient of the signed volume of each tetrahedron wrt the coordinates
    of its four vertices.
    """
    i, j, k, l = 0, 1, 2, 3

    tet_vert_coords: Float[t.Tensor, "tet 4 3"] = vert_coords[tets]

    # For each tet ijkl and for each vertex, find the (inward) area normal of the
    # base triangle, which is proportional to the gradient of the volume wrt the
    # vertex position.
    #
    # vert   base tri   area normal
    # ----------------------------------
    # i      jkl        jl x jk
    # j      ikl        ik x il
    # k      ijl        il x ij
    # l      ijk        ij x ik
    #
    # Note that, if a tet has a negative orientation, the resulting gradient will
    # also carry a negative sign (i.e., it points in the direction that minimizes
    # the unsigned/absolute volume of the tet).
    base_tri_edge_1 = (
        tet_vert_coords[:, [l, k, l, j]] - tet_vert_coords[:, [j, i, i, i]]
    )
    base_tri_edge_2 = (
        tet_vert_coords[:, [k, l, j, k]] - tet_vert_coords[:, [j, i, i, i]]
    )

    dVdV = t.cross(base_tri_edge_1, base_tri_edge_2, dim=-1) / 6.0

    return dVdV

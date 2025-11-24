import torch as t
from jaxtyping import Float, Integer

from ...utils.constants import EPS


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


def _tet_face_vector_areas(
    vert_coords: Float[t.Tensor, "vert 3"], tets: Integer[t.LongTensor, "tet 4"]
) -> tuple[
    Float[t.Tensor, "tet 6 3"], Float[t.Tensor, "tet 6 3"], Float[t.Tensor, "tet 6"]
]:
    """
    Compute the outward pointing vector areas for triangles in a tet mesh and their
    dihedral angles.

    Specifically, for each tet and each edge s, this function computes
    * the double area vectors for the two triangles sharing the opposite edge o,
    * the dihedral angle between these two triangles, weighted by the length of o;
      or, more precisely, the cotan weight -|o|cot(theta_o)/6
    """
    i, j, k, l = 0, 1, 2, 3

    tet_vols = t.abs(_tet_signed_vols(vert_coords, tets))

    tet_vert_coords: Float[t.Tensor, "tet 4 3"] = vert_coords[tets]

    # For each tet ijkl and each edge s, computes the (outward) normal on the two
    # triangles with o as the shared edge (i.e., th x o and hh x o).
    area_vec_to: Float[t.Tensor, "tet 6 3"] = t.cross(
        tet_vert_coords[:, [k, i, i, j, i, l]] - tet_vert_coords[:, [l, l, l, k, j, k]],
        tet_vert_coords[:, [i, j, j, i, k, j]] - tet_vert_coords[:, [l, l, l, k, j, k]],
        dim=-1,
    )
    area_vec_ho: Float[t.Tensor, "tet 6 3"] = t.cross(
        tet_vert_coords[:, [j, j, k, i, l, j]] - tet_vert_coords[:, [l, l, l, k, j, k]],
        tet_vert_coords[:, [k, k, i, l, i, i]] - tet_vert_coords[:, [l, l, l, k, j, k]],
        dim=-1,
    )

    # For each tet ijkl and each edge s, computes the contribution of s to the
    # cotan Laplacian (restricted to ijkl), which is given by -|o|cot(theta_o)/6,
    # where |o| is the length of the opposite edge, and theta_o is the dihedral
    # angle formed by the two triangles with o as the shared edge. This contribution
    # can also be written as <th x o, hh x o> / 36 * vol_ijkl; here, vol_ijkl is
    # the unsigned/absolute volume of the tet ijkl.
    weight_o: Float[t.Tensor, "tet 6"] = (
        t.sum(area_vec_to * area_vec_ho, dim=-1) / (36.0 * tet_vols + EPS)[:, None]
    )

    return area_vec_to, area_vec_ho, weight_o

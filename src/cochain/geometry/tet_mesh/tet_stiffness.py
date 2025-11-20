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
ik    jl   ij    kl    il    jk
jk    il   ij    kl    jl    ik
jl    ik   ij    kl    jk    il
kl    ij   ik    jl    jk    il
li    jk   jl    ik    kl    ij
-------------------------------
"""


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


def _cotan_weights(
    vert_coords: Float[t.Tensor, "vert 3"],
    tets: Integer[t.LongTensor, "tri 3"],
    n_verts: int,
) -> Float[t.Tensor, "vert vert"]:
    i, j, k, l = 0, 1, 2, 3

    tet_vert_coords: Float[t.Tensor, "tet 4 3"] = vert_coords[tets]
    tet_vols = t.abs(_tet_signed_vols(vert_coords, tets))

    edge_o = (
        tet_vert_coords[:, [l, l, l, k, j, k]] - tet_vert_coords[:, [k, j, i, i, i, j]]
    )
    edge_hh = (
        tet_vert_coords[:, [l, l, l, l, l, k]] - tet_vert_coords[:, [j, k, k, k, j, i]]
    )
    edge_th = (
        tet_vert_coords[:, [l, l, l, k, k, l]] - tet_vert_coords[:, [i, i, j, j, j, k]]
    )

    # For each tet ijkl and each edge s, computes the (outward) normal on the two
    # triangles with o as the shared edge.
    norm_tri_to: Float[t.Tensor, "tet 6 3"] = t.cross(edge_th, edge_o)
    norm_tri_ho: Float[t.Tensor, "tet 6 3"] = t.cross(edge_hh, edge_o)

    # For each tet ijkl and each edge s, computes the contribution of s to the
    # cotan Laplacian (restricted to ijkl), which is given by -|o|cot(theta_o)/6,
    # where |o| is the length of the opposite edge, and theta_o is the dihedral
    # angle formed by the two triangles with o as the shared edge. This contribution
    # can also be written as <th x o, hh x o> / 36 * vol_ijkl.
    weight_o: Float[t.Tensor, "tet 6"] = (
        t.sum(norm_tri_to * norm_tri_ho, dim=-1) / (36.0 * tet_vols + EPS)[:, None]
    )

    # For each tet ijkl, each edge s contributes one term w_o to the weight matrix,
    # thus each tet contributes six terms (in COO format):
    #
    # [
    #   (i, j, w_kl = w_0),
    #   (i, k, w_jl = w_1),
    #   (j, k, w_il = w_2),
    #   (j, l, w_ik = w_3),
    #   (k, l, w_ij = w_4),
    #   (l, i, w_jk = w_5),
    # ]

    weights_idx = (
        tets[:, [i, i, j, j, k, l, j, k, k, l, l, i]].T.flatten().reshape(2, -1)
    )
    weights_val = weight_o.T.flatten()

    asym_weights = t.sparse_coo_tensor(weights_idx, weights_val, (n_verts, n_verts))
    sym_weights = (asym_weights + asym_weights.T).coalesce()

    return sym_weights


def stiffness_matrix(
    tet_mesh: SimplicialComplex,
) -> Float[t.Tensor, "vert vert"]:
    """
    Computes the stiffness matrix for a 3D mesh, sometimes also known as the "cotan
    Laplacian".
    """
    # The cotan weight matrix W gives the stiffness matrix except for the diagonal
    # elements.
    sym_stiffness = _cotan_weights(
        tet_mesh.vert_coords, tet_mesh.tets, tet_mesh.n_verts
    )

    # Compute the diagonal elements of the stiffness matrix.
    stiffness_diag = t.sparse.sum(sym_stiffness, dim=-1)
    # laplacian_diag.indices() has shape (1, nnz_diag)
    diag_idx = t.concatenate([stiffness_diag.indices(), stiffness_diag.indices()])

    # Generate the final, complete stiffness matrix.
    stiffness = t.sparse_coo_tensor(
        t.hstack((sym_stiffness.indices(), diag_idx)),
        t.concatenate((sym_stiffness.values(), -stiffness_diag.values())),
    ).coalesce()

    return stiffness

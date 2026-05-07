import torch
from einops import einsum, rearrange
from jaxtyping import Float, Integer
from torch import Tensor

from ...complex import SimplicialMesh
from ...sparse.decoupled_tensor import SparseDecoupledTensor

# Naming conventions for 3D mesh edges
#
# For a given tetrahedron `ijkl`. We define a "local reference frame" for each edge.
# For example, consider the edge `ij` as the "self", or `s`. Then
#
# * the opposite edge, `kl`, can be denoted as `o`.
# * The edge `ik` connecting the tail of `ij` and `kl` can be denoted as `tt`.
# * The edge `jl` connecting the head of `ij` and `kl` can be denoted as `hh`.
# * The edge `il` connecting the tail of `ij` with head of `kl` is `th`.
# * The edge `jk` connecting the head of `ij` with tail of `kl` is `ht`.
#
# For a tet with vertices ijkl, these local indices can be translated into this
# local reference frame as follows:
#
# | s  | o  | tt | hh | th | ht |
# | -- | -- | -- | -- | -- | -- |
# | ij | kl | ik | jl | il | jk |
# | ik | jl | ij | kl | il | jk |
# | il | jk | jl | ik | kl | ij |
# | jk | il | ij | kl | jl | ik |
# | jl | ik | ij | kl | jk | il |
# | kl | ij | ik | jl | jk | il |
#
# In addition, we tabulate the following outward facing normal vectors and their
# Jacobians. Here, the `[]` notation maps a vector `t` to a skew symmetric
# matrix `[t]` that represents its cross product with another vector; i.e.,
# `t x v = [t]v` for all vectors `v`.
#
# First, we tabulate the th x o normal vector (i.e., the double area normal that
# is outward facing if the tet is positively oriented) and its gradient wrt vertex
# coordinates. Note that, in the `th x o` column, the cross product in the parenthesis
# is rearranged to be outward facing and "corner-centered", and this is the version
# implemented in the code.
#
# | s  | th x o  (-> oriented) | grad_i     | grad_j     | grad_k     | grad_l     |
# | -- | --------------------- | ---------- | ---------- | ---------- | ---------- |
# | ij | il x kl (-> lk x li)  | [lk]=[-o]  | [ll]=0     | [il]=[th]  | [ki]=[-tt] |
# | ik | il x jl (-> li x lj)  | [jl]=[o]   | [li]=[-th] | [ll]=0     | [ij]=[tt]  |
# | il | kl x jk (-> kl x kj)  | [kk]=0     | [kl]=[th]  | [lj]=[-tt] | [jk]=[o]   |
# | jk | jl x il (-> li x lj)  | [jl]=[th]  | [li]=[-o]  | [ll]=0     | [ij]=[tt]  |
# | jl | jk x ik (-> kj x ki)  | [kj]=[-th] | [ik]=[o]   | [ji]=[-tt] | [kk]=0     |
# | kl | jk x ij (-> ji x jk)  | [kj]=[-th] | [ik]=[tt]  | [ji]=[-o]  | [jj]=0     |
#
# Then, we tabulate the `hh x o` normal vector and its gradient wrt vertex coordinates:
#
# | s  | hh x o  (-> oriented) | grad_i     | grad_j     | grad_k     | grad_l     |
# | -- | --------------------- | ---------- | ---------- | ---------- | ---------- |
# | ij | jl x kl (-> lj x lk)  | [ll]=0     | [kl]=[o]   | [lj]=[-hh] | [jk]=[ht]  |
# | ik | kl x jl (-> lj x lk)  | [ll]=0     | [kl]=[hh]  | [lj]=[-o]  | [jk]=[ht]  |
# | il | ik x jk (-> kj x ki)  | [kj]=[-o]  | [ik]=[hh]  | [ji]=[-ht] | [kk]=0     |
# | jk | kl x il (-> lk x li)  | [lk]=[-hh] | [ll]=0     | [il]=[o]   | [ki]=[-ht] |
# | jl | kl x ik (-> ki x kl)  | [lk]=[-hh] | [kk]=0     | [il]=[ht]  | [ki]=[-o]  |
# | kl | jl x ij (-> jl x ji)  | [jl]=[hh]  | [li]=[-ht] | [jj]=0     | [ij]=[o]   |


def compute_tet_signed_vols(
    vert_coords: Float[Tensor, "global_vert coord=3"],
    tets: Integer[Tensor, "tet local_vert=4"],
) -> Float[Tensor, " tet"]:
    """
    Compute the signed volume of each tetrahedron in a 3D mesh.

    A tet is assigned a positive volume if it satisfies the right-hand rule. For
    a tet with vertices (v0, v1, v2, v3), curl the right hand following the ordering
    of v1, v2, and v3; if the thumb points outwards away from v0, the tet has a
    positive volume. Mathematically, this sign is computed via a scalar triple
    product of the e01, e02, and e03 edges.

    This volume sign is not to be confused with the orientation sign/parity of a
    tet, which refers to the parity of the permutation required to order the
    vertices in lex order.
    """
    tet_vert_coords: Float[Tensor, "tet 4 3"] = vert_coords[tets]

    # For each tet ijkl, compute the edge vectors ij, ik, and il. The volume of
    # the tet is given by the absolute value of the scalar triple product of these
    # three vectors, divided by 6.
    tet_edges = tet_vert_coords[:, [1, 2, 3], :] - tet_vert_coords[:, [0], :]

    tet_signed_vols = (
        torch.sum(
            torch.cross(tet_edges[:, 0], tet_edges[:, 1], dim=-1) * tet_edges[:, 2],
            dim=-1,
        )
        / 6.0
    )

    return tet_signed_vols


def compute_d_tet_signed_vols_d_vert_coords(
    vert_coords: Float[Tensor, "global_vert coord=3"],
    tets: Integer[Tensor, "tet local_vert=4"],
) -> Float[Tensor, "tet local_vert=4 coord=3"]:
    """Compute the gradient of the signed volume with respect to vertex coordinates."""
    i, j, k, l = 0, 1, 2, 3

    tet_vert_coords: Float[Tensor, "tet 4 3"] = vert_coords[tets]

    # For each tet ijkl and for each vertex, find the (inward) area normal of the
    # base triangle, which is proportional to the gradient of the volume wrt the
    # vertex position.
    #
    # vert   base tri   area normal
    # -----------------------------
    # i      jkl        jl x jk
    # j      ikl        ik x il
    # k      ijl        il x ij
    # l      ijk        ij x ik
    #
    # Note that, if a tet has a negative orientation, the resulting gradient will
    # also carry a negative sign (i.e., it points in the direction that minimizes
    # the unsigned/absolute volume of the tet).
    base_edge_1 = tet_vert_coords[:, [l, k, l, j]] - tet_vert_coords[:, [j, i, i, i]]
    base_edge_2 = tet_vert_coords[:, [k, l, j, k]] - tet_vert_coords[:, [j, i, i, i]]

    # The 1/6 factor comes from the fact that the volume is 1/6 of the scalar
    # triple product.
    dVdV = torch.cross(base_edge_1, base_edge_2, dim=-1) / 6.0

    return dVdV


def compute_bc_grads(
    vert_coords: Float[Tensor, "global_vert coord=3"],
    tets: Integer[Tensor, "tet local_vert=4"],
) -> tuple[Float[Tensor, " tet"], Float[Tensor, "tet local_vert=4 coord=3"]]:
    r"""
    Compute the gradients of the barycentric coordinates.

    Consider a tet $ijkl$. Let $\lambda_i(x)$ be the barycentric coordinate function
    associated with vertex $i$ for a point $x$ on the tet. To find the gradient
    of this function, we use the volume definition of barycentric coordinate functions.
    Let $V_i(x)$ be the volume of the sub-tet formed by vertices $(x, j, k, l)$. Then,
    we can write $\lambda_i(x)=V_i(x)/V$, and thus

    $$\nabla_x\lambda_i(x) = V^{-1}\nabla_x V_i(x)$$

    Note that, since $x$ is a vertex of the tet $V_i(x)$ with vertices $j$, $k$ and
    $l$ fixed, taking the gradient of this function w.r.t. $x$ is equivalent to taking
    the gradient of $V$ w.r.t. vertex coordinate $v_i$ with vertices $j$, $k$ and $l$
    fixed. Therefore, the gradient can be simplified as

    $$\nabla_x\lambda_i(x) = V^{-1}\nabla_i V$$

    Note that this expression is independent of $x$.
    """
    tet_signed_vols = compute_tet_signed_vols(vert_coords, tets)
    d_tet_signed_vols_d_vert_coords = compute_d_tet_signed_vols_d_vert_coords(
        vert_coords, tets
    )
    bc_grads = d_tet_signed_vols_d_vert_coords / tet_signed_vols.view(-1, 1, 1)

    return tet_signed_vols, bc_grads


def compute_bc_grad_dots(
    vert_coords: Float[Tensor, "global_vert coord=3"],
    tets: Integer[Tensor, "tet local_vert=4"],
) -> tuple[Float[Tensor, " tet"], Float[Tensor, "tet local_vert=4 local_vert=3"]]:
    r"""
    Compute the inner products between barycentric coordinate gradients.

    Note that the geometric shortcut for computing the pairwise gradient inner
    products on tri meshes is not applicable to tet meshes.
    """
    tet_signed_vols = compute_tet_signed_vols(vert_coords, tets)
    d_signed_vols_d_vert_coords = compute_d_tet_signed_vols_d_vert_coords(
        vert_coords, tets
    )

    bc_grads: Float[Tensor, "tet 4 3"] = (
        d_signed_vols_d_vert_coords / tet_signed_vols.view(-1, 1, 1)
    )
    bc_grad_dots: Float[Tensor, "tet 4 4"] = einsum(
        bc_grads,
        bc_grads,
        "tet vert_1 coord, tet vert_2 coord -> tet vert_1 vert_2",
    )

    return tet_signed_vols, bc_grad_dots


def compute_cotan_weights(
    tet_mesh: SimplicialMesh,
) -> Float[SparseDecoupledTensor, "global_vert global_vert"]:
    r"""
    Compute the cotan weights associated with edges on a tet mesh.

    For edge $e_{ij}$, the cotan weight is given by

    $$W_{ij} = \frac 1 6 \sum_{kl} \|e_{kl}\| \cot\theta^{ij}_{kl}$$

    where $kl$ sums over all vertices $k$ and $l$ such that $ijkl$ forms a tet,
    $\|e_{kl}\|$ is the length of the edge $kl$, and $\theta^{ij}_{kl}$ is the
    interior dihedral angle formed by the two triangles $ikl$ and $jkl$ that
    shares $kl$ as an edge face.
    """
    i, j, k, l = 0, 1, 2, 3

    tet_vols = torch.abs(compute_tet_signed_vols(tet_mesh.vert_coords, tet_mesh.tets))
    tet_vert_coords = tet_mesh.vert_coords[tet_mesh.tets]

    # For each tet ijkl and each edge s, compute the (outward) normal on the two
    # triangles with o as the shared edge (i.e., th x o and hh x o).
    th_cross_o: Float[Tensor, "tet 6 3"] = torch.cross(
        tet_vert_coords[:, [k, i, l, i, j, i]] - tet_vert_coords[:, [l, l, k, l, k, j]],
        tet_vert_coords[:, [i, j, j, j, i, k]] - tet_vert_coords[:, [l, l, k, l, k, j]],
        dim=-1,
    )
    hh_cross_o: Float[Tensor, "tet 6 3"] = torch.cross(
        tet_vert_coords[:, [j, j, j, k, i, l]] - tet_vert_coords[:, [l, l, k, l, k, j]],
        tet_vert_coords[:, [k, k, i, i, l, i]] - tet_vert_coords[:, [l, l, k, l, k, j]],
        dim=-1,
    )

    # For each tet ijkl and each edge s, compute the cotan weight associated
    # with edge s, |o| * cot(θ_o) / 6, where theta_o is the interior dihedral
    # angle formed by the two triangles with o as the shared edge. This contribution
    # can also be written as <th x o, hh x o> / 36V, where V is the unsigned
    # volume of the tet. To see this, use the fact that cot(θ_o) = <th x o, hh x o>/
    # |(th x o) x (hh x o)| and the quadruple cross product term simplifies to
    # |<th, o x hh>| |o| because of the identity (axb)x(cxd) = (a⋅(bxd))c-(a⋅(bxc)d).
    weight_o: Float[Tensor, "tet 6"] = einsum(
        th_cross_o,
        hh_cross_o,
        1.0 / (36.0 * tet_vols),
        "tet edge coord, tet edge coord, tet -> tet edge",
    )

    # Scatter the local edge s contributions to form the global weight matrix.
    idx_coo_asym = rearrange(
        tet_mesh.tets[:, [[i, i, i, j, j, k], [j, k, l, k, l, l]]],
        "tet vert edge -> vert (tet edge)",
    )
    vals_asym = rearrange(weight_o, "tet edge -> (tet edge)")

    idx_coo = torch.hstack((idx_coo_asym, torch.flip(idx_coo_asym, dims=(0,))))
    vals = torch.cat((vals_asym, vals_asym))

    weights = tet_mesh._sparse_coalesced_matrix(
        operator="tet_compute_cotan_weights",
        indices=idx_coo,
        values=vals,
        size=(tet_mesh.n_verts, tet_mesh.n_verts),
    )

    return weights

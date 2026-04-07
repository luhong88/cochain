import torch
from einops import einsum
from jaxtyping import Float, Integer
from torch import LongTensor, Tensor

# We adopt the following convention for describing the relation between vertices
# locally in a triangle. For a given triangle represented by three vertex indices,
# we refer the first, second, and third vertex with index `i`, `j`, and `k`. This
# effectively assigns an orientation to the triangle, and allows us to distinguish
# the neighbors for each vertex ("self", or `s`) as either the "next" (`n`) and
# "previous" (`p`) vertex. For a triangle `ijk`, the "self"/"next"/"prev" relation
# is defined as follows:
#
# | s | n | p |
# | - | - | - |
# | i | j | k |
# | j | k | i |
# | k | i | j |
#
# We will refer to a triangle as `ijk` or `snp`, depending on the context.


def compute_tri_areas(
    vert_coords: Float[Tensor, "global_vert coord=3"],
    tris: Integer[LongTensor, "tri vert=3"],
) -> Float[Tensor, " tri"]:
    """Compute the area of all triangles in a tri mesh."""
    vert_s_coord: Float[Tensor, "tri 3 3"] = vert_coords[tris]

    edge_ij = vert_s_coord[:, 1] - vert_s_coord[:, 0]
    edge_ik = vert_s_coord[:, 2] - vert_s_coord[:, 0]

    area = 0.5 * torch.linalg.norm(torch.cross(edge_ij, edge_ik, dim=-1), dim=-1)

    return area


def compute_d_tri_areas_d_vert_coords(
    vert_coords: Float[Tensor, "global_vert coord=3"],
    tris: Integer[LongTensor, "tri local_vert=3"],
) -> Float[Tensor, "tri local_vert=3 coord=3"]:
    r"""
    Compute the gradient of the triangle areas with respect to vertex coordinates.

    For a triangle $snp$, we can define its area using the cross product:
    $$A = \frac 1 2 \|e_{sn}\times e_{sp}\|$$
    with some algebra, it can be shown that the gradient of $A$ with respect to $v_s$,
    the position of vertex $s$, is given by
    $$
    \nabla_s A =
    \frac{(e_{sn}\times e_{sp})\times e_{np}}{2\|e_{sn}\times e_{sp}\|}
    $$
    Note that this is a vector in the plane of the triangle and points away from
    the base edge $e_{sp}$ perpendicularly in the direction of the triangle's altitude.
    """
    # For each triangle snp, and each vertex s, find the edge vectors sn, sp, and
    # np, and a vector normal to the triangle at s (sn x sp).
    vert_s_coord: Float[Tensor, "tri 3 3"] = vert_coords[tris]

    edge_sn = vert_s_coord[:, [1, 2, 0], :] - vert_s_coord
    edge_sp = vert_s_coord[:, [2, 0, 1], :] - vert_s_coord
    edge_np = vert_s_coord[:, [2, 0, 1], :] - vert_s_coord[:, [1, 2, 0], :]

    norm_s: Float[Tensor, "tri 3 3"] = torch.cross(edge_sn, edge_sp, dim=-1)
    norm_s_len = torch.linalg.norm(norm_s, dim=-1, keepdim=True)

    unorm_s = norm_s / norm_s_len

    # For each triangle snp, the gradient of its area with respect to each vertex
    # s is given by (unorm_s x edge_np)/2
    dAdV = torch.cross(unorm_s, edge_np, dim=-1) / 2.0

    return dAdV


def compute_bc_grads(
    vert_coords: Float[Tensor, "global_vert coord=3"],
    tris: Integer[LongTensor, "tri local_vert=3"],
) -> tuple[Float[Tensor, " tri"], Float[Tensor, "tri vert=3 coord=3"]]:
    r"""
    Compute the gradients of the barycentric coordinates.

    Consider a triangle $ijk$. Let $\lambda_i(x)$ be the barycentric coordinate function
    associated with vertex $i$ for a point $x$ on the triangle. To find the gradient
    of this function, we use the area definition of barycentric coordinate functions.
    Let $A_i(x)$ be the area of the sub-triangle formed by vertices $(x, j, k)$. Then,
    we can write $\lambda_i(x)=A_i(x)/A$, and thus

    $$\nabla_x\lambda_i(x) = A^{-1}\nabla_x A_i(x)$$

    Note that, since $x$ is a vertex of the triangle $A_i(x)$ with vertices $j$ and
    $k$ fixed, taking the gradient of this function w.r.t. $x$ is equivalent to taking
    the gradient of $A$ w.r.t. vertex coordinate $v_i$ with vertices $j$ and $k$ fixed.
    Therefore, the gradient can be simplified as

    $$
    \nabla_x\lambda_i(x) = A^{-1}\nabla_i A
    $$

    Note that this expression is independent of $x$.
    """
    tri_areas = compute_tri_areas(vert_coords, tris)
    d_tri_areas_d_vert_coords = compute_d_tri_areas_d_vert_coords(vert_coords, tris)
    bc_grads = d_tri_areas_d_vert_coords / tri_areas.view(-1, 1, 1)

    return tri_areas, bc_grads


def compute_bc_grad_dots(
    vert_coords: Float[Tensor, "global_vert coord=3"],
    tris: Integer[LongTensor, "tri local_vert=3"],
) -> tuple[Float[Tensor, " tri"], Float[Tensor, "tri local_vert=3 local_vert=3"]]:
    r"""
    Compute the inner products between barycentric coordinate gradients.

    Consider a triangle $snp$, from `compute_bc_grads()`, we showed that

    $$\nabla_x\lambda_s(x)= A^{-1}\nabla_sA$$

    Furthermore, from `compute_cotan_weights()`, we showed that

    $$
    \nabla_s A = (\hat e_{sn}\times \hat e_{sp})\times e_{np}
    $$
    where $\hat e$ indicates that the edge vector is length-normalized.

    These two equations allow us to compute the inner products between the gradients
    of the barycentric coordinate functions without explicit computation of the gradients.
    For example,

    $$
    \left<\nabla_x\lambda_i, \nabla_x\lambda_j\right> =
    \frac{1}{4A^2}
    \left<
    (\hat e_{ij}\times \hat e_{ik})\times e_{jk},
    (\hat e_{jk}\times \hat e_{ji})\times e_{ki}
    \right> =
    \left<e_{jk}, e_{ki}\right>/4A^2
    $$

    Note that the triple cross products in the inner product cancels out, because
    the terms such as $\hat e_{ij}\times \hat e_{ik}$ effectively rotate the edge
    vectors counter-clockwise by 90 degrees, they have no effect on the final inner
    products between the edge vectors.

    Let us denote the expression $\left<e_{jk}, e_{ki}\right>/4A^2$ as `(jk, ki)`.
    Then, the 9 inner products can be expressed as:

    ```
    (jk, jk)  (jk, ki)  (jk, ij)
    (ki, jk)  (ki, ki)  (ki, ij)
    (ij, jk)  (ij, ki)  (ij, ij)
    ```
    """
    i, j, k = 0, 1, 2

    # To compute the per-triangle 3x3 matrix of inner products, find, for each
    # vertex, the opposite edge, and then use einsum() to compute all pairwise
    # inner products between the three edges.
    tris_verts = vert_coords[tris]
    tris_edges = tris_verts[:, [k, i, j]] - tris_verts[:, [j, k, i]]
    tris_edge_dots = einsum(
        tris_edges,
        tris_edges,
        "tri edge_1 coord, tri edge_2 coord -> tri edge_1 edge_2",
    )

    tri_areas = compute_tri_areas(vert_coords, tris)
    tri_areas_scaled = 4.0 * tri_areas**2

    bc_grad_dots = tris_edge_dots / tri_areas_scaled.view(-1, 1, 1)

    return tri_areas, bc_grad_dots


def compute_cotan_weights(
    vert_coords: Float[Tensor, "global_vert coord=3"],
    tris: Integer[LongTensor, "tri local_vert=3"],
) -> Float[Tensor, "global_vert global_vert"]:
    r"""
    Compute the cotan weights associated with edges on a tri mesh.

    For edge $e_{ij}$, the cotan weight is given by
    $$W_{ij} = -\frac 1 2 \sum_k \cot\alpha_k$$
    where $k$ sums over all vertices such that $ijk$ forms a triangle and $\alpha_k$
    is the interior angle at vertex $k$ opposite to the edge $ij$.
    """
    # For each triangle snp, and each vertex s, find the edge vectors sn and sp,
    # Compute the dot product and cross product between these two vectors, and use
    # their ratio to compute the cotan of the interior angle at s.
    vert_s_coord: Float[Tensor, "tri 3 3"] = vert_coords[tris]

    edge_sn = vert_s_coord[:, [1, 2, 0], :] - vert_s_coord
    edge_sp = vert_s_coord[:, [2, 0, 1], :] - vert_s_coord

    edge_sn_sp_dot = torch.sum(edge_sn * edge_sp, dim=-1)
    edge_sn_sp_cross = torch.linalg.norm(torch.cross(edge_sn, edge_sp, dim=-1), dim=-1)
    cot_s: Float[Tensor, "tri 3"] = edge_sn_sp_dot / edge_sn_sp_cross

    # For each triangle snp, and each vertex s, scatter cot_s to edge np in the
    # weight matrix (W_np) and assemble the asymmetric sparse weight matrix.
    r_idx = tris[:, [1, 2, 0]].flatten()
    c_idx = tris[:, [2, 0, 1]].flatten()
    idx_coo = torch.vstack((r_idx, c_idx))

    vals = -0.5 * cot_s.flatten()

    n_verts = vert_coords.size(0)
    shape = (n_verts, n_verts)

    asym_weights = torch.sparse_coo_tensor(indices=idx_coo, values=vals, size=shape)

    # Symmetrize so that the cotan at i is scattered to both jk and kj.
    sym_weights = (asym_weights + asym_weights.T).coalesce()

    return sym_weights

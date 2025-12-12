import torch as t
from jaxtyping import Float, Integer

from ...complex import SimplicialComplex
from ...utils.constants import EPS
from .tri_geometry import (
    _d_tri_areas_d_vert_coords,
    _tri_areas,
)
from .tri_stiffness import _cotan_weights, _d_cotan_weights_d_vert_coords


def star_2(tri_mesh: SimplicialComplex) -> Float[t.Tensor, "tri"]:
    """
    The Hodge 2-star operator maps the 2-simplices (triangles) in a mesh to their
    dual 0-cells. This function computes the ratio of the "volume" of the dual 0-cells
    (which is 1 by convention) to the area of the primal triangles. The returned tensor
    forms the diagonal of the 2-star tensor.
    """
    return 1.0 / _tri_areas(tri_mesh.vert_coords, tri_mesh.tris)


def d_inv_star_2_d_vert_coords(
    tri_mesh: SimplicialComplex,
) -> Float[t.Tensor, "tri vert 3"]:
    """
    Compute the Jacobian of the inverse Hodge 2-star matrix (diagonal elements)
    with respect to vertex coordinates.
    """
    vert_coords: Float[t.Tensor, "vert 3"] = tri_mesh.vert_coords
    tris: Integer[t.LongTensor, "tri 3"] = tri_mesh.tris

    n_verts = tri_mesh.n_verts
    n_tris = tri_mesh.n_tris

    dAdV = _d_tri_areas_d_vert_coords(vert_coords, tris)

    dSdV_idx = t.vstack(
        (t.repeat_interleave(t.arange(n_tris, device=tris.device), 3), tris.flatten())
    )
    dSdV_val = dAdV.flatten(end_dim=1)
    dSdV = t.sparse_coo_tensor(dSdV_idx, dSdV_val, (n_tris, n_verts, 3)).coalesce()

    return dSdV


def d_star_2_d_vert_coords(
    tri_mesh: SimplicialComplex,
) -> Float[t.Tensor, "tri vert 3"]:
    """
    Compute the Jacobian of the Hodge 2-star matrix (diagonal elements) with respect
    to vertex coordinates.
    """
    d_inv_S_dV = d_inv_star_2_d_vert_coords(tri_mesh)

    s2 = star_2(tri_mesh)[d_inv_S_dV.indices()[0]]
    inv_scale = -s2.square()[:, None]

    dSdV = t.sparse_coo_tensor(
        d_inv_S_dV.indices(), d_inv_S_dV.values() * inv_scale, d_inv_S_dV.shape
    ).coalesce()

    return dSdV


def star_1(tri_mesh: SimplicialComplex) -> Float[t.Tensor, "edge"]:
    """
    The Hodge 1-star operator maps the 1-simplices (edges) in a mesh to the
    circumcentric dual 1-cells. This function computes the length ratio of the dual
    1-cells to the primal edges, which is given by the cotan formula. The returned
    tensor forms the diagonal of the 1-star tensor.
    """
    vert_coords: Float[t.Tensor, "vert 3"] = tri_mesh.vert_coords
    tris: Integer[t.LongTensor, "tri 3"] = tri_mesh.tris

    n_verts = tri_mesh.n_verts

    # The cotan weights matrix (i.e., off-diagonal elements of the stiffness matrix)
    # already contains the desired values; i.e., S_ij = -0.5*sum_k[cot_k] for all
    # k that forms a triangle with i and j.
    weights: Float[t.Tensor, "vert vert"] = _cotan_weights(vert_coords, tris, n_verts)

    # For the weights matrix W_ij in COO format, the first row of its index tensor
    # is for the i dimension, and the second row is for the j dimension. We flatten
    # the two dimensions into a 1D index with (i, j) -> i*n_verts + j.
    all_idx = weights.indices()
    all_idx_flat = all_idx[0] * n_verts + all_idx[1]

    # Similarly, we convert the canonical edges represented by a vertex-indexing tuple
    # (i, j) into a flat index with the same formulae.
    edges: Integer[t.Tensor, "edge 2"] = tri_mesh.edges
    edge_idx_flat = edges[:, 0] * n_verts + edges[:, 1]

    # Identify the location of the canonical edge ij in the sparse W_ij indices,
    # using the flattened indices, and use the location to extract the cotan values.
    subset_idx = t.searchsorted(all_idx_flat, edge_idx_flat)
    subset_vals = weights.values()[subset_idx]

    return -subset_vals  # note the negative sign to get dual edge lengths


def d_star_1_d_vert_coords(
    tri_mesh: SimplicialComplex,
) -> Float[t.Tensor, "edge vert 3"]:
    """
    Compute the Jacobian of the Hodge 1-star matrix (diagonal elements) with respect
    to vertex coordinates.
    """
    vert_coords: Float[t.Tensor, "vert 3"] = tri_mesh.vert_coords
    tris: Integer[t.LongTensor, "tri 3"] = tri_mesh.tris

    n_verts = tri_mesh.n_verts
    n_edges = tri_mesh.n_edges

    # Similar to how the 1-star can be computed by extracting the correct elements
    # from the cotan weights matrix, its jacobian can be computed by extracting
    # the correct elements from the jacobian of the cotan weights matrix. More
    # specifically, We will extract gradient vectors -dWdV_ijk for all canonical
    # edges e = ij, and store them in a new tensor dSdV_ek.
    dWdV: Float[t.Tensor, "vert vert vert 3"] = _d_cotan_weights_d_vert_coords(
        vert_coords, tris, n_verts
    )

    # Compute a flat index for all edges ij represented in dWdV_ijk.
    all_edge_idx = dWdV.indices()
    all_edge_idx_flat = all_edge_idx[0] * n_verts + all_edge_idx[1]

    # Similarly, compute a flat index for all canonical edges.
    canon_edges: Integer[t.Tensor, "edge 2"] = tri_mesh.edges
    canon_edge_idx_flat = canon_edges[:, 0] * n_verts + canon_edges[:, 1]

    # Find the "insertion location" of each edge into the list of canonical edges,
    # in a way that preserves the flat index ordering.
    all_edge_insert_loc = t.searchsorted(canon_edge_idx_flat, all_edge_idx_flat)
    # An edge is canonical iff its flat index matches the flat index of the canonical
    # edge at its insertion location. To perform this check, we need to prevent
    # out-of-bound errors by capping insertion locations to the number of canonical
    # edges.
    edge_insert_loc_clipped = t.clip(all_edge_insert_loc, 0, n_edges - 1)
    canon_edge_mask = canon_edge_idx_flat[edge_insert_loc_clipped] == all_edge_idx_flat

    # Final assembly.
    dSdV_e_idx = all_edge_insert_loc[canon_edge_mask]
    dSdV_k_idx = dWdV.indices()[2, canon_edge_mask]
    dSdV_val = -dWdV.values()[canon_edge_mask]

    dSdV = t.sparse_coo_tensor(
        t.vstack((dSdV_e_idx, dSdV_k_idx)),
        dSdV_val,
        (n_edges, n_verts, 3),
    ).coalesce()

    return dSdV


def d_inv_star_1_d_vert_coords(
    tri_mesh: SimplicialComplex,
) -> Float[t.Tensor, "edge vert 3"]:
    """
    Compute the Jacobian of the inverse Hodge 1-star matrix (diagonal elements)
    with respect to vertex coordinates.
    """
    dSdV = d_star_1_d_vert_coords(tri_mesh)

    s1 = star_1(tri_mesh)[dSdV.indices()[0]]
    inv_scale = -1.0 / (s1.square()[:, None] + EPS)

    d_inv_S_dV = t.sparse_coo_tensor(
        dSdV.indices(), dSdV.values() * inv_scale, dSdV.shape
    ).coalesce()

    return d_inv_S_dV


def star_0(tri_mesh: SimplicialComplex) -> Float[t.Tensor, "vert"]:
    """
    The Hodge 0-star operator maps the 0-simplices (vertices) in a mesh to their
    barycentric dual 2-cells. This function computes the ratio of the area of the
    dual 2-cells to the "size" of the vertices (which is 1 by convention). The
    returned tensor forms the diagonal of the 0-star tensor.

    The barycentric dual area for each vertex is the sum of 1/3 of the areas of
    all triangles that share the vertex as a face.
    """
    tri_area = _tri_areas(tri_mesh.vert_coords, tri_mesh.tris)

    diag = t.zeros(
        tri_mesh.n_verts,
        dtype=tri_mesh.vert_coords.dtype,
        device=tri_mesh.vert_coords.device,
    )
    diag.scatter_add_(
        dim=0,
        index=tri_mesh.tris.flatten(),
        src=t.repeat_interleave(tri_area / 3.0, 3),
    )

    return diag


def d_star_0_d_vert_coords(
    tri_mesh: SimplicialComplex,
) -> Float[t.Tensor, "vert vert 3"]:
    """
    Compute the Jacobian of the Hodge 0-star matrix (diagonal elements) with respect
    to vertex coordinates.
    """
    vert_coords: Float[t.Tensor, "vert 3"] = tri_mesh.vert_coords
    tris: Integer[t.LongTensor, "tri 3"] = tri_mesh.tris
    n_verts = tri_mesh.n_verts

    dAdV: Float[t.Tensor, "tri 3 3"] = _d_tri_areas_d_vert_coords(vert_coords, tris)

    # For each triangle ijk and each vertex s, dAdV_ijk_s contributes to the gradient
    # star0_ll wrt s whenever l = s or js is an edge in the mesh. Therefore, each
    # triangle ijk contributes 9 gradient terms, in COO format:
    # [
    #   (i, i, dAdV_ijk_i/3),
    #   (i, j, dAdV_ijk_j/3),
    #   (i, k, dAdV_ijk_k/3),
    #
    #   (j, i, dAdV_ijk_i/3),
    #   (j, j, dAdV_ijk_j/3),
    #   (j, k, dAdV_ijk_k/3),
    #
    #   (k, i, dAdV_ijk_i/3),
    #   (k, j, dAdV_ijk_j/3),
    #   (k, k, dAdV_ijk_k/3),
    # ]

    # Translate the ijk notation to actual indices to access tensor elements.
    i, j, k = 0, 1, 2

    dSdV_idx = (
        tris[
            :,
            [
                [i, i, i, j, j, j, k, k, k],  # first column/index
                [i, j, k, i, j, k, i, j, k],  # second column/index
            ],
        ]
        .transpose(0, 1)
        .flatten(start_dim=1)
    )

    dSdV_val = t.repeat_interleave(dAdV, repeats=3, dim=0).flatten(end_dim=1) / 3.0
    dSdV = t.sparse_coo_tensor(dSdV_idx, dSdV_val, (n_verts, n_verts, 3)).coalesce()

    return dSdV


def d_inv_star_0_d_vert_coords(
    tri_mesh: SimplicialComplex,
) -> Float[t.Tensor, "vert vert 3"]:
    """
    Compute the Jacobian of the inverse Hodge 0-star matrix (diagonal elements)
    with respect to vertex coordinates.
    """
    dSdV = d_star_0_d_vert_coords(tri_mesh)

    s0 = star_0(tri_mesh)[dSdV.indices()[0]]
    inv_scale = -1.0 / (s0.square()[:, None] + EPS)

    d_inv_S_dV = t.sparse_coo_tensor(
        dSdV.indices(), dSdV.values() * inv_scale, dSdV.shape
    ).coalesce()

    return d_inv_S_dV

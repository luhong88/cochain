from typing import Literal

import torch as t
from einops import einsum, rearrange, repeat
from jaxtyping import Float, Integer

from ...complex import SimplicialMesh
from ...geometry.tet.tet_geometry import (
    d_tet_signed_vols_d_vert_coords,
    get_tet_signed_vols,
)
from ...geometry.tri.tri_geometry import (
    compute_d_tri_areas_d_vert_coords,
    compute_tri_areas,
)
from ...sparse.decoupled_tensor import DiagDecoupledTensor, SparseDecoupledTensor
from ...utils.faces import enumerate_local_faces


def _vertex_based_tri_mixed_mass_matrix(
    n_verts: int,
    n_edges: int,
    tris: Integer[t.LongTensor, "tri local_vert=3"],
    tri_edge_idx: Integer[t.LongTensor, "tri local_edge=3"],
    tri_edge_orientations: Float[t.Tensor, "tri local_edge=3"],
    tri_areas: Float[t.Tensor, " tri"],
    bary_coords_grad: Float[t.Tensor, "tri local_vert=3 coord=3"],
) -> Float[SparseDecoupledTensor, "global_vert*coord global_edge"]:
    """
    Compute the cross/mixed mass matrix. For each triangle,

    P_(ix)_(jk) = int[(λ_i*e_x)(λ_j*∇λ_k - λ_k*∇λ_j)dA]
                = <e_x, M_ij * ∇λ_k - M_ik * ∇λ_j>

    where x index over the Cartesian basis vectors anchored at the vertices and
    (i, j, k) index over the vertices in a triangle, and M is the local, per-tri
    consistent mass-0 matrix.
    """
    ref_local_mass_0 = ((t.ones(3, 3) + t.eye(3)) / 12.0).to(
        dtype=tri_areas.dtype, device=tri_areas.device
    )
    local_mass_0: Float[t.Tensor, "tri vert=3 vert=3"] = tri_areas.view(
        -1, 1, 1
    ) * ref_local_mass_0.view(1, 3, 3)

    local_edge_idx: Integer[t.LongTensor, "edge=3 vert=2"] = enumerate_local_faces(
        splx_dim=2, face_dim=1, device=bary_coords_grad.device
    )

    local_int = einsum(
        local_mass_0,
        bary_coords_grad,
        "tri v_i v_j, tri v_k coord -> tri v_i v_j v_k coord",
    ) - einsum(
        local_mass_0,
        bary_coords_grad,
        "tri v_i v_k, tri v_j coord -> tri v_i v_j v_k coord",
    )

    # local_int contains all possible pairing of vertices (v_i) and edges (v_j, v_k),
    # but we only need the 3 unique edge faces and the orientation sign correction
    # in preparation for scatter-add to global canonical edges.
    local_int_canon_edges = einsum(
        tri_edge_orientations,
        local_int[:, :, local_edge_idx[:, 0], local_edge_idx[:, 1], :],
        "tri e_jk, tri v_i e_jk coord -> tri v_i coord e_jk",
    )

    n_coords = 3
    n_verts_per_tri = 3
    n_edges_per_tri = 3

    row_idx_shaped = repeat(n_coords * tris, "tri v_i -> tri v_i coord", coord=n_coords)
    offset = t.tensor(
        [[[0, 1, 2]]], dtype=row_idx_shaped.dtype, device=row_idx_shaped.device
    )
    row_idx = repeat(
        row_idx_shaped + offset,
        "tri v_i coord -> (tri v_i coord e_jk)",
        e_jk=n_edges_per_tri,
    )

    col_idx = repeat(
        tri_edge_idx,
        "tri e_jk -> (tri v_i coord e_jk)",
        v_i=n_verts_per_tri,
        coord=n_coords,
    )

    coo_idx = t.stack((row_idx, col_idx))

    cross_mass = t.sparse_coo_tensor(
        indices=coo_idx,
        values=local_int_canon_edges.flatten(),
        size=(n_verts * n_coords, n_edges),
        dtype=bary_coords_grad.dtype,
        device=bary_coords_grad.device,
    ).coalesce()

    return SparseDecoupledTensor.from_tensor(cross_mass)


def _vertex_based_vector_mass_matrix(
    mass_0: Float[SparseDecoupledTensor, "vert vert"],
) -> Float[SparseDecoupledTensor, "vert*coord vert*coord"]:
    """
    Compute the vertex-based vector mass matrix M_V_(ix)(jy), where i, j index
    over the vertices and x, y index over the Cartesian basis vectors anchored at
    the vertices.

    M_V_(ix)(jy) = int[(λ_i*e_x)(λ_j*e_y) dV] = δ_xy*M_ij

    where M_ij is the consistent mass-0 matrix. Note that this works identically
    on both tri and tet meshes.

    Example:
    -----
    For the ref triangle consisting of only three vertices, M_0 is given by

      a b b
      b a b
      b b a

    The M_V is then given by

      a 0 0 | b 0 0 | b 0 0
      0 a 0 | 0 b 0 | 0 b 0
      0 0 a | 0 0 b | 0 0 b
      ---------------------
      b 0 0 | a 0 0 | b 0 0
      0 b 0 | 0 a 0 | 0 b 0
      0 0 b | 0 0 a | 0 0 b
      ---------------------
      b 0 0 | b 0 0 | a 0 0
      0 b 0 | 0 b 0 | 0 a 0
      0 0 b | 0 0 b | 0 0 a


    """
    n_coords = 3

    # Each (i, j) index of M_0 translates into three indices for M_V:
    # (3i, 3j), (3i + 1, 3j + 1), and (3i + 2, 3j + 2)
    offset = t.arange(n_coords, dtype=mass_0.pattern.dtype, device=mass_0.device)

    m_v_idx = repeat(
        n_coords * mass_0.pattern.idx_coo,
        "dim vert -> dim (vert coord)",
        coord=n_coords,
    ) + repeat(offset, "coord -> dim (vert coord)", dim=2, vert=mass_0._nnz())

    m_v_val = repeat(mass_0.val, "nnz -> (nnz coord)", coord=n_coords)

    m_v = t.sparse_coo_tensor(
        indices=m_v_idx,
        values=m_v_val,
        size=[n_coords * s for s in mass_0.shape],
        dtype=mass_0.dtype,
        device=mass_0.device,
    ).coalesce()

    return SparseDecoupledTensor.from_tensor(m_v)

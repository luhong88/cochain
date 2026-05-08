from typing import Any

import torch
from einops import einsum, rearrange, repeat
from jaxtyping import Float, Integer
from torch import Tensor

from ...sparse.decoupled_tensor import (
    BaseDecoupledTensor,
    DiagDecoupledTensor,
    SparseDecoupledTensor,
)
from ...sparse.linalg.solvers._inv_sparse_operator import InvSparseOperator
from ...utils.faces import enumerate_local_faces

# TODO: investigate sparse coo tensor mesh caching.


def vertex_based_tri_mixed_mass_matrix(
    n_verts: int,
    n_edges: int,
    tris: Integer[Tensor, "tri local_vert=3"],
    tri_edge_idx: Integer[Tensor, "tri local_edge=3"],
    tri_edge_orientations: Float[Tensor, "tri local_edge=3"],
    tri_areas: Float[Tensor, " tri"],
    bary_coords_grad: Float[Tensor, "tri local_vert=3 coord=3"],
) -> Float[SparseDecoupledTensor, "global_vert*coord global_edge"]:
    """
    Compute the cross/mixed mass matrix. For each triangle,

    P_(ix)_(jk) = int[(λ_i*e_x)(λ_j*∇λ_k - λ_k*∇λ_j)dA]
                = <e_x, M_ij * ∇λ_k - M_ik * ∇λ_j>

    where x index over the Cartesian basis vectors anchored at the vertices and
    (i, j, k) index over the vertices in a triangle, and M is the local, per-tri
    consistent mass-0 matrix.
    """
    # Note that, for this calculation, the local, per-triangle mass matrix is
    # required instead of the global mass matrices computed in metric.tri.
    ref_local_mass_0 = ((torch.ones(3, 3) + torch.eye(3)) / 12.0).to(
        dtype=tri_areas.dtype, device=tri_areas.device
    )
    local_mass_0: Float[Tensor, "tri vert=3 vert=3"] = einsum(
        tri_areas, ref_local_mass_0, "tri, v_i v_j -> tri v_i v_j"
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
    local_edge_idx: Integer[Tensor, "edge=3 vert=2"] = enumerate_local_faces(
        splx_dim=2, face_dim=1, device=bary_coords_grad.device
    )

    local_int_canon_edges = einsum(
        tri_edge_orientations,
        local_int[:, :, local_edge_idx[:, 0], local_edge_idx[:, 1], :],
        "tri e_jk, tri v_i e_jk coord -> tri v_i coord e_jk",
    )

    n_coords = 3
    n_verts_per_tri = 3
    n_edges_per_tri = 3

    # Scatter-add the local_int_canon_edges to convert the (local vert, local edge)
    # relation to the (global vert, global edge) representation.

    # The row index iterates over the flattened (global vert, coordinate basis) dim.
    # The col index iterates over the global edge dim.
    row_idx_shaped = repeat(n_coords * tris, "tri v_i -> tri v_i coord", coord=n_coords)
    offset = torch.tensor(
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

    cross_mass = torch.sparse_coo_tensor(
        indices=torch.stack((row_idx, col_idx)),
        values=local_int_canon_edges.flatten(),
        size=(n_verts * n_coords, n_edges),
        dtype=bary_coords_grad.dtype,
        device=bary_coords_grad.device,
    ).coalesce()

    return SparseDecoupledTensor.from_tensor(cross_mass)


def vertex_based_tet_mixed_mass_matrix(
    n_verts: int,
    n_edges: int,
    tets: Integer[Tensor, "tet local_vert=4"],
    tet_edge_idx: Integer[Tensor, "tet local_edge=6"],
    tet_edge_orientations: Float[Tensor, "tet local_edge=6"],
    tet_unsigned_vols: Float[Tensor, " tet"],
    bary_coords_grad: Float[Tensor, "tet vert=4 coord=3"],
) -> Float[SparseDecoupledTensor, "global_vert*coord global_edge"]:
    """
    Compute the cross/mixed mass matrix for a tet mesh.
    """
    # Note that, for this calculation, the local, per-tet mass matrix is
    # required instead of the global mass matrices computed in metric.tet.
    ref_local_mass_0 = ((torch.ones(4, 4) + torch.eye(4)) / 20.0).to(
        dtype=tet_unsigned_vols.dtype, device=tet_unsigned_vols.device
    )
    local_mass_0: Float[Tensor, "tet vert=4 vert=4"] = einsum(
        tet_unsigned_vols, ref_local_mass_0, "tet, v_i v_j -> tet v_i v_j"
    )

    local_int = einsum(
        local_mass_0,
        bary_coords_grad,
        "tet v_i v_j, tet v_k coord -> tet v_i v_j v_k coord",
    ) - einsum(
        local_mass_0,
        bary_coords_grad,
        "tet v_i v_k, tet v_j coord -> tet v_i v_j v_k coord",
    )

    # local_int contains all possible pairing of vertices (v_i) and edges (v_j, v_k),
    # but we only need the 6 unique edge faces and the orientation sign correction
    # in preparation for scatter-add to global canonical edges.
    local_edge_idx: Integer[Tensor, "edge=6 vert=2"] = enumerate_local_faces(
        splx_dim=3, face_dim=1, device=bary_coords_grad.device
    )

    local_int_canon_edges = einsum(
        tet_edge_orientations,
        local_int[:, :, local_edge_idx[:, 0], local_edge_idx[:, 1], :],
        "tet e_jk, tet v_i e_jk coord -> tet v_i coord e_jk",
    )

    n_coords = 3
    n_verts_per_tet = 4
    n_edges_per_tet = 6

    # Scatter-add the local_int_canon_edges to convert the (local vert, local edge)
    # relation to the (global vert, global edge) representation.

    # The row index iterates over the flattened (global vert, coordinate basis) dim.
    # The col index iterates over the global edge dim.
    row_idx_shaped = repeat(n_coords * tets, "tet v_i -> tet v_i coord", coord=n_coords)
    offset = torch.tensor(
        [[[0, 1, 2]]], dtype=row_idx_shaped.dtype, device=row_idx_shaped.device
    )
    row_idx = repeat(
        row_idx_shaped + offset,
        "tet v_i coord -> (tet v_i coord e_jk)",
        e_jk=n_edges_per_tet,
    )

    col_idx = repeat(
        tet_edge_idx,
        "tet e_jk -> (tet v_i coord e_jk)",
        v_i=n_verts_per_tet,
        coord=n_coords,
    )

    cross_mass = torch.sparse_coo_tensor(
        indices=torch.stack((row_idx, col_idx)),
        values=local_int_canon_edges.flatten(),
        size=(n_verts * n_coords, n_edges),
        dtype=bary_coords_grad.dtype,
        device=bary_coords_grad.device,
    ).coalesce()

    return SparseDecoupledTensor.from_tensor(cross_mass)


def vertex_based_consistent_vector_mass_matrix(
    mass_0: Float[SparseDecoupledTensor, "vert vert"],
) -> Float[SparseDecoupledTensor, "vert*coord vert*coord"]:
    """
    Compute the vertex-based vector mass matrix M_V_(ix)(jy), where i, j index
    over the vertices and x, y index over the Cartesian basis vectors anchored at
    the vertices.

    M_V_(ix)(jy) = int[(λ_i*e_x)(λ_j*e_y) dV] = δ_xy*M_ij

    where M_ij is the (global) consistent mass-0 matrix. Note that this works
    identically on both tri and tet meshes.

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
    offset = torch.arange(n_coords, dtype=mass_0.pattern.dtype, device=mass_0.device)

    m_v_idx = repeat(
        n_coords * mass_0.pattern.idx_coo,
        "dim vert -> dim (vert coord)",
        coord=n_coords,
    ) + repeat(offset, "coord -> dim (vert coord)", dim=2, vert=mass_0._nnz())

    m_v_val = repeat(mass_0.values, "nnz -> (nnz coord)", coord=n_coords)

    m_v = torch.sparse_coo_tensor(
        indices=m_v_idx,
        values=m_v_val,
        size=[n_coords * s for s in mass_0.shape],
        dtype=mass_0.dtype,
        device=mass_0.device,
    ).coalesce()

    return SparseDecoupledTensor.from_tensor(m_v)


def vertex_based_diag_vector_mass_matrix(
    star_0: Float[DiagDecoupledTensor, "vert vert"],
) -> Float[DiagDecoupledTensor, "vert*coord vert*coord"]:
    """
    An approximation of _vertex_based_vector_mass_matrix() by replacing the
    consistent mass-0 matrix with the diagonal Hodge start-0 matrix, which results
    in a diagonal vector mass matrix.
    """
    val = repeat(star_0.values, "vert -> (vert coord)", coord=3)
    return DiagDecoupledTensor(val)


def vertex_based_galerkin_flat(
    vec_field: Float[Tensor, "vert coord"],
    mass_1: Float[BaseDecoupledTensor, "edge edge"]
    | Float[InvSparseOperator, "edge edge"],
    mass_mixed: Float[SparseDecoupledTensor, "vert*coord edge"],
    solver_kwargs: dict[str, Any],
) -> Float[Tensor, " edge"]:
    """
    Compute the flat of a vector field defined over the vertices of the mesh using
    the Galerkin projection method.

    Formula: M_0@η = P.T@v, where M_0 is the vert mass matrix, η is the 1-cochain,
    P is the mixed mass matrix, and v is the vector field.
    """
    rhs = mass_mixed.T @ vec_field.flatten()

    match mass_1:
        case InvSparseOperator():
            return mass_1(rhs, **solver_kwargs)
        case DiagDecoupledTensor():
            return mass_1.inv @ rhs
        case _:
            return torch.linalg.solve(mass_1.to_dense(), rhs)


def vertex_based_galerkin_sharp(
    cochain_1: Float[Tensor, " edge"],
    mass_vec: Float[BaseDecoupledTensor, "vert*coord vert*coord"]
    | Float[InvSparseOperator, "vert*coord vert*coord"],
    mass_mixed: Float[SparseDecoupledTensor, "vert*coord edge"],
    solver_kwargs: dict[str, Any],
) -> Float[Tensor, "vert coord=3"]:
    """
    Compute the sharp of a 1-cochain using the Galerkin projection method. The
    resulting vector field is associated with the vertices of the mesh.

    Formula: M_V@v = P@η, where M_V is the vector mass matrix, v is the vector field,
    P is the mixed mass matrix, and η is the 1-cochain.

    Note that, unlike the element-based approach, M_V may not be diagonal using
    the vertex-based approach. If M_V is derived from the consistent mass-0 matrix,
    then it is not diagonal and the `method` argument must be either "dense" or
    "solver"; if M_V is derived from the diagonal Hodge star-0, then it is diagonal
    and the `method` argument should be "inv_star".
    """
    rhs = mass_mixed @ cochain_1

    match mass_vec:
        case InvSparseOperator():
            vec_field_flat = mass_vec(rhs, **solver_kwargs)
        case DiagDecoupledTensor():
            vec_field_flat = mass_vec.inv @ rhs
        case _:
            vec_field_flat = torch.linalg.solve(mass_vec.to_dense(), rhs)

    vec_field = rearrange(vec_field_flat, "(splx coord) -> splx coord", coord=3)

    return vec_field

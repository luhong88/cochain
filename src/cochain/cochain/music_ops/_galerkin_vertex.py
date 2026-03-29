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

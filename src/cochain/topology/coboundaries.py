__all__ = ["cbd_from_tri_mesh", "cbd_from_tet_mesh"]

import torch
from einops import rearrange, repeat
from jaxtyping import Float, Integer
from torch import Tensor

from ..sparse.decoupled_tensor import SparseDecoupledTensor
from ..utils.faces import enumerate_local_faces
from ..utils.perm_parity import compute_lex_rel_orient


def cbd_from_tri_mesh(
    tris: Integer[Tensor, "tri tri_vert=3"],
    dtype: torch.dtype = torch.float32,
) -> tuple[
    Integer[Tensor, "edge edge_vert=2"],
    Float[SparseDecoupledTensor, "edge vert"],
    Float[SparseDecoupledTensor, "tri edge"],
]:
    """
    Compute the coboundary operators/discrete exterior derivatives for a tri mesh.

    Parameters
    ----------
    tris : [tri, tri_vert=3]
        A list of top-level, 2-simplices that define the mesh. It is assumed that
        the tri vertices are labeled using strictly consecutive, zero-based
        numbering.
    dtype : torch.dtype
        The dtype for the coboundary operators.

    Returns
    -------
    unique_canon_edges : [edge, edge_vert=2]
        The list of all edges (up to vertex permutation) in the mesh. Both the
        edge vertices and the ordering of the edges are in lex order. This tensor
        has the same dtype as tris.
    cbd_0 : [edge, vert]
        The 0-coboundary operator.
    cbd_1 : [tri, edge]
        The 1-coboundary operator.
    """
    int_dtype = tris.dtype
    float_dtype = dtype
    device = tris.device

    # For the triangles ijk, find all of its 1-faces and stack them together; this
    # provides a redundant list of all oriented edges in the mesh.
    all_local_edges = enumerate_local_faces(splx_dim=2, face_dim=1, device=device)
    all_oriented_edges = rearrange(
        tris[:, all_local_edges], "tri edge vert -> (tri edge) vert"
    )

    # For each oriented edge, find its canonical, lex-order representation.
    all_canon_edges = all_oriented_edges.sort(dim=-1).values
    # Generate a lex-ordered list of unique, canonical edges, and get the indices
    # that map this list of canonical edges to the list of all oriented edges
    # (ignoring potential orientation differences).
    unique_canon_edges, all_edge_idx = all_canon_edges.unique(
        dim=0, return_inverse=True
    )

    n_verts = torch.max(tris).item() + 1
    n_edges = unique_canon_edges.size(0)
    n_tris = tris.size(0)

    # Generate the 0-coboundary operator as a sparse tensor. A canonically
    # oriented edge ij at index n in unique_canonical_edges is represented by
    # (n, i, -1) and (n, j, 1) (using COO format).
    d0_idx = torch.stack(
        (
            repeat(
                torch.arange(n_edges, dtype=int_dtype, device=device),
                "edge -> (edge vert)",
                vert=2,
            ),
            rearrange(unique_canon_edges, "edge vert -> (edge vert)"),
        )
    )
    d0_val = repeat(
        torch.tensor([-1.0, 1.0], dtype=float_dtype, device=device),
        "vert -> (edge vert)",
        edge=n_edges,
    )
    cbd_0 = torch.sparse_coo_tensor(
        indices=d0_idx,
        values=d0_val,
        size=(n_edges, n_verts),
    ).coalesce()

    # Generate the 1-coboundary operator.
    # Each edge in the cbd requires two "sign corrections". The first sign correction
    # comes from the boundary operator; for a triangle ijk, ∂(ijk) = ij - ik + jk.
    edge_topo_signs = torch.tensor([1.0, -1.0, 1.0], dtype=float_dtype, device=device)
    # The second sign correction comes from the parity of the vertex permutation
    # required to map a tri edge face to a canonical edge (i.e., the permutation
    # required to lex sort the tri edge faces).
    edge_orientation_signs = compute_lex_rel_orient(
        all_oriented_edges, dtype=float_dtype
    )

    d1_idx = torch.stack(
        [
            repeat(
                torch.arange(n_tris, dtype=int_dtype, device=device),
                "tri -> (tri edge)",
                edge=3,
            ),
            all_edge_idx,
        ]
    )
    # 1 edge, tri edge -> (tri edge)
    d1_val = (
        edge_topo_signs.view(1, -1) * edge_orientation_signs.view(n_tris, 3)
    ).flatten()

    cbd_1 = torch.sparse_coo_tensor(
        indices=d1_idx,
        values=d1_val,
        size=(n_tris, n_edges),
    ).coalesce()

    return (
        unique_canon_edges,
        SparseDecoupledTensor.from_tensor(cbd_0),
        SparseDecoupledTensor.from_tensor(cbd_1),
    )


def cbd_from_tet_mesh(
    tets: Integer[Tensor, "tet tet_vert=4"],
    dtype: torch.dtype = torch.float32,
) -> tuple[
    Integer[Tensor, "edge edge_vert=2"],
    Integer[Tensor, "tri tri_vert=3"],
    Float[SparseDecoupledTensor, "edge vert"],
    Float[SparseDecoupledTensor, "tri edge"],
    Float[SparseDecoupledTensor, "tet tri"],
]:
    """
    Compute the coboundary operators/discrete exterior derivatives for a tri mesh.

    Parameters
    ----------
    tets : [tet, tet_vert=4]
        A list of top-level, 3-simplices that define the mesh. It is assumed that
        the tet vertices are labeled using strictly consecutive, zero-based
        numbering.
    dtype : torch.dtype
        The dtype for the coboundary operators.

    Returns
    -------
    unique_canon_edges : [edge, edge_vert=2]
        The list of all edges (up to vertex permutation) in the mesh. Both the
        edge vertices and the ordering of the edges are in lex order. This tensor
        has the same dtype as tets.
    unique_canon_tris : [tri, tri_vert=2]
        The list of all tris (up to vertex permutation) in the mesh. Both the
        tri vertices and the ordering of the tris are in lex order. This tensor
        has the same dtype as tets.
    cbd_0 : [edge, vert]
        The 0-coboundary operator.
    cbd_1 : [tri, edge]
        The 1-coboundary operator.
    cbd_2 : [tet, tri]
        The 2-coboundary operator.
    """
    int_dtype = tets.dtype
    float_dtype = dtype
    device = tets.device

    # For the tets ijkl, find all of its 2-faces and stack them together; this
    # provides a redundant list of all oriented tris in the mesh.
    all_local_tris = enumerate_local_faces(splx_dim=3, face_dim=2, device=device)
    all_oriented_tris = rearrange(
        tets[:, all_local_tris], "tet tri vert -> (tet tri) vert"
    )

    # For each oriented tri, find its canonical, lex-order representation.
    all_canon_tris = all_oriented_tris.sort(dim=-1).values
    # Generate a lex-ordered list of unique, canonical tris, and get the indices
    # that map this list of canonical tris to the list of all oriented tris
    # (ignoring potential orientation differences).
    unique_canon_tris, all_tri_idx = all_canon_tris.unique(dim=0, return_inverse=True)

    n_tris = unique_canon_tris.size(0)
    n_tets = tets.size(0)

    # Generate the 2-coboundary operator.
    # Each edge in the cbd requires two "sign corrections". The first sign correction
    # comes from the boundary operator; for a tet ijkl, ∂(ijkl) = - ijk + ijl - ikl + jkl.
    tri_topo_signs = torch.tensor(
        [-1.0, 1.0, -1.0, 1.0], dtype=float_dtype, device=device
    )
    # The second sign correction comes from the parity of the vertex permutation
    # required to map a tet tri face to a canonical tri (i.e., the permutation
    # required to lex sort the tet tri faces).
    tri_orientation_signs = compute_lex_rel_orient(all_oriented_tris, dtype=float_dtype)

    d2_idx = torch.stack(
        [
            repeat(
                torch.arange(n_tets, dtype=int_dtype, device=device),
                "tet -> (tet tri)",
                tri=4,
            ),
            all_tri_idx,
        ]
    )
    # 1 tri, tet tri -> (tet tri)
    d2_val = (
        tri_topo_signs.view(1, -1) * tri_orientation_signs.view(n_tets, 4)
    ).flatten()

    cbd_2 = torch.sparse_coo_tensor(
        indices=d2_idx, values=d2_val, size=(n_tets, n_tris)
    ).coalesce()

    # Generate the 1- and 0-coboundary operators. This can be done via
    # cbd_from_tri_mesh(), using unique_canon_tris as the triangle mesh.
    unique_canon_edges, cbd_0, cbd_1 = cbd_from_tri_mesh(unique_canon_tris)

    return (
        unique_canon_edges,
        unique_canon_tris,
        cbd_0,
        cbd_1,
        SparseDecoupledTensor.from_tensor(cbd_2),
    )

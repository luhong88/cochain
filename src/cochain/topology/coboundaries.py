import torch
from einops import einsum, rearrange, reduce, repeat
from jaxtyping import Float, Integer
from torch import LongTensor

from ..sparse.decoupled_tensor import SparseDecoupledTensor
from ..utils.perm_parity import compute_lex_rel_orient


def cbd_from_tri_mesh(
    tris: Integer[LongTensor, "global_tri tri_vert=3"],
    dtype: torch.dtype = torch.float32,
) -> tuple[
    Integer[LongTensor, "global_edge edge_vert=2"],
    Float[SparseDecoupledTensor, "global_edge global_vert"],
    Float[SparseDecoupledTensor, "global_tri global_edge"],
]:
    """
    Compute the coboundary operators/discrete exterior derivatives for a tri mesh.

    Parameters
    ----------
    tris : [global_tri, tri_vert=3]
        A list of top-level, 2-simplices that define the mesh. It is assumed that
        the tri vertices are labeled using strictly consecutive, zero-based
        numbering.
    dtype : torch.dtype
        The dtype for the coboundary operators.

    Returns
    -------
    unique_canon_edges : [global_edge, edge_vert=2]
        The list of all edges (up to vertex permutation) in the mesh. Both the
        edge vertices and the ordering of the edges are in lex order.
    cbd_0 : [global_edge, global_vert]
        The 0-coboundary operator.
    cbd_1 : [global_tri global_edge]
        The 1-coboundary operator.
    """
    int_dtype = tris.dtype
    float_dtype = dtype
    device = tris.device

    # For the triangles ijk, get all the i-th face jk, all the j-th face ik,
    # and all the k-th face ij and stack them together; this provides a redundant
    # list of all oriented edges in the mesh.
    all_oriented_edges = torch.concatenate(
        (tris[:, [1, 2]], tris[:, [0, 2]], tris[:, [0, 1]])
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
    # comes from the boundary operator; for a triangle ijk, ∂(ijk) = jk - ik + ij.
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
                "tri -> (edge tri)",
                edge=3,
            ),
            all_edge_idx,
        ]
    )
    # edge 1, edge tri -> (edge tri)
    d1_val = (
        edge_topo_signs.view(-1, 1) * edge_orientation_signs.view(3, n_tris)
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
    tets: Integer[LongTensor, "tet 4"],
) -> tuple[
    Integer[LongTensor, "edge 2"],
    Integer[LongTensor, "tri 3"],
    Float[SparseDecoupledTensor, "edge vert"],
    Float[SparseDecoupledTensor, "tri edge"],
    Float[SparseDecoupledTensor, "tet tri"],
]:
    tets = tets.long()

    device = tets.device

    # For the tets ijkl, get all the i-th face jkl, all the j-th face ikl, all
    # the k-th face ijl, and all the l-th face ijk and stack them together; this
    # provides a redundant list of all oriented edges in the mesh.
    all_oriented_tris = torch.concatenate(
        (tets[:, [1, 2, 3]], tets[:, [0, 2, 3]], tets[:, [0, 1, 3]], tets[:, [0, 1, 2]])
    )
    # Convert oriented tris to the canonical orientation, and check whether each
    # oriented tris conforms to the canonical orientation.
    all_canon_tris, all_tri_orientations = all_oriented_tris.sort(dim=-1)
    # Generate an ordered list of canonical tris, and get the indices of all the
    # oriented tris in this ordered list (ignoring their orientations).
    unique_canon_tris, all_tri_idx = all_canon_tris.unique(dim=0, return_inverse=True)

    n_tris = unique_canon_tris.shape[0]
    n_tets = tets.shape[0]

    # Generate the 2nd-coboundary operator.
    # For a tet ijkl, d2(ijkl) = jkl - ikl + ijl - ijk, which is represented by
    # the "topological signs".
    tri_topo_signs = torch.repeat_interleave(
        torch.tensor([1.0, -1.0, 1.0, -1.0], device=device), n_tets
    )
    # Each oriented tris that has the opposite orientation as the corresponding
    # canonical tris gets an additional -1 "orientation signs". In total, there
    # are 6 ways to arrange the indices of a triangle, three of which get the
    # positive orientation (ijk, kij, jki), and three of which get the negative
    # orientation (ikj, jik, kji). This orientation sign can be determined by
    # taking the vertex indices returned by sort() in all_tris_orientations, and
    # check how many indices match the canonical order indices [0, 1, 2]; if one
    # of them do, the triangle has a negative orientation, and otherwise it has
    # a positive orientation.
    canon_pos_orientation = torch.tensor([0, 1, 2], dtype=torch.int64, device=device)
    tri_orientation_signs = torch.where(
        condition=torch.sum(all_tri_orientations == canon_pos_orientation, dim=-1) == 1,
        self=-1,
        other=1,
    )

    d2_idx = torch.stack([torch.tile(torch.arange(n_tets), (4,)), all_tri_idx]).to(
        device=device
    )
    d2_val = tri_topo_signs * tri_orientation_signs

    cbd_2 = torch.sparse_coo_tensor(
        d2_idx, d2_val, (n_tets, n_tris), device=device
    ).coalesce()

    # Generate the 1st- and 0th-coboundary operators. This can be done via
    # coboundaries_from_tri_mesh(), using unique_canon_tris as the triangle mesh.
    unique_canon_edges, cbd_0, cbd_1 = cbd_from_tri_mesh(unique_canon_tris)

    return (
        unique_canon_edges,
        unique_canon_tris,
        cbd_0,
        cbd_1,
        SparseDecoupledTensor.from_tensor(cbd_2),
    )

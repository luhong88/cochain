import torch as t
from jaxtyping import Float, Integer

from ..sparse.operators import SparseOperator


def coboundaries_from_tri_mesh(
    tris: Integer[t.LongTensor, "tri 3"],
) -> tuple[
    Integer[t.LongTensor, "edge 2"],
    Float[SparseOperator, "edge vert"],
    Float[SparseOperator, "tri edge"],
]:
    tris = tris.long()

    device = tris.device

    # For the triangles ijk, get all the i-th face jk, all the j-th face ik,
    # and all the k-th face ij and stack them together; this provides a redundant
    # list of all oriented edges in the mesh.
    all_oriented_edges = t.concatenate(
        (tris[:, [1, 2]], tris[:, [0, 2]], tris[:, [0, 1]])
    )
    # Convert oriented edges to the canonical orientation, and check whether
    # each oriented edge conforms to the canonical orientation.
    all_canon_edges, all_edge_orientations = all_oriented_edges.sort(dim=-1)
    # Generate an ordered list of canonical edges, and get the indices of all
    # the oriented edges in this ordered list (ignoring their orientations).
    unique_canon_edges, all_edge_idx = all_canon_edges.unique(
        dim=0, return_inverse=True
    )

    n_verts = t.max(tris).item() + 1
    n_edges = unique_canon_edges.shape[0]
    n_tris = tris.shape[0]

    # Generate the 0th-coboundary operator as a sparse tensor. A canonically
    # oriented edge ij at index n in unique_canonical_edges is represented by
    # (n, i, -1) and (n, j, 1) (using COO format).
    d0_idx = t.stack(
        [
            t.repeat_interleave(t.arange(n_edges), 2),
            unique_canon_edges.flatten(),
        ]
    ).to(device=device)
    d0_val = t.tile(t.tensor([-1.0, 1.0], device=device), (n_edges,))
    coboundary_0 = (
        t.sparse_coo_tensor(
            d0_idx,
            d0_val,
            (n_edges, n_verts),
        )
        .coalesce()
        .to(device)
    )

    # Generate the 1st-coboundary operator.
    # For a triangle ijk, d1(ijk) = jk - ik + ij, which is represented by the
    # "topological signs".
    edge_topo_signs = t.repeat_interleave(
        t.tensor([1.0, -1.0, 1.0], device=device), n_tris
    )
    # Each oriented edge ji that has the opposite orientation as the corresponding
    # canonical edge ij gets an additional -1 "orientation signs"; the orientation
    # sign can be determined by the vertex indices returned by sort() in
    # all_edge_orientations.
    edge_orientation_signs = t.where(
        all_edge_orientations[:, 1] > 0, all_edge_orientations[:, 1], -1
    )

    d1_idx = t.stack(
        [
            t.tile(t.arange(n_tris), (3,)),
            all_edge_idx,
        ]
    ).to(device)
    d1_val = edge_topo_signs * edge_orientation_signs

    coboundary_1 = (
        t.sparse_coo_tensor(
            d1_idx,
            d1_val,
            (n_tris, n_edges),
        )
        .coalesce()
        .to(device)
    )

    return (
        unique_canon_edges,
        SparseOperator.from_tensor(coboundary_0),
        SparseOperator.from_tensor(coboundary_1),
    )


def coboundaries_from_tet_mesh(
    tets: Integer[t.LongTensor, "tet 4"],
) -> tuple[
    Integer[t.LongTensor, "edge 2"],
    Integer[t.LongTensor, "tri 3"],
    Float[SparseOperator, "edge vert"],
    Float[SparseOperator, "tri edge"],
    Float[SparseOperator, "tet tri"],
]:
    tets = tets.long()

    device = tets.device

    # For the tets ijkl, get all the i-th face jkl, all the j-th face ikl, all
    # the k-th face ijl, and all the l-th face ijk and stack them together; this
    # provides a redundant list of all oriented edges in the mesh.
    all_oriented_tris = t.concatenate(
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
    tri_topo_signs = t.repeat_interleave(
        t.tensor([1.0, -1.0, 1.0, -1.0], device=device), n_tets
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
    canon_pos_orientation = t.tensor([0, 1, 2], dtype=t.long, device=device)
    tri_orientation_signs = t.where(
        condition=t.sum(all_tri_orientations == canon_pos_orientation, dim=-1) == 1,
        self=-1,
        other=1,
    )

    d2_idx = t.stack([t.tile(t.arange(n_tets), (4,)), all_tri_idx]).to(device=device)
    d2_val = tri_topo_signs * tri_orientation_signs

    coboundary_2 = t.sparse_coo_tensor(
        d2_idx, d2_val, (n_tets, n_tris), device=device
    ).coalesce()

    # Generate the 1st- and 0th-coboundary operators. This can be done via
    # coboundaries_from_tri_mesh(), using unique_canon_tris as the triangle mesh.
    unique_canon_edges, coboundary_0, coboundary_1 = coboundaries_from_tri_mesh(
        unique_canon_tris
    )

    return (
        unique_canon_edges,
        unique_canon_tris,
        coboundary_0,
        coboundary_1,
        SparseOperator.from_tensor(coboundary_2),
    )

import torch as t
from jaxtyping import Float, Integer


def coboundaries_from_tri_mesh(
    vert_coords: Float[t.Tensor, "vert 3"],
    tris: Integer[t.LongTensor, "tri 3"],
) -> tuple[
    Integer[t.LongTensor, "edge 2"],
    Float[t.Tensor, "edge vert"],
    Float[t.Tensor, "tri edge"],
]:
    device = vert_coords.device

    # For the triangles ijk, get all the i-th face jk, all the j-th face ik,
    # and all the k-th face ij and stack them together; this provide redundant
    # list of all oriented edges in the mesh.
    all_oriented_edges = t.concatenate(
        [tris[:, [1, 2]], tris[:, [0, 2]], tris[:, [0, 1]]]
    )
    # Convert oriented edges to the canonical orientation, and check whether
    # each oriented edge conforms to the canonical orientation.
    all_canon_edges, all_edge_orientations = all_oriented_edges.sort(dim=-1)
    # Generate an ordered list of canonical edges, and get the indices of all
    # the oriented edges in this ordered list (ignoring their orientations).
    unique_canon_edges, all_edge_idx = all_canon_edges.unique(
        dim=0, return_inverse=True
    )

    n_verts = vert_coords.shape[0]
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
    )
    d0_val = t.tile(t.Tensor([-1.0, 1.0]), (n_edges,))
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
    # for a triangle ijk, d1(ijk) = jk - ik + ij, which is represented by the
    # "topological signs".
    edge_topo_signs = t.repeat_interleave(t.Tensor([1.0, -1.0, 1.0]), n_tris)
    # each oriented edge ji that has the opposite orientation as the corresponding
    # canonical edge ij gets an additional -1 "orientation signs"; the orientation
    # sign can be determined by the vertex indices returned sort() in
    # all_edge_orientations.
    edge_orientation_signs = t.where(
        all_edge_orientations[:, 1] > 0, all_edge_orientations[:, 1], -1
    )

    d1_idx = t.stack(
        [
            t.tile(t.arange(n_tris), (3,)),
            all_edge_idx,
        ]
    )
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

    return unique_canon_edges, coboundary_0, coboundary_1

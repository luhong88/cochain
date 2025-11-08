import torch as t
from jaxtyping import Float, Integer, Real


class SimplicialComplex2D:
    """
    A simplicial 2-complex.
    """

    def __init__(
        self,
        coboundary_0: Float[t.Tensor, "edge vert"],
        coboundary_1: Float[t.Tensor, "tri edge"],
        cochain_0: Float[t.Tensor, "vert *vert_feat"] | None = None,
        cochain_1: Float[t.Tensor, "edge *edge_feat"] | None = None,
        cochain_2: Float[t.Tensor, "tri *tri_feat"] | None = None,
        vert_coords: Float[t.Tensor, "vert 3"] | None = None,
    ):
        assert coboundary_0.shape[0] == coboundary_1.shape[1], (
            "Inconsistent edge dimension shape for the 0th and 1st coboundary operators."
        )

        self.coboundary_0 = coboundary_0.to_sparse_csr()
        self.coboundary_1 = coboundary_1.to_sparse_csr()

        self.n_verts = coboundary_0.shape[1]
        self.n_edges = coboundary_0.shape[0]
        self.n_tris = coboundary_1.shape[0]

        self.cochain_0 = cochain_0
        self.cochain_1 = cochain_1
        self.cochain_2 = cochain_2
        self.vert_coords = vert_coords

    def to(self, device: str | t.DeviceObjType):
        raise NotImplementedError()

    # TODO: check for immersion
    @classmethod
    def from_mesh(
        cls,
        vert_coords: Float[t.Tensor, "vert 3"],
        tris: Integer[t.LongTensor, "tri 3"],
        **kwargs,
    ):
        """
        Construct a special geometric simplicial 2-complex as a triangulated 2D
        mesh immersed in 3D Euclidean space; note that this function does not assume
        that the complex is a 2-manifold.

        Since no orientation is assigned to the edges using this constructor, we
        will assign a "canonical" orientation to each edge ij such that i < j.
        """
        device = vert_coords.device

        # For the triangles ijk, get all the i-th face jk, all the j-th face ik,
        # and all the k-th face ij and stack them together; this provide redundant
        # list of all oriented edges in the mesh.
        all_oriented_edges = t.concatenate(
            [tris[:, [1, 2]], tris[:, [0, 2]], tris[:, [0, 1]]]
        )
        # Convert oriented edges to the canonical orientation, and check whether
        # each oriented edge conforms to the canonical orientation.
        all_canonical_edges, all_edge_orientations = all_oriented_edges.sort(dim=-1)
        # Generate an ordered list of canonical edges, and get the indices of all
        # the oriented edges in this ordered list (ignoring their orientations).
        unique_canonical_edges, all_edge_idx = all_canonical_edges.unique(
            dim=0, return_inverse=True
        )

        n_verts = vert_coords.shape[0]
        n_edges = unique_canonical_edges.shape[0]
        n_tris = tris.shape[0]

        # Generate the 0th-coboundary operator as a sparse tensor. A canonically
        # oriented edge ij at index n in unique_canonical_edges is represented by
        # (n, i, -1) and (n, j, 1) (using COO format).
        d0_idx = t.stack(
            [
                t.repeat_interleave(t.arange(n_edges), 2),
                unique_canonical_edges.flatten(),
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
        # for a triangle ijk, d1(ijk) = jk - ik + jk, which is represented by the
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

        return cls(
            coboundary_0=coboundary_0,
            coboundary_1=coboundary_1,
            vert_coords=vert_coords,
            **kwargs,
        )

    @classmethod
    def from_graph(cls, edges: t.Tensor, n_verts: int, **kwargs):
        """
        Creates a SimplicialComplex2D object from a standard graph (i.e., a pure
        1-complex with no tris)
        """
        raise NotImplementedError()

    @classmethod
    def from_simplex_lists(cls, n_verts: int, tris=None, edges=None, **kwargs):
        raise NotImplementedError()


class SimplicialBatch:
    """
    Roadmap:
    * Dataset: a list of SimplicialComplex
    * DataLoader returns a SimplicialBatch, which is just a giant SimplicialCOmplex.
    * The collate_fn to perform block-diagonal magic:
        batch.x_0 = torch.cat([sc1.x_0, sc2.x_0, sc3.x_0], dim=0)
        batch.d0 = torch.block_diag(sc1.d0, sc2.d0, sc3.d0)
        batch.d1 = torch.block_diag(sc1.d1, sc2.d1, sc3.d1)
        etc.
    * The "pointer":
        batch.batch_0: A tensor mapping each node to its original complex (e.g., [0, 0, 0, ..., 1, 1, 1, ..., 2, 2, 2]).
        batch.batch_1: A tensor mapping each edge to its original complex.
        batch.batch_2: A tensor mapping each face to its original complex.
        etc.
    """

    def __init__(
        self,
    ):
        raise NotImplementedError

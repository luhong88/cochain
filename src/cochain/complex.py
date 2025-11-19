import torch as t
from jaxtyping import Float, Integer, Real

from .topology import coboundaries


class SimplicialComplex:
    """
    A simplicial complex.
    """

    def __init__(
        self,
        coboundaries: tuple[
            Float[t.Tensor, "edge vert"],
            Float[t.Tensor, "tri edge"],
            Float[t.Tensor, "tet tri"],
        ],
        simplices: tuple[
            Float[t.Tensor, "edge 2"],
            Float[t.Tensor, "tri 3"],
            Float[t.Tensor, "tet 4"],
        ],
        cochains: tuple[
            Float[t.Tensor, "vert *vert_feat"] | None,
            Float[t.Tensor, "edge *edge_feat"] | None,
            Float[t.Tensor, "tri *tri_feat"] | None,
            Float[t.Tensor, "tet *tet_feat"] | None,
        ],
        vert_coords: Float[t.Tensor, "vert 3"] | None,
    ):
        # Assign input data to class attributes.
        self.coboundary_0, self.coboundary_1, self.coboundary_2 = coboundaries

        self.edges, self.tris, self.tets = simplices

        self.cochain_0, self.cochain_1, self.cochain_2, self.cochain_3 = cochains

        self.vert_coords = vert_coords

        # Check for dim consistency in coboundary operators.
        assert self.coboundary_0.shape[0] == self.coboundary_1.shape[1]
        assert self.coboundary_1.shape[0] == self.coboundary_2.shape[1]

        # Check for dim consistency between the coboundary operators and the simplex
        # definitions.
        assert self.edges.shape[0] == self.coboundary_0.shape[0]
        assert self.tris.shape[0] == self.coboundary_1.shape[0]
        assert self.tets.shape[0] == self.coboundary_2.shape[0]

    def to(self, device: str | t.device):
        for attr, value in self.__dict__.items():
            if t.is_tensor(value):
                setattr(self, attr, value.to(device))
        return self

    @property
    def n_verts(self) -> int:
        return self.coboundary_0.shape[1]

    @property
    def n_edges(self) -> int:
        return self.coboundary_0.shape[0]

    @property
    def n_tris(self) -> int:
        return self.coboundary_1.shape[0]

    @property
    def n_tets(self) -> int:
        return self.coboundary_2.shape[0]

    @property
    def dim(self) -> int:
        return max(
            1 * (self.n_edges != 0), 2 * (self.n_tris != 0), 3 * (self.n_tets != 0)
        )

    # TODO: check for immersion
    @classmethod
    def from_tri_mesh(
        cls,
        vert_coords: Float[t.Tensor, "vert 3"],
        tris: Integer[t.LongTensor, "tri 3"],
        cochains: tuple[
            Float[t.Tensor, "vert *vert_feat"] | None,
            Float[t.Tensor, "edge *edge_feat"] | None,
            Float[t.Tensor, "tri *tri_feat"] | None,
        ]
        | None = None,
    ):
        """
        Construct a special geometric simplicial 2-complex as a triangulated 2D
        mesh immersed in 3D Euclidean space; note that this function does not assume
        that the complex is a 2-manifold.

        Since no orientation is assigned to the edges using this constructor, we
        will assign a "canonical" orientation to each edge ij such that i < j.
        """
        unique_canon_edges, coboundary_0, coboundary_1 = (
            coboundaries.coboundaries_from_tri_mesh(vert_coords, tris)
        )

        coboundary_2 = t.sparse_coo_tensor(
            indices=t.empty((2, 0), dtype=t.long),
            values=t.empty((0,), dtype=t.float32),
            size=(0, tris.shape[0]),
        )

        tets = t.empty((0, 4), dtype=t.long)

        if cochains is None:
            cochains = (None, None, None, None)
        else:
            cochains = cochains + (None,)

        return cls(
            coboundaries=(coboundary_0, coboundary_1, coboundary_2),
            simplices=(unique_canon_edges, tris, tets),
            vert_coords=vert_coords,
            cochains=cochains,
        )

    @classmethod
    def from_graph(cls, edges: t.Tensor, n_verts: int, **kwargs):
        """
        Creates a Simplicial2Complex object from a standard graph (i.e., a pure
        1-complex with no tris)
        """
        raise NotImplementedError()

    @classmethod
    def from_simplex_lists(cls, n_verts: int, tris=None, edges=None, **kwargs):
        raise NotImplementedError()


class SimplicialBatch(SimplicialComplex):
    """
    A "batch" of complexes, represented as a single large,
    disconnected complex. This is what the DataLoader returns.
    """

    def __init__(self, batch_0, batch_1, batch_2, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # "Pointer" tensors that map items back to their original complex
        # A tensor mapping each node to its original complex (e.g., [0, 0, 0, ..., 1, 1, 1, ..., 2, 2, 2]).
        self.batch_0 = batch_0  # [N_total]
        self.batch_1 = batch_1  # [E_total]
        self.batch_2 = batch_2  # [F_total]

    @property
    def num_complexes(self):
        return self.batch_0.max().item() + 1


def collate_fn(complicies: list[SimplicialComplex]) -> SimplicialBatch:
    """
    The "magic" function that creates a SimplicialBatch.
    Uses torch.block_diag() on operators and torch.cat() on features.
    """
    # 1. Cat all x_k
    # 2. Block_diag all d_k
    # 3. Create batch_k pointer tensors
    # 4. Return a new SimplicialBatch
    pass


class DataLoader(t.utils.data.DataLoader):
    """
    A user-facing DataLoader that automatically uses the collate_fn.
    """

    def __init__(self, dataset, batch_size=1, **kwargs):
        super().__init__(
            dataset, batch_size=batch_size, collate_fn=collate_fn, **kwargs
        )

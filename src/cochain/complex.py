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
            Integer[t.LongTensor, "edge 2"],
            Integer[t.LongTensor, "tri 3"],
            Integer[t.LongTensor, "tet 4"],
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
    A "batch" of complexes, represented as a single large, disconnected complex.
    This is what the DataLoader returns.
    """

    def __init__(
        self,
        batch_verts: Integer[t.LongTensor, "vert"],
        batch_edges: Integer[t.LongTensor, "edge"],
        batch_tris: Integer[t.LongTensor, "tri"],
        batch_tets: Integer[t.LongTensor, "tet"],
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        # the "batch" tensor for each dimension maps a simplex back to the original
        # simplicial complex; i.e., batch_edges[i] = k indicates that the i-th edge
        # in the batch comes from the k-th simplicial complex.
        self.batch_verts = batch_verts
        self.batch_edges = batch_edges
        self.batch_tris = batch_tris
        self.batch_tets = batch_tets

    @property
    def n_sc(self) -> int:
        return self.batch_verts.max().item() + 1


def collate_fn(sc_batch: list[SimplicialComplex]) -> SimplicialBatch:
    """
    The "magic" function that creates a SimplicialBatch.
    """
    # Assume that all simplices and their tensors are on the same device and
    # all float tensors have the same dtype.
    device = sc_batch[0].coboundary_0.device
    dtype = sc_batch[0].coboundary_0.dtype

    # Generate a cumsum n_sc list for each simplex dimension
    n_simp_batch = t.Tensor(
        [
            [0] + [sc.n_verts for sc in sc_batch],
            [0] + [sc.n_edges for sc in sc_batch],
            [0] + [sc.n_tris for sc in sc_batch],
            [0] + [sc.n_tets for sc in sc_batch],
        ]
    ).to(dtype=t.long, device=device)

    n_simp_cumsum_batch = t.cumsum(n_simp_batch, dim=-1, dtype=t.long)

    # Generate the batch tensor for each simplex dimension
    batch_tensor_dict = {
        f"batch_{simp_type}": t.repeat_interleave(
            t.arange(len(sc_batch), dtype=t.long, device=device),
            repeats=n_simp_batch[simp_dim, 1:],
        )
        for simp_dim, simp_type in enumerate(["verts", "edges", "tris", "tets"])
    }

    # Collate the coboundary operators into sparse block-diagonal forms.
    # Increment the simplex indices for each sc in the batch.
    coboundaries_batch = [
        t.sparse_coo_tensor(
            indices=t.hstack(
                [
                    getattr(sc, f"coboundary_{dim}").indices()
                    + n_simp_cumsum_batch[[dim + 1, dim], idx][:, None]
                    for idx, sc in enumerate(sc_batch)
                ]
            ),
            values=t.hstack(
                [getattr(sc, f"coboundary_{dim}").values() for sc in sc_batch]
            ),
            size=(n_simp_cumsum_batch[dim + 1, -1], n_simp_cumsum_batch[dim, -1]),
            dtype=dtype,
            device=device,
        ).coalesce()
        for dim in range(3)
    ]

    # Collate the simplices; increment the vertex indices for each sc in the batch.
    simplices_batch = [
        t.vstack(
            [
                getattr(sc, simp_type) + n_simp_cumsum_batch[0, idx]
                for idx, sc in enumerate(sc_batch)
            ]
        )
        for simp_type in ["edges", "tris", "tets"]
    ]

    # Collate the cochains
    cochains_batch = []
    for dim in range(4):
        cochains = [getattr(sc, f"cochain_{dim}") for sc in sc_batch]

        if None not in cochains:
            cochains_batch.append(t.vstack(cochains))

        elif all(cochain is None for cochain in cochains):
            cochains_batch.append(None)

        else:
            raise ValueError(
                f"All {dim}-cochains in a batch must all be either tensors or 'None's."
            )

    # Collate the vertex coordinates
    vert_coords_list = [sc.vert_coords for sc in sc_batch]

    if None not in vert_coords_list:
        vert_coords_batch = t.vstack(vert_coords_list)

    elif all(vert_coords is None for vert_coords in vert_coords_list):
        vert_coords_batch = None

    else:
        raise ValueError(
            "All 'vert_coords' in a batch must all be either tensors or 'None's."
        )

    return SimplicialBatch(
        **batch_tensor_dict,
        coboundaries=tuple(coboundaries_batch),
        simplices=tuple(simplices_batch),
        cochains=tuple(cochains_batch),
        vert_coords=vert_coords_batch,
    )


class DataLoader(t.utils.data.DataLoader):
    """
    A user-facing DataLoader that automatically uses the collate_fn.
    """

    def __init__(self, dataset, batch_size=1, **kwargs):
        # Prevent the user from accidentally passing a custom collate_fn() that
        # conflicts with ours.
        if "collate_fn" in kwargs:
            del kwargs["collate_fn"]

        super().__init__(
            dataset, batch_size=batch_size, collate_fn=collate_fn, **kwargs
        )

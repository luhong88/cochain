import torch as t
from jaxtyping import Float, Integer, Real

from .topology import coboundary


class Simplicial2Complex:
    """
    A simplicial 2-complex.
    """

    def __init__(
        self,
        coboundary_0: Float[t.Tensor, "edge vert"],
        coboundary_1: Float[t.Tensor, "tri edge"],
        edges: Integer[t.LongTensor, "edge 2"],
        tris: Integer[t.LongTensor, "tri 3"],
        cochain_0: Float[t.Tensor, "vert *vert_feat"] | None = None,
        cochain_1: Float[t.Tensor, "edge *edge_feat"] | None = None,
        cochain_2: Float[t.Tensor, "tri *tri_feat"] | None = None,
        vert_coords: Float[t.Tensor, "vert 3"] | None = None,
    ):
        self.coboundary_0 = coboundary_0
        self.coboundary_1 = coboundary_1

        self.edges = edges
        self.tris = tris

        self.n_verts = coboundary_0.shape[1]
        self.n_edges = coboundary_0.shape[0]
        self.n_tris = coboundary_1.shape[0]

        self.cochain_0 = cochain_0
        self.cochain_1 = cochain_1
        self.cochain_2 = cochain_2
        self.vert_coords = vert_coords

    def to(self, device: str | t.device):
        for attr, value in self.__dict__.items():
            if t.is_tensor(value):
                setattr(self, attr, value.to(device))
        return self

    # TODO: check for immersion
    @classmethod
    def from_tri_mesh(
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
        unique_canon_edges, coboundary_0, coboundary_1 = (
            coboundary.coboundary_from_tri_mesh(vert_coords, tris)
        )

        return cls(
            coboundary_0=coboundary_0,
            coboundary_1=coboundary_1,
            edges=unique_canon_edges,
            tris=tris,
            vert_coords=vert_coords,
            **kwargs,
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


class SimplicialBatch(Simplicial2Complex):
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


def collate_fn(complicies: list[Simplicial2Complex]) -> SimplicialBatch:
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

import torch as t
from jaxtyping import Float, Integer


class SimplicialMesh:
    """
    A 2-dimensional simplicial complex (i.e., mesh) immersed in 3-dimensional
    Euclidean space.
    """

    def __init__(
        self, vert: Float[t.Tensor, "vert 3"], face: Integer[t.LongTensor, "face 3"]
    ):
        self.vert = vert
        self.face = face

    @property
    def n_vert(self) -> int:
        return self.vert.shape[0]

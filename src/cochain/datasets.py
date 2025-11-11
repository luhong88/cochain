import torch as t

from .complex import Simplicial2Complex


def load_two_tris_mesh() -> Simplicial2Complex:
    """
    A simple 2D mesh embedded in R^3 composed of two triangles sharing one edge.
    """
    return Simplicial2Complex.from_mesh(
        vert_coords=t.Tensor(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        ),
        tris=t.LongTensor([[0, 1, 2], [1, 2, 3]]),
    )

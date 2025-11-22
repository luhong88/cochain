import numpy as np
import torch as t

from ..complex import SimplicialComplex


def load_two_tets_mesh() -> SimplicialComplex:
    """
    A simple 3D mesh embedded composed of two tetrahedra dharing one triangle.
    """
    return SimplicialComplex.from_tet_mesh(
        vert_coords=t.tensor(
            [
                [-0.5, -0.5, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, -1.5],
                [0.0, 0.0, 1.0],
            ]
        ),
        tets=t.tensor([[0, 1, 2, 4], [2, 1, 0, 3]]),
    )

import numpy as np
import pyvista as pv
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


def load_bcc_mesh(dim: int = 5) -> SimplicialComplex:
    """
    Generates a block of tets based on a BCC lattice.
    """
    x = np.linspace(-1, 1, dim)
    y = np.linspace(-1, 1, dim)
    z = np.linspace(-1, 1, dim)
    grid = pv.StructuredGrid(*np.meshgrid(x, y, z))

    tet_grid = grid.triangulate()

    return SimplicialComplex.from_tet_mesh(
        vert_coords=t.from_numpy(tet_grid.points).to(dtype=t.float),
        tets=t.from_numpy(tet_grid.cells.reshape(-1, 5)[:, 1:].copy()).to(dtype=t.long),
    )

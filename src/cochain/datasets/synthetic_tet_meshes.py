import numpy as np
import pytetwild
import pyvista as pv
import torch as t

from ..complex import SimplicialComplex


def load_regular_tet_mesh() -> SimplicialComplex:
    return SimplicialComplex.from_tet_mesh(
        vert_coords=t.tensor(
            [[1.0, 1.0, 1.0], [1.0, -1.0, -1.0], [-1.0, 1.0, -1.0], [-1.0, -1.0, 1.0]]
        ),
        tets=t.tensor([[0, 1, 2, 3]], dtype=t.long),
    )


def load_two_tets_mesh() -> SimplicialComplex:
    """
    A simple 3D mesh embedded composed of two tetrahedra sharing one triangle.
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


def load_solid_torus(
    major_r: float,
    minor_r: float,
    u_res: int,
    v_res: int,
    edge_length_frac: float,
) -> SimplicialComplex:
    # Generate the torus surface mesh, triangulate and clean with pyvista
    surface = pv.ParametricTorus(
        ringradius=major_r, crosssectionradius=minor_r, u_res=u_res, v_res=v_res
    )

    surface = surface.triangulate()
    surface = surface.clean()

    # 3. Extract vertices and faces for tetrahedralization with pytetwild
    v_surf = surface.points
    f_surf = surface.faces.reshape(-1, 4)[:, 1:]

    v_tet, t_tet = pytetwild.tetrahedralize(
        v_surf, f_surf, edge_length_fac=edge_length_frac
    )

    mesh = SimplicialComplex.from_tet_mesh(
        vert_coords=t.from_numpy(v_tet).to(dtype=t.float),
        tets=t.from_numpy(t_tet).to(dtype=t.long),
    )

    return mesh

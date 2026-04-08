import numpy as np
import pytetwild
import pyvista as pv
import torch

from ..complex import SimplicialMesh


def load_regular_tet_mesh() -> SimplicialMesh:
    return SimplicialMesh.from_tet_mesh(
        vert_coords=torch.tensor(
            [[1.0, 1.0, 1.0], [1.0, -1.0, -1.0], [-1.0, 1.0, -1.0], [-1.0, -1.0, 1.0]]
        ),
        tets=torch.tensor([[0, 1, 2, 3]], dtype=torch.int64),
    )


def load_two_tets_mesh() -> SimplicialMesh:
    """
    A simple 3D mesh embedded composed of two tetrahedra sharing one triangle.
    """
    return SimplicialMesh.from_tet_mesh(
        vert_coords=torch.tensor(
            [
                [-0.5, -0.5, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, -1.5],
                [0.0, 0.0, 1.0],
            ]
        ),
        tets=torch.tensor([[0, 1, 2, 4], [2, 1, 0, 3]]),
    )


def load_bcc_mesh(dim: int = 5) -> SimplicialMesh:
    """
    Generates a block of tets based on a BCC lattice.
    """
    x = np.linspace(-1, 1, dim)
    y = np.linspace(-1, 1, dim)
    z = np.linspace(-1, 1, dim)
    grid = pv.StructuredGrid(*np.meshgrid(x, y, z))

    tet_grid = grid.triangulate()

    return SimplicialMesh.from_tet_mesh(
        vert_coords=torch.from_numpy(tet_grid.points).to(dtype=torch.float32),
        tets=torch.from_numpy(tet_grid.cells.reshape(-1, 5)[:, 1:].copy()).to(
            dtype=torch.int64
        ),
    )


def load_solid_torus(
    major_r: float,
    minor_r: float,
    u_res: int,
    v_res: int,
    edge_length_frac: float,
) -> SimplicialMesh:
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

    mesh = SimplicialMesh.from_tet_mesh(
        vert_coords=torch.from_numpy(v_tet).to(dtype=torch.float32),
        tets=torch.from_numpy(t_tet).to(dtype=torch.int64),
    )

    return mesh

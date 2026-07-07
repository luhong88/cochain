from __future__ import annotations

import numpy as np
import torch

from ..complex import SimplicialMesh

try:
    import pytetwild

    _HAS_PYTETWILD = True

except ImportError:
    _HAS_PYTETWILD = False

try:
    import pyvista as pv

    _HAS_PYVISTA = True

except ImportError:
    _HAS_PYVISTA = False


def load_regular_tet_mesh() -> SimplicialMesh:
    """
    Generate a regular tet from alternating vertices of a standard cube.

    A regular tet is one that satisfies the following four criteria:
    * All 6 edges have the same length.
    * All 4 tris are equilateral with the same area.
    * The internal dihedral angle between any two adjoining tris is exactly arccos(1/3).
    * The solid angles at all four vertices are identical.
    """
    return SimplicialMesh.from_tet_mesh(
        vert_coords=torch.tensor(
            [[1.0, 1.0, 1.0], [-1.0, 1.0, -1.0], [1.0, -1.0, -1.0], [-1.0, -1.0, 1.0]]
        ),
        tets=torch.tensor([[0, 1, 2, 3]], dtype=torch.int64),
    )


def load_two_tets_mesh() -> SimplicialMesh:
    """Generate a simple tet mesh composed of two tets sharing one triangle."""
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


def load_sc_mesh(dim: int = 5) -> SimplicialMesh:
    """
    Generate a tetrahedral mesh block based on a regular rectilinear grid.

    This function generates vertices in a simple cubic (SC) lattice bounded between
    [-1, 1] on all three axes. The hexahedral grid cells are subdivided using an
    alternating 5 tet pattern.

    Parameters
    ----------
    dim
        The number of vertices along each axis (x, y, and z) of the bounding
        cube. The total number of generated vertices will be `dim**3`.

    Returns
    -------
    mesh
        A `SimplicialMesh` object representing the BCC mesh.
    """
    if not _HAS_PYVISTA:
        raise ImportError("PyVista backend required.")

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
    """
    Generate a solid torus tet mesh.

    Parameters
    ----------
    major_r
        The major radius of the torus.
    minor_r
        The minor radius of the torus.
    u_res
        The resolution in the u direction.
    v_res
        The resolution in the v direction.
    edge_length_frac
        Tet edge length as a function of bounding box diagonal.

    Returns
    -------
    mesh
        A `SimplicialMesh` object representing the solid torus mesh.
    """
    if not (_HAS_PYVISTA and _HAS_PYTETWILD):
        raise ImportError("PyVista and PyTetWild backends required.")

    # Generate the torus surface mesh, triangulate and clean with pyvista.
    surface = pv.ParametricTorus(
        ringradius=major_r, crosssectionradius=minor_r, u_res=u_res, v_res=v_res
    )

    surface = surface.triangulate()
    surface = surface.clean()

    # Extract vertices and faces for tetrahedralization with pytetwild.
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


def load_spherical_shell(
    outer_r: float,
    inner_r: float,
    theta_res: int,
    phi_res: int,
    edge_length_frac: float,
) -> SimplicialMesh:
    """
    Generate a spherical shell tet mesh.

    Parameters
    ----------
    outer_r
        The outer radius of the shell.
    inner_r
        The inner radius of the shell.
    theta_res
        The number of points in the azimuthal direction.
    phi_res
        the number of points in the polar direction.
    edge_length_frac
        Tet edge length as a function of bounding box diagonal.

    Returns
    -------
    mesh
        A `SimplicialMesh` object representing the spherical shell mesh.
    """
    if not (_HAS_PYVISTA and _HAS_PYTETWILD):
        raise ImportError("PyVista and PyTetWild backends required.")

    # Generate the outer bounding surface.
    outer_sphere = pv.Sphere(
        radius=outer_r, theta_resolution=theta_res, phi_resolution=phi_res
    )

    # Generate the inner bounding surface (the hollow core).
    inner_sphere = pv.Sphere(
        radius=inner_r, theta_resolution=theta_res, phi_resolution=phi_res
    )

    # Flip the normals of the inner sphere to point towards the origin. This ensures
    # the winding number accurately resolves to 0 inside the cavity.
    inner_sphere.flip_faces(inplace=True)

    # Combine both surfaces into a single disjoint mesh.
    surface = outer_sphere + inner_sphere
    surface = surface.triangulate()
    surface = surface.clean()

    # Extract vertices and faces for PyTetWild.
    v_surf = surface.points
    f_surf = surface.faces.reshape(-1, 4)[:, 1:]

    # Tetrahedralize.
    v_tet, t_tet = pytetwild.tetrahedralize(
        v_surf, f_surf, edge_length_fac=edge_length_frac
    )

    # Wrap in mesh class.
    mesh = SimplicialMesh.from_tet_mesh(
        vert_coords=torch.from_numpy(v_tet).to(dtype=torch.float32),
        tets=torch.from_numpy(t_tet).to(dtype=torch.int64),
    )

    return mesh

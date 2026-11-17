import numpy as np
import torch
from scipy.spatial import Delaunay

from ..complex import SimplicialMesh

try:
    import pyvista as pv

    _HAS_PYVISTA = True

except ImportError:
    _HAS_PYVISTA = False


def load_two_tris_mesh() -> SimplicialMesh:
    """Generate a simple tri mesh composed of two tris sharing one edge."""
    return SimplicialMesh.from_tri_mesh(
        vert_coords=torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        ),
        tris=torch.tensor([[0, 1, 2], [1, 3, 2]], dtype=torch.int64),
    )


def load_two_disjoint_tris_mesh() -> SimplicialMesh:
    """Generate a simple tri mesh composed of two disjoint tris."""
    return SimplicialMesh.from_tri_mesh(
        vert_coords=torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 1.0],
                [0.0, 1.0, 1.0],
            ]
        ),
        tris=torch.tensor([[0, 1, 2], [3, 4, 5]], dtype=torch.int64),
    )


def load_square_mesh() -> SimplicialMesh:
    """Generate a simple triangulated square consisting of 4 triangles in the z = 0 plane."""
    return SimplicialMesh.from_tri_mesh(
        vert_coords=torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.5, 0.5, 0.0],
            ]
        ),
        tris=torch.tensor(
            [[0, 4, 1], [1, 4, 2], [2, 4, 3], [3, 4, 0]], dtype=torch.int64
        ),
    )


def load_tent_mesh() -> SimplicialMesh:
    """Generate a simple triangulated square with the central vertex elevated above z=0."""
    return SimplicialMesh.from_tri_mesh(
        vert_coords=torch.tensor(
            [
                [0.5, 0.5, 1.0],
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
            ]
        ),
        tris=torch.tensor(
            [[0, 1, 2], [0, 2, 3], [0, 3, 4], [0, 4, 1]], dtype=torch.int64
        ),
    )


def load_hollow_tet_mesh() -> SimplicialMesh:
    """Generate A simple 2D mesh for the boundary of an irregular tetrahedron."""
    return SimplicialMesh.from_tri_mesh(
        vert_coords=torch.tensor(
            [
                [0.0, 0.0, 2.0],
                [1.0, 0.0, 0.0],
                [0.0, 0.5, 0.0],
                [-0.5, -1.0, 0.0],
            ]
        ),
        tris=torch.tensor(
            [[0, 1, 2], [0, 3, 1], [0, 2, 3], [1, 3, 2]], dtype=torch.int64
        ),
    )


def load_flat_annulus_mesh(
    r_in: float = 0.5,
    r_out: float = 1.0,
    radial_res: int = 3,
    circum_res: int = 24,
) -> SimplicialMesh:
    """
    Generate a finer Delaunay annulus mesh with interior triangles.

    Concentric vertex rings are staggered by half an angular step on alternating
    rings before Delaunay triangulation to avoid co-circular cyclic quads (which
    would produce zero-length circumcentric dual edges).

    Parameters
    ----------
    r_in
        The inner radius of the annulus.
    r_out
        The outer radius of the annulus.
    radial_res
        The number of radial layers between `r_in` and `r_out`.
    circum_res
        The number of vertices on each concentric circle.

    Returns
    -------
    mesh
        A `SimplicialMesh` object representing the annulus mesh.
    """
    if r_in >= r_out:
        raise ValueError("Inner radius (r_in) must be less than outer radius (r_out).")
    if radial_res < 2:
        raise ValueError("Radial resolution (radial_res) must be at least 2.")
    if circum_res < 3:
        raise ValueError("Circumferential resolution (circum_res) must be at least 3.")

    radii = np.linspace(r_in, r_out, radial_res + 1)
    rings = []
    for ring_idx, r in enumerate(radii):
        angle_offset = (ring_idx % 2) * (np.pi / circum_res)
        theta = np.linspace(0, 2 * np.pi, circum_res, endpoint=False) + angle_offset
        rings.append(np.column_stack((r * np.cos(theta), r * np.sin(theta))))

    vert_coords = np.vstack(rings)
    vert_coords_3d = np.hstack((vert_coords, np.zeros((vert_coords.shape[0], 1))))

    tri = Delaunay(vert_coords)

    # Remove triangles filling the inner hole (all three vertices on the innermost ring).
    all_triangles = tri.simplices
    keep_mask = np.any(all_triangles >= circum_res, axis=1)
    annulus_tris = all_triangles[keep_mask]

    mesh = SimplicialMesh.from_tri_mesh(
        torch.from_numpy(vert_coords_3d).to(dtype=torch.float32),
        torch.from_numpy(annulus_tris).to(dtype=torch.int64),
    )

    return mesh


def load_icosphere_mesh(r: float = 1.0, n_sub: int = 1):
    """
    Generate an icosphere mesh.

    Parameters
    ----------
    r
        The radius of the icosphere.
    n_sub
        The number of times each triangle of the original icosahedron is subdivided.

    Returns
    -------
    mesh
        A `SimplicialMesh` object representing the icosphere mesh.
    """
    if not _HAS_PYVISTA:
        raise ImportError("PyVista backend required.")

    pv_sphere = pv.Icosphere(radius=r, nsub=n_sub)

    vert_coords_np = np.asarray(pv_sphere.points)
    tris_np = np.asarray(pv_sphere.regular_faces)

    vert_coords_t = torch.from_numpy(vert_coords_np).to(dtype=torch.float32)
    tris_t = torch.from_numpy(tris_np).to(dtype=torch.int64)

    cochain_sphere = SimplicialMesh.from_tri_mesh(vert_coords_t, tris_t)

    return cochain_sphere

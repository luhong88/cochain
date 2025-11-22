import numpy as np
import torch as t
from scipy.spatial import Delaunay

from ..complex import SimplicialComplex


def load_two_tris_mesh() -> SimplicialComplex:
    """
    A simple 2D mesh embedded in R^3 composed of two triangles sharing one edge.
    """
    return SimplicialComplex.from_tri_mesh(
        vert_coords=t.tensor(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        ),
        tris=t.tensor([[0, 1, 2], [1, 3, 2]], dtype=t.long),
    )


def load_square_mesh() -> SimplicialComplex:
    """
    A simple triangulated square consisting of 4 triangles in the z = 0 plane.
    """
    return SimplicialComplex.from_tri_mesh(
        vert_coords=t.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.5, 0.5, 0.0],
            ]
        ),
        tris=t.tensor([[0, 4, 1], [1, 4, 2], [2, 4, 3], [3, 4, 0]], dtype=t.long),
    )


def load_tent_mesh() -> SimplicialComplex:
    """
    Similar to the square mesh, but the central vertex is elevated above the z=0
    plane.
    """
    return SimplicialComplex.from_tri_mesh(
        vert_coords=t.tensor(
            [
                [0.5, 0.5, 1.0],
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
            ]
        ),
        tris=t.tensor([[0, 1, 2], [0, 2, 3], [0, 3, 4], [0, 4, 1]], dtype=t.long),
    )


def load_tet_mesh() -> SimplicialComplex:
    """
    A simple 2D mesh for the boundary of a tetrahedron.
    """
    return SimplicialComplex.from_tri_mesh(
        vert_coords=t.tensor(
            [
                [0.0, 0.0, 2.0],
                [1.0, 0.0, 0.0],
                [0.0, 0.5, 0.0],
                [-0.5, -1.0, 0.0],
            ]
        ),
        tris=t.tensor([[0, 1, 2], [0, 3, 1], [0, 2, 3], [1, 3, 2]], dtype=t.long),
    )


def load_flat_annulus_mesh(
    r_in: float = 0.5,
    r_out: float = 1.0,
    n_segments_in: int = 5,
    n_segments_out: int = 10,
) -> SimplicialComplex:
    """
    Generates a 2D annulus mesh using Delaunay triangulation. The mesh is created
    from points on two concentric circles.
    """
    if r_in >= r_out:
        raise ValueError("Inner radius (r_in) must be less than outer radius (r_out).")
    if n_segments_in < 3:
        raise ValueError("Number of segments must be at least 3.")
    if n_segments_out < 3:
        raise ValueError("Number of segments must be at least 3.")

    # Calculate inner circle coordinates.
    # Use endpoint=False to avoid duplicating the 0 and 2*pi point
    theta_in = np.linspace(0, 2 * np.pi, n_segments_in, endpoint=False)
    x_in = r_in * np.cos(theta_in)
    y_in = r_in * np.sin(theta_in)
    points_in = np.vstack((x_in, y_in)).T

    # Calculate outer circle coordinates.
    theta_out = np.linspace(0, 2 * np.pi, n_segments_out, endpoint=False)
    x_out = r_out * np.cos(theta_out)
    y_out = r_out * np.sin(theta_out)
    points_out = np.vstack((x_out, y_out)).T

    # Combine inner and outer points
    vert_coords = np.vstack((points_in, points_out))
    vert_coords_3d = np.hstack((vert_coords, np.zeros((vert_coords.shape[0], 1))))

    # Perform Delaunay triangulation.
    tri = Delaunay(vert_coords)

    # The Delaunay triangulation fills in the center "hole", which need to be
    # removed. A triangle is in the "hole" if all its vertex indices are < n_segments;
    # i.e., if all of its vertices are on the inner circle.
    all_triangles = tri.simplices
    keep_mask = np.any(all_triangles >= n_segments_in, axis=1)
    annulus_tris = all_triangles[keep_mask]

    annulus_mesh = SimplicialComplex.from_tri_mesh(
        t.from_numpy(vert_coords_3d).to(dtype=t.float),
        t.from_numpy(annulus_tris).to(dtype=t.long),
    )

    return annulus_mesh

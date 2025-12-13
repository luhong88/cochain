import numpy as np
import pytest
import pyvista as pv
import torch as t

from cochain.complex import SimplicialComplex
from cochain.datasets import synthetic_tet_meshes, synthetic_tri_meshes


@pytest.fixture
def two_tris_mesh() -> SimplicialComplex:
    return synthetic_tri_meshes.load_two_tris_mesh()


@pytest.fixture
def square_mesh() -> SimplicialComplex:
    return synthetic_tri_meshes.load_square_mesh()


@pytest.fixture
def tent_mesh() -> SimplicialComplex:
    return synthetic_tri_meshes.load_tent_mesh()


@pytest.fixture
def hollow_tet_mesh() -> SimplicialComplex:
    return synthetic_tri_meshes.load_hollow_tet_mesh()


@pytest.fixture
def icosphere_mesh() -> SimplicialComplex:
    pv_sphere = pv.Icosphere(nsub=1)

    vert_coords_np = np.asarray(pv_sphere.points)
    tris_np = np.asarray(pv_sphere.regular_faces)

    vert_coords_t = t.from_numpy(vert_coords_np).to(dtype=t.float)
    tris_t = t.from_numpy(tris_np).to(dtype=t.long)

    cochain_sphere = SimplicialComplex.from_tri_mesh(vert_coords_t, tris_t)

    return cochain_sphere


@pytest.fixture
def flat_annulus_mesh() -> SimplicialComplex:
    return synthetic_tri_meshes.load_flat_annulus_mesh(
        r_in=0.5, r_out=1.0, n_segments_in=5, n_segments_out=10
    )


@pytest.fixture
def reg_tet_mesh() -> SimplicialComplex:
    return synthetic_tet_meshes.load_regular_tet_mesh()


@pytest.fixture
def two_tets_mesh() -> SimplicialComplex:
    return synthetic_tet_meshes.load_two_tets_mesh()


@pytest.fixture
def simple_bcc_mesh() -> SimplicialComplex:
    return synthetic_tet_meshes.load_bcc_mesh(dim=3)

import numpy as np
import pytest
import torch as t
import trimesh

import cochain.datasets as datasets
from cochain.complex import SimplicialComplex


@pytest.fixture
def two_tris_mesh() -> SimplicialComplex:
    return datasets.load_two_tris_mesh()


@pytest.fixture
def square_mesh() -> SimplicialComplex:
    return datasets.load_square_mesh()


@pytest.fixture
def tent_mesh() -> SimplicialComplex:
    return datasets.load_tent_mesh()


@pytest.fixture
def tet_mesh() -> SimplicialComplex:
    return datasets.load_tet_mesh()


@pytest.fixture
def icosphere_mesh() -> SimplicialComplex:
    trimesh_sphere = trimesh.creation.icosphere(subdivisions=1)

    vert_coords_np = np.asarray(trimesh_sphere.vertices)
    tris_np = np.asarray(trimesh_sphere.faces)

    vert_coords_t = t.from_numpy(vert_coords_np).to(dtype=t.float)
    tris_t = t.from_numpy(tris_np)

    cochain_sphere = SimplicialComplex.from_tri_mesh(vert_coords_t, tris_t)

    return cochain_sphere


@pytest.fixture
def flat_annulus_mesh() -> SimplicialComplex:
    return datasets.load_flat_annulus_mesh(
        r_in=0.5, r_out=1.0, n_segments_in=5, n_segments_out=10
    )

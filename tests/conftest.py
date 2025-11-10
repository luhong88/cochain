import numpy as np
import pytest
import torch as t
import trimesh

import cochain.datasets as datasets
from cochain.complex import Simplicial2Complex


@pytest.fixture
def two_tris_mesh():
    return datasets.load_two_tris_mesh()


@pytest.fixture
def icosphere_mesh():
    trimesh_sphere = trimesh.creation.icosphere(subdivisions=1)

    vert_coords_np = np.asarray(trimesh_sphere.vertices)
    tris_np = np.asarray(trimesh_sphere.faces)

    vert_coords_t = t.from_numpy(vert_coords_np).to(dtype=t.float)
    tris_t = t.from_numpy(tris_np)

    cochain_sphere = Simplicial2Complex.from_mesh(vert_coords_t, tris_t)

    return cochain_sphere

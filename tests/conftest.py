import os
import random

import numpy as np
import pytest
import pyvista as pv
import torch as t

from cochain.complex import SimplicialComplex
from cochain.datasets import synthetic_tet_meshes, synthetic_tri_meshes


def pytest_addoption(parser):
    """
    Add a commandline option to specify a global RNG seed.
    """
    parser.addoption(
        "--rng-seed",
        action="store",
        default=0,
        type=int,
        help="Seed for random number generators. Use -1 for a random seed.",
    )


@pytest.fixture(scope="session")
def session_seed(request):
    """
    Determines the RNG seed for the entire session.
    """
    seed_arg = request.config.getoption("--rng-seed")

    if seed_arg == -1:
        seed = int.from_bytes(os.urandom(4), "big")
        print(f"\n[RNG] Using Random Session Seed: {seed}")
    else:
        seed = seed_arg

    return seed


@pytest.fixture(scope="function", autouse=True)
def set_rng(session_seed):
    """
    Resets the RNG state before each test function using the sesion seed. Note that
    'autouse=True' means this runs automatically for every test.
    """
    t.manual_seed(session_seed)
    np.random.seed(session_seed)
    random.seed(session_seed)

    if t.cuda.is_available():
        t.cuda.manual_seed_all(session_seed)

    yield


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


@pytest.fixture
def solid_torus_mesh() -> SimplicialComplex:
    return synthetic_tet_meshes.load_solid_torus(
        major_r=1.0, minor_r=0.5, u_res=5, v_res=5, edge_length_frac=1.0
    )

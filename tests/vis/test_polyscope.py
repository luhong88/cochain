import pytest

from cochain.vis import PolyscopeViewer

ps = pytest.importorskip("polyscope")


def test_tri_mesh_vis_with_polyscope(icosphere_mesh, device):
    ps.init("openGL_mock")

    mesh = icosphere_mesh.to(device)

    cochain_0 = mesh.vert_coords[:, 0]
    cochain_1 = (
        mesh.vert_coords[mesh.edges[:, 0], 0] - mesh.vert_coords[mesh.edges[:, 1], 0]
    )
    cochain_2 = mesh.vert_coords[mesh.tris, 2].mean(dim=-1)

    vec_0 = mesh.vert_coords
    vec_1 = mesh.vert_coords[mesh.edges].mean(dim=1)
    vec_2 = mesh.vert_coords[mesh.tris].mean(dim=1)

    ps_mesh = PolyscopeViewer(name="test_tri_mesh", mesh=mesh)

    ps_mesh.add_k_cochain(k=0, name="cochain_0", cochain=cochain_0)
    ps_mesh.add_k_cochain(k=1, name="cochain_1", cochain=cochain_1)
    ps_mesh.add_k_cochain(k=2, name="cochain_2", cochain=cochain_2)

    ps_mesh.add_vector_field(k=0, name="vec_0", vec_field=vec_0)
    ps_mesh.add_vector_field(k=1, name="vec_1", vec_field=vec_1)
    ps_mesh.add_vector_field(k=2, name="vec_2", vec_field=vec_2)

    ps.show(forFrames=3)

    ps.remove_all_structures()


def test_tet_mesh_vis_with_polyscope(solid_torus_mesh, device):
    ps.init("openGL_mock")

    mesh = solid_torus_mesh.to(device)

    cochain_0 = mesh.vert_coords[:, 0]
    cochain_1 = (
        mesh.vert_coords[mesh.edges[:, 0], 0] - mesh.vert_coords[mesh.edges[:, 1], 0]
    )
    cochain_2 = mesh.vert_coords[mesh.tris, 2].mean(dim=-1)
    cochain_3 = mesh.vert_coords[mesh.tets, 1].mean(dim=-1)

    vec_0 = mesh.vert_coords
    vec_1 = mesh.vert_coords[mesh.edges].mean(dim=1)
    vec_2 = mesh.vert_coords[mesh.tris].mean(dim=1)
    vec_3 = mesh.vert_coords[mesh.tets].mean(dim=1)

    ps_mesh = PolyscopeViewer(name="test_tet_mesh", mesh=mesh)

    ps_mesh.add_k_cochain(k=0, name="cochain_0", cochain=cochain_0)
    ps_mesh.add_k_cochain(k=1, name="cochain_1", cochain=cochain_1)
    ps_mesh.add_k_cochain(k=2, name="cochain_2", cochain=cochain_2)
    ps_mesh.add_k_cochain(k=3, name="cochain_3", cochain=cochain_3)

    ps_mesh.add_vector_field(k=0, name="vec_0", vec_field=vec_0)
    ps_mesh.add_vector_field(k=1, name="vec_1", vec_field=vec_1)
    ps_mesh.add_vector_field(k=2, name="vec_2", vec_field=vec_2)
    ps_mesh.add_vector_field(k=3, name="vec_3", vec_field=vec_3)

    ps.show(forFrames=3)

    ps.remove_all_structures()

import numpy as np
import pytest
import torch
from scipy.sparse import coo_matrix, csgraph

from cochain.complex import SimplicialMesh
from cochain.topology.topo_laplacians import laplacian_k


def _compute_graph_laplacian_scipy(mesh: SimplicialMesh):
    """Compute the graph Laplacian of the 1-skeleton of a mesh via scipy."""
    edges = mesh.edges.detach().cpu().numpy()
    n_verts = mesh.n_verts

    row = edges[:, 0]
    col = edges[:, 1]

    rows = np.concatenate([row, col])
    cols = np.concatenate([col, row])
    data = np.ones_like(rows, dtype=float)

    A_csr = coo_matrix((data, (rows, cols)), shape=(n_verts, n_verts)).tocsr()

    graph_laplacian = torch.from_numpy(
        csgraph.laplacian(A_csr, normed=False).todense(),
    ).to(
        dtype=mesh.dtype,
        device=mesh.device,
    )

    return graph_laplacian


@pytest.mark.parametrize(
    "mesh",
    [
        "two_tris_mesh",
        "hollow_tet_mesh",
        "finer_flat_annulus_mesh",
        "two_tets_mesh",
        "simple_bcc_mesh",
        "solid_torus_mesh",
    ],
)
def test_laplacian_0_equivalence_to_graph_laplacian(mesh, request, device):
    mesh = request.getfixturevalue(mesh).to(device)

    laplacian = laplacian_k(mesh, k=0, component="full").to_dense()
    laplacian_scipy = _compute_graph_laplacian_scipy(mesh)

    torch.testing.assert_close(laplacian, laplacian_scipy)


@pytest.mark.parametrize("mesh", ["two_tris_mesh", "two_tets_mesh"])
def test_laplacian_symmetry(mesh, request, device):
    mesh = request.getfixturevalue(mesh).to(device)

    for k in range(mesh.dim + 1):
        laplacian = laplacian_k(mesh, k=k, component="full")
        torch.testing.assert_close(laplacian.to_dense(), laplacian.T.to_dense())


@pytest.mark.parametrize("mesh", ["two_tris_mesh", "two_tets_mesh"])
def test_laplacian_PSD(mesh, request, device):
    mesh = request.getfixturevalue(mesh).to(dtype=torch.float64, device=device)

    for k in range(mesh.dim + 1):
        laplacian = laplacian_k(mesh, k=k, component="full").to_dense()
        eigs = torch.linalg.eigvalsh(laplacian)

        assert eigs.min() >= -1e-8


@pytest.mark.parametrize("mesh", ["two_tris_mesh", "two_tets_mesh"])
def test_laplacian_orthogonality(mesh, request, device):
    mesh = request.getfixturevalue(mesh).to(device)

    for k in range(1, mesh.dim):
        l_down = laplacian_k(mesh, k=k, component="down")
        l_up = laplacian_k(mesh, k=k, component="up")

        composition_1 = (l_down @ l_up).to_dense()
        composition_2 = (l_up @ l_down).to_dense()

        torch.testing.assert_close(composition_1, torch.zeros_like(composition_1))
        torch.testing.assert_close(composition_2, torch.zeros_like(composition_2))


@pytest.mark.parametrize(
    "k, betti",
    [(0, 1), (1, 0), (2, 0), (3, 0)],
)
def test_sphere_homology_group_dims(k, betti, two_tets_mesh: SimplicialMesh, device):
    mesh = two_tets_mesh.to(device=device)

    operator = laplacian_k(mesh, k=k, component="full").to_dense()
    dim_ker = operator.shape[0] - torch.linalg.matrix_rank(operator)
    torch.testing.assert_close(dim_ker, torch.tensor(betti, device=device))

    # Also test Poincare duality.
    dual_operator = laplacian_k(
        mesh, k=mesh.dim - k, component="full", dual_complex=True
    ).to_dense()
    dual_dim_ker = dual_operator.shape[0] - torch.linalg.matrix_rank(dual_operator)
    torch.testing.assert_close(dim_ker, dual_dim_ker)


@pytest.mark.parametrize(
    "k, betti",
    [(0, 1), (1, 1), (2, 0), (3, 0)],
)
def test_torus_homology_group_dims(k, betti, solid_torus_mesh: SimplicialMesh, device):
    mesh = solid_torus_mesh.to(device)

    operator = laplacian_k(mesh, k=k, component="full").to_dense()
    dim_ker = operator.shape[0] - torch.linalg.matrix_rank(operator)
    torch.testing.assert_close(dim_ker, torch.tensor(betti, device=device))

    # Also test Poincare duality.
    dual_operator = laplacian_k(
        mesh, k=mesh.dim - k, component="full", dual_complex=True
    ).to_dense()
    dual_dim_ker = dual_operator.shape[0] - torch.linalg.matrix_rank(dual_operator)
    torch.testing.assert_close(dim_ker, dual_dim_ker)


@pytest.mark.parametrize(
    "k, betti",
    [(0, 1), (1, 0), (2, 1), (3, 0)],
)
def test_spherical_shell_homology_group_dims(
    k, betti, solid_spherical_shell_mesh: SimplicialMesh, device
):
    mesh = solid_spherical_shell_mesh.to(device)

    operator = laplacian_k(mesh, k=k, component="full").to_dense()
    dim_ker = operator.shape[0] - torch.linalg.matrix_rank(operator)
    torch.testing.assert_close(dim_ker, torch.tensor(betti, device=device))

    # Also test Poincare duality.
    dual_operator = laplacian_k(
        mesh, k=mesh.dim - k, component="full", dual_complex=True
    ).to_dense()
    dual_dim_ker = dual_operator.shape[0] - torch.linalg.matrix_rank(dual_operator)
    torch.testing.assert_close(dim_ker, dual_dim_ker)


@pytest.mark.parametrize(
    "k, betti",
    [(0, 1), (1, 0), (2, 0)],
)
def test_disk_homology_group_dims(k, betti, tent_mesh: SimplicialMesh, device):
    mesh = tent_mesh.to(device)

    operator = laplacian_k(mesh, k=k, component="full").to_dense()
    dim_ker = operator.shape[0] - torch.linalg.matrix_rank(operator)
    torch.testing.assert_close(dim_ker, torch.tensor(betti, device=device))

    # Also test Poincare duality.
    dual_operator = laplacian_k(
        mesh, k=mesh.dim - k, component="full", dual_complex=True
    ).to_dense()
    dual_dim_ker = dual_operator.shape[0] - torch.linalg.matrix_rank(dual_operator)
    torch.testing.assert_close(dim_ker, dual_dim_ker)


@pytest.mark.parametrize(
    "k, betti",
    [(0, 2), (1, 0), (2, 0)],
)
def test_two_disjoint_disk_homology_group_dims(
    k, betti, two_disjoint_tris_mesh: SimplicialMesh, device
):
    mesh = two_disjoint_tris_mesh.to(device)

    operator = laplacian_k(mesh, k=k, component="full").to_dense()
    dim_ker = operator.shape[0] - torch.linalg.matrix_rank(operator)
    torch.testing.assert_close(dim_ker, torch.tensor(betti, device=device))

    # Also test Poincare duality.
    dual_operator = laplacian_k(
        mesh, k=mesh.dim - k, component="full", dual_complex=True
    ).to_dense()
    dual_dim_ker = dual_operator.shape[0] - torch.linalg.matrix_rank(dual_operator)
    torch.testing.assert_close(dim_ker, dual_dim_ker)


@pytest.mark.parametrize(
    "k, betti",
    [(0, 1), (1, 1), (2, 0)],
)
def test_annulus_homology_group_dims(
    k, betti, flat_annulus_mesh: SimplicialMesh, device
):
    mesh = flat_annulus_mesh.to(device)

    operator = laplacian_k(mesh, k=k, component="full").to_dense()
    dim_ker = operator.shape[0] - torch.linalg.matrix_rank(operator)
    torch.testing.assert_close(dim_ker, torch.tensor(betti, device=device))

    # Also test Poincare duality.
    dual_operator = laplacian_k(
        mesh, k=mesh.dim - k, component="full", dual_complex=True
    ).to_dense()
    dual_dim_ker = dual_operator.shape[0] - torch.linalg.matrix_rank(dual_operator)
    torch.testing.assert_close(dim_ker, dual_dim_ker)


@pytest.mark.parametrize(
    "k, betti",
    [(0, 1), (1, 0), (2, 1)],
)
def test_hollow_sphere_homology_group_dims(
    k, betti, icosphere_mesh: SimplicialMesh, device
):
    mesh = icosphere_mesh.to(device)

    operator = laplacian_k(mesh, k=k, component="full").to_dense()
    dim_ker = operator.shape[0] - torch.linalg.matrix_rank(operator)
    torch.testing.assert_close(dim_ker, torch.tensor(betti, device=device))

    # Also test Poincare duality.
    dual_operator = laplacian_k(
        mesh, k=mesh.dim - k, component="full", dual_complex=True
    ).to_dense()
    dual_dim_ker = dual_operator.shape[0] - torch.linalg.matrix_rank(dual_operator)
    torch.testing.assert_close(dim_ker, dual_dim_ker)

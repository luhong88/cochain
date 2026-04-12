import torch

from cochain.complex import SimplicialMesh
from cochain.metric.tri.tri_hodge_stars import star_1
from cochain.metric.tri.tri_laplacians import (
    laplacian_1,
    laplacian_1_curl_curl,
    laplacian_1_grad_div,
)
from cochain.topology.spanning_tree import compute_cotree_mask, compute_tree_mask
from cochain.topology.topo_laplacians import laplacian_k


def test_cbd_0_rank(finer_flat_annulus_mesh: SimplicialMesh, device):
    """
    The tree decomposition for the down 1-Laplacian of a triangular mesh identifies
    the same number of edges as the rank of the 0-coboundary operator (assuming
    that the mesh has no relative boundary conditions).
    """
    mesh = finer_flat_annulus_mesh.to(device)

    cbd_0 = mesh.cbd[0].to_dense()
    cbd_0_rank = torch.linalg.matrix_rank(cbd_0)

    l0 = laplacian_k(mesh, k=0, component="up")
    tree_mask = compute_tree_mask(
        topo_laplacian_0=l0,
        canon_edges=mesh.edges,
    )
    n_tree_edge = torch.sum(tree_mask)

    torch.testing.assert_close(cbd_0_rank, n_tree_edge)


def test_cbd_1_rank(icosphere_mesh: SimplicialMesh, device):
    """
    The cotree decomposition for the up 1-Laplacian of a triangular mesh identifies
    the same number of edges as the rank of the 1-coboundary operator (assuming
    that the mesh has no boundaries).
    """
    mesh = icosphere_mesh.to(device)

    cbd_1 = mesh.cbd[1]
    cbd_1_rank = torch.linalg.matrix_rank(cbd_1.to_dense())

    dual_l0 = laplacian_k(mesh, k=0, component="up", dual=True)
    cotree_mask = compute_cotree_mask(
        dual_topo_laplacian_0=dual_l0,
        cbd_1=cbd_1,
    )
    n_cotree_edge = torch.sum(cotree_mask)

    torch.testing.assert_close(cbd_1_rank, n_cotree_edge)


def test_l1_positive_definite_no_bc(icosphere_mesh: SimplicialMesh, device):
    """
    The 1-Laplacian after tree-cotree gauge fixing should be strictly positive
    definite.
    """
    mesh = icosphere_mesh.to(device)

    l0 = laplacian_k(mesh, k=0, component="up")
    dual_l0 = laplacian_k(mesh, k=0, component="up", dual=True)

    cotree_mask = compute_cotree_mask(dual_topo_laplacian_0=dual_l0, cbd_1=mesh.cbd[1])
    tree_mask = compute_tree_mask(
        topo_laplacian_0=l0, canon_edges=mesh.edges, cotree_mask=cotree_mask
    )

    # Test b_0 and b_2 betti numbers
    assert tree_mask.sum() == mesh.n_verts - 1
    assert cotree_mask.sum() == mesh.n_tris - 1

    # Test tree-cotree disjointness
    assert not (tree_mask & cotree_mask).any()

    free_edge_mask = tree_mask | cotree_mask

    l1 = (star_1(mesh) @ laplacian_1(mesh)).to_dense()
    l1_fixed = l1[free_edge_mask][:, free_edge_mask]
    min_eig = torch.linalg.eigvalsh(l1_fixed).min()

    assert min_eig > 0.0


def test_l1_down_positive_definite_no_bc(icosphere_mesh: SimplicialMesh, device):
    mesh = icosphere_mesh.to(device)

    l0 = laplacian_k(mesh, k=0, component="up")
    tree_mask = compute_tree_mask(topo_laplacian_0=l0, canon_edges=mesh.edges)

    l1_down = (star_1(mesh) @ laplacian_1_grad_div(mesh)).to_dense()
    l1_fixed = l1_down[tree_mask][:, tree_mask]
    min_eig = torch.linalg.eigvalsh(l1_fixed).min()

    assert min_eig > 0.0


def test_l1_up_positive_definite_no_bc(icosphere_mesh: SimplicialMesh, device):
    mesh = icosphere_mesh.to(device)

    dual_l0 = laplacian_k(mesh, k=0, component="up", dual=True)
    cotree_mask = compute_cotree_mask(dual_topo_laplacian_0=dual_l0, cbd_1=mesh.cbd[1])

    l1_up = (star_1(mesh) @ laplacian_1_curl_curl(mesh)).to_dense()
    l1_fixed = l1_up[cotree_mask][:, cotree_mask]
    min_eig = torch.linalg.eigvalsh(l1_fixed).min()

    assert min_eig > 0.0


def test_l1_positive_definite_absolute_bc(
    finer_flat_annulus_mesh: SimplicialMesh, device
):
    mesh = finer_flat_annulus_mesh.to(device)

    l0 = laplacian_k(mesh, k=0, component="up")
    dual_l0 = laplacian_k(mesh, k=0, component="up", dual=True)

    cotree_mask = compute_cotree_mask(dual_topo_laplacian_0=dual_l0, cbd_1=mesh.cbd[1])
    tree_mask = compute_tree_mask(
        topo_laplacian_0=l0, canon_edges=mesh.edges, cotree_mask=cotree_mask
    )

    assert not (tree_mask & cotree_mask).any()

    free_edge_mask = tree_mask | cotree_mask

    l1 = (star_1(mesh) @ laplacian_1(mesh)).to_dense()
    l1_fixed = l1[free_edge_mask][:, free_edge_mask]
    min_eig = torch.linalg.eigvalsh(l1_fixed).min()

    assert min_eig > 0.0


def test_l1_down_positive_definite_absolute_bc(
    finer_flat_annulus_mesh: SimplicialMesh, device
):
    mesh = finer_flat_annulus_mesh.to(device)

    l0 = laplacian_k(mesh, k=0, component="up")
    tree_mask = compute_tree_mask(topo_laplacian_0=l0, canon_edges=mesh.edges)

    l1 = (star_1(mesh) @ laplacian_1_grad_div(mesh)).to_dense()
    l1_fixed = l1[tree_mask][:, tree_mask]
    min_eig = torch.linalg.eigvalsh(l1_fixed).min()

    assert min_eig > 0.0


def test_l1_up_positive_definite_absolute_bc(
    finer_flat_annulus_mesh: SimplicialMesh, device
):
    mesh = finer_flat_annulus_mesh.to(device)

    dual_l0 = laplacian_k(mesh, k=0, component="up", dual=True)
    cotree_mask = compute_cotree_mask(dual_topo_laplacian_0=dual_l0, cbd_1=mesh.cbd[1])

    l1 = (star_1(mesh) @ laplacian_1_curl_curl(mesh)).to_dense()
    l1_fixed = l1[cotree_mask][:, cotree_mask]
    min_eig = torch.linalg.eigvalsh(l1_fixed).min()

    assert min_eig > 0.0


def test_l1_positive_definite_relative_bc(
    finer_flat_annulus_mesh: SimplicialMesh, device
):
    mesh = finer_flat_annulus_mesh.to(device)

    l0 = laplacian_k(mesh, k=0, component="up")
    dual_l0 = laplacian_k(mesh, k=0, component="up", dual=True)

    cotree_mask = compute_cotree_mask(
        dual_topo_laplacian_0=dual_l0,
        cbd_1=mesh.cbd[1],
        edge_rel_bc_mask=mesh.bd_edge_mask,
    )
    tree_mask = compute_tree_mask(
        topo_laplacian_0=l0,
        canon_edges=mesh.edges,
        vert_rel_bc_mask=mesh.bd_vert_mask,
        cotree_mask=cotree_mask,
    )

    assert not (tree_mask & cotree_mask).any()

    # Note that the boundary edges also need to be excluded from the free edges
    # since they are fixed by the boundary condition.
    free_edge_mask = (~mesh.bd_edge_mask) & (tree_mask | cotree_mask)

    l1 = (star_1(mesh) @ laplacian_1(mesh)).to_dense()
    l1_fixed = l1[free_edge_mask][:, free_edge_mask]
    min_eig = torch.linalg.eigvalsh(l1_fixed).min()

    assert min_eig > 0.0


def test_l1_down_positive_definite_relative_bc(
    finer_flat_annulus_mesh: SimplicialMesh, device
):
    mesh = finer_flat_annulus_mesh.to(device)

    l0 = laplacian_k(mesh, k=0, component="up")
    tree_mask = compute_tree_mask(
        topo_laplacian_0=l0,
        canon_edges=mesh.edges,
        vert_rel_bc_mask=mesh.bd_vert_mask,
    )

    free_edge_mask = (~mesh.bd_edge_mask) & tree_mask

    l1 = (star_1(mesh) @ laplacian_1(mesh)).to_dense()
    l1_fixed = l1[free_edge_mask][:, free_edge_mask]
    min_eig = torch.linalg.eigvalsh(l1_fixed).min()

    assert min_eig > 0.0


def test_l1_up_positive_definite_relative_bc(
    finer_flat_annulus_mesh: SimplicialMesh, device
):
    mesh = finer_flat_annulus_mesh.to(device)

    dual_l0 = laplacian_k(mesh, k=0, component="up", dual=True)

    cotree_mask = compute_cotree_mask(
        dual_topo_laplacian_0=dual_l0,
        cbd_1=mesh.cbd[1],
        edge_rel_bc_mask=mesh.bd_edge_mask,
    )

    free_edge_mask = (~mesh.bd_edge_mask) & cotree_mask

    l1 = (star_1(mesh) @ laplacian_1(mesh)).to_dense()
    l1_fixed = l1[free_edge_mask][:, free_edge_mask]
    min_eig = torch.linalg.eigvalsh(l1_fixed).min()

    assert min_eig > 0.0


def test_l1_gauge_fix_condition_number(finer_flat_annulus_mesh: SimplicialMesh, device):
    """
    In general, using a maximum spanning tree weighted by edge mass should result
    in a gauge-fixed linear system with better condition number than a purely
    topological spanning tree.
    """
    mesh = finer_flat_annulus_mesh.to(device)

    l0 = laplacian_k(mesh, k=0, component="up")
    dual_l0 = laplacian_k(mesh, k=0, component="up", dual=True)
    l1 = laplacian_1(mesh).to_dense()

    # First, perform tree-cotree decomposition with no geometric edge weights
    cotree_mask = compute_cotree_mask(
        dual_topo_laplacian_0=dual_l0,
        cbd_1=mesh.cbd[1],
        edge_rel_bc_mask=mesh.bd_edge_mask,
    )
    tree_mask = compute_tree_mask(
        topo_laplacian_0=l0,
        canon_edges=mesh.edges,
        vert_rel_bc_mask=mesh.bd_vert_mask,
        cotree_mask=cotree_mask,
    )
    free_edge_mask = (~mesh.bd_edge_mask) & (tree_mask | cotree_mask)
    l1_fixed_topo = l1[free_edge_mask][:, free_edge_mask]

    # Next, perform the same decomposition but using the hodge star for edge weights
    s1 = star_1(mesh)
    inv_s1 = s1.inv

    cotree_mask = compute_cotree_mask(
        dual_topo_laplacian_0=dual_l0,
        cbd_1=mesh.cbd[1],
        inv_mass_1=inv_s1,
        edge_rel_bc_mask=mesh.bd_edge_mask,
    )
    tree_mask = compute_tree_mask(
        topo_laplacian_0=l0,
        canon_edges=mesh.edges,
        mass_1=s1,
        vert_rel_bc_mask=mesh.bd_vert_mask,
        cotree_mask=cotree_mask,
    )
    free_edge_mask = (~mesh.bd_edge_mask) & (tree_mask | cotree_mask)
    l1_fixed_geo = l1[free_edge_mask][:, free_edge_mask]

    topo_cond = torch.linalg.cond(l1_fixed_topo)
    geo_cond = torch.linalg.cond(l1_fixed_geo)

    assert topo_cond > geo_cond

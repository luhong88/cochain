import torch
from jaxtyping import Float

from cochain.boundary.selection import BoundarySelection
from cochain.complex import SimplicialMesh
from cochain.hodge import laplacians
from cochain.metric.tri import tri_hodge_stars
from cochain.sparse.decoupled_tensor import SparseDecoupledTensor
from cochain.topology.spanning_tree import compute_cotree_mask, compute_tree_mask
from cochain.topology.topo_laplacians import laplacian_k


def _weak_laplacian_1_grad_div(
    tri_mesh: SimplicialMesh,
    bc_sel: BoundarySelection | None = None,
) -> Float[SparseDecoupledTensor, "edge edge"]:
    """Grad-div component of the weak 1-Laplacian for a tri mesh."""
    if bc_sel is None:
        return laplacians.weak_down_laplacian(
            cbd_km1=tri_mesh.cbd[0],
            mass_k=tri_hodge_stars.star_1(tri_mesh),
            inv_mass_km1=tri_hodge_stars.star_0(tri_mesh).inv,
        )
    else:
        vert_mask, edge_mask = bc_sel.retained_mask[:2]
        return laplacians.weak_down_laplacian(
            cbd_km1=tri_mesh.cbd[0].submatrix(edge_mask, vert_mask).tensor,
            mass_k=tri_hodge_stars.star_1(tri_mesh).submatrix(edge_mask).tensor,
            inv_mass_km1=(
                tri_hodge_stars.star_0(tri_mesh).submatrix(vert_mask).tensor.inv
            ),
        )


def _weak_laplacian_1_curl_curl(
    tri_mesh: SimplicialMesh,
    bc_sel: BoundarySelection | None = None,
) -> Float[SparseDecoupledTensor, "edge edge"]:
    """Curl-curl component of the weak 1-Laplacian for a tri mesh."""
    if bc_sel is None:
        return laplacians.weak_up_laplacian(
            tri_mesh.cbd[1], tri_hodge_stars.star_2(tri_mesh)
        )
    else:
        edge_mask, tri_mask = bc_sel.retained_mask[1:3]
        return laplacians.weak_up_laplacian(
            tri_mesh.cbd[1].submatrix(tri_mask, edge_mask).tensor,
            tri_hodge_stars.star_2(tri_mesh).submatrix(tri_mask).tensor,
        )


def _weak_laplacian_1(
    tri_mesh: SimplicialMesh,
    bc_sel: BoundarySelection | None = None,
) -> Float[SparseDecoupledTensor, "edge edge"]:
    """Weak hybrid 1-Laplacian for a tri mesh."""
    return SparseDecoupledTensor.assemble(
        _weak_laplacian_1_grad_div(tri_mesh, bc_sel),
        _weak_laplacian_1_curl_curl(tri_mesh, bc_sel),
    )


def test_cbd_0_rank(finer_flat_annulus_mesh: SimplicialMesh, device):
    """
    Check that the rank of the down 1-Laplacian is equal to that of the 0-cbd.

    The tree decomposition identifies the same number of edges as the rank of the
    down 1-Laplacian of a tri mesh, which is the same as the rank of the 0-coboundary
    operator (assuming that the mesh has no relative boundary conditions).
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
    Check that the rank of the up 1-Laplacian is equal to that of the 1-cbd.

    The cotree decomposition identifies the same number of edges as the rank of the
    up 1-Laplacian of a tri mesh, which is the same as the rank of the 1-coboundary
    operator (assuming that the mesh has no boundaries).
    """
    mesh = icosphere_mesh.to(device)

    cbd_1 = mesh.cbd[1]
    cbd_1_rank = torch.linalg.matrix_rank(cbd_1.to_dense())

    dual_l0 = laplacian_k(mesh, k=0, component="up", dual_complex=True)
    cotree_mask = compute_cotree_mask(
        dual_topo_laplacian_0=dual_l0,
        cbd_1=cbd_1,
    )
    n_cotree_edge = torch.sum(cotree_mask)

    torch.testing.assert_close(cbd_1_rank, n_cotree_edge)


def test_l1_positive_definite_no_bc(icosphere_mesh: SimplicialMesh, device):
    """Check that L-1 after tree-cotree gauge fixing is SPD on a mesh with no boundaries."""
    mesh = icosphere_mesh.to(device)

    l0 = laplacian_k(mesh, k=0, component="up")
    dual_l0 = laplacian_k(mesh, k=0, component="up", dual_complex=True)

    cotree_mask = compute_cotree_mask(dual_topo_laplacian_0=dual_l0, cbd_1=mesh.cbd[1])
    tree_mask = compute_tree_mask(
        topo_laplacian_0=l0, canon_edges=mesh.edges, cotree_mask=cotree_mask
    )

    # Test b_0 and b_2 betti numbers
    assert tree_mask.sum() == mesh.n_verts - 1
    assert cotree_mask.sum() == mesh.n_tris - 1

    # Test tree-cotree disjointness
    assert not (tree_mask & cotree_mask).any()

    # Test positive definiteness
    free_edge_mask = tree_mask | cotree_mask
    l1 = _weak_laplacian_1(mesh)
    l1_fixed = l1.submatrix(free_edge_mask).tensor.to_dense()
    min_eig = torch.linalg.eigvalsh(l1_fixed).min()

    assert min_eig > 0.0


def test_l1_down_positive_definite_no_bc(icosphere_mesh: SimplicialMesh, device):
    """Check that L-1 down after tree gauge fixing is SPD on a mesh with no boundaries."""
    mesh = icosphere_mesh.to(device)

    l0 = laplacian_k(mesh, k=0, component="up")
    tree_mask = compute_tree_mask(topo_laplacian_0=l0, canon_edges=mesh.edges)

    l1_down = _weak_laplacian_1_grad_div(mesh)
    l1_fixed = l1_down.submatrix(tree_mask).tensor.to_dense()
    min_eig = torch.linalg.eigvalsh(l1_fixed).min()

    assert min_eig > 0.0


def test_l1_up_positive_definite_no_bc(icosphere_mesh: SimplicialMesh, device):
    """Check that L-1 up after cotree gauge fixing is SPD on a mesh with no boundaries."""
    mesh = icosphere_mesh.to(device)

    dual_l0 = laplacian_k(mesh, k=0, component="up", dual_complex=True)
    cotree_mask = compute_cotree_mask(dual_topo_laplacian_0=dual_l0, cbd_1=mesh.cbd[1])

    l1_up = _weak_laplacian_1_curl_curl(mesh)
    l1_fixed = l1_up.submatrix(cotree_mask).tensor.to_dense()
    min_eig = torch.linalg.eigvalsh(l1_fixed).min()

    assert min_eig > 0.0


def test_l1_positive_definite_absolute_bc(
    finer_flat_annulus_mesh: SimplicialMesh, device
):
    """Check that L-1 after tree-cotree guage fixing is SPD on a mesh with absolute BC."""
    mesh = finer_flat_annulus_mesh.to(device)

    l0 = laplacian_k(mesh, k=0, component="up")
    dual_l0 = laplacian_k(mesh, k=0, component="up", dual_complex=True)

    cotree_mask = compute_cotree_mask(dual_topo_laplacian_0=dual_l0, cbd_1=mesh.cbd[1])
    tree_mask = compute_tree_mask(
        topo_laplacian_0=l0, canon_edges=mesh.edges, cotree_mask=cotree_mask
    )

    assert not (tree_mask & cotree_mask).any()

    # Absolute BC does not require any further masking.
    free_edge_mask = tree_mask | cotree_mask

    l1 = _weak_laplacian_1(mesh)
    l1_fixed = l1.submatrix(free_edge_mask).tensor.to_dense()
    min_eig = torch.linalg.eigvalsh(l1_fixed).min()

    assert min_eig > 0.0


def test_l1_down_positive_definite_absolute_bc(
    finer_flat_annulus_mesh: SimplicialMesh, device
):
    """Check that L-1 down after tree guage fixing is SPD on a mesh with absolute BC."""
    mesh = finer_flat_annulus_mesh.to(device)

    l0 = laplacian_k(mesh, k=0, component="up")
    tree_mask = compute_tree_mask(topo_laplacian_0=l0, canon_edges=mesh.edges)

    l1 = _weak_laplacian_1_grad_div(mesh)
    # Absolute BC does not require any further masking.
    l1_fixed = l1.submatrix(tree_mask).tensor.to_dense()
    min_eig = torch.linalg.eigvalsh(l1_fixed).min()

    assert min_eig > 0.0


def test_l1_up_positive_definite_absolute_bc(
    finer_flat_annulus_mesh: SimplicialMesh, device
):
    """Check that L-1 up after cotree guage fixing is SPD on a mesh with absolute BC."""
    mesh = finer_flat_annulus_mesh.to(device)

    dual_l0 = laplacian_k(mesh, k=0, component="up", dual_complex=True)
    cotree_mask = compute_cotree_mask(dual_topo_laplacian_0=dual_l0, cbd_1=mesh.cbd[1])

    l1 = _weak_laplacian_1_curl_curl(mesh)
    # Absolute BC does not require any further masking.
    l1_fixed = l1.submatrix(cotree_mask).tensor.to_dense()
    min_eig = torch.linalg.eigvalsh(l1_fixed).min()

    assert min_eig > 0.0


def test_l1_positive_definite_relative_bc(
    finer_flat_annulus_mesh: SimplicialMesh, device
):
    """Check that L-1 after tree-cotree guage fixing is SPD on a mesh with relative BC."""
    mesh = finer_flat_annulus_mesh.to(device)
    bc_sel = BoundarySelection.from_relative_bc(mesh)

    l0 = laplacian_k(mesh, k=0, component="up")
    dual_l0 = laplacian_k(mesh, k=0, component="up", dual_complex=True)

    cotree_mask = compute_cotree_mask(
        dual_topo_laplacian_0=dual_l0,
        cbd_1=mesh.cbd[1],
        bc_sel=bc_sel,
    )
    tree_mask = compute_tree_mask(
        topo_laplacian_0=l0,
        canon_edges=mesh.edges,
        bc_sel=bc_sel,
        cotree_mask=cotree_mask,
    )

    assert not (tree_mask & cotree_mask).any()

    # Note that the boundary edges also need to be excluded from the free edges
    # since they are fixed by the relative boundary condition.
    free_edge_mask = (tree_mask | cotree_mask)[bc_sel.retained_mask[1]]

    l1 = _weak_laplacian_1(mesh, bc_sel)
    l1_fixed = l1.submatrix(free_edge_mask).tensor.to_dense()
    min_eig = torch.linalg.eigvalsh(l1_fixed).min()

    assert min_eig > 0.0


def test_l1_down_positive_definite_relative_bc(
    finer_flat_annulus_mesh: SimplicialMesh, device
):
    """Check that L-1 down after tree guage fixing is SPD on a mesh with relative BC."""
    mesh = finer_flat_annulus_mesh.to(device)
    bc_sel = BoundarySelection.from_relative_bc(mesh)

    l0 = laplacian_k(mesh, k=0, component="up")
    tree_mask = compute_tree_mask(
        topo_laplacian_0=l0,
        canon_edges=mesh.edges,
        bc_sel=bc_sel,
    )

    free_edge_mask = tree_mask[bc_sel.retained_mask[1]]

    l1 = _weak_laplacian_1_grad_div(mesh, bc_sel)
    l1_fixed = l1.submatrix(free_edge_mask).tensor.to_dense()
    min_eig = torch.linalg.eigvalsh(l1_fixed).min()

    assert min_eig > 0.0


def test_l1_up_positive_definite_relative_bc(
    finer_flat_annulus_mesh: SimplicialMesh, device
):
    """Check that L-1 up after cotree guage fixing is SPD on a mesh with relative BC."""
    mesh = finer_flat_annulus_mesh.to(device)
    bc_sel = BoundarySelection.from_relative_bc(mesh)

    dual_l0 = laplacian_k(mesh, k=0, component="up", dual_complex=True)
    cotree_mask = compute_cotree_mask(
        dual_topo_laplacian_0=dual_l0,
        cbd_1=mesh.cbd[1],
        bc_sel=bc_sel,
    )

    free_edge_mask = cotree_mask[bc_sel.retained_mask[1]]

    l1 = _weak_laplacian_1_curl_curl(mesh, bc_sel)
    l1_fixed = l1.submatrix(free_edge_mask).tensor.to_dense()
    min_eig = torch.linalg.eigvalsh(l1_fixed).min()

    assert min_eig > 0.0


def test_l1_weighted_gauge_fix(finer_flat_annulus_mesh: SimplicialMesh, device):
    """
    Check weighted and unweighted gauge fixes on the true relative Laplacian.

    Mass weighting is a conditioning heuristic; improvement is not guaranteed.
    """
    mesh = finer_flat_annulus_mesh.to(device)
    bc_sel = BoundarySelection.from_relative_bc(mesh)

    l0 = laplacian_k(mesh, k=0, component="up")
    dual_l0 = laplacian_k(mesh, k=0, component="up", dual_complex=True)
    weak_l1 = _weak_laplacian_1(mesh, bc_sel)

    # First, perform tree-cotree decomposition with no geometric edge weights
    cotree_mask = compute_cotree_mask(
        dual_topo_laplacian_0=dual_l0,
        cbd_1=mesh.cbd[1],
        bc_sel=bc_sel,
    )
    tree_mask = compute_tree_mask(
        topo_laplacian_0=l0,
        canon_edges=mesh.edges,
        bc_sel=bc_sel,
        cotree_mask=cotree_mask,
    )
    free_edge_mask = (tree_mask | cotree_mask)[bc_sel.retained_mask[1]]
    l1_fixed_topo = weak_l1.submatrix(free_edge_mask).tensor.to_dense()

    # Next, perform the same decomposition but using the hodge star for edge weights
    s1 = tri_hodge_stars.star_1(mesh)
    inv_s1 = s1.inv

    cotree_mask = compute_cotree_mask(
        dual_topo_laplacian_0=dual_l0,
        cbd_1=mesh.cbd[1],
        inv_mass_1=inv_s1,
        bc_sel=bc_sel,
    )
    tree_mask = compute_tree_mask(
        topo_laplacian_0=l0,
        canon_edges=mesh.edges,
        mass_1=s1,
        bc_sel=bc_sel,
        cotree_mask=cotree_mask,
    )
    free_edge_mask = (tree_mask | cotree_mask)[bc_sel.retained_mask[1]]
    l1_fixed_geo = weak_l1.submatrix(free_edge_mask).tensor.to_dense()

    assert l1_fixed_topo.shape == l1_fixed_geo.shape
    assert torch.linalg.eigvalsh(l1_fixed_topo).min() > 0.0
    assert torch.linalg.eigvalsh(l1_fixed_geo).min() > 0.0

    topo_cond = torch.linalg.cond(l1_fixed_topo)
    geo_cond = torch.linalg.cond(l1_fixed_geo)

    assert torch.isfinite(topo_cond) and torch.isfinite(geo_cond)
    assert topo_cond > geo_cond

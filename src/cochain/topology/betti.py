__all__ = ["compute_betti_numbers"]

import torch

from ..boundary.selection import BoundarySelection
from ..complex import SimplicialMesh
from .morse import compute_morse_complex
from .spanning_tree import compute_cotree_mask, compute_tree_mask
from .topo_laplacians import laplacian_k


def _tri_manifold_betti_via_trees(
    tri_mesh: SimplicialMesh, bc_sel: BoundarySelection | None = None
) -> tuple[int, int, int]:
    """
    Compute the first three real (relative) Betti numbers for a tri mesh.

    This function uses the tree-cotree decomposition to compute the Betti numbers.
    The input mesh is assumed to be an orientable manifold tri mesh.
    """
    if bc_sel is None:
        bc_sel = BoundarySelection.from_absolute_bc(tri_mesh)

    # Construct the dual spanning tree over the dual 1-skeleton, which identifies
    # |F_retained| - b_2 edges, which is the rank of d_1 over the retained complex.
    dual_l0 = laplacian_k(tri_mesh, k=0, component="up", dual_complex=True)
    cotree = compute_cotree_mask(dual_l0, tri_mesh.cbd[1], bc_sel=bc_sel)

    rank_1 = int(cotree.sum().item())
    b_2 = bc_sel.n_retained_tris - rank_1

    # Construct the spanning tree on the 1-skeleton, which identifies
    # |V_retained| - b_0 edges, which is the rank of d_0 over the retained complex.
    l0 = laplacian_k(tri_mesh, k=0, component="up")
    tree = compute_tree_mask(l0, tri_mesh.edges, cotree_mask=cotree, bc_sel=bc_sel)

    rank_0 = int(tree.sum().item())
    b_0 = bc_sel.n_retained_verts - rank_0

    # Note that the Euler characteristic χ can be expressed in two ways for a
    # tri mesh:
    #     χ = |V_retained| - |E_retained| + |F_retained|
    #     χ = b_0 - b_1 + b_2
    # it follows that b_1 = |E_retained| - (|V_retained| - b_0) - (|F_retained| - b_2).
    b_1 = bc_sel.n_retained_edges - rank_0 - rank_1

    return b_0, b_1, b_2


def _betti_via_morse(
    mesh: SimplicialMesh, bc_sel: BoundarySelection | None = None
) -> tuple[int, int, int]:
    """
    Compute the first three real (relative) Betti numbers for a tri/tet mesh.

    This function uses the discrete Morse complex to compute the Betti numbers,
    and works on arbitrary immersed meshes.
    """
    morse_cbd, crit_splx = compute_morse_complex(mesh, bc_sel=bc_sel)

    n_crit_splx = [splx.size(0) for splx in crit_splx]

    cbd_dense = [cbd.to_dense() for cbd in morse_cbd]
    cbd_rank = [int(torch.linalg.matrix_rank(cbd).item()) for cbd in cbd_dense]

    # b_0 = |K_0| - rank(d_0)
    b_0 = n_crit_splx[0] - cbd_rank[0]
    # b_1 = |K_1| - rank(d_0) - rank(d_1)
    b_1 = n_crit_splx[1] - cbd_rank[0] - cbd_rank[1]
    # b_2 = |K_2| - rank(d_1) - rank(d_2)
    b_2 = n_crit_splx[2] - cbd_rank[1] - cbd_rank[2]

    return b_0, b_1, b_2


def compute_betti_numbers(
    mesh: SimplicialMesh,
    bc_sel: BoundarySelection | None = None,
    orientable_manifold: bool = False,
) -> tuple[int, int, int]:
    r"""
    Compute the first three Betti numbers over the real numbers.

    Parameters
    ----------
    mesh
        A tri or tet mesh.
    bc_sel
        An optional BoundarySelection object, used to mark the tangential
        boundary subcomplex of the mesh. If provided, compute Betti numbers of the
        relative complex. If set to `None`, assume that the mesh either has no
        boundaries or is subject to absolute boundary conditions.
    orientable_manifold
        If the input mesh is 2D and an orientable manifold, compute the Betti
        numbers via tree-cotree decomposition; otherwise compute the Betti numbers
        via discrete Morse complex. Note that, for a 2D, non-orientable manifold,
        the tree-cotree decomposition computes the Betti numbers over the
        $\mathbb{Z}_2$ field. Orientability and manifold assumptions are not
        explicitly checked.

    Returns
    -------
    b_0
        The dimension of real degree-zero (relative) cohomology.
    b_1
        The dimension of real degree-one (relative) cohomology.
    b_2
        The dimension of real degree-two (relative) cohomology.
    """
    if orientable_manifold and (mesh.dim == 2):
        return _tri_manifold_betti_via_trees(mesh, bc_sel=bc_sel)
    else:
        return _betti_via_morse(mesh, bc_sel=bc_sel)

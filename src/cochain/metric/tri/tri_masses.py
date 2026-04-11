__all__ = ["mass_0", "mass_1", "mass_2"]

import torch
from einops import einsum, repeat
from jaxtyping import Float
from torch import Tensor

from ...complex import SimplicialMesh
from ...sparse.decoupled_tensor import SparseDecoupledTensor
from ...utils.faces import enumerate_local_faces
from ._tri_geometry import (
    compute_bc_grad_dots,
    compute_tri_areas,
)
from .tri_hodge_stars import star_2


def mass_0(tri_mesh) -> Float[SparseDecoupledTensor, "vert vert"]:
    r"""
    Compute the consistent mass matrix for discrete 0-forms.

    Parameters
    ----------
    tri_mesh
        A tri mesh.

    Returns
    -------
    [vert, vert]
        The mass matrix.

    Notes
    -----
    The mass matrix $M$ for discrete 0-forms is defined element-wise as the inner
    product

    $$
    M_{ij} = \int_\Omega W_i(x)W_j(x)\,dA
    $$

    where $W_i(x)$ is the Whitney 0-form basis function associated with vertex $i$,
    which is simply the barycentric coordinate function $\lambda_i(x)$. These basis
    functions are also known as the Lagrange elements or linear hat functions.

    Mass-lumping of the $M_0$ matrix via row-sums is equivalent to the barycentric
    Hodge star-0 matrix implemented in `star_0()`; note that this equivalence
    is not true in general for $M_k$'s for $k > 0$.
    """
    tri_areas = compute_tri_areas(tri_mesh.vert_coords, tri_mesh.tris)

    # Using the magic formula, the integral for M_ij can be solved analytically
    # for each triangle. If i = j, M_ij = A/12; if i != j, M_ij = A/24. This
    # then defines a local 3x3 mass-0 matrix that can be scattered to construct
    # the global mass-0 matrix.
    ref_local_mass_0 = ((torch.ones(3, 3) + torch.eye(3)) / 12.0).to(
        dtype=tri_mesh.dtype, device=tri_mesh.device
    )
    local_mass_0: Float[Tensor, "tri 3 3"] = einsum(
        tri_areas, ref_local_mass_0, "tri, vert_1 vert_2 -> tri vert_1 vert_2"
    )

    # Enumerate the global vert idx pairs for each local 3x3 mass-0 matrix and
    # flatten it for scattering.
    r_idx = repeat(tri_mesh.tris, "tri vert_1 -> (tri vert_1 vert_2)", vert_2=3)
    c_idx = repeat(tri_mesh.tris, "tri vert_2 -> (tri vert_1 vert_2)", vert_1=3)

    mass = torch.sparse_coo_tensor(
        indices=torch.vstack((r_idx, c_idx)),
        values=local_mass_0.flatten(),
        size=(tri_mesh.n_verts, tri_mesh.n_verts),
    ).coalesce()

    return SparseDecoupledTensor.from_tensor(mass)


def mass_1(tri_mesh: SimplicialMesh) -> Float[SparseDecoupledTensor, "edge edge"]:
    r"""
    Compute the consistent mass matrix for discrete 1-forms.

    Parameters
    ----------
    tri_mesh
        A tri mesh.

    Returns
    -------
    [edge, edge]
        The mass matrix.

    Notes
    -----
    The mass matrix $M$ for discrete 1-forms is defined element-wise as the inner
    product

    $$M_{ij,kl} = \int_\Omega \left<W_{ij}(x), W_{kl}(x)\right>\,dA$$

    where $W_{ij}(x)$ is the Whitney 1-form basis function associated with the edge
    $ij$, which is defined (using its sharp) as

    $$W_{ij}(x) = \lambda_i(x)\nabla\lambda_j(x) - \lambda_j(x)\nabla\lambda_i(x)$$

    These basis functions are also known as the lowest-order Nédélec edge elements of
    the first kind.
    """
    # To evaluate the integral for M_ij,kl per triangle, note that the integrand
    # can be decomposed into 4 terms of the form λ_i*λ_k*<∇λ_j, ∇λ_l>, where the
    # part λ_i*λ_k is coordinate-dependent while the <∇λ_j, ∇λ_l> part is a constant.
    # Therefore, we start by computing the coordinate-dependent part (bc_ints)
    # and the coordinate-independent part (bc_grad_dots) separately and locally
    # within each triangle.

    # For each tri, compute all <∇λ_j, ∇λ_l> for each pair of vertices (j, l).
    tri_areas, bc_grad_dots = compute_bc_grad_dots(tri_mesh.vert_coords, tri_mesh.tris)

    # For each tri, compute all area integrals of λ_i*λ_k for each pair of vertices
    # (i, k). As shown in the mass_0() function, this integral evaluates to
    # A*(1 + δ_ik)/12, where δ is the Kronecker delta.
    ref_bc_poly_ints = (torch.ones((3, 3)) + torch.eye(3)).to(
        dtype=tri_mesh.dtype, device=tri_mesh.device
    )
    bc_poly_ints = einsum(
        tri_areas / 12.0, ref_bc_poly_ints, "tri, v_1 v_2 -> tri v_1 v_2"
    )

    # For each tri, each pair of edges ij and kl contributes four terms to the
    # inner products of the Whitney 1-form basis functions:
    #
    # M_ij,kl = I_ik*D_jl - I_il*D_jk - I_jk*D_il + I_jl*D_ik
    #
    # where I refers to bc_poly_ints and D refers to bc_grad_dots. Note that this
    # expression is skew-symmetric wrt the edge orientations (W_ji,kl = -W_ij,kl
    # and W_ij,lk = -W_ij,lk, however, the expression is invariant to switching
    # the two edge indices: M_ij,kl = M_kl,ij.

    # Enumerate all 3 unique local edges per tri.
    all_local_edges = enumerate_local_faces(
        splx_dim=2, face_dim=1, device=tri_mesh.device
    )

    # Given the 3 unique edges per tri, there are six unique edge pairs (e0, e1),
    # 00, 01, 02, 11, 12, 22.
    e0_local_edges = [0, 0, 0, 1, 1, 2]
    e0 = all_local_edges[e0_local_edges]

    e1_local_edges = [0, 1, 2, 1, 2, 2]
    e1 = all_local_edges[e1_local_edges]

    # For each edge pair, identify the start and end vertex local indices of e0,
    # which maps onto indices i, j; similarly, for e1, indices k, l.
    i, j = e0.unbind(dim=-1)
    k, l = e1.unbind(dim=-1)

    # Sum together the 4 contribution terms for each of the 6 edge pairs.
    whitney_dot = torch.zeros(
        (tri_mesh.n_tris, 6), dtype=tri_mesh.dtype, device=tri_mesh.device
    )
    whitney_dot.add_(bc_poly_ints[:, i, k] * bc_grad_dots[:, j, l])
    whitney_dot.sub_(bc_poly_ints[:, i, l] * bc_grad_dots[:, j, k])
    whitney_dot.sub_(bc_poly_ints[:, j, k] * bc_grad_dots[:, i, l])
    whitney_dot.add_(bc_poly_ints[:, j, l] * bc_grad_dots[:, i, k])

    # Before scattering the per-tri edge pair values to form the global mass-1
    # matrix, the edge pair values need to be parity-corrected to account for
    # the difference in local edge orientation relative to the global canonical
    # edges.
    global_edge_parity = tri_mesh.edge_faces.parity
    whitney_dot_parity_corrected = (
        whitney_dot
        * global_edge_parity[:, e0_local_edges]
        * global_edge_parity[:, e1_local_edges]
    )

    # Note that the way the local mass-1 matrix is constructed currently is
    # asymmetric (e.g., it accounts for the edge pair 12, but not 21). Therefore,
    # an additional index mapping step is required to generate symmetrized values
    # and indices for the final global mass-1 matrix construction. Specifically,
    # given the enumeration of local edge pairs 00, 01, 02, 11, 12, and 22,
    asym_to_sym_map = [0, 1, 2, 1, 3, 4, 2, 4, 5]
    whitney_dot_sym = whitney_dot_parity_corrected[:, asym_to_sym_map]

    # Find the global edge indices of the all edge pairs to prepare for scatter.
    global_edge_idx = tri_mesh.edge_faces.idx
    r_idx = repeat(global_edge_idx, "tri e_1 -> (tri e_1 e_2)", e_2=3)
    c_idx = repeat(global_edge_idx, "tri e_2 -> (tri e_1 e_2)", e_1=3)
    idx_coo = torch.vstack((r_idx, c_idx))

    # Assemble the global mass-1 matrix.
    mass = torch.sparse_coo_tensor(
        indices=idx_coo,
        values=whitney_dot_sym.flatten(),
        size=(tri_mesh.n_edges, tri_mesh.n_edges),
    ).coalesce()

    return SparseDecoupledTensor.from_tensor(mass)


def mass_2(tri_mesh) -> Float[SparseDecoupledTensor, "tri tri"]:
    r"""
    Compute the consistent mass matrix for discrete 2-forms.

    Note that this function is equivalent to `star_2()`.

    Parameters
    ----------
    tri_mesh
        A tri mesh.

    Returns
    -------
    [tri, tri]
        The mass matrix.

    Notes
    -----
    The mass matrix $M$ for discrete 2-forms is defined element-wise as the inner
    product

    $$M_{ij} = \int_\Omega \left<W_i(x), W_j(x)\right>\,dA$$

    where $W_i(x)$ is the Whitney 2-form basis function associated with a triangle $i$,
    which is defined (using its sharp) as

    $$W_i(x) = 2 (\nabla\lambda_1(x) \times \nabla\lambda_2(x))$$

    Note that each triangle has only one basis function, and it is constant over the
    triangle.

    For a tri mesh, the mass-2 matrix is identical to the diagonal Hodge 2-star matrix.
    To see why, first note that $M$ is diagonal since the basis functions $W_i(x)$ is
    only nonzero over a single triangle. Furthermore, for any two vertices $i$ and $j$
    of a triangle,

    $$\|\nabla\lambda_i(x) \times \nabla\lambda_j(x)\| = 1/2A$$

    which implies that $M_{ij}= A^{-1}\delta_{ij}$.
    """
    return star_2(tri_mesh)

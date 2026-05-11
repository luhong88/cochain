__all__ = ["mass_0", "mass_1", "mass_2", "mass_3"]

import torch
from einops import einsum, repeat
from jaxtyping import Float
from torch import Tensor

from ...complex import SimplicialMesh
from ...sparse.decoupled_tensor import DiagDecoupledTensor, SparseDecoupledTensor
from ...utils.faces import enumerate_local_faces
from ._tet_geometry import (
    compute_bc_grad_dots,
    compute_tet_signed_vols,
)


def mass_0(tet_mesh: SimplicialMesh) -> Float[SparseDecoupledTensor, "vert vert"]:
    r"""
    Compute the consistent mass matrix for discrete 0-forms.

    Parameters
    ----------
    tet_mesh
        A tet mesh.

    Returns
    -------
    [vert, vert]
        The mass matrix.

    Notes
    -----
    The mass matrix $M$ for discrete 0-forms is defined element-wise as the inner
    product

    $$
    M_{ij} = \int_\Omega W_i(x)W_j(x)\,dV
    $$

    where $W_i(x)$ is the Whitney 0-form basis function associated with vertex $i$,
    which is simply the barycentric coordinate function $\lambda_i(x)$. These basis
    functions are also known as the Lagrange elements or linear hat functions.

    Mass-lumping of the $M_0$ matrix via row-sums is equivalent to the barycentric
    Hodge star-0 matrix implemented in `star_0()`; note that this equivalence
    is not true in general for $M_k$'s for $k > 0$.
    """
    tet_vols = torch.abs(
        compute_tet_signed_vols(
            tet_mesh.vert_coords,
            tet_mesh.tets,
        )
    )

    # Using the magic formula, the integral for M_ij can be solved analytically
    # for each triangle. If i = j, M_ij = V/20; if i != j, M_ij = V/40. This
    # then defines a local 4x4 mass-0 matrix that can be scattered to construct
    # the global mass-0 matrix.
    ref_local_mass_0 = ((torch.ones(4, 4) + torch.eye(4)) / 20.0).to(
        dtype=tet_mesh.dtype, device=tet_mesh.device
    )
    local_mass_0: Float[Tensor, "tet 4 4"] = einsum(
        tet_vols, ref_local_mass_0, "tet, vert_1 vert_2 -> tet vert_1 vert_2"
    )

    # Enumerate the global vert idx pairs for each local 4x4 mass-0 matrix and
    # flatten it for scattering.
    r_idx = repeat(tet_mesh.tets, "tet vert_1 -> (tet vert_1 vert_2)", vert_2=4)
    c_idx = repeat(tet_mesh.tets, "tet vert_2 -> (tet vert_1 vert_2)", vert_1=4)

    mass = tet_mesh._sparse_coalesced_matrix(
        operator="tet_mass_0",
        indices=torch.vstack((r_idx.flatten(), c_idx.flatten())),
        values=local_mass_0.flatten(),
        size=(tet_mesh.n_verts, tet_mesh.n_verts),
    )

    return mass


def mass_1(tet_mesh: SimplicialMesh) -> Float[SparseDecoupledTensor, "edge edge"]:
    r"""
    Compute the consistent mass matrix for discrete 1-forms.

    Parameters
    ----------
    tet_mesh
        A tet mesh.

    Returns
    -------
    [edge, edge]
        The mass matrix.

    Notes
    -----
    The mass matrix $M$ for discrete 1-forms is defined element-wise as the inner
    product

    $$M_{ij,kl} = \int_\Omega \left<W_{ij}(x), W_{kl}(x)\right>\,dV$$

    where $W_{ij}(x)$ is the Whitney 1-form basis function associated with the edge
    $ij$, which is defined (using its sharp) as

    $$W_{ij}(x) = \lambda_i(x)\nabla\lambda_j(x) - \lambda_j(x)\nabla\lambda_i(x)$$

    These basis functions are also known as the lowest-order Nédélec edge elements of
    the first kind.
    """
    # To evaluate the integral for M_ij,kl per tet, note that the integrand
    # can be decomposed into 4 terms of the form λ_i*λ_k*<∇λ_j, ∇λ_l>, where the
    # part λ_i*λ_k is coordinate-dependent while the <∇λ_j, ∇λ_l> part is a constant.
    # Therefore, we start by computing the coordinate-dependent part (bc_ints)
    # and the coordinate-independent part (bc_grad_dots) separately and locally
    # within each tet.

    # For each tet, compute all <∇λ_j, ∇λ_l> for each pair of vertices (j, l).
    tet_signed_vols, bc_grad_dots = compute_bc_grad_dots(
        tet_mesh.vert_coords, tet_mesh.tets
    )
    tet_unsigned_vols = torch.abs(tet_signed_vols)

    # For each tri, compute all volume integrals of λ_i*λ_k for each pair of vertices
    # (i, k). As shown in the mass_0() function, this integral evaluates to
    # V*(1 + δ_ik)/20, where δ is the Kronecker delta.
    ref_bc_poly_ints = (torch.ones((4, 4)) + torch.eye(4)).to(
        dtype=tet_mesh.dtype, device=tet_mesh.device
    )
    bc_poly_ints = einsum(
        tet_unsigned_vols / 20.0, ref_bc_poly_ints, "tet, v_1 v_2 -> tet v_1 v_2"
    )

    # For each tet, each pair of edges ij and kl contributes four terms to the
    # inner products of the Whitney 1-form basis functions:
    #
    # M_ij,kl = I_ik*D_jl - I_il*D_jk - I_jk*D_il + I_jl*D_ik
    #
    # where I refers to bc_poly_ints and D refers to bc_grad_dots. Note that this
    # expression is skew-symmetric wrt the edge orientations (W_ji,kl = -W_ij,kl
    # and W_ij,lk = -W_ij,lk, however, the expression is invariant to switching
    # the two edge indices: M_ij,kl = M_kl,ij.

    # Enumerate all 6 unique local edges per tet.
    all_local_edges = enumerate_local_faces(
        splx_dim=3, face_dim=1, device=tet_mesh.device
    )

    # Given the 6 unique edges per tet, there are 21 unique edge pairs (e0, e1),
    # 00, 01, 02, 03, 04, 05, 11, 12, 13, 14, 15, 22, 23, 24, 25, 33, 34, 35, 44, 45, 55
    e0_local_edges = [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 4, 4, 5]
    e0 = all_local_edges[e0_local_edges]

    e1_local_edges = [0, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 2, 3, 4, 5, 3, 4, 5, 4, 5, 5]
    e1 = all_local_edges[e1_local_edges]

    # For each edge pair, identify the start and end vertex local indices of e0,
    # which maps onto indices i, j; similarly, for e1, indices k, l.
    i, j = e0.unbind(dim=-1)
    k, l = e1.unbind(dim=-1)

    # Sum together the 4 contribution terms for each of the 21 edge pairs.
    whitney_dot = torch.zeros(
        (tet_mesh.n_tets, 21), dtype=tet_mesh.dtype, device=tet_mesh.device
    )
    whitney_dot.add_(bc_poly_ints[:, i, k] * bc_grad_dots[:, j, l])
    whitney_dot.sub_(bc_poly_ints[:, i, l] * bc_grad_dots[:, j, k])
    whitney_dot.sub_(bc_poly_ints[:, j, k] * bc_grad_dots[:, i, l])
    whitney_dot.add_(bc_poly_ints[:, j, l] * bc_grad_dots[:, i, k])

    # Before scattering the per-tet edge pair values to form the global mass-1
    # matrix, the edge pair values need to be parity-corrected to account for
    # the difference in local edge orientation relative to the global canonical
    # edges.
    global_edge_parity = tet_mesh.edge_faces.parity
    whitney_dot_parity_corrected = (
        whitney_dot
        * global_edge_parity[:, e0_local_edges]
        * global_edge_parity[:, e1_local_edges]
    )

    # Note that the way the local mass-1 matrix is constructed currently is
    # asymmetric (e.g., it accounts for the edge pair 12, but not 21). Therefore,
    # an additional index mapping step is required to generate symmetrized values
    # and indices for the final global mass-1 matrix construction.
    # fmt: off
    asym_to_sym_map = [
      # 01  02  03  12  13  23
        0,  1,  2,  3,  4,  5,  # 01
        1,  6,  7,  8,  9,  10, # 02
        2,  7,  11, 12, 13, 14, # 03
        3,  8,  12, 15, 16, 17, # 12
        4,  9,  13, 16, 18, 19, # 13
        5,  10, 14, 17, 19, 20  # 23
    ]
    # fmt: on
    whitney_dot_sym = whitney_dot_parity_corrected[:, asym_to_sym_map]

    # Find the global edge indices of the all edge pairs to prepare for scatter.
    global_edge_idx = tet_mesh.edge_faces.idx
    r_idx = repeat(global_edge_idx, "tet e_1 -> (tet e_1 e_2)", e_2=6)
    c_idx = repeat(global_edge_idx, "tet e_2 -> (tet e_1 e_2)", e_1=6)
    idx_coo = torch.vstack((r_idx, c_idx))

    # Assemble the global mass-1 matrix.
    mass = tet_mesh._sparse_coalesced_matrix(
        operator="tet_mass_1",
        indices=idx_coo,
        values=whitney_dot_sym.flatten(),
        size=(tet_mesh.n_edges, tet_mesh.n_edges),
    )

    return mass


def mass_2(tet_mesh: SimplicialMesh) -> Float[SparseDecoupledTensor, "tri tri"]:
    r"""
    Compute the consistent mass matrix for discrete 2-forms.

    Parameters
    ----------
    tet_mesh
        A tet mesh.

    Returns
    -------
    [tri, tri]
        The mass matrix.

    Notes
    -----
    The mass matrix $M$ for discrete 2-forms is defined element-wise as the inner
    product

    $$M_{ijk,rst} = \int_\Omega \left<W_{ijk}(x), W_{rst}(x)\right>\,dV$$

    where $W_{ijk}(x)$ is the Whitney 2-form basis function associated with the tri
    $ijk$, which is defined (using its sharp) as

    $$
    W_{ijk}(x) = 2(
        \lambda_i\nabla\lambda_j\times\nabla\lambda_k
        - \lambda_j\nabla\lambda_k\times\nabla\lambda_i
        + \lambda_k\nabla\lambda_i\times\nabla\lambda_j)
    $$

    These basis functions are also known as the lowest-order Raviart-Thomas-Nédélec
    face elements. Note that, in this function, instead of working with
    this differential form definition of the basis functions, we use the polynomial
    representation

    $$W_i(x) = \frac{x - v_i}{3V}$$

    where we associate each basis function with the vertex opposite to the triangle;
    i.e., $W_i(x)$ is the basis associated with triangle $jkl$ and $v_i$ is the
    coordinate of vertex $i$. These two representations are mathematically equivalent.
    """
    tet_vert_coords: Float[Tensor, "tet 4 3"] = tet_mesh.vert_coords[tet_mesh.tets]

    tet_signed_vols: Float[Tensor, " tet"] = compute_tet_signed_vols(
        tet_mesh.vert_coords, tet_mesh.tets
    )

    # For each tet ijkl and each 2-form basis function, associate basis function
    # with the opposite vertex; i.e., W_i(x) = (x - v_i)/3V is the basis associated
    # with the tri jkl (oriented such that its right-hand rule area normal vector
    # is outward facing). Using the fact that ∑_i λ_i = 1 and x = ∑_i λ_i*v_i, we
    # can rewrite W_i(x) as ∑_k λ_k*e_ik/3V. Then, the inner product between the
    # basis functions is reduced to
    #
    # int[<W_i(x), W_j(x)> dV] = ∑_kl int[λ_k*λ_l dV]*<e_ik,e_jl>/(9V^2)
    #
    # The first term, int[λ_k*λ_l dV], can be evaluated using the magic formula.
    # For the second term, define the local gram matrix g_ij = <v_i, v_j>, then
    #
    # <e_ik,e_jl> = g_kl - g_kj - g_il + g_ij

    # For each tri, compute all volume integrals of λ_k*λ_l for each pair of vertices
    # (k, l). As shown in the mass_0() function, this integral evaluates to
    # V*(1 + δ_ik)/20, where δ is the Kronecker delta. Here, we ignore the volume
    # term since it will get canceled out by the 1/9V^2 term in the integral.
    ints: Float[Tensor, "4 4"] = (torch.ones((4, 4)) + torch.eye(4)).to(
        dtype=tet_mesh.dtype, device=tet_mesh.device
    ) / 20.0

    # Compute the local gram matrix.
    g = einsum(
        tet_vert_coords,
        tet_vert_coords,
        "tet vert_1 coord, tet vert_2 coord -> tet vert_1 vert_2",
    )

    # For each tet and vertex pair (k, l), expand out ∑_kl int[λ_k*λ_l dV]*<e_ik,e_jl>
    # using the gram matrix as
    #
    # ∑_kl I_kl*g_kl - ∑_kl I_kl*g_kj - ∑_kl I_kl*g_il + ∑_kl I_kl*g_ij
    #
    # where we have used I_kl to denote int[λ_k*λ_l dV].

    whitney_dot = torch.zeros(
        (tet_mesh.n_tets, 4, 4), dtype=tet_mesh.dtype, device=tet_mesh.device
    )
    whitney_dot.add_(repeat(torch.einsum("kl,tkl->t", ints, g), "t -> t i j", i=4, j=4))
    whitney_dot.sub_(repeat(torch.einsum("kl,tkj->tj", ints, g), "t j -> t i j", i=4))
    whitney_dot.sub_(repeat(torch.einsum("kl,til->ti", ints, g), "t i -> t i j", j=4))
    whitney_dot.add_(torch.einsum("kl,tij->tij", ints, g))

    # Scale the dot product by the 1/9V term.
    whitney_dot_scaled = whitney_dot / (9.0 * tet_signed_vols.view(-1, 1, 1))

    # Note that, for the purpose of computing the mass-2 matrix, the definition
    # of tri faces is different from the global, "canonical" definitions used in
    # utils.faces.enumerate_local_faces(). More specifically, the canonical
    # tri faces of a tet is defined locally as
    #
    # [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]]
    #
    # However, here, we enumerate the tri faces by first enumerating the vertices
    # that oppose the tri faces, and the tri vertices are enumerated such that
    # its (right-hand rule) area normal is outward-facing (for positively oriented
    # tets; this is why whitney_dot_scaled uses the signed volume, to correct for
    # the flipped area normal direction in a negatively oriented tet). More
    # specifically, this results in the following tri face definitions:
    #
    # [[1, 2, 3], [0, 3, 2], [0, 1, 3], [0, 2, 1]]
    #
    # Therefore, the canonical 2-face definitions and orientations need to be
    # adjusted specifically for this function.
    global_tri_idx = torch.flip(tet_mesh.tri_faces.idx, dims=(-1,))

    boundary_sign = torch.tensor(
        [[1.0, -1.0, 1.0, -1.0]],
        dtype=tet_mesh.dtype,
        device=tet_mesh.device,
    )
    global_tri_parity = boundary_sign * torch.flip(
        tet_mesh.tri_faces.parity, dims=(-1,)
    )

    # Mapping the local basis function to the global basis function requires
    # correction of both the triangle face orientation as well as the tet orientations.
    whitney_inner_prod_signed: Float[Tensor, "tet tri=4 tri=4"] = einsum(
        whitney_dot_scaled,
        global_tri_parity,
        global_tri_parity,
        "tet tri_1 tri_2, tet tri_1, tet tri_2 -> tet tri_1 tri_2",
    )

    # Assemble the mass matrix by scattering the inner products according to the
    # tri indices.
    r_idx = repeat(global_tri_idx, "tet tri_1 -> (tet tri_1 tri_2)", tri_2=4)
    c_idx = repeat(global_tri_idx, "tet tri_2 -> (tet tri_1 tri_2)", tri_1=4)
    idx_coo = torch.vstack((r_idx, c_idx))

    n_tris = tet_mesh.n_tris

    mass = tet_mesh._sparse_coalesced_matrix(
        operator="tet_mass_2",
        indices=idx_coo,
        values=whitney_inner_prod_signed.flatten(),
        size=(n_tris, n_tris),
    )

    return mass


def mass_3(tet_mesh: SimplicialMesh) -> Float[DiagDecoupledTensor, "tet tet"]:
    r"""
    Compute the consistent mass matrix for discrete 3-forms.

    Parameters
    ----------
    tet_mesh
        A tet mesh.

    Returns
    -------
    [tet, tet]
        The mass matrix.

    Notes
    -----
    The mass matrix $M$ for discrete 3-forms is defined element-wise as the inner
    product

    $$M_{ij} = \int_\Omega W_i(x)W_j(x)\,dV$$

    where $W_i(x)$ is the Whitney 3-form basis function associated with a tet $i$,
    which is defined as $W_i(x) = 1/V$. Note that each tet has only one basis
    function, and it is constant over the tet.
    """
    return DiagDecoupledTensor.from_tensor(
        1.0 / torch.abs(compute_tet_signed_vols(tet_mesh.vert_coords, tet_mesh.tets))
    )

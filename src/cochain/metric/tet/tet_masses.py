__all__ = ["mass_0", "mass_1", "mass_2", "mass_3"]

import torch
from einops import einsum, rearrange, reduce, repeat
from jaxtyping import Float, Integer
from torch import LongTensor, Tensor

from ...complex import SimplicialMesh
from ...sparse.decoupled_tensor import DiagDecoupledTensor, SparseDecoupledTensor
from ._tet_geometry import (
    compute_bc_grad_dots,
    compute_d_tet_signed_vols_d_vert_coords,
    compute_tet_signed_vols,
    whitney_2_form_inner_prods,
)


def mass_0(tet_mesh) -> Float[SparseDecoupledTensor, "vert vert"]:
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
    M_{ij} = \int_\Omega W_i(x)W_j(x)\,dA
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
    # for each triangle. If i = j, M_ij = A/20; if i != j, M_ij = A/40. This
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
    r_idx = repeat(tet_mesh.tets, "tri vert_1 -> (tri vert_1 vert_2)", vert_2=4)
    c_idx = repeat(tet_mesh.tets, "tri vert_2 -> (tri vert_1 vert_2)", vert_1=4)

    mass = torch.sparse_coo_tensor(
        indices=torch.vstack((r_idx.flatten(), c_idx.flatten())),
        values=local_mass_0.flatten(),
        size=(tet_mesh.n_verts, tet_mesh.n_verts),
    ).coalesce()

    return SparseDecoupledTensor.from_tensor(mass)


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

    $$M_{ij,kl} = \int_\Omega \left<W_{ij}(x), W_{kl}(x)\right>\,dA$$

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

    # For each tri, compute all <∇λ_j, ∇λ_l> for each pair of vertices (j, l).

    # For each tet ijkl, compute all pairwise inner products of the barycentric
    # coordinate gradients wrt each pair of vertices.
    tet_signed_vols, bc_grad_dots = compute_bc_grad_dots(
        tet_mesh.vert_coords, tet_mesh.tets
    )

    # For each tet ijkl, compute all pairwise integrals of the barycentric coordinates;
    # i.e., int[lambda_i(p)lambda_j(p)dvol_ijkl]. Using the "magic formula", this
    # integral is vol_ijkl*(1 + delta_ij)/20, where delta is the Kronecker delta
    # function.
    bary_coords_int: Float[Tensor, "tet 4 4"] = torch.abs(
        tet_signed_vols.view(-1, 1, 1) / 20.0
    ) * (
        torch.ones(
            (tet_mesh.n_tets, 4, 4), dtype=tet_mesh.dtype, device=tet_mesh.device
        )
        + torch.eye(4, dtype=tet_mesh.dtype, device=tet_mesh.device).view(1, 4, 4)
    )

    # For each tet ijkl, each pair of its edges e1=xy and e2=pq contributes the
    # following term to the mass matrix element M[e1,e2]:
    #
    #         W_xy,pq = I_xp*D_yq - I_xq*D_yp - I_yp*D_xq + I_yq*D_xp
    #
    # Here, I is the barycentric integral (bary_coords_int) and D is the barycentric
    # gradient inner product (barry_coords_grad_dot). Note that, since this expression
    # is "skew-symmetric" wrt the edge orientations (W_yx,pq = -W_xy,pq and
    # W_xy,qp = -W_xy,pq), each non-canonical edge orientation also contributes
    # an overall negative sign.

    # Enumerate all unique edges via their vertex position in the tet.
    i, j, k, l = 0, 1, 2, 3
    unique_edges = torch.tensor(
        [[i, j], [i, k], [j, k], [j, l], [k, l], [i, l]],
        dtype=torch.int64,
        device=tet_mesh.device,
    )

    x_idx = unique_edges[:, 0][:, None]
    y_idx = unique_edges[:, 1][:, None]
    p_idx = unique_edges[:, 0][None, :]
    q_idx = unique_edges[:, 1][None, :]

    # For each tet, find all pairs of Whitney 1-form basis function inner products,
    # i.e., the W_xy,pq.
    whitney_inner_prod: Float[Tensor, "tet 6 6"] = (
        bary_coords_int[:, x_idx, p_idx] * bc_grad_dots[:, y_idx, q_idx]
        - bary_coords_int[:, x_idx, q_idx] * bc_grad_dots[:, y_idx, p_idx]
        - bary_coords_int[:, y_idx, p_idx] * bc_grad_dots[:, x_idx, q_idx]
        + bary_coords_int[:, y_idx, q_idx] * bc_grad_dots[:, x_idx, p_idx]
    )

    # For each tet and each unique edge pair, find the orientations of the edges
    # and their indices on the list of unique, canonical edges (tet_mesh.edges).
    # Note that, given the specific unique_edges definition used in this function
    # (related to the tet edge local ref frame definitions), the way the unique
    # 1-faces is enumerated in unique_edges differs from the canonical edge
    # definition used in utils.faces.enumerate_local_faces(), which gives
    #
    # [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]]
    #
    # Therefore, the canonical edge definitions need to be adjusted for use in
    # this function; however, note that, since unique_edges define 1-faces
    # with the same orientation as the canonical 1-faces, no adjustment of edge
    # orientation signs is required.
    canon_edge_perm = [0, 1, 3, 4, 5, 2]

    edge_faces = tet_mesh.edge_faces
    whitney_edge_signs = edge_faces.parity[:, canon_edge_perm]
    whitney_edges_idx = edge_faces.idx[:, canon_edge_perm]

    # Multiply the Whitney 1-form inner product by the edge orientation signs
    # to get the contribution from canonical edges.
    whitney_flat_signed: Float[Tensor, "tet 36"] = (
        whitney_inner_prod
        * whitney_edge_signs.view(-1, 1, 6)
        * whitney_edge_signs.view(-1, 6, 1)
    ).flatten(start_dim=-2)

    # Get the canonical edge index pairs for the Whitney 1-form inner products of
    # all 36 edge pairs per tet.
    whitney_flat_r_idx: Float[Tensor, " tet*36"] = (
        whitney_edges_idx.view(-1, 6, 1).expand(-1, 6, 6).flatten()
    )
    whitney_flat_c_idx: Float[Tensor, " tet*36"] = (
        whitney_edges_idx.view(-1, 1, 6).expand(-1, 6, 6).flatten()
    )

    # Assemble the mass matrix.
    mass = torch.sparse_coo_tensor(
        torch.vstack((whitney_flat_r_idx, whitney_flat_c_idx)),
        whitney_flat_signed.flatten(),
        (tet_mesh.n_edges, tet_mesh.n_edges),
    ).coalesce()

    return SparseDecoupledTensor.from_tensor(mass)


def mass_2(tet_mesh: SimplicialMesh) -> Float[SparseDecoupledTensor, "tri tri"]:
    """
    Compute the Galerkin triangle/2-form mass matrix.

    For each tet, each (canonical) triangle pairs xyz and rst and their associated
    Whitney 2-form basis functions W_xyz and W_rst contribute the inner product
    term int[W_xyz*W_rst*dV] to the mass matrix element M[xyz, rst], where

    W_xyz(p) = sign[xyz]*(p - v[-xyz])/3V

    Here, v[-xyz] is the coordinate vector of the vertex opposite to xyz. sign[xyz]
    is +1 whenever the triangle xyz satisfies the right-hand rule (i.e., the normal
    vector formed by the right hand points out of the tet), and -1 if not.
    """
    n_tris = tet_mesh.n_tris

    # Note that, for the purpose of computing the mass-2 matrix, the definition
    # of triangle faces is different from the global, "canonical" definitions
    # used in utils.faces.enumerate_local_faces(). More specifically, the canonical
    # triangle faces of a tet is defined locally as
    #
    # [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]]
    #
    # However, here, we enumerate the triangle faces by first enumerating the
    # vertices that oppose the triangle faces, and the triangle vertices are
    # enumerated such that its area normal is outward-facing (note that the way
    # the triangles are indexed here satisfies the right-hand rule for positively
    # oriented tets. More specifically, this results in the following triangle
    # face definitions:
    #
    # [[1, 2, 3], [0, 3, 2], [0, 1, 3], [0, 2, 1]]
    #
    # Therefore, the canonical 2-face definitions and orientations need to be
    # adjusted prior for this function.
    tri_face_idx = torch.flip(tet_mesh.tri_faces.idx, dims=(-1,))
    tri_face_orientations = torch.tensor(
        [[1.0, -1.0, 1.0, -1.0]], dtype=tet_mesh.dtype, device=tet_mesh.device
    ) * torch.flip(tet_mesh.tri_faces.parity, dims=(-1,))

    # First, compute the inner products of the Whitney 2-form basis functions.
    _, whitney_inner_prod_signed = whitney_2_form_inner_prods(
        tet_mesh.vert_coords, tet_mesh.tets, tri_face_orientations
    )

    # Assemble the mass matrix by scattering the inner products according to the
    # triangle indices.
    mass_idx = torch.vstack(
        (
            tri_face_idx.view(-1, 4, 1).expand(-1, 4, 4).flatten(),
            tri_face_idx.view(-1, 1, 4).expand(-1, 4, 4).flatten(),
        )
    )
    mass_val = whitney_inner_prod_signed.flatten()
    mass = torch.sparse_coo_tensor(mass_idx, mass_val, (n_tris, n_tris)).coalesce()

    return SparseDecoupledTensor.from_tensor(mass)


def mass_3(tet_mesh: SimplicialMesh) -> Float[DiagDecoupledTensor, "tet tet"]:
    """
    Compute the diagonal of the tet/3-form mass matrix, which is a diagonal matrix
    containing the inverse of the unsigned tet volumes, which is equivalent to
    the 3-star.
    """
    return DiagDecoupledTensor.from_tensor(
        1.0 / torch.abs(compute_tet_signed_vols(tet_mesh.vert_coords, tet_mesh.tets))
    )

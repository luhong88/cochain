from typing import Literal

import torch as t
from jaxtyping import Float

from ...complex import SimplicialComplex
from ...utils.linalg import diag_sp_mm, sp_diag_mm
from .tet_hodge_stars import star_1, star_2
from .tet_masses import mass_0, mass_1, mass_2, mass_3
from .tet_stiffness import stiffness_matrix

# For the weak k-Laplacian,
#
# S_k = (
#   M_k @ d_{k-1} @ inv_M_{k-1} @ d_{k-1}.T @ M_k +
#   d_k.T @ M_{k+1} @ d_k
# )
#
# The weak k-Laplacian (also known as the stiffness matrix) is related to the
# k-Laplacian via
#
# L_k = inv_M_k @ S_k


def weak_laplacian_0(
    tet_mesh: SimplicialComplex, method: Literal["cotan", "consistent"]
) -> Float[t.Tensor, "vert vert"]:
    """
    Compute the weak 0-Laplacian (vertex Laplacian)
    S0= d0.T @ M_1 @ d0

    If method is `cotan`, use the cotan formula, which is equivalent to using
    the lumped/diagonal circumcentric 1-star matrix for the mass matrix. If method
    is `consistent`, use the FEM mass-1 matrix instead.
    """
    match method:
        case "cotan":
            return stiffness_matrix(tet_mesh)
        case "consistent":
            d0 = tet_mesh.coboundary_0
            d0_T = d0.transpose(0, 1).coalesce()

            m_1 = mass_1(tet_mesh)

            return (d0_T @ m_1 @ d0).coalesce()
        case _:
            raise ValueError()


def weak_laplacian_1_div_grad(
    tet_mesh: SimplicialComplex,
) -> Float[t.Tensor, "edge edge"]:
    """
    Compute the div grad component of the weak 1-Laplacian
    M_1 @ d_0 @ inv_M_0 @ d_0.T @ M_1
    """
    d0 = tet_mesh.coboundary_0
    d0_T = d0.transpose(0, 1).coalesce()

    m_1 = mass_1(tet_mesh)
    inv_m_0 = 1.0 / mass_0(tet_mesh)

    return (m_1 @ d0 @ diag_sp_mm(inv_m_0, d0_T @ m_1)).coalesce()


def weak_laplacian_1_curl_curl(
    tet_mesh: SimplicialComplex,
) -> Float[t.Tensor, "edge edge"]:
    """
    Compute the curl curl component of the weak 1-Laplacian
    d_1.T @ M_2 @ d_1
    """
    d1 = tet_mesh.coboundary_1
    d1_T = d1.transpose(0, 1).coalesce()

    m_2 = mass_2(tet_mesh)

    return (d1_T @ m_2 @ d1).coalesce()


def weak_laplacian_1(tet_mesh: SimplicialComplex) -> Float[t.Tensor, "edge edge"]:
    """
    Compute the weak 1-Laplacian (edge/vector Laplacian)
    S1 = d_1.T @ M_2 @ d_1 + M_1 @ d_0 @ inv_M_0 @ d_0.T @ M_1
    """
    return (
        weak_laplacian_1_div_grad(tet_mesh) + weak_laplacian_1_curl_curl(tet_mesh)
    ).coalesce()


def weak_laplacian_2_div_grad(
    tet_mesh: SimplicialComplex,
    method: Literal[
        "dense",
        "solver",
        "inv_star",
        "row_sum",
    ],
) -> Float[t.Tensor, "tri tri"]:
    """
    Compute the div grad component of the weak 2-Laplacian
    M_2 @ d_1 @ inv_M_1 @ d_1.T @ M_2

    In general, the inverse of the sparse mass-1 matrix is not guaranteed to have
    a similar sparse structure. The `method` argument determines how inv_M_1 is
    handled:

    If method is `dense`, convert the mass-1 matrix to a dense tensor and compute
    its inverse using Cholesky decomposition.

    If method is `inv_star`, use the inverse of the barycentric 1-star in place
    of inv_M_1.

    If method is `row_sum`, coerce the mass-1 matrix into a diagonal matrix through
    row sum, and then take its inverse.
    """
    d1 = tet_mesh.coboundary_1
    d1_T = d1.transpose(0, 1).coalesce()

    match method:
        case "dense":
            m_1 = mass_1(tet_mesh).to_dense()
            m_1_cho = t.linalg.cholesky(m_1)
            inv_m_1 = t.cholesky_inverse(m_1_cho)
            m_2 = mass_2(tet_mesh)

            return m_2 @ d1 @ inv_m_1 @ d1_T @ m_2

        case "inv_star":
            m_1 = mass_1(tet_mesh)
            inv_m_1 = 1.0 / star_1(tet_mesh)
            m_2 = mass_2(tet_mesh)

            return (m_2 @ d1 @ diag_sp_mm(inv_m_1, d1_T @ m_2)).coalesce()

        case "row_sum":
            m_1 = mass_1(tet_mesh)
            inv_m_1 = 1.0 / t.sum(m_1, dim=-1).to_dense()
            m_2 = mass_2(tet_mesh)

            return (m_2 @ d1 @ diag_sp_mm(inv_m_1, d1_T @ m_2)).coalesce()

        case "solver":
            raise NotImplementedError()

        case _:
            raise ValueError()


def weak_laplacian_2_curl_curl(
    tet_mesh: SimplicialComplex,
) -> Float[t.Tensor, "tri tri"]:
    """
    Compute the curl curl component of the weak 1-Laplacian
    d_2.T @ M_3 @ d_2
    """
    d2 = tet_mesh.coboundary_2
    d2_T = d2.transpose(0, 1).coalesce()

    m_3 = mass_3(tet_mesh)

    return (d2_T @ diag_sp_mm(m_3, d2)).coalesce()


def weak_laplacian_2(
    tet_mesh: SimplicialComplex,
    method: Literal[
        "dense",
        "solver",
        "inv_star",
        "row_sum",
    ],
) -> Float[t.Tensor, "tri tri"]:
    """
    Compute the weak 2-Laplacian (face Laplacian)
    S2 = d_2.T @ M_3 @ d_2 + M_2 @ d_1 @ inv_M_1 @ d_1.T @ M_2

    If method is `solver`, returns a mixed finite element method solver function.
    """
    if method == "solver":
        raise NotImplementedError()

    elif method in ["dense", "inv_star", "row_sum"]:
        curl_curl = weak_laplacian_2_curl_curl(tet_mesh)
        div_grad = weak_laplacian_2_div_grad(tet_mesh, method)

        return div_grad + curl_curl

    else:
        raise ValueError()


def weak_laplacian_3(
    tet_mesh: SimplicialComplex,
    method: Literal[
        "dense",
        "solver",
        "inv_star",
        "row_sum",
    ],
) -> Float[t.Tensor, "tri tri"]:
    """
    Compute the weak 3-Laplacian (tet Laplacian)
    M_3 @ d_2 @ inv_M_2 @ d_2.T @ M_3

    In general, the inverse of the sparse mass-2 matrix is not guaranteed to have
    a similar sparse structure. The `method` argument determines how inv_M_2 is
    handled:

    If method is `dense`, convert the mass-2 matrix to a dense tensor and compute
    its inverse using Cholesky decomposition.

    If method is `solver`, returns a mixed finite element method solver function.

    If method is `inv_star`, use the inverse of the barycentric 2-star in place
    of inv_M_2.

    If method is `row_sum`, coerce the mass-2 matrix into a diagonal matrix through
    row sum, and then take its inverse.
    """
    d2 = tet_mesh.coboundary_2
    d2_T = d2.transpose(0, 1).coalesce()

    match method:
        case "dense":
            m_2 = mass_2(tet_mesh).to_dense()
            inv_m_2 = t.cholesky_inverse(m_2)
            m_3 = mass_3(tet_mesh)

            return diag_sp_mm(m_3, d2) @ inv_m_2 @ sp_diag_mm(d2_T, m_3)

        case "solver":
            raise NotImplementedError()

        case "inv_star":
            m_2 = mass_2(tet_mesh)
            inv_m_2 = 1.0 / star_2(tet_mesh)
            m_3 = mass_3(tet_mesh)

            return (
                diag_sp_mm(m_3, d2) @ diag_sp_mm(inv_m_2, sp_diag_mm(d2_T, m_3))
            ).coalesce()

        case "row_sum":
            m_2 = mass_2(tet_mesh)
            inv_m_2 = 1.0 / t.sum(m_2, dim=-1).to_dense()
            m_3 = mass_3(tet_mesh)

            return (
                diag_sp_mm(m_3, d2) @ diag_sp_mm(inv_m_2, sp_diag_mm(d2_T, m_3))
            ).coalesce()

        case _:
            raise ValueError()

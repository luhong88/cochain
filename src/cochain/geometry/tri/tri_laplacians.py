from typing import Literal

import torch as t
from jaxtyping import Float

from ...complex import SimplicialComplex
from ...utils.linalg import diag_sp_mm, sp_diag_mm
from .tri_hodge_stars import star_0, star_1, star_2
from .tri_stiffness import stiffness_matrix

# Laplacian_k = (
#   d_{k-1} @ inv_star_{k-1} @ d_{k-1}.T @ star_k +
#   inv_star_k @ d_k.T @ star_{k+1} @ d_k
# )
#
# or, equivalently,
#
# Laplacian_k = d_{k-1} @ codiff_k + codiff_{k+1} @ d_k
#
# where
#
# codiff_k = inv_star_{k-1} @ d_{k-1}.T @ star_k
#
# Note here that d_k stands for the k-coboundary operator,
# while d_k.T stands for the k-boundary operator.


def codifferential_1(
    tri_mesh: SimplicialComplex,
    dual_complex: Literal["circumcentric", "barycentric"] = "circumcentric",
) -> Float[t.Tensor, "vert edge"]:
    """
    Compute the codifferential on 1-forms, `star_0_inv @ d0_T @ star_1`
    """
    d0_T = tri_mesh.coboundary_0.transpose(0, 1).coalesce()

    s0 = star_0(tri_mesh)
    s1 = star_1(tri_mesh, dual_complex)

    codiff_1 = diag_sp_mm(1.0 / s0, sp_diag_mm(d0_T, s1))

    return codiff_1


def codifferential_2(
    tri_mesh: SimplicialComplex,
    dual_complex: Literal["circumcentric", "barycentric"] = "circumcentric",
) -> Float[t.Tensor, "edge tri"]:
    """
    Compute the codifferential on 2-forms, `star_1_inv @ d1_T @ star_2`
    """
    d1_T = tri_mesh.coboundary_1.transpose(0, 1).coalesce()

    s1 = star_1(tri_mesh, dual_complex)
    s2 = star_2(tri_mesh)

    codiff_2 = diag_sp_mm(1.0 / s1, sp_diag_mm(d1_T, s2))

    return codiff_2


def laplacian_0(tri_mesh: SimplicialComplex) -> Float[t.Tensor, "vert vert"]:
    """
    Compute the 0-Laplacian (vertex Laplacian).
    L0 = codiff_1 @ d0 = inv_star_0 @ d0.T @ star_1 @ d0

    This function uses the cotan weights to compute `d0.T @ star_1 @ d0`,
    i.e., the stiffness matrix.
    """
    return diag_sp_mm(1.0 / star_0(tri_mesh), stiffness_matrix(tri_mesh))


def laplacian_1_div_grad(
    tri_mesh: SimplicialComplex,
    codiff_1: Float[t.Tensor, "vert edge"] | None = None,
) -> Float[t.Tensor, "edge edge"]:
    """
    Compute the div grad component of the 1-Laplacian, `d0 @ codiff_1`.

    If codiff_1 is not provided, construct it using the circumcentric 1-star.
    """
    d0 = tri_mesh.coboundary_0

    if codiff_1 is None:
        codiff_1 = codifferential_1(tri_mesh)

    return (d0 @ codiff_1).coalesce()


def laplacian_1_curl_curl(
    tri_mesh: SimplicialComplex,
    codiff_2: Float[t.Tensor, "edge tri"] | None = None,
) -> Float[t.Tensor, "edge edge"]:
    """
    Computes the curl curl component of the 1-Laplacian, `codiff_2 @ d1`.

    If codiff_2 is not provided, construct it using the circumcentric 1-star.
    """
    d1 = tri_mesh.coboundary_1

    if codiff_2 is None:
        codiff_2 = codifferential_2(tri_mesh)

    return (codiff_2 @ d1).coalesce()


def laplacian_1(
    tri_mesh: SimplicialComplex,
    codiff_1: Float[t.Tensor, "vert edge"] | None = None,
    codiff_2: Float[t.Tensor, "edge tri"] | None = None,
) -> Float[t.Tensor, "edge edge"]:
    """
    Compute the 1-Laplacian (edge/vector Laplacian).
    L1 = (codiff_2 @ d1) + (d0 @ codiff_1)

    If the codifferentials are not provided, construct them using the circumcentric
    1-star.
    """
    laplacian_1 = (
        laplacian_1_div_grad(tri_mesh, codiff_1)
        + laplacian_1_curl_curl(tri_mesh, codiff_2)
    ).coalesce()

    return laplacian_1


def laplacian_2(
    tri_mesh: SimplicialComplex,
    codiff_2: Float[t.Tensor, "edge tri"] | None = None,
) -> Float[t.Tensor, "tri tri"]:
    """
    Compute the 2-Laplacian (face Laplacian).
    L2 = d1 @ codiff_2
    """
    d1 = tri_mesh.coboundary_1

    if codiff_2 is None:
        codiff_2 = codifferential_2(tri_mesh)

    return (d1 @ codiff_2).coalesce()

from typing import Literal

from jaxtyping import Float

from ...complex import SimplicialComplex
from ...sparse.operators import SparseOperator
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
    dual_complex: Literal["circumcentric", "barycentric"] = "barycentric",
) -> Float[SparseOperator, "vert edge"]:
    """
    Compute the codifferential on 1-forms, `star_0_inv @ d0_T @ star_1`
    """
    d0_T = tri_mesh.coboundary[0].T

    s0 = star_0(tri_mesh)
    s1 = star_1(tri_mesh, dual_complex)

    codiff_1 = s0.inv @ d0_T @ s1

    return codiff_1


def codifferential_2(
    tri_mesh: SimplicialComplex,
    dual_complex: Literal["circumcentric", "barycentric"] = "barycentric",
) -> Float[SparseOperator, "edge tri"]:
    """
    Compute the codifferential on 2-forms, `star_1_inv @ d1_T @ star_2`
    """
    d1_T = tri_mesh.coboundary[1].T

    s1 = star_1(tri_mesh, dual_complex)
    s2 = star_2(tri_mesh)

    codiff_2 = s1.inv @ d1_T @ s2

    return codiff_2


def laplacian_0(
    tri_mesh: SimplicialComplex,
    dual_complex: Literal["circumcentric", "barycentric"] = "barycentric",
    codiff_1: Float[SparseOperator, "vert edge"] | None = None,
) -> Float[SparseOperator, "vert vert"]:
    """
    Compute the 0-Laplacian (vertex Laplacian).
    L0 = codiff_1 @ d0 = inv_star_0 @ d0.T @ star_1 @ d0

    If 'codiff_1' is provided, construct L0 via codiff_1 @ d0; otherwise,
    construct L0 using the cotan Laplacian/stiffness matrix if 'dual_complex' is
    circumcentric, or the barycentric 1-star if 'dual_complex' is barycentric.
    """
    if codiff_1 is not None:
        return codiff_1 @ tri_mesh.coboundary[0]

    match dual_complex:
        case "circumcentric":
            return star_0(tri_mesh).inv @ stiffness_matrix(tri_mesh)

        case "barycentric":
            return codifferential_1(tri_mesh, dual_complex) @ tri_mesh.coboundary[0]

        case _:
            raise ValueError()


def laplacian_1_div_grad(
    tri_mesh: SimplicialComplex,
    dual_complex: Literal["circumcentric", "barycentric"] = "barycentric",
    codiff_1: Float[SparseOperator, "vert edge"] | None = None,
) -> Float[SparseOperator, "edge edge"]:
    """
    Compute the div grad component of the 1-Laplacian, `d0 @ codiff_1`.

    If 'codiff_1' is not provided, construct it using 1-star specified by 'dual_complex'.
    """
    d0 = tri_mesh.coboundary[0]

    if codiff_1 is None:
        codiff_1 = codifferential_1(tri_mesh, dual_complex)

    return d0 @ codiff_1


def laplacian_1_curl_curl(
    tri_mesh: SimplicialComplex,
    dual_complex: Literal["circumcentric", "barycentric"] = "barycentric",
    codiff_2: Float[SparseOperator, "edge tri"] | None = None,
) -> Float[SparseOperator, "edge edge"]:
    """
    Computes the curl curl component of the 1-Laplacian, `codiff_2 @ d1`.

    If codiff_2 is not provided, construct it using 1-star specified by 'dual_complex'.
    """
    d1 = tri_mesh.coboundary[1]

    if codiff_2 is None:
        codiff_2 = codifferential_2(tri_mesh, dual_complex)

    return codiff_2 @ d1


def laplacian_1(
    tri_mesh: SimplicialComplex,
    dual_complex: Literal["circumcentric", "barycentric"] = "barycentric",
    codiff_1: Float[SparseOperator, "vert edge"] | None = None,
    codiff_2: Float[SparseOperator, "edge tri"] | None = None,
) -> Float[SparseOperator, "edge edge"]:
    """
    Compute the 1-Laplacian (edge/vector Laplacian).
    L1 = (codiff_2 @ d1) + (d0 @ codiff_1)

    If the codifferentials are not provided, construct them using 1-star specified
    by 'dual_complex'.
    """
    laplacian_1 = SparseOperator.assemble(
        laplacian_1_div_grad(tri_mesh, dual_complex, codiff_1),
        laplacian_1_curl_curl(tri_mesh, dual_complex, codiff_2),
    )

    return laplacian_1


def laplacian_2(
    tri_mesh: SimplicialComplex,
    dual_complex: Literal["circumcentric", "barycentric"] = "barycentric",
    codiff_2: Float[SparseOperator, "edge tri"] | None = None,
) -> Float[SparseOperator, "tri tri"]:
    """
    Compute the 2-Laplacian (face Laplacian).
    L2 = d1 @ codiff_2

    If codiff_2 is not provided, construct it using 1-star specified by 'dual_complex'.
    """
    d1 = tri_mesh.coboundary[1]

    if codiff_2 is None:
        codiff_2 = codifferential_2(tri_mesh, dual_complex)

    return d1 @ codiff_2

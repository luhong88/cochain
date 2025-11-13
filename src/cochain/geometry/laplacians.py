import torch as t
from jaxtyping import Float, Integer

from ..complex import Simplicial2Complex
from .hodge_stars import _star_inv, star_0, star_1, star_2
from .stiffness import stiffness_matrix

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


def codifferential_1(simplicial_mesh) -> Float[t.Tensor, "vert edge"]:
    """
    Compute the codifferential on 1-forms, `star_0_inv @ d0_T @ star_1`
    """
    d0 = simplicial_mesh.coboundary_0.to_sparse_csr()
    d0_T = d0.transpose(0, 1).to_sparse_csr()

    s0 = star_0(simplicial_mesh)
    s1 = star_1(simplicial_mesh)
    s0_inv = _star_inv(s0)

    codiff_1 = s0_inv @ d0_T @ s1

    return codiff_1


def codifferential_2(simplicial_mesh) -> Float[t.Tensor, "edge tri"]:
    """
    Compute the codifferential on 2-forms, `star_1_inv @ d1_T @ star_2`
    """
    d1 = simplicial_mesh.coboundary_1.to_sparse_csr()
    d1_T = d1.transpose(0, 1).to_sparse_csr()

    s1 = star_1(simplicial_mesh)
    s2 = star_2(simplicial_mesh)
    s1_inv = _star_inv(s1)

    codiff_2 = s1_inv @ d1_T @ s2

    return codiff_2


def laplacian_0(simplicial_mesh: Simplicial2Complex) -> Float[t.Tensor, "vert vert"]:
    """
    Compute the 0-Laplacian (vertex Laplacian).
    L0 = codiff_1 @ d0 = inv_star_0 @ d0.T @ star_1 @ d0

    This function uses the cotan weights to compute `d0.T @ star_1 @ d0`,
    i.e., the stiffness matrix.
    """
    s0 = star_0(simplicial_mesh)
    s0_inv = _star_inv(s0)

    return s0_inv @ stiffness_matrix(simplicial_mesh)


def laplacian_1_div_grad(
    simplicial_mesh: Simplicial2Complex,
    codiff_1: Float[t.Tensor, "vert edge"] | None = None,
) -> Float[t.Tensor, "edge edge"]:
    """
    Compute the div grad component of the 1-Laplacian, `d0 @ codiff_1`.
    """
    d0 = simplicial_mesh.coboundary_0.to_sparse_csr()

    if codiff_1 is None:
        codiff_1 = codifferential_1(simplicial_mesh)

    return (d0 @ codiff_1).coalesce()


def laplacian_1_curl_curl(
    simplicial_mesh: Simplicial2Complex,
    codiff_2: Float[t.Tensor, "edge tri"] | None = None,
) -> Float[t.Tensor, "edge edge"]:
    """
    Computes the curl curl component of the 1-Laplacian, `codiff_2 @ d1`.
    """
    d1 = simplicial_mesh.coboundary_1.to_sparse_csr()

    if codiff_2 is None:
        codiff_2 = codifferential_2(simplicial_mesh)

    return (codiff_2 @ d1).coalesce()


def laplacian_1(
    simplicial_mesh: Simplicial2Complex,
    codiff_1: Float[t.Tensor, "vert edge"] | None = None,
    codiff_2: Float[t.Tensor, "edge tri"] | None = None,
) -> Float[t.Tensor, "edge edge"]:
    """
    Compute the 1-Laplacian (edge/vector Laplacian).
    L1 = (codiff_2 @ d1) + (d0 @ codiff_1)
    """
    laplacian_1 = (
        laplacian_1_div_grad(simplicial_mesh, codiff_1)
        + laplacian_1_curl_curl(simplicial_mesh, codiff_2)
    ).coalesce()

    return laplacian_1


def laplacian_2(
    simplicial_mesh: Simplicial2Complex,
    codiff_2: Float[t.Tensor, "edge tri"] | None = None,
) -> Float[t.Tensor, "tri tri"]:
    """
    Compute the 2-Laplacian (face Laplacian).
    L2 = d1 @ codiff_2
    """
    d1 = simplicial_mesh.coboundary_1.to_sparse_csr()

    if codiff_2 is None:
        codiff_2 = codifferential_2(simplicial_mesh)

    return (d1 @ codiff_2).coalesce()

from typing import Literal

from jaxtyping import Float

from ...complex import SimplicialMesh
from ...sparse.decoupled_tensor import SparseDecoupledTensor
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


def codiff_1(
    tri_mesh: SimplicialMesh,
    dual_complex: Literal["circumcentric", "barycentric"] = "barycentric",
) -> Float[SparseDecoupledTensor, "vert edge"]:
    r"""
    Compute the codifferential on discrete 1-forms.

    The 1-codifferential is defined as $\delta_1 = \star_0^{-1}d_0^T \star_1$,
    where $\star_k$ is the Hodge $k$-star and $d_k$ is the $k$-coboundary operator/
    discrete exterior derivative.

    Parameters
    ----------
    tri_mesh
        A tri mesh.
    dual_complex
        The type of dual complex over which to compute the Hodge 1-star.

    Returns
    -------
    [vert, edge]
        The codifferential operator.

    """
    d0_T = tri_mesh.cbd[0].T

    s0 = star_0(tri_mesh)
    s1 = star_1(tri_mesh, dual_complex)

    codiff_1 = s0.inv @ d0_T @ s1

    return codiff_1


def codiff_2(
    tri_mesh: SimplicialMesh,
    dual_complex: Literal["circumcentric", "barycentric"] = "barycentric",
) -> Float[SparseDecoupledTensor, "edge tri"]:
    r"""
    Compute the codifferential on discrete 2-forms.

    The 2-codifferential is defined as $\delta_2 = \star_1^{-1}d_1^T \star_2$,
    where $\star_k$ is the Hodge $k$-star and $d_k$ is the $k$-coboundary operator/
    discrete exterior derivative.

    Parameters
    ----------
    tri_mesh
        A tri mesh.
    dual_complex
        The type of dual complex over which to compute the Hodge 1-star.

    Returns
    -------
    [edge, tri]
        The codifferential operator.

    """
    d1_T = tri_mesh.cbd[1].T

    s1 = star_1(tri_mesh, dual_complex)
    s2 = star_2(tri_mesh)

    codiff_2 = s1.inv @ d1_T @ s2

    return codiff_2


def laplacian_0(
    tri_mesh: SimplicialMesh,
    dual_complex: Literal["circumcentric", "barycentric"] = "barycentric",
    codiff_1: Float[SparseDecoupledTensor, "vert edge"] | None = None,
) -> Float[SparseDecoupledTensor, "vert vert"]:
    r"""
    Compute the Hodge 0-Laplacian.

    The 0-Laplacian is defined as

    $$ L_0=\delta_1 d_0 = \star_0^{-1} d_0^T \star_1 d_0$$

    where $\delta_k$ is the $k$-codifferential, $\star_k$ is the Hodge $k$-star and
    $d_k$ is the $k$-coboundary operator/discrete exterior derivative. This operator
    is also sometimes known as the vertex Laplacian.

    Parameters
    ----------
    tri_mesh
        A tri mesh.
    dual_complex
        If `codiff_1` is `None`, this argument determines the type of dual complex
        over which to compute the Hodge 1-star. In particular, if the `dual_complex`
        is "circumcentric", $L_0$ is computed via $\star_0^{-1} S$, where $S$ is the
        stiffness matrix/cotan Laplacian.
    codiff_1 : [vert, edge]
        If provided, compute $L_0$ via $\delta_1 d_0$; if `None`, compute $L_0$ from the Hodge stars and coboundary operators.

    Returns
    -------
    [vert, vert]
        The Laplacian operator.
    """
    if codiff_1 is not None:
        return codiff_1 @ tri_mesh.cbd[0]

    match dual_complex:
        case "circumcentric":
            return star_0(tri_mesh).inv @ stiffness_matrix(tri_mesh)

        case "barycentric":
            return codiff_1(tri_mesh, dual_complex) @ tri_mesh.cbd[0]

        case _:
            raise ValueError("Unknown 'dual_complex' argument.")


def laplacian_1_grad_div(
    tri_mesh: SimplicialMesh,
    dual_complex: Literal["circumcentric", "barycentric"] = "barycentric",
    codiff_1: Float[SparseDecoupledTensor, "vert edge"] | None = None,
) -> Float[SparseDecoupledTensor, "edge edge"]:
    r"""
    Compute the grad-div component of the Hodge 1-Laplacian.

    The grad-div component of the 1-Laplacian, also known as the "down" Laplacian,
    is defined as

    $$ L_1^\text{down} = d_0 \delta_1$$

    where $\delta_k$ is the $k$-codifferential, and $d_k$ is the $k$-coboundary
    operator/discrete exterior derivative.

    Parameters
    ----------
    tri_mesh
        A tri mesh.
    dual_complex
        If `codiff_1` is `None`, this argument determines the type of dual complex
        over which to compute the Hodge 1-star.
    codiff_1 : [vert, edge]
        If provided, compute $L_1^\text{down}$ via $d_0 \delta_1$; if `None`, compute
        $L_1^\text{down}$ from the Hodge stars and coboundary operators.

    Returns
    -------
    [edge, edge]
        The Laplacian operator.
    """
    d0 = tri_mesh.cbd[0]

    if codiff_1 is None:
        codiff_1 = codiff_1(tri_mesh, dual_complex)

    return d0 @ codiff_1


def laplacian_1_curl_curl(
    tri_mesh: SimplicialMesh,
    dual_complex: Literal["circumcentric", "barycentric"] = "barycentric",
    codiff_2: Float[SparseDecoupledTensor, "edge tri"] | None = None,
) -> Float[SparseDecoupledTensor, "edge edge"]:
    r"""
    Compute the curl-curl component of the Hodge 1-Laplacian.

    The curl-curl component of the 1-Laplacian, also known as the "up" Laplacian,
    is defined as

    $$ L_1^\text{up} = \delta_2 d_1$$

    where $\delta_k$ is the $k$-codifferential, and $d_k$ is the $k$-coboundary
    operator/discrete exterior derivative.

    Parameters
    ----------
    tri_mesh
        A tri mesh.
    dual_complex
        If `codiff_2` is `None`, this argument determines the type of dual complex
        over which to compute the Hodge 1-star.
    codiff_2 : [edge, tri]
        If provided, compute $L_1^\text{up}$ via $\delta_2 d_1$; if `None`, compute
        $L_1^\text{up}$ from the Hodge stars and coboundary operators.

    Returns
    -------
    [edge, edge]
        The Laplacian operator.
    """
    d1 = tri_mesh.cbd[1]

    if codiff_2 is None:
        codiff_2 = codiff_2(tri_mesh, dual_complex)

    return codiff_2 @ d1


def laplacian_1(
    tri_mesh: SimplicialMesh,
    dual_complex: Literal["circumcentric", "barycentric"] = "barycentric",
    codiff_1: Float[SparseDecoupledTensor, "vert edge"] | None = None,
    codiff_2: Float[SparseDecoupledTensor, "edge tri"] | None = None,
) -> Float[SparseDecoupledTensor, "edge edge"]:
    r"""
    Compute the Hodge 1-Laplacian.

    The 1-Laplacian, is defined as

    $$ L_1 = \delta_2 d_1 + d_0 \delta_1$$

    where $\delta_k$ is the $k$-codifferential, and $d_k$ is the $k$-coboundary
    operator/discrete exterior derivative.

    Parameters
    ----------
    tri_mesh
        A tri mesh.
    dual_complex
        If `codiff_1` or `codiff_2` is `None`, this argument determines the type of
        dual complex over which to compute the Hodge 1-star.
    codiff_1 : [vert, edge]
        If provided, compute the $L_1^\text{down}$ component via $d_0 \delta_1$; if
        `None`, compute $L_1^\text{down}$ from the Hodge stars and coboundary operators.
    codiff_2 : [edge, tri]
        If provided, compute the $L_1^\text{up}$ component via $\delta_2 d_1$; if
        `None`, compute $L_1^\text{up}$ from the Hodge stars and coboundary operators.

    Returns
    -------
    [edge, edge]
        The Laplacian operator.
    """
    laplacian_1 = SparseDecoupledTensor.assemble(
        laplacian_1_grad_div(tri_mesh, dual_complex, codiff_1),
        laplacian_1_curl_curl(tri_mesh, dual_complex, codiff_2),
    )

    return laplacian_1


def laplacian_2(
    tri_mesh: SimplicialMesh,
    dual_complex: Literal["circumcentric", "barycentric"] = "barycentric",
    codiff_2: Float[SparseDecoupledTensor, "edge tri"] | None = None,
) -> Float[SparseDecoupledTensor, "tri tri"]:
    r"""
    Compute the Hodge 2-Laplacian.

    The 2-Laplacian, is defined as

    $$ L_2 = d_1 \delta_2$$

    where $\delta_k$ is the $k$-codifferential, and $d_k$ is the $k$-coboundary
    operator/discrete exterior derivative.  Note that, unlike $L_1$, $L_2$ on a
    tri mesh only contains a single "down" component.

    Parameters
    ----------
    tri_mesh
        A tri mesh.
    dual_complex
        If `codiff_2` is `None`, this argument determines the type of dual complex
        over which to compute the Hodge 1-star.
    codiff_2 : [edge, tri]
        If provided, compute $L_2$ via $d+1 \delta_2$; if  `None`, compute $L_2$
        from the Hodge stars and coboundary operators.

    Returns
    -------
    [tri, tri]
        The Laplacian operator.
    """
    d1 = tri_mesh.cbd[1]

    if codiff_2 is None:
        codiff_2 = codiff_2(tri_mesh, dual_complex)

    return d1 @ codiff_2

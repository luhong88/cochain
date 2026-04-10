from typing import Any, Literal

import torch
from jaxtyping import Float
from torch import Tensor

from ...complex import SimplicialMesh
from ...sparse.decoupled_tensor import SparseDecoupledTensor
from ...sparse.linalg.solvers._inv_sparse_operator import InvSparseOperator
from ._mixed_weak_laplacian_operator import MixedWeakLaplacianOperator
from .tet_hodge_stars import star_0, star_1, star_2
from .tet_masses import mass_1, mass_2, mass_3
from .tet_stiffness import stiffness_matrix

__all__ = [
    "weak_laplacian_0",
    "weak_laplacian_1_grad_div",
    "weak_laplacian_1_curl_curl",
    "weak_laplacian_1",
    "weak_laplacian_2_curl_curl",
    "weak_laplacian_2_grad_div",
    "weak_laplacian_2",
    "weak_laplacian_3",
]

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
    tet_mesh: SimplicialMesh, method: Literal["cotan", "consistent"]
) -> Float[SparseDecoupledTensor, "vert vert"]:
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
            d0 = tet_mesh.cbd[0]
            d0_T = d0.T

            m_1 = mass_1(tet_mesh)

            return d0_T @ m_1 @ d0
        case _:
            raise ValueError()


def weak_laplacian_1_grad_div(
    tet_mesh: SimplicialMesh,
) -> Float[SparseDecoupledTensor, "edge edge"]:
    """
    Compute the div grad component of the weak 1-Laplacian
    M_1 @ d_0 @ inv_M_0 @ d_0.T @ M_1
    """
    d0 = tet_mesh.cbd[0]
    d0_T = d0.T

    m_1 = mass_1(tet_mesh)
    inv_m_0 = star_0(tet_mesh).inv

    return m_1 @ d0 @ inv_m_0 @ d0_T @ m_1


def weak_laplacian_1_curl_curl(
    tet_mesh: SimplicialMesh,
) -> Float[SparseDecoupledTensor, "edge edge"]:
    """
    Compute the curl curl component of the weak 1-Laplacian
    d_1.T @ M_2 @ d_1
    """
    d1 = tet_mesh.cbd[1]
    d1_T = d1.T

    m_2 = mass_2(tet_mesh)

    return d1_T @ m_2 @ d1


def weak_laplacian_1(
    tet_mesh: SimplicialMesh,
) -> Float[SparseDecoupledTensor, "edge edge"]:
    """
    Compute the weak 1-Laplacian (edge/vector Laplacian)
    S1 = d_1.T @ M_2 @ d_1 + M_1 @ d_0 @ inv_M_0 @ d_0.T @ M_1
    """
    return SparseDecoupledTensor.assemble(
        weak_laplacian_1_grad_div(tet_mesh), weak_laplacian_1_curl_curl(tet_mesh)
    )


# TODO: update docstring to remove reference to cholesky
def weak_laplacian_2_curl_curl(
    tet_mesh: SimplicialMesh,
    method: Literal[
        "dense",
        "solver",
        "inv_star",
    ],
    solver_cls: InvSparseOperator | None = None,
    solver_init_kwargs: dict[str, Any] | None = None,
) -> (
    Float[Tensor, "tri tri"]
    | Float[SparseDecoupledTensor, "tri tri"]
    | Float[MixedWeakLaplacianOperator, " tri tri"]
):
    """
    Compute the curl curl component of the weak 2-Laplacian
    M_2 @ d_1 @ inv_M_1 @ d_1.T @ M_2

    In general, the inverse of the sparse mass-1 matrix is not guaranteed to have
    a similar sparse structure. The `method` argument determines how inv_M_1 is
    handled:

    If method is `dense`, convert the mass-1 matrix to a dense tensor and compute
    its inverse using Cholesky decomposition.

    If method is `inv_star`, use the inverse of the barycentric 1-star in place
    of inv_M_1.
    """
    d1 = tet_mesh.cbd[1]
    d1_T = d1.T

    m1 = mass_1(tet_mesh)
    m2 = mass_2(tet_mesh)

    match method:
        case "dense":
            return (m2 @ d1) @ torch.linalg.solve(m1.to_dense(), (d1_T @ m2).to_dense())

        case "inv_star":
            m1 = mass_1(tet_mesh)
            inv_m_1 = star_1(tet_mesh).inv
            m2 = mass_2(tet_mesh)

            return m2 @ d1 @ inv_m_1 @ d1_T @ m2

        case "solver":
            return MixedWeakLaplacianOperator(
                cbd_km1=d1,
                cbd_k=None,
                mass_km1=m1,
                mass_k=m2,
                mass_kp1=None,
                solver_cls=solver_cls,
                solver_init_kwargs=solver_init_kwargs,
            )

        case _:
            raise ValueError(f"Unknown 'method' argument ('{method}').")


def weak_laplacian_2_grad_div(
    tet_mesh: SimplicialMesh,
) -> Float[SparseDecoupledTensor, "tri tri"]:
    """
    Compute the div grad component of the weak 1-Laplacian
    d_2.T @ M_3 @ d_2
    """
    d2 = tet_mesh.cbd[2]
    d2_T = d2.T

    m_3 = mass_3(tet_mesh)

    return d2_T @ m_3 @ d2


def weak_laplacian_2(
    tet_mesh: SimplicialMesh,
    method: Literal[
        "dense",
        "solver",
        "inv_star",
    ],
    solver_cls: InvSparseOperator | None = None,
    solver_init_kwargs: dict[str, Any] | None = None,
) -> (
    Float[Tensor, "tri tri"]
    | Float[SparseDecoupledTensor, "tri tri"]
    | Float[MixedWeakLaplacianOperator, " tri tri"]
):
    """
    Compute the weak 2-Laplacian (face Laplacian)
    S2 = d_2.T @ M_3 @ d_2 + M_2 @ d_1 @ inv_M_1 @ d_1.T @ M_2

    If method is `solver`, returns a mixed finite element method solver function.
    """
    match method:
        case "solver":
            d1 = tet_mesh.cbd[1]
            d2 = tet_mesh.cbd[2]

            m1 = mass_1(tet_mesh)
            m2 = mass_2(tet_mesh)
            m3 = mass_3(tet_mesh)

            return MixedWeakLaplacianOperator(
                cbd_km1=d1,
                cbd_k=d2,
                mass_km1=m1,
                mass_k=m2,
                mass_kp1=m3,
                solver_cls=solver_cls,
                solver_init_kwargs=solver_init_kwargs,
            )

        case "dense" | "inv_star":
            curl_curl = weak_laplacian_2_curl_curl(tet_mesh, method)
            div_grad = weak_laplacian_2_grad_div(tet_mesh)

            match curl_curl:
                case SparseDecoupledTensor():
                    return SparseDecoupledTensor.assemble(div_grad, curl_curl)
                case Tensor():
                    return div_grad + curl_curl.to_dense()
                case _:
                    raise TypeError()

        case _:
            raise ValueError(f"Unknown 'method' argument ('{method}').")


# TODO: update docstring to remove reference to cholesky
def weak_laplacian_3(
    tet_mesh: SimplicialMesh,
    method: Literal[
        "dense",
        "solver",
        "inv_star",
    ],
    solver_cls: InvSparseOperator | None = None,
    solver_init_kwargs: dict[str, Any] | None = None,
) -> (
    Float[Tensor, "tet tet"]
    | Float[SparseDecoupledTensor, "tet tet"]
    | Float[MixedWeakLaplacianOperator, " tet tet"]
):
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
    """
    d2 = tet_mesh.cbd[2]
    d2_T = d2.T

    m2 = mass_2(tet_mesh)
    m3 = mass_3(tet_mesh)

    match method:
        case "dense":
            return (m3 @ d2) @ torch.linalg.solve(m2.to_dense(), (d2_T @ m3).to_dense())

        case "inv_star":
            m2 = mass_2(tet_mesh)
            inv_m2 = star_2(tet_mesh).inv
            m3 = mass_3(tet_mesh)

            return m3 @ d2 @ inv_m2 @ d2_T @ m3

        case "solver":
            return MixedWeakLaplacianOperator(
                cbd_km1=d2,
                cbd_k=None,
                mass_km1=m2,
                mass_k=m3,
                mass_kp1=None,
                solver_cls=solver_cls,
                solver_init_kwargs=solver_init_kwargs,
            )

        case _:
            raise ValueError(f"Unknown 'method' argument ('{method}').")

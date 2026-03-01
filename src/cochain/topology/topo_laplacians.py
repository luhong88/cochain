from typing import Literal

from jaxtyping import Float

from ..complex import SimplicialComplex
from ..sparse.operators import SparseOperator


def laplacian_k(
    sc: SimplicialComplex,
    *,
    k: int,
    component: Literal["up", "down", "full"],
    dual: bool = False,
) -> Float[SparseOperator, "k_simp k_simp"]:
    """
    Laplacian_k = d_j @ d_j.T + d_k.T @ d_k, where d_k is the k-coboundary
    operator, d_k.T is the k-boundary operator, and j = k - 1.

    If dual = True, compute the topological k-Laplacian on the dual complex.
    """
    if dual:
        cbd = sc.dual_cbd
    else:
        cbd = sc.cbd

    match component:
        case "up":
            d_k = cbd[k]
            up_laplacian = d_k.T @ d_k
            return up_laplacian

        case "down":
            d_j = cbd[k - 1]
            down_laplacian = d_j @ d_j.T
            return down_laplacian

        case "full":
            d_k = cbd[k]
            up_laplacian = d_k.T @ d_k

            d_j = cbd[k - 1]
            down_laplacian = d_j @ d_j.T

            full_laplacian = SparseOperator.assemble(up_laplacian, down_laplacian)

            return full_laplacian

        case _:
            raise ValueError()

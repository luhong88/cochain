import torch as t
from jaxtyping import Float

from ..complex import SimplicialComplex
from ..sparse.operators import SparseOperator


def laplacian_k(
    sc: SimplicialComplex, k: int, dual: bool = False
) -> tuple[
    Float[SparseOperator, "k_simp k_simp"],
    Float[SparseOperator, "k_simp k_simp"],
    Float[SparseOperator, "k_simp k_simp"],
]:
    """
    Laplacian_k = d_j @ d_j.T + d_k.T @ d_k, where d_k is the k-coboundary
    operator, d_k.T is the k-boundary operator, and j = k - 1.

    If dual = True, compute the topological k-Laplacian on the dual complex.
    """

    if dual:
        coboundary = sc.coboundary
    else:
        coboundary = sc.dual_coboundary

    # Get the k-th and (k-1)th- coboundary operator (or, generate an empty one with
    # the appropriate dimensions for "out-of-bound" values of k), and use them to
    # construct the Laplacian.
    d_k = coboundary[k]
    up_laplacian = d_k.T @ d_k

    d_j = coboundary[k - 1]
    down_laplacian = d_j @ d_j.T

    laplacian_k = SparseOperator.assemble(up_laplacian, down_laplacian)

    return down_laplacian, up_laplacian, laplacian_k

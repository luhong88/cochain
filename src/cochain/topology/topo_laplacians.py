__all__ = ["laplacian_k"]

from typing import Literal

from jaxtyping import Float

from ..complex import SimplicialMesh
from ..sparse.decoupled_tensor import SparseDecoupledTensor


def laplacian_k(
    sc: SimplicialMesh,
    *,
    k: int,
    component: Literal["up", "down", "full"],
    dual_complex: bool = False,
) -> Float[SparseDecoupledTensor, "k_splx k_splx"]:
    """
    Compute the topological/combinatorial $k$-Laplacian.

    The topological $k$-Laplacian is defined as:

    $$L_k = d_{k-1} d_{k-1}^T + d_k^T d_k$$

    where $d_k$ is the $k$-coboundary operator.

    Parameters
    ----------
    sc
        A mesh object.
    k
        Which Laplacian to compute.
    component
        If "up", compute the up component of the $K$-Laplacian ($d_k^T d_k$);
        if "down", compute the down component ($d_{k-1} d_{k-1}^T$); if "full",
        compute the full Laplacian.
    dual_complex
        If True, compute the $k$-Laplacian on the dual complex by using the
        dual coboundary operators.

    Returns
    -------
    [k_splx, k_splx]
        The $K$-Laplacian.

    Notes
    -----
    For $k = 0$, the topological/combinatorial Laplacian is equivalent to the
    (unweighted) graph Laplacian in graph theory.
    """
    if k > sc.dim:
        raise ValueError(
            f"The argument k ({k}) cannot be higher than the mesh dimension ({sc.dim})."
        )
    if k < 0:
        raise ValueError(f"The argument k ({k}) must be a nonnegative integer.")

    if dual_complex:
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
            if k < sc.dim:
                d_k = cbd[k]
                up_laplacian = d_k.T @ d_k
            else:
                up_laplacian = None

            if k > 0:
                d_j = cbd[k - 1]
                down_laplacian = d_j @ d_j.T
            else:
                down_laplacian = None

            match down_laplacian, up_laplacian:
                case None, None:
                    raise ValueError()
                case None, _:
                    full_laplacian = up_laplacian
                case _, None:
                    full_laplacian = down_laplacian
                case _:
                    full_laplacian = SparseDecoupledTensor.assemble(
                        up_laplacian, down_laplacian
                    )

            return full_laplacian

        case _:
            raise ValueError(f"Unknown 'component' argument ('{component}').")

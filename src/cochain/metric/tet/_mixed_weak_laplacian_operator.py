from typing import Any

import torch
from jaxtyping import Float
from torch import Tensor

from ...sparse.decoupled_tensor import SparseDecoupledTensor
from ...sparse.linalg.solvers._inv_sparse_operator import InvSparseOperator


# TODO: account for interactions with boundary conditions
class MixedWeakLaplacianOperator:
    r"""
    Construct a mixed formulation solver for a weak k-Laplacian.

    Parameters
    ----------
    cbd_km1: [k_splx, km1_splx]
        The k-coboundary operator.
    cbd_k: [kp1_splx, k_splx]
        The (k+1)-coboundary operator.
    mass_km1: [km1_splx, km1_splx]
        The consistent mass matrix for discrete (k-1)-forms.
    mass_k: [k_splx, k_splx]
        The consistent mass matrix for discrete k-forms.
    mass_kp1: [kp1_splx, kp1_splx]
        The consistent mass matrix for discrete (k+1)-forms.
    solver_cls
        A subclass of InvSparseOperator that represents a persistent sparse solver.
    solver_init_kwargs
        Keyword arguments as a dict passed to the `solver_cls` constructor.

    Attributes
    ----------
    dtype: torch.dtype
        The dtype of the mixed formulation sparse tensor.
    device: torch.device
        The device of the mixed formulation sparse tensor.
    shape: torch.Size
        The shape of the k-Laplacian operator.

    Notes
    -----
    Consider a weak k-Laplacian

    $$S_k = d_k^T M_{k+1} d_k + M_k d_{k-1} M_{k-1}^{-1} d_{k-1}^T M_k$$

    To solve the sparse linear system $S_k x = b$ for the $k$-cochain $x$ with the
    mixed formulation, define an auxiliary $(k-1)$-cochain $y$ as the codifferential
    of $x$ (i.e., $y = M_{k-1}^{-1} d_{k-1}^T M_k x$), which transforms the linear system
    into a new system

    $$
    \begin{bmatrix}
        -M_{k-1}    & d_{k-1}^T M_k \\
        M_k d_{k-1} & d_k^T M_{k+1} d_k
    \end{bmatrix}
    \begin{bmatrix}
        y \\ x
    \end{bmatrix}
    =
    \begin{bmatrix}
    0 \\ b
    \end{bmatrix}
    $$

    This approach removes the need for matrix inverse ($M_{k-1}^{-1}$) required
    to construct $S_k$ explicitly; however, the block matrix is now symmetric indefinite
    compared to $S_k$, which is symmetric positive semidefinite. This approach is called
    "mixed" formulation because, instead of solving a system $S_k x = b$ for a $k$-cochain $x$,
    an auxiliary $(k-1)$-cochain $y$ is introduced and we solve for a concatenated,
    "mixed" cochain $[y, x]$.

    Note that, this approach also works for the down component of $S_k$ ($M_k d_{k-1}
    M_{k-1}^{-1} d_{k-1}^T M_k$), in which case the block matrix simplies to 

    $$
    \begin{bmatrix}
        -M_{k-1}    & d_{k-1}^T M_k \\
        M_k d_{k-1} & 0
    \end{bmatrix}
    $$
    """

    def __init__(
        self,
        cbd_km1: Float[SparseDecoupledTensor, "k_splx km1_splx"],
        cbd_k: Float[SparseDecoupledTensor, "kp1_splx k_splx"] | None,
        mass_km1: Float[SparseDecoupledTensor, "km1_splx km1_splx"],
        mass_k: Float[SparseDecoupledTensor, "k_splx k_splx"],
        mass_kp1: Float[SparseDecoupledTensor, "kp1_splx kp1_splx"] | None,
        solver_cls: type[InvSparseOperator],
        solver_init_kwargs: dict[str, Any],
    ):
        block_00 = -mass_km1
        block_10 = mass_k @ cbd_km1
        block_01 = block_10.T

        if (cbd_k is None) and (mass_kp1 is None):
            block_11 = None
        else:
            block_11 = cbd_k.T @ mass_kp1 @ cbd_k

        mixed_op = SparseDecoupledTensor.bmat(
            [[block_00, block_01], [block_10, block_11]]
        )

        self.solver = solver_cls(mixed_op, **solver_init_kwargs)

        self.dtype = mass_k.dtype
        self.device = mass_k.device
        self.shape = mass_k.shape
        self._n_km1_splx = mass_km1.size(0)

    def size(self, dim: int | None = None) -> int | torch.Size:
        if dim is None:
            return self.shape
        else:
            return self.shape[dim]

    def __call__(
        self,
        b: Float[Tensor, " k_splx *ch"],
        solver_kwargs: dict[str, Any] | None = None,
    ) -> tuple[Float[Tensor, " k_splx *ch"], Float[Tensor, " km1_splx *ch"]]:
        r"""
        Solve the mixed formulation linear system.

        Parameters
        ----------
        b: [k_splx, *ch]
            The RHS vector, arbitrary trailing channel dimensions are allowed
            but treated independently.
        solver_kwargs
            Additional keyword arguments to be passed to the sparse solver call.

        Returns
        -------
        x: [k_splx, *ch]
            The $k$-cochain as the solution to $S_k x = b$.
        y: [km1_splx, *ch]
            The codifferential of $x$ (i.e., $y = \delta x$)
        """
        b_pad = torch.zeros(
            (self._n_km1_splx, *b.shape[1:]), dtype=b.dtype, device=b.device
        )
        b_full = torch.cat((b_pad, b), dim=0)

        if solver_kwargs is None:
            solver_kwargs = {}

        x_full = self.solver(b_full, **solver_kwargs)

        x = x_full[self._n_km1_splx :]
        y = x_full[: self._n_km1_splx]

        return x, y

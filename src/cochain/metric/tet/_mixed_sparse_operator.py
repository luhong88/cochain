from typing import Any

import torch
from jaxtyping import Float
from torch import Tensor

from ...sparse.decoupled_tensor import SparseDecoupledTensor
from ...sparse.linalg.solvers._inv_sparse_operator import InvSparseOperator


# TODO: account for interactions with boundary conditions
class MixedSparseOperator:
    def __init__(
        self,
        mass_k: Float[SparseDecoupledTensor, "k_splx k_splx"],
        mass_km1: Float[SparseDecoupledTensor, "km1_splx km1_splx"],
        mass_kp1: Float[SparseDecoupledTensor, "kp1_splx kp1_splx"] | None,
        cbd_km1: Float[SparseDecoupledTensor, "k_splx km1_splx"],
        cbd_k: Float[SparseDecoupledTensor, "kp1_splx k_splx"] | None,
        solver_cls: type[InvSparseOperator],
        solver_init_kwargs: dict[str, Any],
    ):
        # Consider a weak k-Laplacian
        #
        # S_k = (
        #   M_k @ d_{k-1} @ inv_M_{k-1} @ d_{k-1}.T @ M_k +
        #   d_k.T @ M_{k+1} @ d_k
        # )
        #
        # To solve the sparse linear system S_k @ x = b, define an auxiliary variable
        # y = inv_M_{k-1} @ d_{k-1}.T @ M_k, which transforms the linear system
        # into a new system B_k @ x' = b', where x' = cat(y, x), b' = cat(0, b),
        # and B_k is the block-symmetric matrix
        #
        # | -M_{k-1}        d_{k-1}.T @ M_k       |
        # | M_k @ d_{k-1}   d_k^T @ M_{k+1} @ d_k |
        #
        # Note that, this approach also works for the up component of S_k,
        # in which case B_11 is simply zero.

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
    ) -> Float[Tensor, " k_splx *ch"]:
        b_pad = torch.zeros(
            (self._n_km1_splx, *b.shape[1:]), dtype=b.dtype, device=b.device
        )
        b_full = torch.cat((b_pad, b), dim=0)

        if solver_kwargs is None:
            solver_kwargs = {}

        x_full = self.solver(b_full, **solver_kwargs)
        x = x_full[self._n_km1_splx :]

        return x

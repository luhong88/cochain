from typing import Literal

import torch
from jaxtyping import Float, Integer
from torch import LongTensor, Tensor

from ...complex import SimplicialMesh
from ...sparse.decoupled_tensor import SparseDecoupledTensor
from ..ext_prod.whitney import WhitneyWedgeL2Projector


def galerkin_contract(
    cochain_k: Float[Tensor, " k_splx *ch"],
    cochain_1: Float[Tensor, " edge *ch"],
    mass_k: Float[SparseDecoupledTensor, "k_splx k_splx"],
    mass_km1: Float[SparseDecoupledTensor, "km1_splx km1_splx"],
    wedge_op: WhitneyWedgeL2Projector,
    method: Literal["dense", "solver"],
) -> Float[Tensor, " km1_splx *ch"]:
    n_km1_splx = mass_km1.size(0)
    ch_dims = cochain_k.shape[1:]

    dummy_cochain_km1: Float[Tensor, " km1_splx *ch"] = torch.zeros(
        (n_km1_splx, *ch_dims),
        dtype=cochain_k.dtype,
        device=cochain_k.device,
    )

    mass_cochain: Float[Tensor, " k_splx *ch"] = mass_k @ cochain_k

    def _wedge_forward(cochain_km1: Tensor) -> Tensor:
        # pairing="scalar" preserves the *ch dimension in the output
        return wedge_op(cochain_1, cochain_km1, pairing="scalar")

    # Because the WhitneyWedgeL2Projector forward pass is a pure function, it is
    # safe for vjp(); note that, even though the wedge product __init__() does
    # create store intermediate tensors in the buffer, these are considered
    # immutable for the purpose of vjp().
    _, vjp_fxn = torch.func.vjp(_wedge_forward, dummy_cochain_km1)
    rhs = vjp_fxn(mass_cochain)[0]

    match method:
        case "dense":
            return torch.linalg.solve(mass_km1.to_dense(), rhs)

        case "solver":
            raise NotImplementedError()

        case _:
            raise ValueError()

from typing import Literal

import torch
from jaxtyping import Float, Integer
from torch import LongTensor, Tensor

from ..complex import SimplicialMesh
from ..sparse.decoupled_tensor import BaseDecoupledTensor, DiagDecoupledTensor
from .ext_prod.cup import AntisymmetricCupProduct
from .ext_prod.whitney import WhitneyWedgeL2Projector


def galerkin_contract(
    cochain_k: Float[Tensor, " k_splx *ch"],
    cochain_1: Float[Tensor, " edge *ch"],
    mass_k: Float[BaseDecoupledTensor, "k_splx k_splx"],
    mass_km1: Float[BaseDecoupledTensor, "km1_splx km1_splx"],
    wedge_op: WhitneyWedgeL2Projector,
    method: Literal["dense", "solver"],
) -> Float[Tensor, " km1_splx *ch"]:
    """
    Compute the interior product i_v(η) between a vector field V (represented by
    its flat v = ♭V) and a discrete k-form/k-cochain η using the Galerkin approach.

    By taking advantage of the adjoint relation between the interior product and
    the wedge product, the problem of finding the (k-1)-cochain ξ = i_v(η) is
    reduced to solving the linear system

    M_(k-1) @ ξ = W(v, *).T @ M_k @ η

    where M_(k-1) and M_k are the consistent mass matrices and W(v, *).T is the
    matrix representation of the adjoint of the wedge product between v and a
    (k-1)-cochain.

    Note that, if the input cochains contain batch/channel dimensions, then the
    `cochain_k`, `cochain_1`, and the output (k-1)-cochain should all have the same
    batch/channel dimensions.
    """
    n_km1_splx = mass_km1.size(0)
    ch_dims = cochain_k.shape[1:]

    # To compute the RHS W(v, *).T@M_k@η, note that this expression is equivalent
    # to the vector-jacobian product between the vector M_k@η and the jacobian
    # W(v, *). Therefore, if a "forward pass" is defined as W(v, x) for some
    # (dummy) (k-1)-cochain x, then the reverse-mode gradient using M_k@η as the
    # cotangent computes exactly the RHS. This gradient can be computed using
    # the functional VJP implemented in torch.func.vjp().
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


def algebraic_contract(
    k: int,
    cochain_k: Float[Tensor, " k_splx *ch"],
    cochain_1: Float[Tensor, " edge *ch"],
    star_k: Float[DiagDecoupledTensor, "k_splx k_splx"],
    star_km1: Float[DiagDecoupledTensor, "km1_splx km1_splx"],
    cup_op: AntisymmetricCupProduct,
) -> Float[Tensor, " km1_splx *ch"]:
    """
    Compute the interior product i_v(η) between a vector field V (represented by
    its flat v = ♭V) and a discrete k-form/k-cochain η using an algebraic/DEC
    approach by replacing wedge product in the definition of the interior product
    (in the smooth setting) with the (antisymmetric) cup product:

    ξ = i_v(η) = (-1)^(k(n-k)) * star_(k-1).inv @ cup(v, star_k @ η)

    where n is the dimension of the ambience space (3) and star_(k-1) and star_k
    are the diagonal Hodge stars. The antisymmetric cup product is preferred here
    over the standard cup product because it satisfies the graded commutativity
    property and is invariant to vertex permutations.

    Note that, if the input cochains contain batch/channel dimensions, then the
    `cochain_k`, `cochain_1`, and the output (k-1)-cochain should all have the same
    batch/channel dimensions.
    """
    cup_prod = cup_op(cochain_1, star_k @ cochain_k, pairing="scalar")

    ambient_dim = 3  # The mesh is embedded in R^3
    sign = (-1.0) ** (k * (ambient_dim - k))

    int_prod = sign * star_km1.inv @ cup_prod

    return int_prod

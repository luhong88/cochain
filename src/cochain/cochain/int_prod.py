from typing import Any

import torch
from jaxtyping import Float
from torch import Tensor

from ..sparse.decoupled_tensor import SparseDecoupledTensor
from ..sparse.linalg.solvers._inv_sparse_operator import InvSparseOperator
from .ext_prod.whitney import WhitneyWedgeL2Projector


def galerkin_contract(
    vec_field_flat: Float[Tensor, " edge *ch"],
    cochain_k: Float[Tensor, " k_splx *ch"],
    mass_km1: Float[SparseDecoupledTensor, "km1_splx km1_splx"]
    | Float[InvSparseOperator, "km1_splx km1_splx"],
    wedge_op: WhitneyWedgeL2Projector,
    solver_kwargs: dict[str, Any] | None = None,
) -> Float[Tensor, " km1_splx *ch"]:
    """
    Compute the Galerkin interior product between a vector field and a k-form.

    Compute the interior product i_v(η) between a vector field V (represented by
    its flat v = ♭V) and a discrete k-form/k-cochain η using the Galerkin approach.

    By taking advantage of the adjoint relation between the interior product and
    the wedge product, the problem of finding the (k-1)-cochain ξ = i_v(η) is
    reduced to solving the linear system

    M_(k-1) @ ξ = W(v, *).T @ M_k @ η

    where M_(k-1) and M_k are the consistent mass matrices and W(v, *).T is the
    matrix representation of the adjoint of the wedge product between v and a
    (k-1)-cochain. Note that the WhitneyWedgeL2Projector implementation returns
    the load vector rather than the wedge product directly; i.e., the `wedge_op`
    effectively acts as M_k @ W(v, *) rather than W(v, *), an explicit M_k matrix
    is not required.

    The input `wedge_op` should be setup to compute the load vector for the wedge
    product between a 1-cochain and a (k-1)-cochain.

    If the input `mass_km1` is a callable InvSparseOperator, the RHS will be
    passed to the operator to solve for ξ; otherwise, `mass_km1` will be converted
    to a dense tensor and torch.linalg.solve() will be used to solve for ξ.

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

    def _wedge_forward(cochain_km1: Tensor) -> Tensor:
        # pairing="scalar" preserves the *ch dimension in the output
        return wedge_op(vec_field_flat, cochain_km1, pairing="scalar")

    # Because the WhitneyWedgeL2Projector forward pass is a pure function, it is
    # safe for vjp(); note that, even though the wedge product __init__() does
    # create store intermediate tensors in the buffer, these are considered
    # immutable for the purpose of vjp().
    _, vjp_fxn = torch.func.vjp(_wedge_forward, dummy_cochain_km1)
    rhs: Float[Tensor, " km1_splx *ch"] = vjp_fxn(cochain_k)[0]

    if isinstance(mass_km1, InvSparseOperator):
        if solver_kwargs is None:
            solver_kwargs = {}

        return mass_km1(rhs, **solver_kwargs)

    else:
        return torch.linalg.solve(mass_km1.to_dense(), rhs)

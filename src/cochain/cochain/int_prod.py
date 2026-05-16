__all__ = ["galerkin_contract"]

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
    r"""
    Compute the Galerkin interior product between a vector field and a k-form.

    Let $v$ be a 1-cochain representing the flat of a vector field, $\eta$ be a
    $k$-cochain representing a discretized $k$-form, and $i_v(\eta)$ be their
    interior product. By taking advantage of the adjoint relation between the
    interior product and the wedge product, the problem of finding the $(k-1)$-cochain
    $\xi = i_v(\eta)$ is reduced to solving the linear system

    $$M_{k-1} \xi = W(v, *)^T M_k \eta$$

    where $M_{k-1}$ and $M_k$ are the consistent mass matrices and $W(v, *)^T$ is
    the matrix representation of the adjoint of the wedge product between $v$ and
    an arbitrary $(k-1)$-cochain.

    Parameters
    ----------
    vec_field_flat : [edge, *ch]
        The flat of the vector field. The channel dimensions, if there are any,
        need to match those of `cochain_k`.
    cochain_k : [k_splx, *ch]
        The k-cochain. The channel dimensions, if there are any, need to match
        those of `vec_field_flat`.
    mass_km1 : [km1_splx, km1_spl]
        The $(k-1)$-mass matrix. If this is a callable `InvSparseOperator`, the
        RHS ($W(v, *)^T M_k \eta$) will be passed to the operator to solve for
        $\xi$; otherwise, `mass_km1` will be converted to a dense tensor and
        `torch.linalg.solve()` will be used to solve for $\xi$.
    wedge_op
        An instance of `WhitneyWedgeL2Projector` configured to compute the load
        vector for the wedge product between a 1-cochain and a $(k-1)$-cochain.

    Returns
    -------
    [km1_splx, *ch]
        The $(k-1)$-cochain $\xi$.

    Notes
    -----
    Let $V$ be a vector field with its flat defined as $v = \flat V$. Let $\eta$ be
    an arbitrary $k$-form and $\zeta$ an arbitrary $(k-1)$-form. Then the interior
    product $i$ and the wedge product $\wedge$ satisfies the following adjoint relation:

    $$\left<\xi, \zeta\right> = \left<\eta, v\wedge\zeta\right>$$

    where we have defined $\xi = i_V(\eta)$. In the discrete setting, by replacing the
    forms $\eta$, $\zeta$, and $\xi$ with their corresponding cochains, the adjoint
    relation can be re-written as a linear system,

    $$\xi^T M_{k-1} \zeta = \eta^T M_k W(v)\zeta$$

    where $W(v)$ represents the operator that takes the wedge product between $v$ and
    a $k$-cochain. Since this linear system is satisfied for arbitrary $\zeta$, it
    can be dropped from the equation; then, taking the transpose of both sides gives

    $$M_{k-1} \xi = W(v)^T M_k \eta$$

    Although $M_k$ appears in the linear system and is required to solve for
    $\xi$, it is not explicitly required as an input to this function. To see why, note
    that the `WhitneyWedgeL2Projector` implementation returns the load vector rather
    than the wedge product directly; i.e., the `wedge_op` effectively acts as
    $M_k W(v)$ rather than $W(v)$. If we define the "forward pass" of the wedge
    product as

    $$y = F(x) = M_k W(v) x$$

    then for a hypothetical loss function $L(y)$, its derivative w.r.t. $x$ is given by

    $$
    \frac{\partial L}{\partial x} =
    \frac{\partial L}{\partial y}\frac{\partial y}{\partial x} =
    \bar y^T J_F
    $$

    where $\bar y = \partial L/\partial y$ is the sensitivity/cotangent of $L$ w.r.t.
    $y$ and $J_F = M_k W(v)$ is the Jacobian matrix of $F$. Therefore, if we
    assign $\bar y \leftarrow \eta$, then the reverse-mode autograd over $F$
    computes the VJP $\eta^T M_k W(v)$ which gives the transpose of the RHS vector
    required to solve $M_{k-1} \xi = W(v)^T M_k \eta$ (strictly speaking, PyTorch
    returns VJP that matches the shape of $x$ so no explicit transposition
    is performed).
    """
    n_km1_splx = mass_km1.size(0)
    ch_dims = cochain_k.shape[1:]

    # Compute the RHS W(v, *).T@M_k@η as a reverse-mode gradient using the
    # functional VJP implemented in torch.func.vjp().
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

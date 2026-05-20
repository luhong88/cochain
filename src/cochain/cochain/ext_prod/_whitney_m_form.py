import torch
from jaxtyping import Float
from torch import Tensor

from ...complex import SimplicialMesh
from ._whitney_utils import (
    compute_moments,
    compute_whitney_router,
    dispatch_bc_grad_dot,
)


def _inv_metric_det(
    bc_grad_dot: Float[Tensor, "splx vert vert"], form_deg: int
) -> Float[Tensor, " splx *d_lambda"]:
    r"""
    Compute the scalar inner products between barycentric differentials.

    Parameters
    ----------
    bc_grad_dot : [splx, vert, vert]
        The pairwise inner products between the gradients of barycentric
        coordinate functions within each top-level simplex.
    form_deg
        The degree of the form.

    Returns
    -------
    [splx, *d_lambda]

    Notes
    -----
    Let $d$ be the `form_deg`. For $d=0$, this function returns 1 for each
    simplex, since the Whitney 0-form basis functions do not involve any
    $d\lambda$'s. For $d=1$, this function returns all pairwise

    $$
    \left<d\lambda_i, d\lambda_j\right> =
    \left<\nabla \lambda_i, \nabla \lambda_j\right>
    $$

    for each top-level simplex, which is simply the input `bc_grad_dot`. For 
    $d=2$, this function returns all pairwise

    $$
    \left<d\lambda_i \wedge d\lambda_j, d\lambda_k \wedge d\lambda_l\right> =
    \begin{vmatrix}
        \left<\nabla \lambda_i, \nabla \lambda_k\right> &
        \left<\nabla \lambda_i, \nabla \lambda_l\right> \\
        \left<\nabla \lambda_j, \nabla \lambda_k\right> &
        \left<\nabla \lambda_j, \nabla \lambda_l\right>
    \end{vmatrix}
    $$

    for each top-level simplex; this equality follows from the Binet-Cauchy identity.
    """
    match form_deg:
        case 0:
            d_bc_wedge_dot = torch.ones(
                bc_grad_dot.shape[0],
                dtype=bc_grad_dot.dtype,
                device=bc_grad_dot.device,
            )

        case 1:
            d_bc_wedge_dot = bc_grad_dot

        # <dλ_i ⋀ dλ_j, dλ_k ⋀ dλ_l>
        case 2:
            n_splx = bc_grad_dot.size(0)
            n_vert = bc_grad_dot.size(-1)

            d_bc_wedge_dot = torch.zeros(
                n_splx,
                n_vert,
                n_vert,
                n_vert,
                n_vert,
                dtype=bc_grad_dot.dtype,
                device=bc_grad_dot.device,
            )

            # <dλ_i, dλ_k><dλ_j, dλ_l>
            d_bc_wedge_dot.add_(
                torch.einsum("tik,tjl->tijkl", bc_grad_dot, bc_grad_dot)
            )
            # <dλ_i, dλ_l><dλ_j, dλ_k>
            d_bc_wedge_dot.sub_(
                torch.einsum("til,tjk->tijkl", bc_grad_dot, bc_grad_dot)
            )

    return d_bc_wedge_dot


def _get_triple_tensor_prod_einsum_str(k: int, l: int) -> str:
    """
    Generate the string used in einsum() to compute the triple tensor product
    from constituent components.
    """
    m = k + l
    d_lambda_input_vars = "xyz"
    d_lambda_output_vars = "pqr"

    k_d_lambda_vars = d_lambda_input_vars[:k]
    l_d_lambda_vars = d_lambda_input_vars[k : k + l]
    m_d_lambda_vars = d_lambda_output_vars[:m]

    einsum_inputs = [
        "ua" + k_d_lambda_vars,  # k-form router
        "vb" + l_d_lambda_vars,  # l-form router
        "wc" + m_d_lambda_vars,  # m-form router
        "t",  # simplex size
        "abc",  # moments
        "t" + k_d_lambda_vars + l_d_lambda_vars + m_d_lambda_vars,  # wedge dot
    ]

    einsum_output = "tuvw"

    einsum_str = ",".join(einsum_inputs) + "->" + einsum_output

    return einsum_str


def triple_tensor_prod(
    k: int,
    l: int,
    mesh: SimplicialMesh,
) -> Float[Tensor, "top_splx k_face l_face m_face"]:
    """
    Compute the triple product tensor T_ijk required for computing the load vector.

    T_uvw is defined as the L^2 inner product <ϕ_u ⋀ ϕ_v, ϕ_w>, where ϕ_u is the
    Whitney k-form defined on the k-simplex u, ϕ_v is the Whitney l-form defined
    on the l-simplex v, and ϕ_w is the Whitney (k+l)-form defined on the
    (k+l)-simplex w.
    """
    device = mesh.device
    dtype = mesh.dtype

    k_form_router = compute_whitney_router(mesh.dim, k, device, dtype)
    l_form_router = compute_whitney_router(mesh.dim, l, device, dtype)
    kl_form_router = compute_whitney_router(mesh.dim, k + l, device, dtype)

    moments = compute_moments(3, mesh.dim, device, dtype)

    bc_grad_dot, splx_size = dispatch_bc_grad_dot(mesh)
    wedge_dot = _inv_metric_det(bc_grad_dot, k + l)

    einsum_str = _get_triple_tensor_prod_einsum_str(k, l)

    return torch.einsum(
        einsum_str,
        k_form_router,
        l_form_router,
        kl_form_router,
        splx_size,
        moments,
        wedge_dot,
    )

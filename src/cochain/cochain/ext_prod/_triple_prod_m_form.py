import torch
from einops import einsum
from jaxtyping import Float
from torch import Tensor

from ...complex import SimplicialMesh
from ._whitney_utils import (
    compute_moments,
    compute_whitney_router,
    dispatch_bc_grad_dot,
)


def _compute_inv_metric_det(
    bc_grad_dot: Float[Tensor, "top_splx vert vert"], form_deg: int
) -> Float[Tensor, " top_splx *d_lambda"]:
    r"""
    Compute the scalar inner products between barycentric differentials.

    Parameters
    ----------
    bc_grad_dot : [splx, vert, vert]
        The pairwise inner products between the gradients of barycentric
        coordinate functions within each top-level simplex.
    form_deg
        The degree of the form. If the `form_deg` is $k$, compute all possible
        pairwise inner products between the wedge products of $k$ barycentric
        differentials.

    Returns
    -------
    [splx, *d_lambda]
        The output inner product tensor. The `splx` dimension refers to the top-level
        simplices, and the `*d_lambda` dimension(s) refer to the barycentric
        differentials. If the `form_deg` is $k$ and the dimension of the top-level
        simplices is $d$, then there are $2^k$ such `d_lambda` dimensions, and each 
        `d_\lambda` dimension is of size $d+1$.

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


def _get_triple_prod_einsum_str(k: int, l: int) -> str:
    """Generate the einsum string for computing the triple product."""
    m = k + l
    d_lambda_input_vars = ["dl_x", "dl_y", "dl_z"]
    d_lambda_output_vars = ["dl_p", "dl_q", "dl_r"]

    k_d_lambdas = d_lambda_input_vars[:k]
    l_d_lambdas = d_lambda_input_vars[k : k + l]
    m_d_lambdas = d_lambda_output_vars[:m]

    einsum_inputs = [
        " ".join(["k_face", "l_a"] + k_d_lambdas),  # k-form router
        " ".join(["l_face", "l_b"] + l_d_lambdas),  # l-form router
        " ".join(["m_face", "l_c"] + m_d_lambdas),  # m-form router
        "top_splx",  # simplex size
        "l_a l_b l_c",  # moments
        " ".join(["top_splx"] + k_d_lambdas + l_d_lambdas + m_d_lambdas),  # wedge dot
    ]
    einsum_output = "top_splx k_face l_face m_face"

    einsum_str = ",".join(einsum_inputs) + " -> " + einsum_output

    return einsum_str


def compute_triple_prod_tensor(
    k: int,
    l: int,
    mesh: SimplicialMesh,
) -> Float[Tensor, "top_splx k_face l_face m_face"]:
    r"""
    Compute the triple product tensor required for the Galerkin wedge product.

    Parameters
    ----------
    k
        The order of the k-cochain.
    l
        The order of the l-cochain.
    mesh
        A simplicial mesh over which the cochains are defined.

    Returns
    -------
    [top_splx, k_face, l_face, m_face]
        The triple product tensor.

    Notes
    -----
    Let $m = k + l$. To compute the wedge product between a $k$-cochain and an
    $l$-cochain using the Galerkin approach requires computing, for each top-level
    simplex, the triple product

    $$
    T_{rst} = \int_\Omega \left<W_{kr} \wedge W_{ls}, W_{mt}\right> dV
    $$

    where $r$ iterates over the $k$-faces, $s$ iterates over the $l$-faces, and
    $t$ iterates over the $m$-faces; $W_{kr}$ is the Whitney $k$-form basis function
    defined on the $k$-simplex $r$, $W_{ls}$ is the Whitney $l$-form basis function
    defined on the $l$-simplex $s$, and $W_{mt}$ is the Whitney $(k+l)$-form basis
    function defined on the $(k+l)$-simplex $t$.

    In general, a Whitney basis form (of the lowest order) consists of multiple
    terms with a barycentric weight coefficient (which is coordinate dependent)
    and a wedge product of a variable number of barycentric differentials (which
    is coordinate independent). Therefore, the integral for $T_{rst}$ can be split
    into a sum of products of two terms:

    * An inner product between wedge products of barycentric differentials, which
      is computed by the `_inv_metric_det()` function. Specifically, the inner
      product is between two Whitney $m$-form basis functions.
    * An area/volume integral of products of barycentric weights, which is computed
      using the magic formula by the `compute_moments()` function. However, note
      that `compute_moments()` performs the integral over a reference simplex,
      and therefore a third term is required to account for the area/volume of
      the top-level simplex.

    This function achieves this by dynamically computing "router" tensors for
    the Whitney $k$-, $l$-, and $m$-basis forms, which, when contracted with the
    outputs from the `_inv_metric_det()` and `compute_moments()` functions,
    implicitly assemble the correct basis forms and compute their triple product
    tensor.
    """
    device = mesh.device
    dtype = mesh.dtype

    k_form_router = compute_whitney_router(mesh.dim, k, device, dtype)
    l_form_router = compute_whitney_router(mesh.dim, l, device, dtype)
    m_form_router = compute_whitney_router(mesh.dim, k + l, device, dtype)

    moments = compute_moments(3, mesh.dim, device, dtype)

    bc_grad_dot, splx_size = dispatch_bc_grad_dot(mesh)
    wedge_dot = _compute_inv_metric_det(bc_grad_dot, k + l)

    einsum_str = _get_triple_prod_einsum_str(k, l)

    return einsum(
        k_form_router,
        l_form_router,
        m_form_router,
        splx_size,
        moments,
        wedge_dot,
        einsum_str,
    )

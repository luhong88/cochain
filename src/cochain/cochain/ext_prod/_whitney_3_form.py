import itertools

import torch
from einops import einsum
from jaxtyping import Float
from torch import Tensor

from ...complex import SimplicialMesh
from ...utils.perm_parity import compute_lex_rel_orient
from ._whitney_utils import (
    compute_moments,
    compute_whitney_router,
    dispatch_bc_grad_dot,
)


def _compute_unit_tet_3_form_perms(
    device: torch.device, dtype: torch.dtype
) -> Float[Tensor, "d_lambda d_lambda d_lambda"]:
    r"""
    Enumerate the Whitney 3-form permutations over a tet with unit volume.

    Parameters
    ----------
    device
        The device of the output tensor.
    dtype
        The dtype of the output tensor.

    Returns
    -------
    form3_perms : [d_lambda, d_lambda, d_lambda]
        The output tensor containing the Whitney 3-form permutation signs.

    Notes
    -----
    The space of Whitney 3-forms in $\mathbb R^3$ is one-dimensional, and all
    3-forms can be written as:

    $$
    d\lambda_i \wedge d\lambda_j \wedge d\lambda_k =
    C_{ijk} (d\lambda_0 \wedge d\lambda_1 \wedge d\lambda_2)
    $$

    This function computes the $C_{ijk}$ coefficient tensor.

    The $C_{ijk}$ tensor contains only $1$, $0$, and $-1$ and is constructed
    using the following rules:

    * If there are any duplicates in $i$, $j$, $k$, then $C_{ijk} = 0$.
    * If $i$, $j$, and $k$ are distinct, then $C_{ijk} = \text{sign}(ijk)(-1)^{l+1}$.
      Here, $\text{sign}(ijk)$ is the parity of permutation required to put $ijk$
      back to lex order, and $l$ denotes the missing index.

    To see why the missing index factors into the calculation, consider the example
    $dd\lambda_0 \wedge d\lambda_1 \wedge d\lambda_3$. If $C_{ijk}$ is solely
    determined by $\text{sign}(ijk)$, then $C_{013} = 1$. However, since the complete
    set of barycentric coordinates satisfies the partition of unity (i.e.,
    $\lambda_0 + \lambda_1 + \lambda_2 + \lambda_3 = 1$), we can show that

    $$
    d\lambda_0 \wedge d\lambda_1 \wedge (- d\lambda_0 - d\lambda_1 - d\lambda_2)
    = -d\lambda_0 \wedge d\lambda_1 \wedge d\lambda_2
    $$

    and thus $C_{013} = -1$ (i.e., $l=2$ and $(-1)^{2+1} = -1$).

    A convenient way to account for both $\text{sign}(ijk)$ and $(-1)^{l+1}$ is
    by computing $C_{ijk} = \text{sign}(ijkl)$ (i.e., appending the missing $l$
    to the end). The tensor represented by $\text{sign}(ijkl)$ is known as the
    4D Levi-Civita symbol $\epsilon_{ijkl}$, and $C_{ijk}$ is related to
    $\epsilon_{ijkl}$ by contracting out the last $l$ dimension (note that any
    combination of $i$, $j$, and $k$ leaves at most one $l$ with nonzero values).
    """
    form3_perms = torch.zeros(4, 4, 4, dtype=dtype, device=device)

    perm = torch.tensor(
        list(itertools.permutations(range(4), r=4)), dtype=torch.int64, device=device
    )
    signs = compute_lex_rel_orient(perm).to(dtype=dtype)

    form3_perms[perm[:, :-1].T.unbind(0)] = signs

    return form3_perms


def _compute_3_form_squared_norm(
    bc_grad_dot: Float[Tensor, "tet vert vert"],
) -> Float[Tensor, " tet"]:
    r"""
    Compute the squared norm of the Whitney 3-form basis on the tets.

    This function is a specialized version of `_compute_inv_metric_det()` that
    optimizes the calculation for Whitney 3-forms.

    Parameters
    ----------
    bc_grad_dot : [tet, vert, vert]
        The pairwise inner products between the gradients of barycentric
        coordinate functions within each tet.

    Returns
    -------
    form3_norm2 : [tet,]
        The output Whitney 3-form squared norms tensor.

    Notes
    -----
    The space of Whitney 3-forms in $\mathbb R^3$ is one-dimensional. If one
    takes $d\lambda_0 \wedge d\lambda_1 \wedge d\lambda_2$ as the basis function,
    then the only unique inner product (up to sign) between 3-forms is between
    $d\lambda_0 \wedge d\lambda_1 \wedge d\lambda_2$ and itself.

    For the sake of simplicity, let us write

    $$
    g_{ij} = \left< d\lambda_i, d\lambda_j \right>
    $$

    Then, one can show, using the Cauchy-Binet formula, that

    $$
    \|d\lambda_0 \wedge d\lambda_1 \wedge d\lambda_2\|^2 = 
    \begin{vmatrix}
        g_{00} & g_{01} & g_{02} \\
        g_{10} & g_{11} & g_{12} \\
        g_{20} & g_{21} & g_{22}
    \end{vmatrix}
    $$
    """
    form3_norm2 = torch.linalg.det(bc_grad_dot[:, :-1, :-1])
    return form3_norm2


def _get_3_form_triple_prod_einsum_str(k: int, l: int) -> str:
    """
    Generate the einsum string for computing the triple product when $k + l = 3$.

    This function is a specialized version of `_get_triple_prod_einsum_str()`
    for when $k + l = 3$. If one were to use the logic in `_get_triple_prod_einsum_str()`,
    the `d_bc_wedge_dot` tensor would have the shape `(tet, vert, vert, vert, vert, vert, vert)`.
    Instead, when $k + l = 3$, the `d_bc_wedge_dot` consists of inner products
    between permutations of the Whitney 3-form, which consists of a single
    unique element (per tet) as computed by `_compute_3_form_squared_norm()` up
    to sign, and the sign for each operand of the inner product is computed by
    `_compute_unit_tet_3_form_perms()`. In other words, the dimensions of
    `d_bc_wedge_dot` are "separable" into three groups:

    `d_bc_wedge_dot[t,x,y,z,p,q,r]=form3_norm2[t]*form3_perms[x,y,z]*form3_perms[p,q,r]`

    Given this separation, there are only two tensors in the final einsum that
    depends on the `tet` dimension and only on the `tet` dimension: the tet volumes
    and the `form3_norm2` tensor. We therefore left the `tet` dimension out of
    the einsum and treat it separately in `compute_3_form_triple_prod_tensor()`.
    """
    d_lambda_input_vars = ["dl_x", "dl_y", "dl_z"]
    d_lambda_output_vars = ["dl_p", "dl_q", "dl_r"]

    k_d_lambda_vars = d_lambda_input_vars[:k]
    l_d_lambda_vars = d_lambda_input_vars[k : k + l]

    einsum_inputs = [
        " ".join(["k_face", "l_a"] + k_d_lambda_vars),  # k-form router
        " ".join(["l_face", "l_b"] + l_d_lambda_vars),  # l-form router
        " ".join(["m_face", "l_c"] + d_lambda_output_vars),  # m-form router
        " ".join(d_lambda_input_vars),  # C_xyz
        " ".join(d_lambda_output_vars),  # C_pqr
        "l_a l_b l_c",  # moments
    ]
    einsum_output = "k_face l_face m_face"

    einsum_str = ",".join(einsum_inputs) + " -> " + einsum_output

    return einsum_str


def compute_3_form_triple_prod_tensor(
    k: int,
    l: int,
    mesh: SimplicialMesh,
) -> Float[Tensor, "tet k_face l_face 3_face"]:
    """Perform a specialized `compute_triple_prod_tensor()` for when $k + l = 3$."""
    device = mesh.device
    dtype = mesh.dtype

    k_form_router = compute_whitney_router(mesh.dim, k, device, dtype)
    l_form_router = compute_whitney_router(mesh.dim, l, device, dtype)
    m_form_router = compute_whitney_router(mesh.dim, k + l, device, dtype)

    moments = compute_moments(3, mesh.dim, device, dtype)

    bc_grad_dot, splx_size = dispatch_bc_grad_dot(mesh)
    form3_norm2 = _compute_3_form_squared_norm(bc_grad_dot)
    form3_perms = _compute_unit_tet_3_form_perms(device=device, dtype=dtype)

    einsum_str = _get_3_form_triple_prod_einsum_str(k, l)

    # As noted in _get_3_form_triple_prod_einsum_str(), we split the calculation
    # into two products: one that depends only on the tet dimension, and one that
    # does not depend on the tet dimension, and piece them together afterwards.
    tet_prod = splx_size * form3_norm2
    face_prod = einsum(
        k_form_router,
        l_form_router,
        m_form_router,
        form3_perms,
        form3_perms,
        moments,
        einsum_str,
    )

    triple_prod = einsum(
        tet_prod,
        face_prod,
        "tet, k_face l_face m_face -> tet k_face l_face m_face",
    )

    return triple_prod

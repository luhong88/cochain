import itertools

import torch as t
from jaxtyping import Float

from ..complex import SimplicialComplex
from ..utils.perm_parity import compute_lex_rel_orient
from ._whitney_utils import compute_bc_grad_dot, compute_moments, compute_whitney_router


def _compute_mask_3_form(
    device: t.device, dtype: t.dtype
) -> Float[t.Tensor, "d_lambda d_lambda d_lambda"]:
    """
    The space of 3-forms in R^3 is one-dimensional, and all 3-forms can be related
    to the reference form via:

    dλ_i ⋀ dλ_j ⋀ dλ_k = C_ijk * (dλ_0 ⋀ dλ_1 ⋀ dλ_2)

    This function computes the C_ijk tensor.

    The C_ijk tensor contains only 1, 0, and -1 and is constructed using the
    following rules:

    * If there is any duplicates in i, j, k, then C_ijk = 0.
    * If i, j, and k are distinct, then C_ijk = sign(ijk)*(-1)**(l+1). Here, sign(ijk)
      is the parity of permutation required to put ijk back to lex order, and l
      denotes the missing index.

    To see why the missing index is required, consider the example dλ_0 ⋀ dλ_1 ⋀ dλ_3,
    since λ_0 + λ_1 + λ_2 + λ_3 = 1, this 3-form is related to the reference form by

    dλ_0 ⋀ dλ_1 ⋀ (-dλ_0 - dλ_1 - dλ_2) = -dλ_0 ⋀ dλ_1 ⋀ dλ_2

    This sign is equivalent to (-1)**(2 + 1) = -1. A convenient way to compute
    the combined sign is as C_ijk = sign(ijkl) (i.e., appending l to the end).
    """
    mask = t.zeros(4, 4, 4, dtype=dtype, device=device)

    perm = t.tensor(
        list(itertools.permutations(range(4), r=4)), dtype=t.int64, device=device
    )
    signs = compute_lex_rel_orient(perm).to(dtype=dtype)

    mask[perm[:, :-1].T.unbind(0)] = signs

    return mask


def _inv_metric_det_3_form(
    bc_grad_dot: Float[t.Tensor, "tet vert vert"],
) -> Float[t.Tensor, " tet"]:
    """
    A specialized version of _inv_metric_det() that optimizes the calculation for
    3-forms; specifically, it computes <dλ_0 ⋀ dλ_1 ⋀ dλ_2, dλ_0 ⋀ dλ_1 ⋀ dλ_2>
    for all tets.
    """
    ref_det_012 = t.linalg.det(bc_grad_dot[:, :-1, :-1])
    return ref_det_012


def _get_triple_tensor_prod_einsum_str_3_form(k: int, l: int) -> str:
    """
    A specialized version of _get_triple_tensor_prod_einsum_str() for when
    k + l = 3. In this case, instead of using the d_bc_wedge_dot tensor from
    _inv_metric_det(), which has shape (tet, vert, vert, vert, vert, vert, vert),
    decompose this tensor as

    T_txyzpqr= ref_det_012_t * C_xyz * C_pqr

    Here, ref_det_012_t comes from _inv_metric_det_3_form(), and C_xyz and C_pqr
    comes from _compute_mask_3_form().

    Since there are only two terms involving the tet dimension (tet volume and
    rer_det_012), they are left out of this einsum to be treated separately.
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
        k_d_lambda_vars + l_d_lambda_vars,  # C_xyz
        m_d_lambda_vars,  # C_pqr
        "abc",  # moments
    ]
    einsum_str = ",".join(einsum_inputs) + "->uvw"

    return einsum_str


def triple_tensor_prod_3_form(
    k: int,
    l: int,
    mesh: SimplicialComplex,
) -> Float[t.Tensor, "tet k_face l_face 3_face"]:
    """
    A specialized version of triple_tensor_prod() for when k + l = 3.
    """
    device = mesh.vert_coords.device
    dtype = mesh.vert_coords.dtype

    k_form_router = compute_whitney_router(mesh.dim, k, device, dtype)
    l_form_router = compute_whitney_router(mesh.dim, l, device, dtype)
    kl_form_router = compute_whitney_router(mesh.dim, k + l, device, dtype)

    moments = compute_moments(3, mesh.dim, device, dtype)

    bc_grad_dot, simp_size = compute_bc_grad_dot(mesh)
    ref_det_012 = _inv_metric_det_3_form(bc_grad_dot)

    mask_3_form = _compute_mask_3_form(device=device, dtype=dtype)

    einsum_str = _get_triple_tensor_prod_einsum_str_3_form(k, l)

    prod1: Float[t.Tensor, " tet"] = simp_size * ref_det_012
    prod2: Float[t.Tensor, "k_face l_face m_face"] = t.einsum(
        einsum_str,
        k_form_router,
        l_form_router,
        kl_form_router,
        mask_3_form,
        mask_3_form,
        moments,
    )

    return prod1.view(-1, 1, 1, 1) * prod2.unsqueeze(0)

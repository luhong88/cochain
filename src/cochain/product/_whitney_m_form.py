import torch as t
from jaxtyping import Float

from ..complex import SimplicialComplex
from ._whitney_utils import compute_bc_grad_dot, compute_moments, compute_whitney_router


def _inv_metric_det(
    bc_grad_dot: Float[t.Tensor, "simp vert vert"], form_deg: int
) -> Float[t.Tensor, " simp *d_lambda"]:
    """
    Compute the scalar inner products between wedge products of dλ's, which is
    equivalent to computing the determinant of the inner products of the gradients
    of λ's.
    """
    match form_deg:
        # Because Whitney 0-forms do not involve dλ terms, return 1 for each simplex.
        case 0:
            d_bc_wedge_dot = t.ones(
                bc_grad_dot.shape[0], dtype=bc_grad_dot.dtype, device=bc_grad_dot.device
            )

        # <dλ_i, dλ_j> = <grad[λ_i], grad[λ_j]>
        case 1:
            d_bc_wedge_dot = bc_grad_dot

        # <dλ_i ⋀ dλ_j, dλ_k ⋀ dλ_l> is equal to the determinant of
        #
        # | <grad[λ_i], grad[λ_k]> <grad[λ_i], grad[λ_l]> |
        # | <grad[λ_j], grad[λ_k]> <grad[λ_j], grad[λ_l]> |
        case 2:
            n_simp = bc_grad_dot.size(0)
            n_vert = bc_grad_dot.size(-1)

            d_bc_wedge_dot = t.zeros(
                n_simp,
                n_vert,
                n_vert,
                n_vert,
                n_vert,
                dtype=bc_grad_dot.dtype,
                device=bc_grad_dot.device,
            )

            d_bc_wedge_dot.add_(t.einsum("tik,tjl->tijkl", bc_grad_dot, bc_grad_dot))
            d_bc_wedge_dot.sub_(t.einsum("til,tjk->tijkl", bc_grad_dot, bc_grad_dot))

        # TODO: memory optimization

        # <dλ_i ⋀ dλ_j ⋀ dλ_k, dλ_a ⋀ dλ_b ⋀ dλ_c> is equal to the determinant of
        #
        # | <grad[λ_i], grad[λ_a]> <grad[λ_i], grad[λ_b]> <grad[λ_i], grad[λ_c]> |
        # | <grad[λ_j], grad[λ_a]> <grad[λ_j], grad[λ_b]> <grad[λ_j], grad[λ_c]> |
        # | <grad[λ_k], grad[λ_a]> <grad[λ_k], grad[λ_b]> <grad[λ_k], grad[λ_c]> |
        case 3:
            n_simp = bc_grad_dot.size(0)
            n_vert = bc_grad_dot.size(-1)

            d_bc_wedge_dot = t.zeros(
                n_simp,
                n_vert,
                n_vert,
                n_vert,
                n_vert,
                n_vert,
                n_vert,
                dtype=bc_grad_dot.dtype,
                device=bc_grad_dot.device,
            )

            # Denote <grad[λ_i], grad[λ_a]> as g_ia. Computing the determinant
            # is equivalent to summing over all permutations of all products of
            # the form sign(xyz)*g_ix*g_jy*gkz where x, y, z denotes permutations
            # of a, b, c and the sign(xyz) denotes the parity of the permutation
            # to arnage x, y, z back to lex order.
            for x, y, z in ["abc", "bca", "cab"]:
                d_bc_wedge_dot.add_(
                    t.einsum(
                        f"ti{x},tj{y},tk{z}->tijkabc",
                        bc_grad_dot,
                        bc_grad_dot,
                        bc_grad_dot,
                    )
                )

            for x, y, z in ["acb", "bac", "cba"]:
                d_bc_wedge_dot.sub_(
                    t.einsum(
                        f"ti{x},tj{y},tk{z}->tijkabc",
                        bc_grad_dot,
                        bc_grad_dot,
                        bc_grad_dot,
                    )
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
    mesh: SimplicialComplex,
) -> Float[t.Tensor, "top_simp k_face l_face m_face"]:
    """
    Compute the triple product tensor T_ijk required for computing the load vector.

    T_uvw is defined as the L^2 inner product <ϕ_u ⋀ ϕ_v, ϕ_w>, where ϕ_u is the
    Whitney k-form defined on the k-simplex u, ϕ_v is the Whitney l-form defined
    on the l-simplex v, and ϕ_w is the Whitney (k+l)-form defined on the
    (k+l)-simplex w.
    """
    device = mesh.vert_coords.device
    dtype = mesh.vert_coords.dtype

    k_form_router = compute_whitney_router(mesh.dim, k, device, dtype)
    l_form_router = compute_whitney_router(mesh.dim, l, device, dtype)
    kl_form_router = compute_whitney_router(mesh.dim, k + l, device, dtype)

    moments = compute_moments(3, mesh.dim, device, dtype)

    bc_grad_dot, simp_size = compute_bc_grad_dot(mesh)
    wedge_dot = _inv_metric_det(bc_grad_dot, k + l)

    einsum_str = _get_triple_tensor_prod_einsum_str(k, l)

    return t.einsum(
        einsum_str,
        k_form_router,
        l_form_router,
        kl_form_router,
        simp_size,
        moments,
        wedge_dot,
    )

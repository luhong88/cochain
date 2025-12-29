import itertools
import math
from typing import Literal

import torch as t
from jaxtyping import Float, Integer

from ..complex import SimplicialComplex
from ..geometry.tet import tet_geometry
from ..geometry.tri import tri_geometry
from ..utils.faces import enumerate_faces
from ..utils.perm_parity import compute_lex_rel_orient
from ..utils.search import simplex_search


def _compute_whitney_router(
    simp_dim: int, form_deg: int, device: t.device, dtype: t.dtype = t.float
) -> Float[t.Tensor, "face lambda *d_lambda"]:
    """
    Compute the coefficients required to construct the Whitney forms from the
    λ's and the dλ's.
    """
    faces = enumerate_faces(simp_dim, form_deg, device="cpu").tolist()

    router_shape = (len(faces),) + (simp_dim + 1,) * (form_deg + 1)
    router = t.zeros(router_shape, dtype=dtype, device=device)

    for simp_idx, simp in enumerate(faces):
        perms = t.tensor(
            list(itertools.permutations(simp)), dtype=t.int64, device=device
        )
        signs = compute_lex_rel_orient(perms).to(dtype=dtype, device=device)
        router[simp_idx][perms.T.unbind(0)] = signs

    return router


def _compute_moments(
    order: int, simp_dim: int, device: t.device, dtype: t.dtype = t.float
) -> t.Tensor:
    """
    For an n-simplex with unit area/volume and n + 1 barycentric coordinate functions
    λ_i, use the magic formula

    int[prod_i[λ_i^m_i]dV] = (n! * prod_i[m_i!]) / (n + sum_i[m_i])!

    to compute the moment tensors. The output tensor is of shape
    (simp_dim + 1,) * order.
    """
    verts = list(range(simp_dim + 1))
    moments = t.zeros((len(verts),) * order)

    for lambdas in itertools.product(verts, repeat=order):
        exponents = [lambdas.count(i) for i in verts]
        numerator = math.factorial(simp_dim) * math.prod(
            [math.factorial(i) for i in exponents]
        )
        denominator = math.factorial(simp_dim + sum(exponents))
        moments[lambdas] = numerator / denominator

    return moments.to(device=device, dtype=dtype)


def _compute_bc_grad_dot(
    mesh: SimplicialComplex,
) -> tuple[Float[t.Tensor, "simp vert vert"], Float[t.Tensor, " simp"]]:
    """
    A wrapper function for dispatching the correct bary_coord_grad_inner_prods()
    function for either tri or tet meshes.
    """
    match mesh.dim:
        case 2:
            simp_size = tri_geometry.compute_tri_areas(mesh.vert_coords, mesh.tris)
            simp_size_grad = tri_geometry.compute_d_tri_areas_d_vert_coords(
                mesh.vert_coords, mesh.tris
            )
            bc_grad_dot = tri_geometry.bary_coord_grad_inner_prods(
                simp_size.view(-1, 1, 1), simp_size_grad
            )

        case 3:
            signed_simp_size = tet_geometry.get_tet_signed_vols(
                mesh.vert_coords, mesh.tets
            )
            simp_size = t.abs(signed_simp_size)
            signed_simp_size_grad = tet_geometry.d_tet_signed_vols_d_vert_coords(
                mesh.vert_coords, mesh.tets
            )
            bc_grad_dot = tet_geometry.bary_coord_grad_inner_prods(
                signed_simp_size.view(-1, 1, 1), signed_simp_size_grad
            )

        case _:
            raise NotImplementedError()

    return bc_grad_dot, simp_size


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

    k_form_router = _compute_whitney_router(mesh.dim, k, device, dtype)
    l_form_router = _compute_whitney_router(mesh.dim, l, device, dtype)
    kl_form_router = _compute_whitney_router(mesh.dim, k + l, device, dtype)

    moments = _compute_moments(3, mesh.dim, device, dtype)

    bc_grad_dot, simp_size = _compute_bc_grad_dot(mesh)
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


def _find_top_simp_faces(
    face_dim: int,
    mesh_dim: int,
    mesh: SimplicialComplex,
    simp_map: dict[int, Integer[t.Tensor, "simp vert"]],
) -> tuple[
    Integer[t.LongTensor, "top_simp k_face"], Float[t.Tensor, "top_simp k_face"]
]:
    """
    Given a simplicial n-complex, for each top level n-simplex, find all of its
    k-faces, their indices in the list of k-simplices in the mesh, and their
    orientation sign corrections.
    """
    k = face_dim
    # Identify the k-faces of the top level simplices and their sign corrections.
    k_faces: Float[t.Tensor, "top_simp k_face k+1"] = simp_map[mesh_dim][
        :, enumerate_faces(mesh_dim, k, device=mesh.vert_coords.device)
    ]
    k_faces_flat = k_faces.view(-1, k + 1)
    k_faces_idx_flat = simplex_search(
        key_simps=simp_map[k],
        query_simps=k_faces_flat,
        sort_key_simp=True if k == mesh.dim else False,
        sort_key_vert=True if k == mesh.dim else False,
        sort_query_vert=True,
    )
    k_faces_idx = k_faces_idx_flat.view(*k_faces.shape[:-1])

    # Note that, in the implementation of the cup product, the parental simplices
    # are sorted before extracting their faces; as such, the faces automatically
    # possesse the canonical orientation, and we only need to correct for the
    # permutation parity required to sort the parental simplices. Here, since the
    # parental simplices are not sorted first, we need two parity corrections, one
    # for the permutation parity of the unsorted faces (induced parity), and one
    # for the permutation parity of the unsorted parental (global parity).
    if k == mesh.dim:
        k_face_parity_global = compute_lex_rel_orient(simp_map[k][k_faces_idx_flat])
    else:
        k_face_parity_global = t.ones(
            1, dtype=mesh.vert_coords.dtype, device=mesh.vert_coords.device
        ).expand_as(k_faces_idx_flat)

    k_face_parity_induced = compute_lex_rel_orient(k_faces_flat)

    k_face_parity = (
        (k_face_parity_induced * k_face_parity_global)
        .to(dtype=mesh.vert_coords.dtype, device=mesh.vert_coords.device)
        .view(*k_faces.shape[:-1])
    )

    return k_faces_idx, k_face_parity


class WhitneyWedgeProjection(t.nn.Module):
    def __init__(self, k: int, l: int, mesh: SimplicialComplex):
        """
        Compute the load vector required to compute the Whitney L2 wedge product.
        """
        super().__init__()

        simp_map = {
            dim: simp
            for dim, simp in enumerate([mesh.verts, mesh.edges, mesh.tris, mesh.tets])
        }

        m = k + l

        # Identify the k-faces of the top level simplices and their sign corrections.
        k_face_idx, k_face_parity = _find_top_simp_faces(k, mesh.dim, mesh, simp_map)

        self.k_face_idx: Integer[t.LongTensor, "top_simp k_face"]
        self.register_buffer("k_face_idx", k_face_idx)

        self.k_face_parity: Float[t.Tensor, "top_simp k_face"]
        self.register_buffer("k_face_parity", k_face_parity)

        # Identify the l-faces of the top level simplices and their sign corrections.
        l_face_idx, l_face_parity = _find_top_simp_faces(l, mesh.dim, mesh, simp_map)

        self.l_face_idx: Integer[t.LongTensor, "top_simp l_face"]
        self.register_buffer("l_face_idx", l_face_idx)

        self.l_face_parity: Float[t.Tensor, "top_simp l_face"]
        self.register_buffer("l_face_parity", l_face_parity)

        # Identify the (k+l)-faces of the top level simplices and their sign corrections.
        m_face_idx, m_face_parity = _find_top_simp_faces(m, mesh.dim, mesh, simp_map)

        self.m_face_idx: Integer[t.LongTensor, "top_simp m_face"]
        self.register_buffer("m_face_idx", m_face_idx)

        self.m_face_parity: Float[t.Tensor, "top_simp m_face"]
        self.register_buffer("m_face_parity", m_face_parity)

        self.n_m_simp = simp_map[m].size(0)

        # Compute the triple tensor product.
        triple_prod = triple_tensor_prod(k, l, mesh)

        self.triple_prod: Float[t.Tensor, "top_simp k_face l_face m_face"]
        self.register_buffer("triple_prod", triple_prod)

    def forward(
        self,
        k_cochain: Float[t.Tensor, " k_simp *ch_in"],
        l_cochain: Float[t.Tensor, " l_simp *ch_in"],
        pairing: Literal["scalar", "dot", "cross"] = "scalar",
    ) -> Float[t.Tensor, " m_simp *ch_out"]:
        k_cochain_at_k_face = t.einsum(
            "tf,tf...->tf...", self.k_face_parity, k_cochain[self.k_face_idx]
        )
        l_cochain_at_l_face = t.einsum(
            "tf,tf...->tf...", self.l_face_parity, l_cochain[self.l_face_idx]
        )

        # If pairing='scalar', *ch_in can match to an arbitrary number of channel
        # dimensions; for other pairing method, *ch_in need to match to one dimension.
        match pairing:
            case "scalar":
                m_cochain_at_m_face = t.einsum(
                    "tuvw,tu...,tv...,tw->tw...",
                    self.triple_prod,
                    k_cochain_at_k_face,
                    l_cochain_at_l_face,
                    self.m_face_parity,
                )

            case "dot":
                m_cochain_at_m_face = t.einsum(
                    "tuvw,tuc,tvc,tw->tw",
                    self.triple_prod,
                    k_cochain_at_k_face,
                    l_cochain_at_l_face,
                    self.m_face_parity,
                ).unsqueeze(-1)

            case "cross":
                epsilon = t.tensor(
                    [
                        [[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]],
                        [[0.0, 0.0, -1.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
                        [[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                    ],
                    device=self.triple_prod.device,
                    dtype=self.triple_prod.dtype,
                )

                m_cochain_at_m_face = t.einsum(
                    "tuvw,tuc,tvd,tw,cde->twe",
                    self.triple_prod,
                    k_cochain_at_k_face,
                    l_cochain_at_l_face,
                    self.m_face_parity,
                    epsilon,
                )

            case _:
                raise NotImplementedError()

        ch_out_dims = m_cochain_at_m_face.shape[2:]
        load = t.zeros(
            (self.n_m_simp,) + ch_out_dims,
            device=self.triple_prod.device,
            dtype=self.triple_prod.dtype,
        )
        load.index_add_(
            0,
            self.m_face_idx.flatten(),
            m_cochain_at_m_face.flatten(end_dim=1),
        )

        return load

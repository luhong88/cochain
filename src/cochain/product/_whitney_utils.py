import itertools
import math

import torch as t
from jaxtyping import Float, Integer

from ..complex import SimplicialMesh
from ..geometry.tet import tet_geometry
from ..geometry.tri import tri_geometry
from ..utils.faces import enumerate_local_faces
from ..utils.perm_parity import compute_lex_rel_orient
from ..utils.search import splx_search


def compute_whitney_router(
    splx_dim: int, form_deg: int, device: t.device, dtype: t.dtype = t.float
) -> Float[t.Tensor, "face lambda *d_lambda"]:
    """
    Compute the coefficients required to construct the Whitney forms from the
    λ's and the dλ's.
    """
    faces = enumerate_local_faces(splx_dim, form_deg, device="cpu").tolist()

    router_shape = (len(faces),) + (splx_dim + 1,) * (form_deg + 1)
    router = t.zeros(router_shape, dtype=dtype, device=device)

    for splx_idx, splx in enumerate(faces):
        perms = t.tensor(
            list(itertools.permutations(splx)), dtype=t.int64, device=device
        )
        signs = compute_lex_rel_orient(perms).to(dtype=dtype, device=device)
        router[splx_idx][perms.T.unbind(0)] = signs

    return router


def compute_moments(
    order: int, splx_dim: int, device: t.device, dtype: t.dtype = t.float
) -> t.Tensor:
    """
    For an n-simplex with unit area/volume and n + 1 barycentric coordinate functions
    λ_i, use the magic formula

    int[prod_i[λ_i^m_i]dV] = (n! * prod_i[m_i!]) / (n + sum_i[m_i])!

    to compute the moment tensors. The output tensor is of shape
    (splx_dim + 1,) * order.
    """
    verts = list(range(splx_dim + 1))
    moments = t.zeros((len(verts),) * order)

    for lambdas in itertools.product(verts, repeat=order):
        exponents = [lambdas.count(i) for i in verts]
        numerator = math.factorial(splx_dim) * math.prod(
            [math.factorial(i) for i in exponents]
        )
        denominator = math.factorial(splx_dim + sum(exponents))
        moments[lambdas] = numerator / denominator

    return moments.to(device=device, dtype=dtype)


def compute_bc_grad_dot(
    mesh: SimplicialMesh,
) -> tuple[Float[t.Tensor, "splx vert vert"], Float[t.Tensor, " splx"]]:
    """
    A wrapper function for dispatching the correct bary_coord_grad_inner_prods()
    function for either tri or tet meshes.
    """
    match mesh.dim:
        case 2:
            splx_size = tri_geometry.compute_tri_areas(mesh.vert_coords, mesh.tris)
            splx_size_grad = tri_geometry.compute_d_tri_areas_d_vert_coords(
                mesh.vert_coords, mesh.tris
            )
            bc_grad_dot = tri_geometry.bary_coord_grad_inner_prods(
                splx_size.view(-1, 1, 1), splx_size_grad
            )

        case 3:
            signed_splx_size = tet_geometry.compute_tet_signed_vols(
                mesh.vert_coords, mesh.tets
            )
            splx_size = t.abs(signed_splx_size)
            signed_splx_size_grad = (
                tet_geometry.dompute_d_tet_signed_vols_d_vert_coords(
                    mesh.vert_coords, mesh.tets
                )
            )
            bc_grad_dot = tet_geometry.bary_coord_grad_inner_prods(
                signed_splx_size.view(-1, 1, 1), signed_splx_size_grad
            )

        case _:
            raise NotImplementedError()

    return bc_grad_dot, splx_size


def find_top_splx_faces(
    face_dim: int,
    mesh: SimplicialMesh,
) -> tuple[
    Integer[t.LongTensor, "top_splx k_face"], Float[t.Tensor, "top_splx k_face"]
]:
    """
    Given a simplicial n-complex, for each top level n-simplex, find all of its
    k-faces, their indices in the list of k-simplices in the mesh, and their
    orientation sign corrections.
    """
    k = face_dim
    # Identify the k-faces of the top level simplices and their sign corrections.
    k_faces: Float[t.Tensor, "top_splx k_face k+1"] = mesh.splx[mesh.dim][
        :, enumerate_local_faces(mesh.dim, k, device=mesh.vert_coords.device)
    ]
    k_faces_flat = k_faces.view(-1, k + 1)
    k_faces_idx_flat = splx_search(
        key_splx=mesh.splx[k],
        query_splx=k_faces_flat,
        sort_key_splx=True if k == mesh.dim else False,
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
        k_face_parity_global = compute_lex_rel_orient(mesh.splx[k][k_faces_idx_flat])
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

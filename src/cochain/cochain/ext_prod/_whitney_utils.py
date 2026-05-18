import itertools
import math

import torch
from jaxtyping import Float
from torch import Tensor

from ...complex import SimplicialMesh
from ...metric.tet import _tet_geometry
from ...metric.tri import _tri_geometry
from ...utils.faces import enumerate_local_faces
from ...utils.perm_parity import compute_lex_rel_orient


def compute_whitney_router(
    splx_dim: int,
    form_deg: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> Float[Tensor, "face lambda *d_lambda"]:
    """
    Compute the coefficients required to construct the Whitney forms from the
    λ's and the dλ's.
    """
    faces = enumerate_local_faces(splx_dim, form_deg, device="cpu").tolist()

    router_shape = (len(faces),) + (splx_dim + 1,) * (form_deg + 1)
    router = torch.zeros(router_shape, dtype=dtype, device=device)

    for splx_idx, splx in enumerate(faces):
        perms = torch.tensor(
            list(itertools.permutations(splx)), dtype=torch.int64, device=device
        )
        signs = compute_lex_rel_orient(perms).to(dtype=dtype, device=device)
        router[splx_idx][perms.T.unbind(0)] = signs

    return router


def compute_moments(
    order: int, splx_dim: int, device: torch.device, dtype: torch.dtype = torch.float32
) -> Tensor:
    """
    For an n-simplex with unit area/volume and n + 1 barycentric coordinate functions
    λ_i, use the magic formula

    int[prod_i[λ_i^m_i]dV] = (n! * prod_i[m_i!]) / (n + sum_i[m_i])!

    to compute the moment tensors. The output tensor is of shape
    (splx_dim + 1,) * order.
    """
    verts = list(range(splx_dim + 1))
    moments = torch.zeros((len(verts),) * order)

    for lambdas in itertools.product(verts, repeat=order):
        exponents = [lambdas.count(i) for i in verts]
        numerator = math.factorial(splx_dim) * math.prod(
            [math.factorial(i) for i in exponents]
        )
        denominator = math.factorial(splx_dim + sum(exponents))
        moments[lambdas] = numerator / denominator

    return moments.to(device=device, dtype=dtype)


def dispatch_bc_grad_dot(
    mesh: SimplicialMesh,
) -> tuple[Float[Tensor, "splx vert vert"], Float[Tensor, " splx"]]:
    """
    A wrapper function for dispatching the correct bary_coord_grad_inner_prods()
    function for either tri or tet meshes.
    """
    match mesh.dim:
        case 2:
            splx_size, bc_grad_dot = _tri_geometry.compute_bc_grad_dots(
                mesh.vert_coords, mesh.tris
            )

        case 3:
            signed_splx_size, bc_grad_dot = _tet_geometry.compute_bc_grad_dots(
                mesh.vert_coords, mesh.tets
            )
            splx_size = torch.abs(signed_splx_size)

        case _:
            raise NotImplementedError()

    return bc_grad_dot, splx_size

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
    r"""
    Compute the coefficient routers for constructing Whitney forms.

    Parameters
    ----------
    splx_dim
        The dimension of the simplex.
    form_deg
        The degree of the Whitney form.
    device
        The device of the output tensor.
    dtype
        The dtype of the output tensor.

    Returns
    -------
    [face, lambda, *d_lambda]
        A coefficient matrix, such that contraction over the `lambda` dimension
        with barycentric weights and contraction over the `*d_lambda` dimension
        with barycentric differentials construct the appropriate Whitney basis
        form of the given degree on the given simplex.

    Notes
    -----
    In general, a Whitney k-form basis function defined on the $k$-simplex
    $012\cdots k$ is given by

    $$
    W = k!\sum_{i=0}^{k} (-1)^i \lambda_i d{\lambda}_0 \wedge
    \cdots  \widehat{d\lambda_i} \cdots \wedge d\lambda_k
    $$

    where the hat indicates an omitted term. There is, however, a somewhat more
    verbose but equivalent definition that is more amenable to algorithmic
    implementation: For a given k-simplex, enumerate all permutations of its
    vertices and determine the parity of the permutations. Then, each permutation
    can be interpreted as a term in the Whitney k-form basis function. Specifically,
    the first vertex is a barycentric weight while the rest are barycentric
    differentials, and the permutation parities are the signs of the terms.

    For example, consider the Whitney 2-form basis function on the 2-simplex 012.
    Then, the six permutations correspond to six terms:

    * 012 : $+\lambda_0 d\lambda_1 \wedge d\lambda_2$
    * 021 : $-\lambda_0 d\lambda_2 \wedge d\lambda_1$
    * 102 : $-\lambda_1 d\lambda_0 \wedge d\lambda_2$
    * 120 : $+\lambda_1 d\lambda_2 \wedge d\lambda_0$
    * 201 : $+\lambda_2 d\lambda_0 \wedge d\lambda_1$
    * 210 : $-\lambda_2 d\lambda_1 \wedge d\lambda_0$

    Since the wedge product is antisymmetric, the six term simplifies to

    $$
    W_{012} = 2(\lambda_0 d\lambda_1 \wedge d\lambda_2 +
    \lambda_1 d\lambda_2 \wedge d\lambda_0 +
    \lambda_2 d\lambda_0 \wedge d\lambda_1)
    $$

    as per the first definition. This function constructs the "router" tensor
    using this method that, when contracted with the appropriate barycentric
    weights and differentials tensors, construct the correct Whitney basis forms.
    """
    # To illustrate this function, let us consider a 2-simplex and the 1-forms
    # defined on its edges. The whitney 1-form basis functions are
    # W_ij = λ_i dλ_j - λ_j dλ_i.

    # Enumerate all the 1-faces: 01, 02, 12.
    faces = enumerate_local_faces(splx_dim, form_deg, device="cpu").tolist()

    # The router has shape (3, 3, 3), because there are three 1-faces, 3 vertices/
    # barycentric weights (λ), and three barycentric differentials (dλ).
    router_shape = (len(faces),) + (splx_dim + 1,) * (form_deg + 1)
    router = torch.zeros(router_shape, dtype=dtype, device=device)

    # Consider face 12. There are two permutations of its vertices: 12 and 21,
    # with the permutation signs +1 and -1, and the router gets
    # router[2, 1, 2] = 1 (+λ_1 dλ_2) and router[2, 2, 1] = -1 (-λ_2 dλ_1).
    for splx_idx, splx in enumerate(faces):
        perms = torch.tensor(
            list(itertools.permutations(splx)), dtype=torch.int64, device=device
        )
        signs = compute_lex_rel_orient(perms).to(dtype=dtype, device=device)
        # Note that the perms.T is required for advanced multi-dim indexing.
        # For example, to assign (i, j) -> +1, (i, k) -> -1 and (k, j) -> +1,
        # the pytorch expect router[[i, i, k], [j, k, j]] = [+1, -1, +1].
        router[splx_idx][perms.T.unbind(0)] = signs

    return router


def compute_moments(
    order: int,
    splx_dim: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    r"""
    Compute all moment integrals of a given order over a reference simplex.

    Parameters
    ----------
    order
        The order of the moment integrals.
    splx_dim
        The dimension of the reference simplex.
    device
        The device of the output tensor.
    dtype
        The dtype of the output tensor.

    Returns
    -------
    Given a reference simplex of dimension $n$ and moment order $d$, this function
    returns a tensor of shape `(n + 1, ..., n + 1)` (`d` times). An element at
    index `(i_1, ..., i_d)` corresponds to the integral of the product of barycentric
    coordinate functions $\lambda_{i_1} \cdots \lambda_{i_d}$ over the reference
    simplex.

    Notes
    -----
    For a ref $n$-simplex with unit volume and $n + 1$ barycentric coordinate
    functions $\lambda_i$, the magic formula gives

    $$
    \int_\Omega \prod_i \lambda_i^{m_i}\,dV =
    \frac{n! \prod_i m_i!}{(n + \sum_i m_i)!}
    $$
    """
    verts = list(range(splx_dim + 1))
    moments = torch.zeros((len(verts),) * order)

    for lambdas in itertools.product(verts, repeat=order):
        exponents = [lambdas.count(v) for v in verts]
        numerator = math.factorial(splx_dim) * math.prod(
            [math.factorial(m) for m in exponents]
        )
        denominator = math.factorial(splx_dim + sum(exponents))
        moments[lambdas] = numerator / denominator

    return moments.to(device=device, dtype=dtype)


def dispatch_bc_grad_dot(
    mesh: SimplicialMesh,
) -> tuple[Float[Tensor, "splx vert vert"], Float[Tensor, " splx"]]:
    """Dispatch the correct bary_coord_grad_inner_prods() for tri or tet meshes."""
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

__all__ = ["mixed_mass", "vector_mass", "galerkin_flat", "galerkin_sharp"]

from typing import Any, Literal

import torch
from jaxtyping import Float
from torch import Tensor

from ...complex import SimplicialMesh
from ...metric.tet import _tet_geometry, tet_hodge_stars, tet_masses
from ...metric.tri import _tri_geometry, tri_hodge_stars, tri_masses
from ...sparse.decoupled_tensor import (
    BaseDecoupledTensor,
    SparseDecoupledTensor,
)
from ...sparse.linalg.solvers._sparse_solver import InvSparseOperator
from . import _galerkin_element, _galerkin_vertex


def mixed_mass(
    mesh: SimplicialMesh, mode: Literal["element", "vertex"]
) -> Float[SparseDecoupledTensor, "splx*coord edge"]:
    r"""
    Compute the mixed/cross mass matrix.

    The mixed mass matrix consists of inner products between the basis functions
    of the vector space and the Whitney 1-form basis functions of the lowest forms.
    This matrix is also sometimes known as the Galerkin projection matrix or the
    coupling matrix.

    Parameters
    ----------
    mesh
        A simplicial mesh.
    mode
        If `mode` is "element", then the vector space basis functions are defined
        to be piecewise-constant per top-level simplex; if `mode` is "vertex",
        then the vector space basis functions are defined to be piecewise-linear
        and anchored at the vertices.

    Returns
    -------
    [splx*coord, edge]
        The mixed/cross mass matrix. If `mode` is "element", then `splx` refers
        to the number of top-level simplices; if `mode` is "vertex", then `splx`
        refers to the number of vertices. In either case, `coord` is 3.

    Notes
    -----
    Let $W_{ij}(r)$ be the (sharp of the) Whitney 1-form basis function of the
    lowest order associated with the edge $ij$, which is defined as

    $$
    W_{ij}(r) = \lambda_i(r)\nabla\lambda_j(r) - \lambda_j(r)\nabla\lambda_i(r)
    $$

    where $\lambda_i(r)$ is the barycentric coordinate function of point $r$
    associated with the vertex $i$. Note that the gradients of the barycentric
    coordinate functions are constant vectors independent of $r$.

    If `mode` is "element", the vector space basis functions are attached to
    the top-level simplices and defined as the set $\{1_\sigma e_c\}$, where
    $1_\sigma$ is the indicator function for the top-level simplex $\sigma$,
    and $e_c$ is one of the three standard Cartesian basis vectors
    $\{e_x, e_y, e_z\}$. Then, the mixed mass matrix $P$ can be defined
    elementwise for each combination of top-level simplex $\sigma$, Cartesian
    basis vector $e_c$, and edge $ij$ as

    $$
    P_{\sigma c, ij} =
    \int_\sigma \left<e_c, W_{ij}(r)\right> dV =
    \left<e_c, \int_\sigma W_{ij}(r)\,dV\right>
    $$

    Using the magic formula, one can show that the integral of $W_{ij}(r)$ over a
    top-level simplex $\sigma$ is given by

    $$
    \int_\sigma W_{ij}(r) \,dV =
    \frac{\left|\sigma\right|}{d+1} (\nabla\lambda_j - \nabla\lambda_i)
    $$

    where $|\sigma|$ is the area/volume of $\sigma$ and $d$ is the mesh dimension.

    If `mode` is "vertex", then the vector space basis functions are attached to the
    vertices and defined as the set $\{\lambda_v(r)e_c\}$. Then, the mixed mass
    matrix $P$ can be defined elementwise for each combination of vertex $v$,
    Cartesian basis vector $e_c$, and edge $ij$ as

    $$
    P_{vc, ij} = \int_\Omega \left<\lambda_v(r) e_c, W_{ij}(r) \right> dV =
    \left<e_c, \int_\Omega \lambda_v(r) W_{ij}(r)\,dV\right>
    $$

    In particular,

    $$
    \int_\Omega \lambda_v(r) W_{ij}(r)\,dV =
    M_{0,vi}\nabla\lambda_j - M_{0,vj}\nabla\lambda_i
    $$

    where $M_0$ is the consistent mass matrix for discrete 0-forms.
    """
    match mesh.dim:
        case 2:
            tri_areas, bary_coords_grad = _tri_geometry.compute_bc_grads(
                vert_coords=mesh.vert_coords, tris=mesh.tris
            )

        case 3:
            tet_signed_vols, bary_coords_grad = _tet_geometry.compute_bc_grads(
                vert_coords=mesh.vert_coords, tets=mesh.tets
            )
            tet_unsigned_vols = torch.abs(tet_signed_vols)

        case _:
            raise ValueError(f"Unsupported mesh dimension {mesh.dim}.")

    match (mode, mesh.dim):
        case ("element", 2):
            return _galerkin_element.element_based_tri_mixed_mass_matrix(
                mesh=mesh,
                tri_areas=tri_areas,
                bary_coords_grad=bary_coords_grad,
            )

        case ("element", 3):
            return _galerkin_element.element_based_tet_mixed_mass_matrix(
                mesh=mesh,
                tet_unsigned_vols=tet_unsigned_vols,
                bary_coords_grad=bary_coords_grad,
            )

        case ("vertex", 2):
            return _galerkin_vertex.vertex_based_tri_mixed_mass_matrix(
                mesh=mesh,
                tri_areas=tri_areas,
                bary_coords_grad=bary_coords_grad,
            )

        case ("vertex", 3):
            return _galerkin_vertex.vertex_based_tet_mixed_mass_matrix(
                mesh=mesh,
                tet_unsigned_vols=tet_unsigned_vols,
                bary_coords_grad=bary_coords_grad,
            )

        case _:
            raise ValueError(f"Unknown mode argument '{mode}'.")


def vector_mass(
    mesh: SimplicialMesh,
    mode: Literal["element", "vertex"],
    diagonal: bool = False,
) -> Float[BaseDecoupledTensor, "splx*coord splx*coord"]:
    r"""
    Compute the vector mass matrix.

    Parameters
    ----------
    mesh
        A simplicial mesh.
    mode
        If `mode` is "element", then the vector space basis functions are defined
        to be piecewise-constant per top-level simplex; if `mode` is "vertex",
        then the vector space basis functions defined to be piecewise-linear and
        anchored at the vertices.
    diagonal
        If `mode` is "vertex", `diagonal=False` computes the exact vector mass
        matrix from the consistent 0-mass matrix, which is in general not diagonal,
        while `diagonal=True` computes an approximate, diagonal vector mass matrix
        from the Hodge 0-star matrix. If `mode` is "element", the vector mass
        matrix is always diagonal and this argument is ignored.

    Returns
    -------
    [splx*coord, splx*coord]
        The vector mass matrix. If `mode` is "element", then `splx` refers to
        the number of top-level simplices; if `mode` is "vertex", then `splx`
        refers to the number of vertices. In either case, `coord` is 3.

    Notes
    -----
    If `mode` is "element", the vector space basis functions are attached to
    the top-level simplices and defined as the set $\{1_\sigma e_i\}$, where
    $1_\sigma$ is the indicator function for the top-level simplex $\sigma$,
    and $e_i$ is one of the three standard Cartesian basis vectors
    $\{e_x, e_y, e_z\}$. Then, the vector mass matrix $M_V$ can be defined
    elementwise for each simplex-basis pair as

    $$
    M_{\sigma i, \tau j} =
    \int_{\sigma\cap\tau} \left<e_i, e_j\right> dV =
    \left|\sigma\cap\tau\right| \delta_{ij}
    $$

    where $\left|\sigma\cap\tau\right|$ is the volume of the intersection between the
    top-level simplex $\sigma$ and $\tau$. We assume that the mesh is an immersion
    in $\mathbb R^3$, in which case $\left|\sigma\cap\tau\right|$ is nonzero iff
    $\sigma=\tau$. Therefore, the element-based $M_V$ is a diagonal matrix scaled
    by the top-level simplex areas/volumes.

    If `mode` is "vertex", then the vector space basis functions are attached to the
    vertices and defined as the set $\{\lambda_v(r)e_c\}$, where $\lambda_i(r)$ is
    the barycentric coordinate function of point $r$ associated with the vertex
    $i$. The vector mass matrix $M_V$ can be defined elementwise for each
    vertex-basis pair as

    $$
    M_{u i, v j} = \int_\Omega \left<\lambda_u(r) e_i, \lambda_v(r) e_j \right> dV =
    \delta_{ij}M_{0,uv}
    $$

    where $M_0$ is the consistent mass matrix for discrete 0-forms. This vertex-based
    $M_V$ has a block structure where each $(u, v)$ block is a 3x3 diagonal matrix.
    For example, For a ref triangle consisting of three vertices with $M_0$ is given by

    ```
    a b b
    b a b
    b b a
    ```

    The $M_V$ is then given by

    ```
    a 0 0 | b 0 0 | b 0 0
    0 a 0 | 0 b 0 | 0 b 0
    0 0 a | 0 0 b | 0 0 b
    ---------------------
    b 0 0 | a 0 0 | b 0 0
    0 b 0 | 0 a 0 | 0 b 0
    0 0 b | 0 0 a | 0 0 b
    ---------------------
    b 0 0 | b 0 0 | a 0 0
    0 b 0 | 0 b 0 | 0 a 0
    0 0 b | 0 0 b | 0 0 a
    ```
    """
    match (mode, mesh.dim):
        case ("element", 2):
            tri_areas = _tri_geometry.compute_tri_areas(mesh.vert_coords, mesh.tris)
            return _galerkin_element.element_based_tri_vector_mass_matrix(tri_areas)

        case ("element", 3):
            tet_unsigned_vols = torch.abs(
                _tet_geometry.compute_tet_signed_vols(mesh.vert_coords, mesh.tets)
            )
            return _galerkin_element.element_based_tet_vector_mass_matrix(
                tet_unsigned_vols
            )

        case ("vertex", 2):
            if diagonal:
                star_0 = tri_hodge_stars.star_0(mesh)
                return _galerkin_vertex.vertex_based_diag_vector_mass_matrix(star_0)
            else:
                mass_0 = tri_masses.mass_0(mesh)
                return _galerkin_vertex.vertex_based_consistent_vector_mass_matrix(
                    mesh=mesh, mass_0=mass_0
                )

        case ("vertex", 3):
            if diagonal:
                star_0 = tet_hodge_stars.star_0(mesh)
                return _galerkin_vertex.vertex_based_diag_vector_mass_matrix(star_0)
            else:
                mass_0 = tet_masses.mass_0(mesh)
                return _galerkin_vertex.vertex_based_consistent_vector_mass_matrix(
                    mesh=mesh, mass_0=mass_0
                )

        case _:
            raise ValueError(
                f"Unsupported mesh dimension ({mesh.dim}) and/or mode '{mode}' argument."
            )


def galerkin_flat(
    vec_field: Float[Tensor, "splx coord=3"],
    mass_1: Float[BaseDecoupledTensor, "edge edge"]
    | Float[InvSparseOperator, "edge edge"],
    mass_mixed: Float[SparseDecoupledTensor, "splx*coord edge"],
    mode: Literal["element", "vertex"],
    solver_kwargs: dict[str, Any] | None = None,
) -> Float[Tensor, " edge"]:
    r"""
    Compute the flat of a vector field using the Galerkin projection method.

    For a vector field $V$, this function finds a 1-cochain $\eta$ that best
    approximates the $V^\flat$ by solving a linear system of the form

    $$M_1 \eta = P^T V$$

    where $M_1$ is the 1-mass matrix and $P$ is the mixed mass matrix.

    Parameters
    ----------
    vec_field : [splx, coord]
        The input vector field. The interpretation of the first `splx` dimension
        depends on the `mode` argument. If `mode` is "element", then `splx` refers
        to the number of top-level simplices; if `mode` is "vertex", then `splx`
        refers to the number of vertices. In either case, `coord` is 3.
    mass_1 : [edge, edge]
        The 1-mass matrix. If this is a callable `InvSparseOperator`, the RHS
        ($P^T V$) will be passed to the operator to solve for $\eta$; if the mass
        matrix is approximated with a `DiagDecoupledTensor` Hodge 0-star matrix,
        the matrix is directly inverted to solve for $\eta$; if the mass matrix
        is a `SparseDecoupledTensor`, it will be converted to a dense tensor and
        `torch.linalg.solve()` will be used to solve for $\eta$.
    mass_mixed : [splx*coord, edge]
        The mixed mass matrix computed via `mixed_mass()` using the same `mode`
        argument to this function.
    mode
        If `mode` is "element", the input vector field should be piecewise-constant
        and defined over the top-level-simplices of the mesh; if `mode` is "vertex",
        the input vector field should be piecewise-linear and defined over the
        vertices of the mesh. The input `mass_mixed` matrix should be computed
        using the same `mode`
        argument.
    solver_kwargs
        If `mass_1` is a callable `InvSparseOperator`, additional keyword arguments
        can be passed to the sparse solver here.

    Returns
    -------
    [edge,]
        A 1-cochain representing the flat of the input vector field.

    Notes
    -----
    Given a vector field $V$, we want to find a discrete 1-form $\eta$ that best
    approximates $V^\flat$. Using the Galerkin projection approach, this is equivalent
    to asserting that the error $\epsilon = V^\flat - \eta$ is orthogonal to the space of
    test functions, which is spanned by the Whitney 1-form bases; i.e.,

    $$
    \int_\Omega \left<V^\flat, W^i\right> dV = \int_\Omega \left<\eta, W^i\right> dV
    $$

    for all basis functions $W^i$. To further simplify this equation, we expand
    $\eta$ using the bases of the trial space, which are the same Whitney 1-form bases.
    With the Einstein notation, this expansion can be written as $\eta = \eta_j W^j$.
    In addition, we expand $V$ using the vector space basis functions (see
    `vector_mass()` for more details), $V = V^k\phi_k$. Taken together, the orthogonality
    condition can be written as

    $$
    \int_\Omega \left<\left(V^k\phi_k\right)^\flat, W^i\right> dV =
    \int_\Omega \left<\eta_j W^j, W^i\right> dV
    $$

    Recall that the integral of $\left<W^j, W^i\right>$ defines the elements of the
    consistent 1-mass matrix $M_1$ and the integral of $\left<\phi_k^\flat, W^i\right>$,
    which is equivalent to $\left<\phi_k, (W^i)^\sharp\right>$, defines the elements
    of the mixed mass matrix $P$ (see `mixed_mass()` for more details). Therefore,
    this equation can more concisely written as the linear system

    $$M_1 \eta = P^T V$$

    where $\eta$ denotes the unknown 1-cochain (i.e., a vector containing the $\eta_j$
    coefficients).
    """
    if solver_kwargs is None:
        solver_kwargs = {}

    match mode:
        case "element":
            return _galerkin_element.element_based_galerkin_flat(
                vec_field, mass_1, mass_mixed, solver_kwargs
            )

        case "vertex":
            return _galerkin_vertex.vertex_based_galerkin_flat(
                vec_field, mass_1, mass_mixed, solver_kwargs
            )

        case _:
            raise ValueError(f"Unknown mode argument '{mode}'.")


def galerkin_sharp(
    cochain_1: Float[Tensor, " edge"],
    mass_vec: Float[BaseDecoupledTensor, "splx*coord splx*coord"]
    | Float[InvSparseOperator, "splx*coord splx*coord"],
    mass_mixed: Float[SparseDecoupledTensor, "splx*coord edge"],
    mode: Literal["element", "vertex"],
    solver_kwargs: dict[str, Any] | None = None,
) -> Float[Tensor, "splx coord=3"]:
    r"""
    Compute the sharp of a 1-cochain using the Galerkin projection method.

    For a 1-cochain $\eta$, this function finds a vector field $V$ that best
    approximates the $\eta^\sharp$ by solving a linear system of the form

    $$M_V V = P \eta$$

    where $M_V$ is the vector mass matrix and $P$ is the mixed mass matrix.

    Parameters
    ----------
    cochain_1 : [edge,]
        The input 1-cochain.
    mass_vec : [splx*coord, splx*coord]
        The vector mass matrix computed via `vector_mass()` using the same `mode`
        argument to this function. If this is a callable `InvSparseOperator`, the
        RHS ($P@η$) will be passed to the operator to solve for $V$; if it is a
        `DiagDecoupledTensor`, the vector mass matrix is directly inverted to
        solve for $V$; if it is a `SparseDecoupledTensor`, it will be converted
        to a dense tensor and `torch.linalg.solve()` will be used to solve for $V$.
        If `mode` is "element", the vector mass matrix should always be a
        `DiagDecoupledTensor`.
    mass_mixed : [splx*coord, edge]
        The mixed mass matrix computed via `mixed_mass()` using the same `mode`
        argument to this function.
    mode
        If `mode` is "element", the output vector field will be piecewise-constant
        and defined over the top-level-simplices of the mesh; if `mode` is "vertex",
        the output vector field will be piecewise-linear and defined over the
        vertices of the mesh. The input `mass_vec` and `mass_mixed` matrix should
        be computed using the same `mode` argument.
    solver_kwargs
        If `mass_vec` is a callable `InvSparseOperator`, additional keyword
        arguments can be passed to the sparse solver here.

    Returns
    -------
    [splx, coord]
        A vector field representing the sharp of the input 1-cochain. If `mode`
        is "element", then `splx` refers to the number of top-level simplices;
        if `mode` is "vertex", then `splx` refers to the number of vertices.
        In either case, `coord` is 3.

    Notes
    -----
    Given a discrete 1 form $\eta$, we want to find a vector field $V$ that best
    approximates $\eta^\sharp$. Using the Galerkin projection approach, this is
    equivalent to asserting that the error $\epsilon = \eta^\sharp - V$ is orthogonal
    to the space of test functions, $\{\phi_i\}$ (see `vector_mass()` for more
    details); i.e.,

    $$
    \int_\Omega \left<\eta^\sharp, \phi_i\right> dV = \int_\Omega \left<V, \phi_i\right> dV
    $$

    for all basis functions $\phi_i$. To further simplify this equation, we expand
    $\eta$ using the bases of the trial space, which are the Whitney 1-form bases.
    With the Einstein notation, this expansion can be written as $\eta = \eta_j W^j$.
    In addition, we expand $V$ using the same vector space basis functions,
    $V = V^k\phi_k$. Taken together, the orthogonality condition can be written as

    $$
    \int_\Omega \left<\left(\eta_j W^j\right)^\sharp, \phi_i\right> dV =
    \int_\Omega \left<V^k\phi_k, \phi_j\right> dV
    $$

    Recall that the integral of $\left<\phi_i, (W^j)^\sharp\right>$ defines the elements
    of the mixed mass matrix $P$ (see `mixed_mass()` for more details) and the integral
    of $\left<\phi_k, \phi_j\right>$ defines the elements of the vector mass matrix
    $M_V$ (see `vector_mass()` for more details). Therefore, this equation can more
    concisely written as the linear system

    $$M_V V = P \eta$$

    where $\eta$ denotes a 1-cochain (i.e., a vector containing the $\eta_j$
    coefficients).
    """
    match mode:
        case "element":
            return _galerkin_element.element_based_galerkin_sharp(
                cochain_1, mass_vec, mass_mixed
            )

        case "vertex":
            if solver_kwargs is None:
                solver_kwargs = {}

            return _galerkin_vertex.vertex_based_galerkin_sharp(
                cochain_1, mass_vec, mass_mixed, solver_kwargs
            )

        case _:
            raise ValueError(f"Unknown mode argument '{mode}'.")

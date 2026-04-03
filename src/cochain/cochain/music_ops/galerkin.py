from typing import Literal

import torch as t
from jaxtyping import Float

from ...complex import SimplicialMesh
from ...geometry.tet import tet_hodge_stars, tet_masses
from ...geometry.tet.tet_geometry import (
    compute_tet_signed_vols,
    dompute_d_tet_signed_vols_d_vert_coords,
)
from ...geometry.tri import tri_hodge_stars, tri_masses
from ...geometry.tri.tri_geometry import (
    compute_d_tri_areas_d_vert_coords,
    compute_tri_areas,
)
from ...sparse.decoupled_tensor import (
    BaseDecoupledTensor,
    SparseDecoupledTensor,
)
from . import _galerkin_element, _galerkin_vertex


def mixed_mass(
    mesh: SimplicialMesh, mode: Literal["element", "vertex"]
) -> Float[SparseDecoupledTensor, "splx*coord edge"]:
    """
    Compute the cross/mixed mass matrix consisting of the inner products between
    the basis functions of the vector space and the Whitney 1-form basis functions
    of the lowest forms.

    If `mode` is "element", then the vector space basis functions are defined to
    be piecewise-constant per top-level simplex; i.e., the basis vectors are
    (ϕ_σ*e_1, ϕ_σ*e_2, ϕ_σ*e_3), where ϕ_σ is the indicator function for the top
    level simplex σ, and the e's are the standard Cartesian basis vectors. If
    `mode` is "vertex", then the vector space basis functions are anchored at
    the vertices and defined as ϕ_vi(p) = λ_v(p)e_i, where λ_v(p) is the barycentric
    coordinate of point p associated with vertex v and e_i is one of the standard
    Cartesian basis functions.

    This matrix is also sometimes known as the Galerkin projection matrix or the
    coupling matrix.
    """
    match mesh.dim:
        case 2:
            tri_areas = compute_tri_areas(mesh.vert_coords, mesh.tris)

            d_tri_areas_d_vert_coords = compute_d_tri_areas_d_vert_coords(
                mesh.vert_coords, mesh.tris
            )
            bary_coords_grad: Float[t.Tensor, "tri vert=3 coord=3"] = (
                d_tri_areas_d_vert_coords / tri_areas.view(-1, 1, 1)
            )

        case 3:
            tet_signed_vols = compute_tet_signed_vols(mesh.vert_coords, mesh.tets)
            tet_unsigned_vols = t.abs(tet_signed_vols)

            d_signed_vols_d_vert_coords = dompute_d_tet_signed_vols_d_vert_coords(
                mesh.vert_coords, mesh.tets
            )
            bary_coords_grad: Float[t.Tensor, "tet vert=4 coord=3"] = (
                d_signed_vols_d_vert_coords / tet_signed_vols.view(-1, 1, 1)
            )

    match (mode, mesh.dim):
        case ("element", 2):
            return _galerkin_element.element_based_tri_mixed_mass_matrix(
                n_edges=mesh.n_edges,
                tri_edge_idx=mesh.edge_faces.idx,
                tri_edge_orientations=mesh.edge_faces.parity,
                tri_areas=tri_areas,
                bary_coords_grad=bary_coords_grad,
            )

        case ("element", 3):
            return _galerkin_element.element_based_tet_mixed_mass_matrix(
                n_edges=mesh.n_edges,
                tet_edge_idx=mesh.edge_faces.idx,
                tet_edge_orientations=mesh.edge_faces.parity,
                tet_unsigned_vols=tet_unsigned_vols,
                bary_coords_grad=bary_coords_grad,
            )

        case ("vertex", 2):
            return _galerkin_vertex.vertex_based_tri_mixed_mass_matrix(
                n_verts=mesh.n_verts,
                n_edges=mesh.n_edges,
                tris=mesh.tris,
                tri_edge_idx=mesh.edge_faces.idx,
                tri_edge_orientations=mesh.edge_faces.parity,
                tri_areas=tri_areas,
                bary_coords_grad=bary_coords_grad,
            )

        case ("vertex", 3):
            return _galerkin_vertex.vertex_based_tet_mixed_mass_matrix(
                n_verts=mesh.n_verts,
                n_edges=mesh.n_edges,
                tets=mesh.tets,
                tet_edge_idx=mesh.edge_faces.idx,
                tet_edge_orientations=mesh.edge_faces.parity,
                tet_unsigned_vols=tet_unsigned_vols,
                bary_coords_grad=bary_coords_grad,
            )

        case _:
            raise ValueError()


def vector_mass(
    mesh: SimplicialMesh, mode: Literal["element", "vertex"], diagonal: bool = False
) -> Float[BaseDecoupledTensor, "splx*coord splx*coord"]:
    """
    Compute the vector mass matrix consisting of the inner products between the
    basis functions of the vector space. The `mode` argument controls the definition
    of the basis functions (see `mixed_mass()` for more details).

    If `mode` is "element", then the returned matrix is always diagonal. If `mode`
    is "vertex", `diagonal=False` computes the exact vector mass matrix from the
    consistent mass-0 matrix, which is in general not diagonal, and `diagonal=True`
    computes an approximate, diagonal mass matrix from the Hodge star-0 matrix.
    """
    match (mode, mesh.dim):
        case ("element", 2):
            tri_areas = compute_tri_areas(mesh.vert_coords, mesh.tris)
            return _galerkin_element.element_based_tri_vector_mass_matrix(tri_areas)

        case ("element", 3):
            tet_signed_vols = compute_tet_signed_vols(mesh.vert_coords, mesh.tets)
            tet_unsigned_vols = t.abs(tet_signed_vols)
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
                    mass_0
                )

        case ("vertex", 3):
            if diagonal:
                star_0 = tet_hodge_stars.star_0(mesh)
                return _galerkin_vertex.vertex_based_diag_vector_mass_matrix(star_0)
            else:
                mass_0 = tet_masses.mass_0(mesh)
                return _galerkin_vertex.vertex_based_consistent_vector_mass_matrix(
                    mass_0
                )

        case _:
            raise ValueError()


def galerkin_flat(
    vec_field: Float[t.Tensor, "splx coord"],
    mass_1: Float[BaseDecoupledTensor, "edge edge"],
    mass_mixed: Float[SparseDecoupledTensor, "splx*coord edge"],
    mode: Literal["element", "vertex"],
    method: Literal["dense", "solver", "inv_star"],
) -> Float[t.Tensor, " edge"]:
    """
    Compute the flat of a vector field using the Galerkin projection method.

    If `mode` is "element", the input vector field should be piecewise-constant
    and defined over the top-level-simplices of the mesh; if `mode` is "vertex",
    the input vector field should be defined over the vertices of the mesh. The
    input `mass_mixed` matrix should be computed using the same `mode` argument.

    This function requires solving a linear system of the form M_1@η = P.T@v, where
    M_1 is the edge mass matrix, η is the 1-cochain, P is the mixed mass matrix,
    and v is the vector field. If `method` is "dense", the `mass_1` matrix is
    converted to a dense tensor first before invoking `torch.linalg.solve()`;
    if `method` is "solver", the linear system is passed to a sparse system solver;
    if `method` is "inv_star", the `mass_1` matrix is assumed to be diagonal (e.g.,
    a Hodge star-1 matrix) and directly inverted to solve the linear system.
    """
    match mode:
        case "element":
            return _galerkin_element.element_based_galerkin_flat(
                vec_field, mass_1, mass_mixed, method
            )

        case "vertex":
            return _galerkin_vertex.vertex_based_galerkin_flat(
                vec_field, mass_1, mass_mixed, method
            )

        case _:
            raise ValueError()


def galerkin_sharp(
    cochain_1: Float[t.Tensor, " edge"],
    mass_vec: Float[BaseDecoupledTensor, "splx*coord splx*coord"],
    mass_mixed: Float[SparseDecoupledTensor, "splx*coord edge"],
    mode: Literal["element", "vertex"],
    method: Literal["dense", "solver", "inv_star"] | None = None,
) -> Float[t.Tensor, "splx coord=3"]:
    """
    Compute the sharp of a 1-cochain using the Galerkin projection method.

    If `mode` is "element", the output vector field will be piecewise-constant
    and defined over the top-level-simplices of the mesh; if `mode` is "vertex",
    the output vector field will be defined over the vertices of the mesh. The
    input `mass_vec` and `mass_mixed` matrix should be computed using the same
    `mode` argument.

    This function requires solving a linear system of the form M_V@v = P@η, where
    M_V is the vector mass matrix, v is the vector field, P is the mixed mass matrix,
    and η is the 1-cochain. If `mode` is "element", the input `mass_vec` is assumed
    to be diagonal and directly inverted to solve the system. If `mode` is "vertex",
    the solution method is controlled by the `method` argument. If `method` is "dense",
    the `mass_vec` matrix is converted to a dense tensor first before invoking
    `torch.linalg.solve()`; if `method` is "solver", the linear system is passed
    to a sparse system solver; if `method` is "inv_star", the `mass_vec` matrix is
    assumed to be diagonal (e.g., derived from a Hodge star-0 matrix) and directly
    inverted to solve the linear system.
    """
    match mode:
        case "element":
            return _galerkin_element.element_based_galerkin_sharp(
                cochain_1, mass_vec, mass_mixed
            )

        case "vertex":
            if method is None:
                raise ValueError()

            return _galerkin_vertex.vertex_based_galerkin_sharp(
                cochain_1, mass_vec, mass_mixed, method
            )

        case _:
            raise ValueError()

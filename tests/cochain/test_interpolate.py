from random import random

import pytest
import torch as t
from einops import einsum, rearrange, repeat
from jaxtyping import Float, Integer

from cochain.cochain.discretize import DeRhamMap
from cochain.cochain.interpolate import barycentric_whitney_map
from cochain.geometry.tet.tet_geometry import (
    d_tet_signed_vols_d_vert_coords,
    get_tet_signed_vols,
)
from cochain.geometry.tri.tri_geometry import (
    compute_d_tri_areas_d_vert_coords,
    compute_tri_areas,
)
from cochain.utils.faces import enumerate_local_faces
from cochain.utils.quadrature import Dunavant, GaussLegendre, Keast


@pytest.mark.parametrize(
    "mesh, k, quad",
    [
        ("hollow_tet_mesh", 1, GaussLegendre),
        ("hollow_tet_mesh", 2, Dunavant),
        ("two_tets_mesh", 1, GaussLegendre),
        ("two_tets_mesh", 2, Dunavant),
        ("two_tets_mesh", 3, Keast),
    ],
)
def test_interpolate_discretize_left_inverse(mesh, k, quad, device, request):
    """
    Test that the de Rham map is the left inverse of the Whitney map; i.e., applying
    the Whitney map to interpolate a k-cochain, followed by applying the de Rham
    map to discretize the k-form, gives back the same k-cochain.
    """
    mesh = request.getfixturevalue(mesh).to(device)
    k_cochain_true = t.randn(mesh.splx[k].size(0)).to(device)

    # First, interpolate the discrete k-cochain
    bary_coords, weights = quad(
        dtype=mesh.vert_coords.dtype, device=mesh.vert_coords.device
    ).get_rule(degree=3)

    k_form = barycentric_whitney_map(
        k=k,
        k_cochain=k_cochain_true,
        bary_coords=bary_coords.unsqueeze(0),
        mesh=mesh,
        mode="boundary",
        boundary_reduction="mean",
    )

    # Then, discretize the interpolated k-form
    de_rham = DeRhamMap(k=k, quad_degree=3, mesh=mesh, allow_neg_weights=True)
    k_cochain_reconstructed = de_rham.discretize(k_form)

    t.testing.assert_close(k_cochain_reconstructed, k_cochain_true)


@pytest.mark.parametrize("mesh", ["hollow_tet_mesh", "two_tets_mesh"])
def test_commutativity_with_d_on_0_form(mesh, request, device):
    """
    Test that the Whitney map W commutes with the exterior derivative d using
    0-cochains; i.e., for any 0-cochain η, W(dη) = d(Wη).
    """
    mesh = request.getfixturevalue(mesh).to(device)

    # Generate a common set of sampled points on the interior of the top-level simplices.
    match mesh.dim:
        case 2:
            bary_coords, _ = Dunavant(
                dtype=mesh.vert_coords.dtype, device=device
            ).get_rule(degree=3)
        case 3:
            bary_coords, _ = Keast(
                dtype=mesh.vert_coords.dtype, device=device
            ).get_rule(degree=3)

    # Test a random 0-cochain with 2 channel dimensions.
    cochain_0 = t.randn((mesh.n_verts, 2), dtype=mesh.vert_coords.dtype, device=device)
    d_0 = mesh.cbd[0]

    # First, compute the interpolation of the exterior derivative.
    cochain_1 = d_0 @ cochain_0
    w_d_cochain = barycentric_whitney_map(
        k=1,
        k_cochain=cochain_1,
        bary_coords=bary_coords.unsqueeze(0),
        mesh=mesh,
        mode="interior",
    )

    # Then, compute the exterior derivative of the interpolation. Recall that
    # the interpolated 0-form is expressed as ω(p) = sum[η_i*W_i(p)], where,
    # for 0-forms, W_i = λ_i. THe application of the exterior derivative to this
    # expression is equivalent to taking the gradient: ∇ω(p) = sum[η_i*∇W_i(p)].
    # Therefore, to compute the exterior derivative of the interpolation, we
    # repeat the same logic as in _bary_whitney_tri_cochain_0() and
    # _bary_whitney_tet_cochain_0(), but with the original basis function W_i(p)
    # replaced by ∇W_i(p).
    match mesh.dim:
        case 2:
            tri_areas = rearrange(
                compute_tri_areas(mesh.vert_coords, mesh.tris), "tri -> tri 1 1"
            )
            d_tri_areas_d_vert_coords = compute_d_tri_areas_d_vert_coords(
                mesh.vert_coords, mesh.tris
            )
            bary_coords_grad = d_tri_areas_d_vert_coords / tri_areas
            cochain_0_at_vert_faces = cochain_0[mesh.tris]

        case 3:
            tet_signed_vols = get_tet_signed_vols(mesh.vert_coords, mesh.tets)
            d_signed_vols_d_vert_coords = d_tet_signed_vols_d_vert_coords(
                mesh.vert_coords, mesh.tets
            )
            bary_coords_grad = d_signed_vols_d_vert_coords / tet_signed_vols.view(
                -1, 1, 1
            )
            cochain_0_at_vert_faces = cochain_0[mesh.tets]

        case _:
            raise ValueError()

    # Note that the exterior derivative of the interpolated 0-cochain has no pt
    # dimension (since it should be constant within the top-level simplices).
    d_w_cochain = einsum(
        bary_coords_grad,
        cochain_0_at_vert_faces,
        "top_splx vert coord, top_splx vert ch -> top_splx ch coord",
    )
    d_w_cochain_formed = repeat(
        d_w_cochain, "top_splx ch coord -> top_splx pt ch coord", pt=bary_coords.size(0)
    )

    # Check that the two approaches give the same results.
    t.testing.assert_close(w_d_cochain, d_w_cochain_formed)


@pytest.mark.parametrize("mesh", ["hollow_tet_mesh", "two_tets_mesh"])
def test_commutativity_with_d_on_1_form(mesh, request, device):
    """
    Test that the Whitney map W commutes with the exterior derivative d using
    1-cochains; i.e., for any 1-cochain η, W(dη) = d(Wη).
    """
    mesh = request.getfixturevalue(mesh).to(device)

    # Generate a common set of sampled points on the interior of the top-level simplices.
    match mesh.dim:
        case 2:
            bary_coords, _ = Dunavant(
                dtype=mesh.vert_coords.dtype, device=device
            ).get_rule(degree=3)
        case 3:
            bary_coords, _ = Keast(
                dtype=mesh.vert_coords.dtype, device=device
            ).get_rule(degree=3)

    # Test a random 1-cochain with 2 channel dimensions.
    cochain_1 = t.randn((mesh.n_edges, 2), dtype=mesh.vert_coords.dtype, device=device)
    d_1 = mesh.cbd[1]

    # First, compute the interpolation of the exterior derivative.
    cochain_2 = d_1 @ cochain_1
    w_d_cochain = barycentric_whitney_map(
        k=2,
        k_cochain=cochain_2,
        bary_coords=bary_coords.unsqueeze(0),
        mesh=mesh,
        mode="interior",
    )

    # Then, compute the exterior derivative of the interpolation. Recall that
    # the interpolated 1-form is expressed as ω(p) = sum[η_ij*W_ij(p)], where,
    # for 1-forms, W_ij = λ_i∇λ_j - λ_j∇λ_i. The application of the exterior
    # derivative to this expression is equivalent to taking the curl:
    #
    # ∇ x ω(p) = sum[η_ij * ∇ x W_ij(p)] = sum[η_ij * 2(∇λ_i x ∇λ_j)]
    #
    # Therefore, to compute the exterior derivative of the interpolation, we
    # repeat the same logic as in _bary_whitney_tri_cochain_1() and
    # _bary_whitney_tet_cochain_1(), but with the original basis function W_ij(p)
    # replaced by ∇ x W_ij(p).
    match mesh.dim:
        case 2:
            tri_areas = rearrange(
                compute_tri_areas(mesh.vert_coords, mesh.tris), "tri -> tri 1 1"
            )
            d_tri_areas_d_vert_coords = compute_d_tri_areas_d_vert_coords(
                mesh.vert_coords, mesh.tris
            )
            bary_coords_grad = d_tri_areas_d_vert_coords / tri_areas

            local_edge_idx = enumerate_local_faces(
                simp_dim=2, face_dim=1, device=device
            )

            cochain_1_at_edge_faces = cochain_1[mesh.tri_edge_idx]
            sign_correction = mesh.tri_edge_orientations

        case 3:
            tet_signed_vols = get_tet_signed_vols(mesh.vert_coords, mesh.tets)
            d_signed_vols_d_vert_coords = d_tet_signed_vols_d_vert_coords(
                mesh.vert_coords, mesh.tets
            )
            bary_coords_grad = d_signed_vols_d_vert_coords / tet_signed_vols.view(
                -1, 1, 1
            )

            local_edge_idx = enumerate_local_faces(
                simp_dim=3, face_dim=1, device=device
            )

            cochain_1_at_edge_faces = cochain_1[mesh.tet_edge_idx]
            sign_correction = mesh.tet_edge_orientations

        case _:
            raise ValueError()

    basis = 2.0 * t.cross(
        bary_coords_grad[:, local_edge_idx[:, 0]],
        bary_coords_grad[:, local_edge_idx[:, 1]],
        dim=-1,
    )

    # Note that (1) the exterior derivative of the interpolated 1-cochain has no
    # pt dimension (since it should be constant within the top-level simplices),
    # and (2) we need a sign correction here, in comparison to the 0-form test case,
    # because of potential mismatch between the orientations of local 1-faces vs
    # global canonical 1-simplices.
    d_w_cochain = einsum(
        basis,
        sign_correction,
        cochain_1_at_edge_faces,
        "top_splx edge coord, top_splx edge, top_splx edge ch -> top_splx ch coord",
    )
    d_w_cochain_formed = repeat(
        d_w_cochain, "top_splx ch coord -> top_splx pt ch coord", pt=bary_coords.size(0)
    )

    # Check that the two approaches give the same results.
    t.testing.assert_close(w_d_cochain, d_w_cochain_formed)


def test_commutativity_with_d_on_2_form(two_tets_mesh, request, device):
    """
    Test that the Whitney map W commutes with the exterior derivative d using
    2-cochains; i.e., for any 2-cochain η, W(dη) = d(Wη).
    """
    mesh = two_tets_mesh.to(device)

    # Generate a common set of sampled points on the interior of the top-level simplices.
    bary_coords, _ = Keast(dtype=mesh.vert_coords.dtype, device=device).get_rule(
        degree=3
    )

    # Test a random 1-cochain with 2 channel dimensions.
    cochain_2 = t.randn((mesh.n_tris, 2), dtype=mesh.vert_coords.dtype, device=device)
    d_2 = mesh.cbd[2]

    # First, compute the interpolation of the exterior derivative.
    cochain_3 = d_2 @ cochain_2
    w_d_cochain = barycentric_whitney_map(
        k=3,
        k_cochain=cochain_3,
        bary_coords=bary_coords.unsqueeze(0),
        mesh=mesh,
        mode="interior",
    )

    # Then, compute the exterior derivative of the interpolation. Recall that
    # the interpolated 2-form is expressed as ω(p) = sum[η_ijk*W_ijk(p)], where,
    # for 2-forms, W_ijk = 2(λ_i ∇λ_jx∇λ_k + λ_j ∇λ_kx∇λ_i + λ_k ∇λ_ix∇ λ_j). The
    # application of the exterior derivative to this expression is equivalent to
    # taking the divergence:
    #
    # ∇ ⋅ ω(p) = sum[η_ijk * ∇ ⋅ W_ijk(p)] = sum[η_ij * 6(∇λ_i ⋅ (∇ λ_j x ∇ λ_k))]
    #
    # Therefore, to compute the exterior derivative of the interpolation, we
    # repeat the same logic as in _bary_whitney_tri_cochain_2() and
    # _bary_whitney_tet_cochain_2(), but with the original basis function W_ijk(p)
    # replaced by ∇ ⋅ W_ijk(p).
    tet_signed_vols = get_tet_signed_vols(mesh.vert_coords, mesh.tets)
    d_signed_vols_d_vert_coords = d_tet_signed_vols_d_vert_coords(
        mesh.vert_coords, mesh.tets
    )
    bary_coords_grad = d_signed_vols_d_vert_coords / tet_signed_vols.view(-1, 1, 1)

    local_tri_idx = enumerate_local_faces(simp_dim=3, face_dim=2, device=device)

    cochain_2_at_edge_faces = cochain_2[mesh.tet_tri_idx]
    sign_correction = mesh.tet_tri_orientations

    # Note that the scalar triple product can be more conveniently computed
    # using the matrix determinant; bary_coords_grad[:, local_tri_idx] has the
    # shape (tet, 2_face, vert, coord). The final unsqueeze preserves the coord
    # dimension. It is also possible to write this out more verbosely as
    #
    # basis = 6.0 * t.sum(
    #     bary_coords_grad[:, local_tri_idx[:, 0]]
    #     * t.cross(
    #         bary_coords_grad[:, local_tri_idx[:, 1]],
    #         bary_coords_grad[:, local_tri_idx[:, 2]],
    #         dim=-1,
    #     ),
    #     dim=-1,
    #     keepdim=True,
    # )
    basis = 6.0 * t.linalg.det(bary_coords_grad[:, local_tri_idx]).unsqueeze(-1)

    # Note that (1) the exterior derivative of the interpolated 2-cochain has no
    # pt dimension (since it should be constant within the top-level simplices),
    # and (2) we need a sign correction here, in comparison to the 0-form test case,
    # because of potential mismatch between the orientations of local 2-faces vs
    # global canonical 2-simplices.
    d_w_cochain = einsum(
        basis,
        sign_correction,
        cochain_2_at_edge_faces,
        "top_splx tri coord, top_splx tri, top_splx tri ch -> top_splx ch coord",
    )
    d_w_cochain_formed = repeat(
        d_w_cochain, "top_splx ch coord -> top_splx pt ch coord", pt=bary_coords.size(0)
    )

    # Check that the two approaches give the same results.
    t.testing.assert_close(w_d_cochain, d_w_cochain_formed)


@pytest.mark.parametrize("mesh", ["hollow_tet_mesh", "two_tets_mesh"])
def test_0_form_interpolate_discretize_right_project(mesh, request, device):
    """
    Test that the Whitney interpolation of a 0-cochain discretized from a constant
    0-form returns the original 0-form. This is a consequence of the fact that
    W ∘ π is a projection operator that projects k-forms to the subspace spanned
    by the Whitney basis functions.
    """
    mesh = request.getfixturevalue(mesh).to(device)

    constant = random()

    # The 0-cochain is simply the assignment of the constant to the vertices.
    cochain_0 = constant * t.ones(
        (mesh.n_verts, 1), dtype=mesh.vert_coords.dtype, device=device
    )

    match mesh.dim:
        case 2:
            bary_coords, _ = Dunavant(
                dtype=mesh.vert_coords.dtype, device=device
            ).get_rule(degree=3)
        case 3:
            bary_coords, _ = Keast(
                dtype=mesh.vert_coords.dtype, device=device
            ).get_rule(degree=3)

    form_0 = barycentric_whitney_map(
        k=0,
        k_cochain=cochain_0,
        bary_coords=bary_coords.unsqueeze(0),
        mesh=mesh,
        mode="interior",
    )

    form_0_true = constant * t.ones_like(form_0)

    t.testing.assert_close(form_0, form_0_true)


@pytest.mark.parametrize("mesh", ["hollow_tet_mesh", "two_tets_mesh"])
def test_1_form_interpolate_discretize_right_project(mesh, request, device):
    """
    Test that the Whitney interpolation of a 1-cochain discretized from a constant
    1-form is exact.
    """
    mesh = request.getfixturevalue(mesh).to(device)

    const_vec = t.randn(3, dtype=mesh.vert_coords.dtype, device=device)

    # Discretize the constant 1-form via de Rham map.
    de_rham = DeRhamMap(k=1, quad_degree=3, mesh=mesh)
    pts = de_rham.sample_points()
    n_splx, n_pts, _ = pts.shape

    cochain_1 = de_rham.discretize(
        k_forms=repeat(const_vec, "coord -> simp pt coord", simp=n_splx, pt=n_pts)
    )

    # Interpolate the discretized 1-form via Whitney map.
    match mesh.dim:
        case 2:
            bary_coords, _ = Dunavant(
                dtype=mesh.vert_coords.dtype, device=device
            ).get_rule(degree=3)
        case 3:
            bary_coords, _ = Keast(
                dtype=mesh.vert_coords.dtype, device=device
            ).get_rule(degree=3)

    form_1 = barycentric_whitney_map(
        k=1,
        k_cochain=cochain_1,
        bary_coords=bary_coords.unsqueeze(0),
        mesh=mesh,
        mode="interior",
    )

    # Compare the interpolated 1-form with the true 1-form.
    form_1_true = const_vec.view(1, 1, -1).expand_as(form_1)

    match mesh.dim:
        case 2:
            # For a 2D mesh embedded in 3D space, the component of the 1-form that
            # is perpendicular to the triangles cannot be reconstructed using
            # the Whitney basis functions; therefore, the perp component needs
            # to be removed from the 1-form before comparing against the
            # interpolated 1-form.
            tri_verts = mesh.vert_coords[mesh.tris]

            area_normal = t.cross(
                tri_verts[:, 1] - tri_verts[:, 0],
                tri_verts[:, 2] - tri_verts[:, 0],
                dim=-1,
            )
            area_unormal = area_normal / t.linalg.norm(
                area_normal, dim=-1, keepdim=True
            )
            form_1_perp_comp = einsum(
                const_vec, area_unormal, "coord, tri coord -> tri"
            )
            form_1_perp = form_1_perp_comp.view(-1, 1) * area_unormal
            form_1_tangent = form_1_true - form_1_perp.unsqueeze(1)

            t.testing.assert_close(form_1, form_1_tangent)

        case 3:
            t.testing.assert_close(form_1, form_1_true)


@pytest.mark.parametrize("mesh", ["hollow_tet_mesh", "two_tets_mesh"])
def test_2_form_interpolate_discretize_right_project(mesh, request, device):
    """
    Test that the Whitney interpolation of a 2-cochain discretized from a constant
    2-form is exact.
    """
    mesh = request.getfixturevalue(mesh).to(device)

    const_vec = t.randn(3, dtype=mesh.vert_coords.dtype, device=device)

    # Discretize the constant 2-form via de Rham map.
    de_rham = DeRhamMap(k=2, quad_degree=3, mesh=mesh)
    pts = de_rham.sample_points()
    n_splx, n_pts, _ = pts.shape

    cochain_2 = de_rham.discretize(
        k_forms=repeat(const_vec, "coord -> simp pt coord", simp=n_splx, pt=n_pts)
    )

    # Interpolate the discretized 2-form via Whitney map.
    match mesh.dim:
        case 2:
            bary_coords, _ = Dunavant(
                dtype=mesh.vert_coords.dtype, device=device
            ).get_rule(degree=3)
        case 3:
            bary_coords, _ = Keast(
                dtype=mesh.vert_coords.dtype, device=device
            ).get_rule(degree=3)

    form_2 = barycentric_whitney_map(
        k=2,
        k_cochain=cochain_2,
        bary_coords=bary_coords.unsqueeze(0),
        mesh=mesh,
        mode="interior",
    )

    # Compare the interpolated 2-form with the true 2-form.

    match mesh.dim:
        case 2:
            # For a 2D mesh embedded in 3D space, the component of the 2-form that
            # is tangential to the triangles cannot be reconstructed using
            # the Whitney basis functions; therefore, the tangential component needs
            # to be removed from the 2-form before comparing against the
            # interpolated 2-form.
            tri_verts = mesh.vert_coords[mesh.tris]

            area_normal = t.cross(
                tri_verts[:, 1] - tri_verts[:, 0],
                tri_verts[:, 2] - tri_verts[:, 0],
                dim=-1,
            )
            area_unormal = area_normal / t.linalg.norm(
                area_normal, dim=-1, keepdim=True
            )
            form_2_perp_comp = einsum(
                const_vec, area_unormal, "coord, tri coord -> tri"
            )
            form_2_perp = form_2_perp_comp.view(-1, 1) * area_unormal
            form_2_perp_shaped = form_2_perp.unsqueeze(1).expand_as(form_2)

            t.testing.assert_close(form_2, form_2_perp_shaped)

        case 3:
            form_2_true = const_vec.view(1, 1, -1).expand_as(form_2)
            t.testing.assert_close(form_2, form_2_true)


def test_3_form_interpolate_discretize_right_project(two_tets_mesh, device):
    """
    Test that the Whitney interpolation of a 3-cochain discretized from a constant
    3-form is exact.
    """
    mesh = two_tets_mesh.to(device)

    const_scalar = t.randn(1, dtype=mesh.vert_coords.dtype, device=device)

    # Discretize the constant 3-form via de Rham map.
    de_rham = DeRhamMap(k=3, quad_degree=3, mesh=mesh)
    pts = de_rham.sample_points()
    n_splx, n_pts, _ = pts.shape

    cochain_3 = de_rham.discretize(
        k_forms=repeat(const_scalar, "coord -> simp pt coord", simp=n_splx, pt=n_pts)
    )

    # Interpolate the discretized 3-form via Whitney map.
    bary_coords, _ = Keast(dtype=mesh.vert_coords.dtype, device=device).get_rule(
        degree=3
    )

    form_3 = barycentric_whitney_map(
        k=3,
        k_cochain=cochain_3,
        bary_coords=bary_coords.unsqueeze(0),
        mesh=mesh,
        mode="interior",
    )

    # Compare the interpolated 2-form with the true 3-form.
    form_3_true = const_scalar.view(1, 1, -1).expand_as(form_3)
    t.testing.assert_close(form_3, form_3_true)


def test_1_form_tangential_continuity_on_tri_mesh(two_tris_mesh, device):
    """
    Testing that the interpolated 1-form on the shared 1-face of two triangles
    must agree in their tangential components.
    """
    mesh = two_tris_mesh.to(device)

    cochain_1 = t.randn(mesh.n_edges, dtype=mesh.vert_coords.dtype, device=device)

    bary_coords, _ = GaussLegendre(
        dtype=mesh.vert_coords.dtype, device=device
    ).get_rule(degree=3)

    form_1: Float[t.Tensor, "tri edge pt coord"] = barycentric_whitney_map(
        k=1,
        k_cochain=cochain_1,
        bary_coords=bary_coords.unsqueeze(0),
        mesh=mesh,
        mode="boundary",
        boundary_reduction="none",
    )

    # For the two_tris_mesh, the triangles are defined as [[0, 1, 2], [1, 3, 2]]
    # so the last edge in the first triangle and the second edge in the second
    # triangle are shared.
    form_1_shared_edge_1 = form_1[0, 2]
    form_1_shared_edge_2 = form_1[1, 1]

    shared_edge_vec = mesh.vert_coords[2] - mesh.vert_coords[1]

    form_1_tangent_1 = einsum(
        form_1_shared_edge_1, shared_edge_vec, "pt coord, coord -> pt"
    )
    form_1_tangent_2 = einsum(
        form_1_shared_edge_2, shared_edge_vec, "pt coord, coord -> pt"
    )

    t.testing.assert_close(form_1_tangent_1, form_1_tangent_2)


def test_1_form_tangential_continuity_on_tet_mesh(two_tets_mesh, device):
    """
    Testing that the interpolated 1-form on the shared 1-face of two tets
    must agree in their tangential components.
    """
    mesh = two_tets_mesh.to(device)

    cochain_1 = t.randn(mesh.n_edges, dtype=mesh.vert_coords.dtype, device=device)

    bary_coords, _ = GaussLegendre(
        dtype=mesh.vert_coords.dtype, device=device
    ).get_rule(degree=3)

    form_1: Float[t.Tensor, "tet edge pt coord"] = barycentric_whitney_map(
        k=1,
        k_cochain=cochain_1,
        bary_coords=bary_coords.unsqueeze(0),
        mesh=mesh,
        mode="boundary",
        boundary_reduction="none",
    )

    # For the two_tets_mesh, the triangles are defined as [[0, 1, 2, 4], [2, 1, 0, 3]],
    # so the edges 01, 02, 12 are shared. By enumerate_local_faces(), these three
    # edges correspond to edge 0, 1, 3, and 3, 1, 0 on the two tets, respectively.
    form_1_shared_edge_1 = form_1[0, [0, 1, 3]]
    form_1_shared_edge_2 = form_1[1, [3, 1, 0]]

    shared_edge_vec = mesh.vert_coords[[1, 2, 2]] - mesh.vert_coords[[0, 0, 1]]

    form_1_tangent_1 = einsum(
        form_1_shared_edge_1, shared_edge_vec, "edge pt coord, edge coord -> edge pt"
    )
    form_1_tangent_2 = einsum(
        form_1_shared_edge_2, shared_edge_vec, "edge pt coord, edge coord -> edge pt"
    )

    t.testing.assert_close(form_1_tangent_1, form_1_tangent_2)


def test_2_form_normal_continuity_on_tet_mesh(two_tets_mesh, device):
    """
    Testing that the interpolated 2-form on the shared 2-face of two tets
    must agree in their normal components.
    """
    mesh = two_tets_mesh.to(device)

    cochain_2 = t.randn(mesh.n_tris, dtype=mesh.vert_coords.dtype, device=device)

    bary_coords, _ = Dunavant(dtype=mesh.vert_coords.dtype, device=device).get_rule(
        degree=3
    )

    form_2: Float[t.Tensor, "tet tri pt coord"] = barycentric_whitney_map(
        k=2,
        k_cochain=cochain_2,
        bary_coords=bary_coords.unsqueeze(0),
        mesh=mesh,
        mode="boundary",
        boundary_reduction="none",
    )

    # For the two_tets_mesh, the triangles are defined as [[0, 1, 2, 4], [2, 1, 0, 3]],
    # so the triangle 012 is shared. By enumerate_local_faces(), this is the first
    # triangle face in each tet.
    form_2_shared_tri_1 = form_2[0, 0]
    form_2_shared_tri_2 = form_2[1, 0]

    shared_tri_normal_vec = t.cross(
        mesh.vert_coords[1] - mesh.vert_coords[0],
        mesh.vert_coords[2] - mesh.vert_coords[0],
    )

    form_2_normal_1 = einsum(
        form_2_shared_tri_1, shared_tri_normal_vec, "pt coord, coord -> pt"
    )
    form_2_normal_2 = einsum(
        form_2_shared_tri_2, shared_tri_normal_vec, "pt coord, coord -> pt"
    )

    # Note that, although the triangle faces 012 and 210 have different local
    # orientations, the function barycentric_whitney_map() returns interpolated
    # forms assuming that the 2-simplices assume the canonical orientation; therefore
    # the normal components of the 2-form on the shared 2-faces should point in
    # the same, not the opposite, directions.
    t.testing.assert_close(form_2_normal_1, form_2_normal_2)

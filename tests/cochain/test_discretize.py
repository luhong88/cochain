import pytest
import torch as t
from einops import einsum, repeat

from cochain.cochain.discretize import DeRhamMap
from cochain.geometry.tet.tet_geometry import get_tet_signed_vols


@pytest.mark.parametrize("mesh", ["hollow_tet_mesh", "two_tets_mesh"])
def test_const_1_form_integration(mesh, request, device):
    mesh = request.getfixturevalue(mesh).to(device)

    de_rham = DeRhamMap(k=1, quad_degree=1, mesh=mesh)

    pts = de_rham.sample_points()
    n_simps, n_pts, _ = pts.shape

    const_form = t.randn((2, 3)).to(
        dtype=mesh.vert_coords.dtype, device=mesh.vert_coords.device
    )
    sampled_form = repeat(
        const_form, "ch coord -> simp pt ch coord", simp=n_simps, pt=n_pts
    )
    discretized_cochain = de_rham.discretize(sampled_form)

    edge_verts = mesh.vert_coords[mesh.edges]
    edge_vecs = edge_verts[:, 1] - edge_verts[:, 0]
    dot_prod = einsum(edge_vecs, const_form, "edge coord, ch coord -> edge ch")

    t.testing.assert_close(discretized_cochain, dot_prod)


@pytest.mark.parametrize("mesh", ["hollow_tet_mesh", "two_tets_mesh"])
def test_const_2_form_integration(mesh, request, device):
    mesh = request.getfixturevalue(mesh).to(device)

    de_rham = DeRhamMap(k=2, quad_degree=1, mesh=mesh)

    pts = de_rham.sample_points()
    n_simps, n_pts, _ = pts.shape

    const_form = t.randn((2, 3)).to(
        dtype=mesh.vert_coords.dtype, device=mesh.vert_coords.device
    )
    sampled_form = repeat(
        const_form, "ch coord -> simp pt ch coord", simp=n_simps, pt=n_pts
    )
    discretized_cochain = de_rham.discretize(sampled_form)

    tri_verts = mesh.vert_coords[mesh.tris]
    tri_area_norm = 0.5 * t.cross(
        tri_verts[:, 1] - tri_verts[:, 0], tri_verts[:, 2] - tri_verts[:, 0], dim=-1
    )
    dot_prod = einsum(tri_area_norm, const_form, "tri coord, ch coord -> tri ch")

    t.testing.assert_close(discretized_cochain, dot_prod)


def test_const_3_form_integration(two_tets_mesh, device):
    mesh = two_tets_mesh.to(device)

    de_rham = DeRhamMap(k=3, quad_degree=1, mesh=mesh)

    pts = de_rham.sample_points()
    n_simps, n_pts, _ = pts.shape

    const_form = t.randn((2, 1)).to(
        dtype=mesh.vert_coords.dtype, device=mesh.vert_coords.device
    )
    sampled_form = repeat(
        const_form, "ch coord -> simp pt ch coord", simp=n_simps, pt=n_pts
    )
    discretized_cochain = de_rham.discretize(sampled_form)

    tet_signed_vols = get_tet_signed_vols(mesh.vert_coords, mesh.tets)
    dot_prod = einsum(tet_signed_vols, const_form, "tet, ch coord -> tet ch")

    t.testing.assert_close(discretized_cochain, dot_prod)


@pytest.mark.parametrize("mesh", ["hollow_tet_mesh", "two_tets_mesh"])
def test_commutativity_with_d_on_0_form(mesh, request, device):
    """
    Test that the de Rham map π commutes with the exterior derivative d using
    0-forms; i.e., for any 0-form ω, π(dω) = d(πω). Recall that dω for a 0-form
    is analogous to the gradient of scalar functions. The test is restricted to
    polynomial basis functions of degree 1.
    """
    mesh = request.getfixturevalue(mesh).to(device)

    d_0 = mesh.cbd[0]

    # For the commutativity to hold, the numerical integration needs to be exact.
    # dω is always one degree lower than ω.
    de_rham_1 = DeRhamMap(k=1, quad_degree=0, mesh=mesh)

    pts_1 = de_rham_1.sample_points()
    n_1_simps, n_1_pts, _ = pts_1.shape

    # The space of 0-forms with polynomial degree 1 consists of 3 basis functions
    # x, y, z. To discretize a 0-form is to simply sample the 0-forms at the
    # mesh vertices. Therefore, the mesh.vert_coords can already be considered
    # as a sample of the three basis functions at the mesh vertices.
    # (1_simp, 0_simp) @ (0_simp, basis) -> (2_simp, basis)
    d_pi_form = d_0 @ mesh.vert_coords

    # Compute dω analytically.
    grad_1_forms = t.eye(
        3,
        dtype=mesh.vert_coords.dtype,
        device=device,
    )

    sampled_1_forms = repeat(
        grad_1_forms, "basis coord -> simp pt basis coord", simp=n_1_simps, pt=n_1_pts
    )

    pi_d_form = de_rham_1.discretize(sampled_1_forms)

    t.testing.assert_close(d_pi_form, pi_d_form)


@pytest.mark.parametrize("mesh", ["hollow_tet_mesh", "two_tets_mesh"])
def test_commutativity_with_d_on_1_form(mesh, request, device):
    """
    Test that the de Rham map π commutes with the exterior derivative d using
    1-forms; i.e., for any 1-form ω, π(dω) = d(πω). Recall that dω for a 1-form
    is analogous to the curl of vector fields. The test is restricted to polynomial
    basis functions of degree 1.
    """
    mesh = request.getfixturevalue(mesh).to(device)

    d_1 = mesh.cbd[1]

    # For the commutativity to hold, the numerical integration needs to be exact.
    de_rham_1 = DeRhamMap(k=1, quad_degree=1, mesh=mesh)
    # dω is always one degree lower than ω.
    de_rham_2 = DeRhamMap(k=2, quad_degree=0, mesh=mesh)

    pts_1 = de_rham_1.sample_points()
    pts_2 = de_rham_2.sample_points()

    n_2_simps, n_2_pts, _ = pts_2.shape

    # The space of 1-forms with polynomial degree 1 consists of 9 basis functions
    # of the form (x, 0, 0), (0, y, 0), (0, 0, z), etc., and each of these basis
    # functions can be written as a 3x3 matrix that, when multiplied by a coordinate
    # vector, returns the value of the 1-form at the coordinate location.
    matrix_1_forms = t.tensor(
        [
            [[1, 0, 0], [0, 0, 0], [0, 0, 0]],  # (x, 0, 0)
            [[0, 1, 0], [0, 0, 0], [0, 0, 0]],  # (y, 0, 0)
            [[0, 0, 1], [0, 0, 0], [0, 0, 0]],  # (z, 0, 0)
            [[0, 0, 0], [1, 0, 0], [0, 0, 0]],  # (0, x, 0)
            [[0, 0, 0], [0, 1, 0], [0, 0, 0]],  # (0, y, 0)
            [[0, 0, 0], [0, 0, 1], [0, 0, 0]],  # (0, z, 0)
            [[0, 0, 0], [0, 0, 0], [1, 0, 0]],  # (0, 0, x)
            [[0, 0, 0], [0, 0, 0], [0, 1, 0]],  # (0, 0, y)
            [[0, 0, 0], [0, 0, 0], [0, 0, 1]],  # (0, 0, z)
        ],
        dtype=mesh.vert_coords.dtype,
        device=device,
    )

    sampled_1_forms = einsum(
        matrix_1_forms,
        pts_1,
        "basis coord1 coord2, simp pt coord2 -> simp pt basis coord1",
    )

    # (2_simp, 1_simp) @ (1_simp, basis) -> (2_simp, basis)
    d_pi_form = d_1 @ de_rham_1.discretize(sampled_1_forms)

    # Compute dω analytically.
    # fmt:off
    curl_2_forms = t.tensor(
        [
            [0, 0, 0],  # ∇ x (x, 0, 0)
            [0, 0, -1], # ∇ x (y, 0, 0)
            [0, 1, 0],  # ∇ x (z, 0, 0)
            [0, 0, 1],  # ∇ x (0, x, 0)
            [0, 0, 0],  # ∇ x (0, y, 0)
            [-1, 0, 0], # ∇ x (0, z, 0)
            [0, -1, 0], # ∇ x (0, 0, x)
            [1, 0, 0],  # ∇ x (0, 0, y)
            [0, 0, 0],  # ∇ x (0, 0, z)
        ],
        dtype=mesh.vert_coords.dtype,
        device=device,
    )
    # fmt:on

    sampled_2_forms = repeat(
        curl_2_forms, "basis coord -> simp pt basis coord", simp=n_2_simps, pt=n_2_pts
    )

    pi_d_form = de_rham_2.discretize(sampled_2_forms)

    t.testing.assert_close(d_pi_form, pi_d_form)


def test_commutativity_with_d_on_2_form(two_tets_mesh, device):
    """
    Test that the de Rham map π commutes with the exterior derivative d using
    2-forms; i.e., for any 2-form ω, π(dω) = d(πω). Recall that dω for a 2-form
    is analogous to the divergence of vector fields. The test is restricted to
    polynomial basis functions of degree 1.
    """
    mesh = two_tets_mesh.to(device)

    d_2 = mesh.cbd[2]

    # For the commutativity to hold, the numerical integration needs to be exact.
    de_rham_2 = DeRhamMap(k=2, quad_degree=1, mesh=mesh)
    # dω is always one degree lower than ω.
    de_rham_3 = DeRhamMap(k=3, quad_degree=0, mesh=mesh)

    pts_2 = de_rham_2.sample_points()
    pts_3 = de_rham_3.sample_points()

    n_3_simps, n_3_pts, _ = pts_3.shape

    # The space of 2-forms with polynomial degree 1 consists of 9 basis functions
    # of the form (x, 0, 0), (0, y, 0), (0, 0, z), etc. under Hodge star isomorphism,
    # and each of these basis functions can be written as a 3x3 matrix that, when
    # multiplied by a coordinate vector, returns the value of the 2-form at the
    # coordinate location.
    matrix_2_forms = t.tensor(
        [
            [[1, 0, 0], [0, 0, 0], [0, 0, 0]],  # (x, 0, 0)
            [[0, 1, 0], [0, 0, 0], [0, 0, 0]],  # (y, 0, 0)
            [[0, 0, 1], [0, 0, 0], [0, 0, 0]],  # (z, 0, 0)
            [[0, 0, 0], [1, 0, 0], [0, 0, 0]],  # (0, x, 0)
            [[0, 0, 0], [0, 1, 0], [0, 0, 0]],  # (0, y, 0)
            [[0, 0, 0], [0, 0, 1], [0, 0, 0]],  # (0, z, 0)
            [[0, 0, 0], [0, 0, 0], [1, 0, 0]],  # (0, 0, x)
            [[0, 0, 0], [0, 0, 0], [0, 1, 0]],  # (0, 0, y)
            [[0, 0, 0], [0, 0, 0], [0, 0, 1]],  # (0, 0, z)
        ],
        dtype=mesh.vert_coords.dtype,
        device=device,
    )

    sampled_2_forms = einsum(
        matrix_2_forms,
        pts_2,
        "basis coord1 coord2, simp pt coord2 -> simp pt basis coord1",
    )

    # (3_simp, 2_simp) @ (2_simp, basis) -> (3_simp, basis)
    d_pi_form = d_2 @ de_rham_2.discretize(sampled_2_forms)

    # Compute dω analytically.
    # fmt:off
    div_3_forms = t.tensor(
        [
            [1], # ∇ ⋅ (x, 0, 0)
            [0], # ∇ ⋅ (y, 0, 0)
            [0], # ∇ ⋅ (z, 0, 0)
            [0], # ∇ ⋅ (0, x, 0)
            [1], # ∇ ⋅ (0, y, 0)
            [0], # ∇ ⋅ (0, z, 0)
            [0], # ∇ ⋅ (0, 0, x)
            [0], # ∇ ⋅ (0, 0, y)
            [1], # ∇ ⋅ (0, 0, z)
        ],
        dtype=mesh.vert_coords.dtype,
        device=device,
    )
    # fmt:on

    sampled_3_forms = repeat(
        div_3_forms, "basis coord -> simp pt basis coord", simp=n_3_simps, pt=n_3_pts
    )

    pi_d_form = de_rham_3.discretize(sampled_3_forms)

    t.testing.assert_close(d_pi_form, pi_d_form)

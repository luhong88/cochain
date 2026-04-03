import pytest
import torch
from einops import einsum, repeat

from cochain.cochain.discretize import DeRhamMap
from cochain.geometry.tet.tet_geometry import compute_tet_signed_vols

# TODO: test gradients


@pytest.mark.parametrize("mesh", ["hollow_tet_mesh", "two_tets_mesh"])
def test_const_1_form_integration(mesh, request, device):
    mesh = request.getfixturevalue(mesh).to(device)

    de_rham = DeRhamMap(k=1, quad_degree=1, mesh=mesh)

    pts = de_rham.sample_points()
    n_splx, n_pts, _ = pts.shape

    const_form = torch.randn((2, 3)).to(
        dtype=mesh.vert_coords.dtype, device=mesh.vert_coords.device
    )
    sampled_form = repeat(
        const_form, "ch coord -> splx pt ch coord", splx=n_splx, pt=n_pts
    )
    discretized_cochain = de_rham.discretize(sampled_form)

    edge_verts = mesh.vert_coords[mesh.edges]
    edge_vecs = edge_verts[:, 1] - edge_verts[:, 0]
    dot_prod = einsum(edge_vecs, const_form, "edge coord, ch coord -> edge ch")

    torch.testing.assert_close(discretized_cochain, dot_prod)


@pytest.mark.parametrize("mesh", ["hollow_tet_mesh", "two_tets_mesh"])
def test_const_2_form_integration(mesh, request, device):
    mesh = request.getfixturevalue(mesh).to(device)

    de_rham = DeRhamMap(k=2, quad_degree=1, mesh=mesh)

    pts = de_rham.sample_points()
    n_splx, n_pts, _ = pts.shape

    const_form = torch.randn((2, 3)).to(
        dtype=mesh.vert_coords.dtype, device=mesh.vert_coords.device
    )
    sampled_form = repeat(
        const_form, "ch coord -> splx pt ch coord", splx=n_splx, pt=n_pts
    )
    discretized_cochain = de_rham.discretize(sampled_form)

    tri_verts = mesh.vert_coords[mesh.tris]
    tri_area_norm = 0.5 * torch.cross(
        tri_verts[:, 1] - tri_verts[:, 0], tri_verts[:, 2] - tri_verts[:, 0], dim=-1
    )
    dot_prod = einsum(tri_area_norm, const_form, "tri coord, ch coord -> tri ch")

    torch.testing.assert_close(discretized_cochain, dot_prod)


def test_const_3_form_integration(two_tets_mesh, device):
    mesh = two_tets_mesh.to(device)

    de_rham = DeRhamMap(k=3, quad_degree=1, mesh=mesh)

    pts = de_rham.sample_points()
    n_splx, n_pts, _ = pts.shape

    const_form = torch.randn((2, 1)).to(
        dtype=mesh.vert_coords.dtype, device=mesh.vert_coords.device
    )
    sampled_form = repeat(
        const_form, "ch coord -> splx pt ch coord", splx=n_splx, pt=n_pts
    )
    discretized_cochain = de_rham.discretize(sampled_form)

    tet_signed_vols = compute_tet_signed_vols(mesh.vert_coords, mesh.tets)
    dot_prod = einsum(tet_signed_vols, const_form, "tet, ch coord -> tet ch")

    torch.testing.assert_close(discretized_cochain, dot_prod)


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
    n_1_splx, n_1_pts, _ = pts_1.shape

    # The space of 0-forms with polynomial degree 1 consists of 3 basis functions
    # x, y, z. To discretize a 0-form is to simply sample the 0-forms at the
    # mesh vertices. Therefore, the mesh.vert_coords can already be considered
    # as a sample of the three basis functions at the mesh vertices.
    # (1_splx, 0_splx) @ (0_splx, basis) -> (2_splx, basis)
    d_pi_form = d_0 @ mesh.vert_coords

    # Compute dω analytically.
    grad_1_forms = torch.eye(
        3,
        dtype=mesh.vert_coords.dtype,
        device=device,
    )

    sampled_1_forms = repeat(
        grad_1_forms, "basis coord -> splx pt basis coord", splx=n_1_splx, pt=n_1_pts
    )

    pi_d_form = de_rham_1.discretize(sampled_1_forms)

    torch.testing.assert_close(d_pi_form, pi_d_form)


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

    n_2_splx, n_2_pts, _ = pts_2.shape

    # The space of 1-forms with polynomial degree 1 consists of 9 basis functions
    # of the form (x, 0, 0), (0, y, 0), (0, 0, z), etc., and each of these basis
    # functions can be written as a 3x3 matrix that, when multiplied by a coordinate
    # vector, returns the value of the 1-form at the coordinate location.
    matrix_1_forms = torch.tensor(
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
        "basis coord1 coord2, splx pt coord2 -> splx pt basis coord1",
    )

    # (2_splx, 1_splx) @ (1_splx, basis) -> (2_splx, basis)
    d_pi_form = d_1 @ de_rham_1.discretize(sampled_1_forms)

    # Compute dω analytically.
    # fmt:off
    curl_2_forms = torch.tensor(
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
        curl_2_forms, "basis coord -> splx pt basis coord", splx=n_2_splx, pt=n_2_pts
    )

    pi_d_form = de_rham_2.discretize(sampled_2_forms)

    torch.testing.assert_close(d_pi_form, pi_d_form)


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

    n_3_splx, n_3_pts, _ = pts_3.shape

    # The space of 2-forms with polynomial degree 1 consists of 9 basis functions
    # of the form (x, 0, 0), (0, y, 0), (0, 0, z), etc. under Hodge star isomorphism,
    # and each of these basis functions can be written as a 3x3 matrix that, when
    # multiplied by a coordinate vector, returns the value of the 2-form at the
    # coordinate location.
    matrix_2_forms = torch.tensor(
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
        "basis coord1 coord2, splx pt coord2 -> splx pt basis coord1",
    )

    # (3_splx, 2_splx) @ (2_splx, basis) -> (3_splx, basis)
    d_pi_form = d_2 @ de_rham_2.discretize(sampled_2_forms)

    # Compute dω analytically.
    # fmt:off
    div_3_forms = torch.tensor(
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
        div_3_forms, "basis coord -> splx pt basis coord", splx=n_3_splx, pt=n_3_pts
    )

    pi_d_form = de_rham_3.discretize(sampled_3_forms)

    torch.testing.assert_close(d_pi_form, pi_d_form)


@pytest.mark.parametrize("mesh", ["hollow_tet_mesh", "two_tets_mesh"])
def test_1_form_polynomial_deg_2_exact_integration(mesh, request, device):
    """
    Test that the numerical integration of a polynomial 1-form of degree 2 should
    be exact using a quadrature rule of the same degree.
    """
    mesh = request.getfixturevalue(mesh).to(device)
    edge_verts = mesh.vert_coords[mesh.edges]

    # In general, polynomial functions f(x, y, z) with 3 variables and of degree 2
    # can be described by 6 basis functions: x^2, y^2, z^2, xy, xz, yz. In addition,
    # for a polynomial 1-form of degree 2, the basis functions can be placed as
    # coefficient in front of the three basis covectors, thus resulting in 18 basis
    # covectors. The integration of these basis covectors is equivalent to the
    # line integral of basis vectors. Since the tangent vector on a 1-simplex
    # is constant, the dot product of the field with the tangent can be pulled
    # out of the integral; i.e., <int_0^1 F dt, v_1 - v_0> (note that, with the
    # reference element parametrization, the integral is over the unit length/
    # reference 1-simplex and the tangent vector v_1 - v_0 is not normalized; this
    # is to be contrasted with the standard arc-length parametrization, where the
    # integral is over [0, L] and the tangent vector is unit length). This expression
    # can be simplified further. Because each F has only one nonzero component,
    # the 18 basis vector fields reduces to only 6 unique integrals of  the form,
    # e.g., int[xy*dt], which can be computed using the magic formula after converting
    # the cartesian coordinates into barycentric coordinates. The final 18 line
    # integrals can then be computed as an outer product between the 6 (scalar)
    # integrals and the 3 (x, y, z) components of the tangent v_1 - v_0.

    # Using the magic formula, the integral of the scalar basis functions over
    # a 1-simplex are:
    #
    # int[x^2] = L * (x_0^2 + x_0*x_1 + x_1^2) / 3
    # int[xy] = L * (2*x_0*y_0 + 2*x_1*y_1 + x_0*y_1 + x_1*y_0) / 6
    #
    # Note that the arc length L is simply 1 with reference element parametrization.
    v0, v1 = edge_verts.unbind(1)
    x_0, y_0, z_0 = v0.unbind(-1)
    x_1, y_1, z_1 = v1.unbind(-1)

    scalar_basis_int = torch.stack(
        [
            (x_0**2 + x_0 * x_1 + x_1**2) / 3.0,
            (y_0**2 + y_0 * y_1 + y_1**2) / 3.0,
            (z_0**2 + z_0 * z_1 + z_1**2) / 3.0,
            (2 * x_0 * y_0 + 2 * x_1 * y_1 + x_0 * y_1 + x_1 * y_0) / 6.0,
            (2 * x_0 * z_0 + 2 * x_1 * z_1 + x_0 * z_1 + x_1 * z_0) / 6.0,
            (2 * y_0 * z_0 + 2 * y_1 * z_1 + y_0 * z_1 + y_1 * z_0) / 6.0,
        ]
    )

    edge_vecs = edge_verts[:, 1] - edge_verts[:, 0]

    dot_prod = einsum(
        scalar_basis_int,
        edge_vecs,
        "scalar_basis edge, edge vec_basis -> edge scalar_basis vec_basis",
    )

    # Compute the 18 basis vector field integrals using numerical quadrature.
    de_rham = DeRhamMap(k=1, quad_degree=2, mesh=mesh)

    pts = de_rham.sample_points()
    x_pts, y_pts, z_pts = pts.unbind(-1)

    # To obtain the values of the 18 basis vector fields over the sampled points,
    # first evaluate the 6 basis functions at the sampled points, and then use
    # the same outer product trick to broadcast the scalar values into the three
    # coordinate slots (here, the "tangent vectors" are simply the standard
    # Cartesian basis vectors).
    sampled_form_scalar_basis = torch.stack(
        [x_pts**2, y_pts**2, z_pts**2, x_pts * y_pts, x_pts * z_pts, y_pts * z_pts]
    )
    vec_basis = torch.eye(3, dtype=pts.dtype, device=pts.device)

    sampled_form = einsum(
        sampled_form_scalar_basis,
        vec_basis,
        "scalar_basis edge pt, vec_basis coord -> edge pt scalar_basis vec_basis coord",
    )

    discretized_cochain = de_rham.discretize(sampled_form)

    # Check that the analytical line integrals agree with the numerical quadratures.
    torch.testing.assert_close(discretized_cochain, dot_prod)


@pytest.mark.parametrize("mesh", ["hollow_tet_mesh", "two_tets_mesh"])
def test_2_form_polynomial_deg_2_exact_integration(mesh, request, device):
    """
    Test that the numerical integration of a polynomial 2-form of degree 2 should
    be exact using a quadrature rule of the same degree. See the test for 1-form
    for more information.
    """
    mesh = request.getfixturevalue(mesh).to(device)
    tri_verts = mesh.vert_coords[mesh.tris]

    # Using the magic formula, the integral of the scalar basis functions over
    # a 2-simplex are:
    #
    # int[x^2] = (x_0^2 + x_1^2 + x_2^2 + x_0*x_1 + x_0*x_2 + x_1*x_2) / 6
    # int[xy] = (2(x_0*y_0 + x_1*y_1 + x_2*y_2) +
    #            x_0*y_1 + x_1*y_0 + x_1*y_2 + x_2*y_1 + x_2*y_0 + x_0*y_2) / 12
    v0, v1, v2 = tri_verts.unbind(1)
    x_0, y_0, z_0 = v0.unbind(-1)
    x_1, y_1, z_1 = v1.unbind(-1)
    x_2, y_2, z_2 = v2.unbind(-1)

    scalar_basis_int = torch.stack(
        [
            (x_0**2 + x_1**2 + x_2**2 + x_0 * x_1 + x_0 * x_2 + x_1 * x_2) / 6.0,
            (y_0**2 + y_1**2 + y_2**2 + y_0 * y_1 + y_0 * y_2 + y_1 * y_2) / 6.0,
            (z_0**2 + z_1**2 + z_2**2 + z_0 * z_1 + z_0 * z_2 + z_1 * z_2) / 6.0,
            (
                2 * (x_0 * y_0 + x_1 * y_1 + x_2 * y_2)
                + x_0 * y_1
                + x_1 * y_0
                + x_1 * y_2
                + x_2 * y_1
                + x_2 * y_0
                + x_0 * y_2
            )
            / 12.0,
            (
                2 * (x_0 * z_0 + x_1 * z_1 + x_2 * z_2)
                + x_0 * z_1
                + x_1 * z_0
                + x_1 * z_2
                + x_2 * z_1
                + x_2 * z_0
                + x_0 * z_2
            )
            / 12.0,
            (
                2 * (y_0 * z_0 + y_1 * z_1 + y_2 * z_2)
                + y_0 * z_1
                + y_1 * z_0
                + y_1 * z_2
                + y_2 * z_1
                + y_2 * z_0
                + y_0 * z_2
            )
            / 12.0,
        ]
    )

    tri_area_norms = 0.5 * torch.cross(
        tri_verts[:, 1] - tri_verts[:, 0], tri_verts[:, 2] - tri_verts[:, 0], dim=-1
    )

    dot_prod = einsum(
        scalar_basis_int,
        tri_area_norms,
        "scalar_basis tri, tri vec_basis -> tri scalar_basis vec_basis",
    )

    # Compute the 18 basis vector field integrals using numerical quadrature.
    de_rham = DeRhamMap(k=2, quad_degree=2, mesh=mesh)

    pts = de_rham.sample_points()
    x_pts, y_pts, z_pts = pts.unbind(-1)

    sampled_form_scalar_basis = torch.stack(
        [x_pts**2, y_pts**2, z_pts**2, x_pts * y_pts, x_pts * z_pts, y_pts * z_pts]
    )
    vec_basis = torch.eye(3, dtype=pts.dtype, device=pts.device)

    sampled_form = einsum(
        sampled_form_scalar_basis,
        vec_basis,
        "scalar_basis tri pt, vec_basis coord -> tri pt scalar_basis vec_basis coord",
    )

    discretized_cochain = de_rham.discretize(sampled_form)

    # Check that the analytical line integrals agree with the numerical quadratures.
    torch.testing.assert_close(discretized_cochain, dot_prod)


def test_3_form_polynomial_deg_2_exact_integration(two_tets_mesh, device):
    """
    Test that the numerical integration of a polynomial 3-form of degree 2 should
    be exact using a quadrature rule of the same degree. See the test for 1-form
    for more information.
    """
    mesh = two_tets_mesh.to(device)
    tet_verts = mesh.vert_coords[mesh.tets]

    v0, v1, v2, v3 = tet_verts.unbind(1)
    x_0, y_0, z_0 = v0.unbind(-1)
    x_1, y_1, z_1 = v1.unbind(-1)
    x_2, y_2, z_2 = v2.unbind(-1)
    x_3, y_3, z_3 = v3.unbind(-1)

    scalar_basis_int = torch.stack(
        [
            (
                x_0**2
                + x_1**2
                + x_2**2
                + x_3**2
                + x_0 * x_1
                + x_0 * x_2
                + x_0 * x_3
                + x_1 * x_2
                + x_1 * x_3
                + x_2 * x_3
            )
            / 10.0,
            (
                y_0**2
                + y_1**2
                + y_2**2
                + y_3**2
                + y_0 * y_1
                + y_0 * y_2
                + y_0 * y_3
                + y_1 * y_2
                + y_1 * y_3
                + y_2 * y_3
            )
            / 10.0,
            (
                z_0**2
                + z_1**2
                + z_2**2
                + z_3**2
                + z_0 * z_1
                + z_0 * z_2
                + z_0 * z_3
                + z_1 * z_2
                + z_1 * z_3
                + z_2 * z_3
            )
            / 10.0,
            (
                2 * (x_0 * y_0 + x_1 * y_1 + x_2 * y_2 + x_3 * y_3)
                + x_0 * y_1
                + x_1 * y_0
                + x_0 * y_2
                + x_2 * y_0
                + x_0 * y_3
                + x_3 * y_0
                + x_1 * y_2
                + x_2 * y_1
                + x_1 * y_3
                + x_3 * y_1
                + x_2 * y_3
                + x_3 * y_2
            )
            / 20.0,
            (
                2 * (x_0 * z_0 + x_1 * z_1 + x_2 * z_2 + x_3 * z_3)
                + x_0 * z_1
                + x_1 * z_0
                + x_0 * z_2
                + x_2 * z_0
                + x_0 * z_3
                + x_3 * z_0
                + x_1 * z_2
                + x_2 * z_1
                + x_1 * z_3
                + x_3 * z_1
                + x_2 * z_3
                + x_3 * z_2
            )
            / 20.0,
            (
                2 * (y_0 * z_0 + y_1 * z_1 + y_2 * z_2 + y_3 * z_3)
                + y_0 * z_1
                + y_1 * z_0
                + y_0 * z_2
                + y_2 * z_0
                + y_0 * z_3
                + y_3 * z_0
                + y_1 * z_2
                + y_2 * z_1
                + y_1 * z_3
                + y_3 * z_1
                + y_2 * z_3
                + y_3 * z_2
            )
            / 20.0,
        ]
    )

    # Since 3-forms are scalar-valued, the vec_basis dimension is trivial.
    tet_signed_vols = compute_tet_signed_vols(mesh.vert_coords, mesh.tets).unsqueeze(-1)

    dot_prod = einsum(
        scalar_basis_int,
        tet_signed_vols,
        "scalar_basis tri, tri vec_basis -> tri scalar_basis vec_basis",
    )

    # Compute the basis integrals using numerical quadrature.
    de_rham = DeRhamMap(k=3, quad_degree=2, mesh=mesh)

    pts = de_rham.sample_points()
    x_pts, y_pts, z_pts = pts.unbind(-1)

    sampled_form_scalar_basis = torch.stack(
        [x_pts**2, y_pts**2, z_pts**2, x_pts * y_pts, x_pts * z_pts, y_pts * z_pts]
    )
    vec_basis = torch.tensor([[1]], dtype=pts.dtype, device=pts.device)

    sampled_form = einsum(
        sampled_form_scalar_basis,
        vec_basis,
        "scalar_basis tri pt, vec_basis coord -> tri pt scalar_basis vec_basis coord",
    )

    discretized_cochain = de_rham.discretize(sampled_form)

    # Check that the analytical line integrals agree with the numerical quadratures.
    torch.testing.assert_close(discretized_cochain, dot_prod)

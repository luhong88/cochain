import random

import pytest
import torch
from einops import einsum, repeat

from cochain.cochain.discretize import DeRhamMap
from cochain.cochain.ext_prod.whitney import WhitneyWedgeL2Projector
from cochain.cochain.int_prod import galerkin_contract
from cochain.complex import SimplicialMesh
from cochain.metric.tet import tet_masses
from cochain.metric.tri import tri_masses
from cochain.sparse.linalg.solvers import SuperLU


def test_galerkin_contraction_1_form_on_tet_mesh(two_tets_mesh: SimplicialMesh, device):
    """
    Test that the interior product between a constant vector field and a constant
    k-form can be exactly reproduced on a tet mesh using the Galerkin method.

    Note that we do not test this property on tri meshes, because a random vector
    field is in general not in the tangent space of a tri mesh. Since the
    galerkin_contract() function is built on top of mass matrices and wedge
    products that have already been tested for both tri and tet meshes, we avoid
    the issue with tangent space here in these tests by using tet meshes only.
    """
    mesh = two_tets_mesh.to(device)

    # Set up a constant vector field and a constant 1-form and compute their
    # interior product analytically, which is simply the inner product.
    const_vec = torch.randn(3, dtype=mesh.dtype, device=mesh.device)
    const_1_form = torch.randn(3, dtype=mesh.dtype, device=mesh.device)

    # Note that a constant 0-form is identical to its discretization, so there is
    # no need to convert the 0-form to a 0-cochain.
    int_prod_true = torch.sum(const_vec * const_1_form).view(1).expand(mesh.n_verts)

    # Compute the flat of the constant vector field.
    edge_verts = mesh.vert_coords[mesh.edges]
    edge_vecs = edge_verts[:, 1] - edge_verts[:, 0]
    const_vec_flat = einsum(const_vec, edge_vecs, "coord, edge coord -> edge")

    # Compute the flat of the constant 1-form using the de Rham map.
    de_rham_1 = DeRhamMap(k=1, quad_degree=1, mesh=mesh)

    sampled_points = de_rham_1.sample_points()
    n_edges, n_pts, _ = sampled_points.shape
    const_1_form_shaped = repeat(
        const_1_form, "coord -> edge pt coord", edge=n_edges, pt=n_pts
    )

    const_1_cochain = de_rham_1.discretize(const_1_form_shaped)

    # Compute the interior product using the galerkin_contract() method.
    wedge_op = WhitneyWedgeL2Projector(k=1, l=0, mesh=mesh)
    mass_0 = tet_masses.mass_0(mesh)

    int_prod = galerkin_contract(
        vec_field_flat=const_vec_flat,
        cochain_k=const_1_cochain,
        mass_km1=mass_0,
        wedge_op=wedge_op,
    )

    torch.testing.assert_close(int_prod, int_prod_true)

    # Also test the InvSparseOperator route.
    mass_0_op = SuperLU(mass_0, backend="scipy")

    int_prod = galerkin_contract(
        vec_field_flat=const_vec_flat,
        cochain_k=const_1_cochain,
        mass_km1=mass_0_op,
        wedge_op=wedge_op,
    )

    torch.testing.assert_close(int_prod, int_prod_true)


def test_galerkin_contraction_2_form_on_tet_mesh(two_tets_mesh: SimplicialMesh, device):
    mesh = two_tets_mesh.to(device)

    # Set up a constant vector field and a constant 2-form and compute their
    # interior product analytically, which is simply the cross product.
    const_vec = torch.randn(3, dtype=mesh.dtype, device=mesh.device)
    const_2_form = torch.randn(3, dtype=mesh.dtype, device=mesh.device)

    # Note that, the interior product must be computed as ω x v, rather than v x ω.
    int_prod_form = torch.cross(const_2_form, const_vec, dim=-1)

    # Discretize the true interior product into a 1-cochain.
    de_rham_1 = DeRhamMap(k=1, quad_degree=1, mesh=mesh)

    sampled_points = de_rham_1.sample_points()
    n_edges, n_pts, _ = sampled_points.shape
    int_prod_form_shaped = repeat(
        int_prod_form, "coord -> edge pt coord", edge=n_edges, pt=n_pts
    )

    int_prod_cochain_true = de_rham_1.discretize(int_prod_form_shaped)

    # Compute the flat of the constant vector field.
    edge_verts = mesh.vert_coords[mesh.edges]
    edge_vecs = edge_verts[:, 1] - edge_verts[:, 0]
    const_vec_flat = einsum(const_vec, edge_vecs, "coord, edge coord -> edge")

    # Compute the flat of the constant 2-form using the de Rham map.
    de_rham_2 = DeRhamMap(k=2, quad_degree=1, mesh=mesh)

    sampled_points = de_rham_2.sample_points()
    n_tris, n_pts, _ = sampled_points.shape
    const_2_form_shaped = repeat(
        const_2_form, "coord -> tri pt coord", tri=n_tris, pt=n_pts
    )

    const_2_cochain = de_rham_2.discretize(const_2_form_shaped)

    # Compute the interior product using the galerkin_contract() method.
    wedge_op = WhitneyWedgeL2Projector(k=1, l=1, mesh=mesh)
    mass_1 = tet_masses.mass_1(mesh)

    int_prod = galerkin_contract(
        vec_field_flat=const_vec_flat,
        cochain_k=const_2_cochain,
        mass_km1=mass_1,
        wedge_op=wedge_op,
    )

    torch.testing.assert_close(int_prod, int_prod_cochain_true)

    # Also test the InvSparseOperator route.
    mass_1_op = SuperLU(mass_1, backend="scipy")

    int_prod = galerkin_contract(
        vec_field_flat=const_vec_flat,
        cochain_k=const_2_cochain,
        mass_km1=mass_1_op,
        wedge_op=wedge_op,
    )

    torch.testing.assert_close(int_prod, int_prod_cochain_true)


def test_galerkin_contraction_3_form_on_tet_mesh(two_tets_mesh: SimplicialMesh, device):
    mesh = two_tets_mesh.to(device)

    # Set up a constant vector field and a constant 3-form and compute their
    # interior product analytically, which is simply the scalar product.
    const_vec = torch.randn(3, dtype=mesh.dtype, device=mesh.device)
    const_3_form = torch.randn(1, dtype=mesh.dtype, device=mesh.device)
    int_prod_form = const_3_form * const_vec

    # Discretize the true interior product into a 2-cochain.
    de_rham_2 = DeRhamMap(k=2, quad_degree=1, mesh=mesh)

    sampled_points = de_rham_2.sample_points()
    n_tris, n_pts, _ = sampled_points.shape
    int_prod_form_shaped = repeat(
        int_prod_form, "coord -> tri pt coord", tri=n_tris, pt=n_pts
    )

    int_prod_cochain_true = de_rham_2.discretize(int_prod_form_shaped)

    # Compute the flat of the constant vector field.
    edge_verts = mesh.vert_coords[mesh.edges]
    edge_vecs = edge_verts[:, 1] - edge_verts[:, 0]
    const_vec_flat = einsum(const_vec, edge_vecs, "coord, edge coord -> edge")

    # Compute the flat of the constant 3-form using the de Rham map.
    de_rham_3 = DeRhamMap(k=3, quad_degree=1, mesh=mesh)

    sampled_points = de_rham_3.sample_points()
    n_tets, n_pts, _ = sampled_points.shape
    const_3_form_shaped = repeat(
        const_3_form, "coord -> tet pt coord", tet=n_tets, pt=n_pts
    )

    const_3_cochain = de_rham_3.discretize(const_3_form_shaped)

    # Compute the interior product using the galerkin_contract() method.
    wedge_op = WhitneyWedgeL2Projector(k=1, l=2, mesh=mesh)
    mass_2 = tet_masses.mass_2(mesh)

    int_prod = galerkin_contract(
        vec_field_flat=const_vec_flat,
        cochain_k=const_3_cochain,
        mass_km1=mass_2,
        wedge_op=wedge_op,
    )

    torch.testing.assert_close(int_prod, int_prod_cochain_true)

    # Also test the InvSparseOperator route.
    mass_2_op = SuperLU(mass_2, backend="scipy")

    int_prod = galerkin_contract(
        vec_field_flat=const_vec_flat,
        cochain_k=const_3_cochain,
        mass_km1=mass_2_op,
        wedge_op=wedge_op,
    )

    torch.testing.assert_close(int_prod, int_prod_cochain_true)


def test_galerkin_contraction_nilpotency_2_form(two_tets_mesh: SimplicialMesh, device):
    """
    Test that contracting a k-cochain twice against the same vector field gives
    zero.

    Note that, due to L^2 projection errors, the Galerkin method does not strictly
    satisfy the nilpotency property unless tested on constant vector fields and
    constant k-forms.
    """
    k = 2
    mesh = two_tets_mesh.to(device)

    # Set up a constant vector field and a constant 2-form.
    const_vec = torch.randn(3, dtype=mesh.dtype, device=mesh.device)
    const_2_form = torch.randn(3, dtype=mesh.dtype, device=mesh.device)

    # Compute the flat of the constant vector field.
    edge_verts = mesh.vert_coords[mesh.edges]
    edge_vecs = edge_verts[:, 1] - edge_verts[:, 0]
    const_vec_flat = einsum(const_vec, edge_vecs, "coord, edge coord -> edge")

    # Compute the flat of the constant 2-form using the de Rham map.
    de_rham_2 = DeRhamMap(k=k, quad_degree=1, mesh=mesh)

    sampled_points = de_rham_2.sample_points()
    n_tris, n_pts, _ = sampled_points.shape
    const_2_form_shaped = repeat(
        const_2_form, "coord -> tri pt coord", tri=n_tris, pt=n_pts
    )

    const_2_cochain = de_rham_2.discretize(const_2_form_shaped)

    # Perform two consecutive interior products
    wedge_op_r1 = WhitneyWedgeL2Projector(k=1, l=k - 1, mesh=mesh)
    mass_km1 = getattr(tet_masses, f"mass_{k - 1}")(mesh)

    int_prod_r1 = galerkin_contract(
        vec_field_flat=const_vec_flat,
        cochain_k=const_2_cochain,
        mass_km1=mass_km1,
        wedge_op=wedge_op_r1,
    )

    wedge_op_r2 = WhitneyWedgeL2Projector(k=1, l=k - 2, mesh=mesh)
    mass_km2 = getattr(tet_masses, f"mass_{k - 2}")(mesh)

    int_prod_r2 = galerkin_contract(
        vec_field_flat=const_vec_flat,
        cochain_k=int_prod_r1,
        mass_km1=mass_km2,
        wedge_op=wedge_op_r2,
    )

    # The final (k-2)-cochain should be close to zero.
    int_prod_r2_true = torch.zeros_like(int_prod_r2)

    torch.testing.assert_close(int_prod_r2, int_prod_r2_true)


def test_galerkin_contraction_nilpotency_3_form(two_tets_mesh: SimplicialMesh, device):
    k = 3
    mesh = two_tets_mesh.to(device)

    # Set up a constant vector field and a constant 3-form.
    const_vec = torch.randn(3, dtype=mesh.dtype, device=mesh.device)
    const_3_form = torch.randn(1, dtype=mesh.dtype, device=mesh.device)

    # Compute the flat of the constant vector field.
    edge_verts = mesh.vert_coords[mesh.edges]
    edge_vecs = edge_verts[:, 1] - edge_verts[:, 0]
    const_vec_flat = einsum(const_vec, edge_vecs, "coord, edge coord -> edge")

    # Compute the flat of the constant 3-form using the de Rham map.
    de_rham_3 = DeRhamMap(k=3, quad_degree=1, mesh=mesh)

    sampled_points = de_rham_3.sample_points()
    n_tets, n_pts, _ = sampled_points.shape
    const_3_form_shaped = repeat(
        const_3_form, "coord -> tet pt coord", tet=n_tets, pt=n_pts
    )

    const_3_cochain = de_rham_3.discretize(const_3_form_shaped)

    # Perform two consecutive interior products
    wedge_op_r1 = WhitneyWedgeL2Projector(k=1, l=k - 1, mesh=mesh)
    mass_km1 = getattr(tet_masses, f"mass_{k - 1}")(mesh)

    int_prod_r1 = galerkin_contract(
        vec_field_flat=const_vec_flat,
        cochain_k=const_3_cochain,
        mass_km1=mass_km1,
        wedge_op=wedge_op_r1,
    )

    wedge_op_r2 = WhitneyWedgeL2Projector(k=1, l=k - 2, mesh=mesh)
    mass_km2 = getattr(tet_masses, f"mass_{k - 2}")(mesh)

    int_prod_r2 = galerkin_contract(
        vec_field_flat=const_vec_flat,
        cochain_k=int_prod_r1,
        mass_km1=mass_km2,
        wedge_op=wedge_op_r2,
    )

    # The final (k-2)-cochain should be close to zero.
    int_prod_r2_true = torch.zeros_like(int_prod_r2)

    torch.testing.assert_close(int_prod_r2, int_prod_r2_true)


@pytest.mark.parametrize(
    "k, mesh",
    [
        (1, "two_tets_mesh"),
        (2, "two_tets_mesh"),
        (3, "two_tets_mesh"),
        (1, "two_tris_mesh"),
        (2, "two_tris_mesh"),
    ],
)
def test_galerkin_contraction_linearity(k: int, mesh, request, device):
    mesh = request.getfixturevalue(mesh).to(device)

    a = random.random()
    b = random.random()

    vec_field_flat_1 = torch.randn(mesh.n_edges, dtype=mesh.dtype, device=mesh.device)
    vec_field_flat_2 = torch.randn(mesh.n_edges, dtype=mesh.dtype, device=mesh.device)

    k_cochain_1 = torch.randn(mesh.n_splx[k], dtype=mesh.dtype, device=mesh.device)
    k_cochain_2 = torch.randn(mesh.n_splx[k], dtype=mesh.dtype, device=mesh.device)

    wedge_op_r1 = WhitneyWedgeL2Projector(k=1, l=k - 1, mesh=mesh)

    match mesh.dim:
        case 3:
            mass_km1 = getattr(tet_masses, f"mass_{k - 1}")(mesh)
        case 2:
            mass_km1 = getattr(tri_masses, f"mass_{k - 1}")(mesh)

    # First, test linearity in the vector field argument.
    int_prod_sum = galerkin_contract(
        vec_field_flat=a * vec_field_flat_1 + b * vec_field_flat_2,
        cochain_k=k_cochain_1,
        mass_km1=mass_km1,
        wedge_op=wedge_op_r1,
    )

    int_prod_1 = a * galerkin_contract(
        vec_field_flat=vec_field_flat_1,
        cochain_k=k_cochain_1,
        mass_km1=mass_km1,
        wedge_op=wedge_op_r1,
    )

    int_prod_2 = b * galerkin_contract(
        vec_field_flat=vec_field_flat_2,
        cochain_k=k_cochain_1,
        mass_km1=mass_km1,
        wedge_op=wedge_op_r1,
    )

    torch.testing.assert_close(int_prod_1 + int_prod_2, int_prod_sum)

    # Then, test linearity in the k-form argument.
    int_prod_sum = galerkin_contract(
        vec_field_flat=vec_field_flat_1,
        cochain_k=a * k_cochain_1 + b * k_cochain_2,
        mass_km1=mass_km1,
        wedge_op=wedge_op_r1,
    )

    int_prod_1 = a * galerkin_contract(
        vec_field_flat=vec_field_flat_1,
        cochain_k=k_cochain_1,
        mass_km1=mass_km1,
        wedge_op=wedge_op_r1,
    )

    int_prod_2 = b * galerkin_contract(
        vec_field_flat=vec_field_flat_1,
        cochain_k=k_cochain_2,
        mass_km1=mass_km1,
        wedge_op=wedge_op_r1,
    )

    torch.testing.assert_close(int_prod_1 + int_prod_2, int_prod_sum)

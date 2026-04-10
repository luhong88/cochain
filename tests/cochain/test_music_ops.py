import pytest
import torch
from einops import repeat

from cochain.cochain import music_ops
from cochain.metric.tet import tet_masses
from cochain.metric.tri import tri_masses
from cochain.sparse.linalg.solvers import SuperLU


@pytest.mark.parametrize(
    "mesh, mode, diagonal",
    [
        ("hollow_tet_mesh", "element", False),
        ("hollow_tet_mesh", "vertex", True),
        ("hollow_tet_mesh", "vertex", False),
        ("two_tets_mesh", "element", False),
        ("two_tets_mesh", "vertex", True),
        ("two_tets_mesh", "vertex", False),
    ],
)
def test_vector_mass_spd(mesh, mode, diagonal, request, device):
    mesh = request.getfixturevalue(mesh).to(device)

    # Note that the diagonal argument is only relevant for mode="vertex".
    mass_v = music_ops.vector_mass(mesh, mode, diagonal).to_dense()

    torch.testing.assert_close(mass_v, mass_v.T)

    eigs = torch.linalg.eigvalsh(mass_v)
    assert eigs.min() >= 1e-6


@pytest.mark.parametrize("mesh", ["hollow_tet_mesh", "two_tets_mesh"])
def test_local_vertex_based_const_vec_field_reconstruction(mesh, request, device):
    """
    Taking the flat of a constant vector field, followed by sharp, should reproduce
    the same constant vector field. Note that the inverse of this process (cochain
    -> vector field -> cochain) does not work.

    Note that, because of the pathologies of the vertex-based local sharp
    operator, exact reconstruction is not expected; instead, we check that the
    reconstructed vector fields point in roughly the same direction as the original.
    """
    mesh = request.getfixturevalue(mesh).to(device)

    vec_field = repeat(
        torch.randn(3, dtype=mesh.dtype, device=mesh.device),
        "coord -> vert coord",
        vert=mesh.n_verts,
    )

    cochain = music_ops.local_flat(vec_field=vec_field, mesh=mesh, mode="vertex")

    vec_field_reconstructed = music_ops.local_sharp(
        cochain_1=cochain, mesh=mesh, mode="vertex"
    )

    vec_field_cos_dist = torch.sum(vec_field * vec_field_reconstructed, dim=-1) / (
        torch.linalg.norm(vec_field, dim=-1)
        * torch.linalg.norm(vec_field_reconstructed, dim=-1)
    )

    assert (vec_field_cos_dist > 0.7).all()


@pytest.mark.parametrize(
    "mesh, location",
    [
        ("hollow_tet_mesh", "barycenter"),
        ("hollow_tet_mesh", "circumcenter"),
        ("two_tets_mesh", "barycenter"),
    ],
)
def test_local_element_based_const_vec_field_reconstruction(
    mesh, location, request, device
):
    """
    Taking the flat of a constant vector field, followed by sharp, should reproduce
    the same constant vector field.

    Note that, for the tri mesh, the reconstructed vector field is strictly inside
    the tangent spaces of the triangles.
    """
    mesh = request.getfixturevalue(mesh).to(device)

    vec_field = repeat(
        torch.randn(3, dtype=mesh.dtype, device=mesh.device),
        "coord -> splx coord",
        splx=mesh.n_splx[mesh.dim],
    )

    cochain = music_ops.local_flat(vec_field=vec_field, mesh=mesh, mode="element")

    vec_field_reconstructed = music_ops.local_sharp(
        cochain_1=cochain, mesh=mesh, mode="element", location=location
    )

    match mesh.dim:
        case 2:
            tri_verts = mesh.vert_coords[mesh.tris]

            area_normal = torch.cross(
                tri_verts[:, 1] - tri_verts[:, 0],
                tri_verts[:, 2] - tri_verts[:, 0],
                dim=-1,
            )
            area_unormal = area_normal / torch.linalg.norm(
                area_normal, dim=-1, keepdim=True
            )

            vec_field_normal = (
                torch.sum(vec_field * area_unormal, dim=-1, keepdim=True) * area_unormal
            )
            vec_field_tangent = vec_field - vec_field_normal

            torch.testing.assert_close(vec_field_reconstructed, vec_field_tangent)

        case 3:
            torch.testing.assert_close(vec_field_reconstructed, vec_field)


@pytest.mark.parametrize("mesh", ["square_mesh", "two_tets_mesh"])
@pytest.mark.parametrize("diagonal", [True, False])
def test_galerkin_vertex_based_const_vec_field_reconstruction(
    mesh, diagonal, request, device
):
    """
    Taking the flat of a constant vector field, followed by sharp, should reproduce
    the same constant vector field.

    Note that, for this test to pass on a tri mesh, we setup the mesh to be
    flat and entirely within the z = 0 plane, and test a random vector field with
    zero z-component. This is partly because the flat and sharp operators operate
    on the tangent spaces of the triangles (as is with the local approaches).
    Furthermore, if a tri mesh has non-zero curvature, projecting a constant vector
    field onto the local tangent spaces creates discontinuities at the edge, which
    the vertex-based Galerkin formulation cannot represent and smoothes over.

    Therefore, the reconstructed vector field will only match the original on a
    tri mesh if (1) the mesh has no curvature and (2) the starting vector field
    has no component normal to the mesh. Note that this restriction does not apply
    to a tet mesh because it is a flat 3-manifold embedded in R^3 and its tangent
    space spans R^3.

    Lastly, note that, for a constant vector field v, M_V@v gives the same result
    whether M_V is derived from the consistent mass-0 matrix or the diagonal
    Hodge star-0 matrix.
    """
    mesh = request.getfixturevalue(mesh).to(device)

    match mesh.dim:
        case 2:
            mass_1 = tri_masses.mass_1(mesh)
        case 3:
            mass_1 = tet_masses.mass_1(mesh)

    mass_vec = music_ops.vector_mass(mesh, mode="vertex", diagonal=diagonal)
    mass_mixed = music_ops.mixed_mass(mesh, mode="vertex")

    random_vec = torch.randn(3, dtype=mesh.dtype, device=mesh.device)
    if mesh.dim == 2:
        random_vec[-1] = 0.0

    vec_field = repeat(
        random_vec,
        "coord -> vert coord",
        vert=mesh.n_verts,
    )

    cochain = music_ops.galerkin_flat(
        vec_field=vec_field,
        mass_1=mass_1,
        mass_mixed=mass_mixed,
        mode="vertex",
    )

    vec_field_reconstructed = music_ops.galerkin_sharp(
        cochain_1=cochain,
        mass_vec=mass_vec,
        mass_mixed=mass_mixed,
        mode="vertex",
    )

    torch.testing.assert_close(vec_field_reconstructed, vec_field)

    # Also test the InvSparseOperator route
    if not diagonal:
        mass_1_op = SuperLU(mass_1, backend="scipy")
        mass_vec_op = SuperLU(mass_vec, backend="scipy")

        cochain = music_ops.galerkin_flat(
            vec_field=vec_field,
            mass_1=mass_1_op,
            mass_mixed=mass_mixed,
            mode="vertex",
        )

        vec_field_reconstructed = music_ops.galerkin_sharp(
            cochain_1=cochain,
            mass_vec=mass_vec_op,
            mass_mixed=mass_mixed,
            mode="vertex",
        )

        torch.testing.assert_close(vec_field_reconstructed, vec_field)


@pytest.mark.parametrize("mesh", ["hollow_tet_mesh", "two_tets_mesh"])
def test_galerkin_element_based_const_vec_field_reconstruction(mesh, request, device):
    mesh = request.getfixturevalue(mesh).to(device)

    match mesh.dim:
        case 2:
            mass_1 = tri_masses.mass_1(mesh)
        case 3:
            mass_1 = tet_masses.mass_1(mesh)

    mass_vec = music_ops.vector_mass(mesh, mode="element")
    mass_mixed = music_ops.mixed_mass(mesh, mode="element")

    vec_field = repeat(
        torch.randn(3, dtype=mesh.dtype, device=mesh.device),
        "coord -> splx coord",
        splx=mesh.n_splx[mesh.dim],
    )

    mass_1_op = SuperLU(mass_1, backend="scipy")

    # Test both the dense solver and InvSparseOperator routes.
    for m1 in [mass_1, mass_1_op]:
        cochain = music_ops.galerkin_flat(
            vec_field=vec_field,
            mass_1=m1,
            mass_mixed=mass_mixed,
            mode="element",
        )

        vec_field_reconstructed = music_ops.galerkin_sharp(
            cochain_1=cochain,
            mass_vec=mass_vec,
            mass_mixed=mass_mixed,
            mode="element",
        )

        match mesh.dim:
            case 2:
                # As before, the reconstructed vector field is in the tangent space
                # of the triangles and the normal component of the original field
                # needs to be projected out prior to comparison.
                tri_verts = mesh.vert_coords[mesh.tris]

                area_normal = torch.cross(
                    tri_verts[:, 1] - tri_verts[:, 0],
                    tri_verts[:, 2] - tri_verts[:, 0],
                    dim=-1,
                )
                area_unormal = area_normal / torch.linalg.norm(
                    area_normal, dim=-1, keepdim=True
                )

                vec_field_normal = (
                    torch.sum(vec_field * area_unormal, dim=-1, keepdim=True)
                    * area_unormal
                )
                vec_field_tangent = vec_field - vec_field_normal

                torch.testing.assert_close(vec_field_reconstructed, vec_field_tangent)

            case 3:
                torch.testing.assert_close(vec_field_reconstructed, vec_field)


@pytest.mark.parametrize("mesh", ["hollow_tet_mesh", "two_tets_mesh"])
def test_galerkin_element_based_adjoint_relation(mesh, request, device):
    """
    Test the adjoint relation between the sharp and flat operators. Specifically,
    for a given vector field v and 1-cochain η,

    <v, ♯η> = <♭v, η>

    where the first inner product is defined in the vector space L^2(V) and evaluated
    as v.T @ M_V @ ♯η, and the second inner product is defined in the edge space
    L^2(E) and evaluated as ♭v.T @ M_1 @ η.

    Note that this adjoint relation (as defined above) is not satisfied by
    the local approaches.
    """
    mesh = request.getfixturevalue(mesh).to(device)

    match mesh.dim:
        case 2:
            mass_1 = tri_masses.mass_1(mesh)
        case 3:
            mass_1 = tet_masses.mass_1(mesh)

    mass_vec = music_ops.vector_mass(mesh, mode="element")
    mass_mixed = music_ops.mixed_mass(mesh, mode="element")

    vec_field = torch.randn(
        (mesh.n_splx[mesh.dim], 3), dtype=mesh.dtype, device=mesh.device
    )
    vec_field_flat = music_ops.galerkin_flat(
        vec_field=vec_field,
        mass_1=mass_1,
        mass_mixed=mass_mixed,
        mode="element",
    )

    cochain = torch.randn(mesh.n_edges, dtype=mesh.dtype, device=mesh.device)
    cochain_sharp = music_ops.galerkin_sharp(
        cochain_1=cochain, mass_vec=mass_vec, mass_mixed=mass_mixed, mode="element"
    )

    lhs = torch.sum(vec_field.flatten() * (mass_vec @ cochain_sharp.flatten()))
    rhs = torch.sum(vec_field_flat * (mass_1 @ cochain))

    torch.testing.assert_close(lhs, rhs)


@pytest.mark.parametrize("mesh", ["hollow_tet_mesh", "two_tets_mesh"])
def test_galerkin_vertex_based_adjoint_relation(mesh, request, device):
    """
    Test the adjoint relation between the sharp and flat operators.
    """
    mesh = request.getfixturevalue(mesh).to(device)

    match mesh.dim:
        case 2:
            mass_1 = tri_masses.mass_1(mesh)
        case 3:
            mass_1 = tet_masses.mass_1(mesh)

    mass_vec = music_ops.vector_mass(mesh, mode="vertex", diagonal=False)
    mass_mixed = music_ops.mixed_mass(mesh, mode="vertex")

    vec_field = torch.randn((mesh.n_verts, 3), dtype=mesh.dtype, device=mesh.device)
    vec_field_flat = music_ops.galerkin_flat(
        vec_field=vec_field,
        mass_1=mass_1,
        mass_mixed=mass_mixed,
        mode="vertex",
    )

    cochain = torch.randn(mesh.n_edges, dtype=mesh.dtype, device=mesh.device)
    cochain_sharp = music_ops.galerkin_sharp(
        cochain_1=cochain,
        mass_vec=mass_vec,
        mass_mixed=mass_mixed,
        mode="vertex",
    )

    lhs = torch.sum(vec_field.flatten() * (mass_vec @ cochain_sharp.flatten()))
    rhs = torch.sum(vec_field_flat * (mass_1 @ cochain))

    torch.testing.assert_close(lhs, rhs)

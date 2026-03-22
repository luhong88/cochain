import pytest
import torch as t

from cochain.cochain.discretize import DeRhamMap
from cochain.cochain.interpolate import barycentric_whitney_map
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
    k_cochain_true = t.randn(mesh.simplices[k].size(0)).to(device)

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

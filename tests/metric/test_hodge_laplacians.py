import pytest
import torch
from torch import Tensor

from cochain.complex import SimplicialMesh
from cochain.metric.hodge_laplacians import (
    codifferential,
    weak_down_laplacian,
    weak_up_laplacian,
)
from cochain.metric.tri import tri_masses
from cochain.sparse.decoupled_tensor import SparseDecoupledTensor
from cochain.sparse.linalg.solvers import SuperLU


def _operators(
    tri_mesh: SimplicialMesh,
) -> tuple[
    SparseDecoupledTensor,
    Tensor,
    SparseDecoupledTensor,
    Tensor,
    SparseDecoupledTensor,
]:
    """Construct compatible degree-zero and degree-one operators on a tri mesh."""
    mesh = tri_mesh.to(dtype=torch.float64)
    cbd_km1 = mesh.cbd[0]
    mass_km1 = tri_masses.mass_0(mesh)
    mass_k = tri_masses.mass_1(mesh)

    return (
        cbd_km1,
        mass_km1.to_dense(),
        mass_km1,
        mass_k.to_dense(),
        mass_k,
    )


@pytest.mark.parametrize(
    "route",
    ["dense_mass", "sparse_mass", "sparse_solver", "dense_inverse", "sparse_inverse"],
)
def test_codifferential_inverse_mass_routes_agree(
    route: str, hollow_tet_mesh: SimplicialMesh, device
):
    """All exact inverse-mass routes should produce the same codifferential."""
    cbd_km1, mass_km1_dense, mass_km1, mass_k_dense, mass_k = _operators(
        hollow_tet_mesh.to(device)
    )
    expected = torch.linalg.solve(mass_km1_dense, cbd_km1.to_dense().T @ mass_k_dense)

    match route:
        case "dense_mass":
            result = codifferential(cbd_km1, mass_k, mass_km1=mass_km1_dense)
            assert isinstance(result, Tensor)
        case "sparse_mass":
            result = codifferential(cbd_km1, mass_k, mass_km1=mass_km1)
            assert isinstance(result, Tensor)
        case "sparse_solver":
            solver = SuperLU(mass_km1, backend="scipy")
            result = codifferential(cbd_km1, mass_k, mass_km1=solver)
            assert isinstance(result, Tensor)
        case "dense_inverse":
            inv_mass_km1 = torch.linalg.inv(mass_km1_dense)
            result = codifferential(cbd_km1, mass_k, inv_mass_km1=inv_mass_km1)
            assert isinstance(result, Tensor)
        case "sparse_inverse":
            inv_mass_km1 = SparseDecoupledTensor.from_tensor(
                torch.linalg.inv(mass_km1_dense).to_sparse()
            )
            result = codifferential(cbd_km1, mass_k, inv_mass_km1=inv_mass_km1)
            assert isinstance(result, SparseDecoupledTensor)
        case _:
            raise AssertionError(f"Unknown test route: {route}")

    torch.testing.assert_close(result.to_dense(), expected)


@pytest.mark.parametrize(
    "route", ["dense_mass", "sparse_mass", "sparse_solver", "sparse_inverse"]
)
def test_weak_down_laplacian_inverse_mass_routes_agree(
    route: str, hollow_tet_mesh: SimplicialMesh, device
):
    """All exact inverse-mass routes should produce the same weak down term."""
    cbd_km1, mass_km1_dense, mass_km1, mass_k_dense, mass_k = _operators(
        hollow_tet_mesh.to(device)
    )
    m_d = mass_k_dense @ cbd_km1.to_dense()
    expected = m_d @ torch.linalg.solve(mass_km1_dense, m_d.T)

    match route:
        case "dense_mass":
            result = weak_down_laplacian(cbd_km1, mass_k, mass_km1=mass_km1_dense)
            assert isinstance(result, Tensor)
        case "sparse_mass":
            result = weak_down_laplacian(cbd_km1, mass_k, mass_km1=mass_km1)
            assert isinstance(result, Tensor)
        case "sparse_solver":
            solver = SuperLU(mass_km1, backend="scipy")
            result = weak_down_laplacian(cbd_km1, mass_k, mass_km1=solver)
            assert isinstance(result, Tensor)
        case "sparse_inverse":
            inv_mass_km1 = SparseDecoupledTensor.from_tensor(
                torch.linalg.inv(mass_km1_dense).to_sparse()
            )
            result = weak_down_laplacian(cbd_km1, mass_k, inv_mass_km1=inv_mass_km1)
            assert isinstance(result, SparseDecoupledTensor)
        case _:
            raise AssertionError(f"Unknown test route: {route}")

    torch.testing.assert_close(result.to_dense(), expected)


def test_weak_up_laplacian_matches_dense_formula(
    hollow_tet_mesh: SimplicialMesh, device
):
    """The weak up term should match its direct dense formula and remain sparse."""
    cbd_k, _, _, mass_kp1_dense, mass_kp1 = _operators(hollow_tet_mesh.to(device))
    result = weak_up_laplacian(cbd_k, mass_kp1)
    expected = cbd_k.to_dense().T @ mass_kp1_dense @ cbd_k.to_dense()

    assert isinstance(result, SparseDecoupledTensor)
    torch.testing.assert_close(result.to_dense(), expected)


@pytest.mark.parametrize("operator", [codifferential, weak_down_laplacian])
@pytest.mark.parametrize("provided", ["neither", "both"])
def test_inverse_mass_arguments_are_mutually_exclusive(
    operator, provided: str, hollow_tet_mesh: SimplicialMesh, device
):
    """Exactly one mass or explicit inverse must be supplied."""
    cbd_km1, _, mass_km1, _, mass_k = _operators(hollow_tet_mesh.to(device))
    kwargs = {}
    if provided == "both":
        kwargs = {"mass_km1": mass_km1, "inv_mass_km1": mass_km1}

    with pytest.raises(
        ValueError, match="Exactly one of 'mass' and 'inv_mass' must be provided"
    ):
        operator(cbd_km1, mass_k, **kwargs)


@pytest.mark.parametrize("operator", [codifferential, weak_down_laplacian])
def test_solver_kwargs_rejected_for_matrix_mass(
    operator, hollow_tet_mesh: SimplicialMesh, device
):
    """Sparse-solver keyword arguments must not be silently ignored."""
    cbd_km1, _, mass_km1, _, mass_k = _operators(hollow_tet_mesh.to(device))

    with pytest.raises(
        ValueError,
        match="'solver_kwargs' is only valid when 'mass' is an InvSparseOperator",
    ):
        operator(
            cbd_km1,
            mass_k,
            mass_km1=mass_km1,
            solver_kwargs={"trans": "N"},
        )


def test_codifferential_rejects_incompatible_mass_shape(
    hollow_tet_mesh: SimplicialMesh, device
):
    """An incompatible mass matrix should fail rather than broadcast silently."""
    cbd_km1, _, _, _, mass_k = _operators(hollow_tet_mesh.to(device))
    incompatible_mass = torch.eye(3, dtype=mass_k.dtype)

    with pytest.raises(RuntimeError):
        codifferential(cbd_km1, mass_k, mass_km1=incompatible_mass)


def test_sparse_solver_route_propagates_finite_gradients(
    hollow_tet_mesh: SimplicialMesh, device
):
    """The sparse solve should differentiate through mass and RHS values."""
    cbd_km1, _, mass_km1, _, mass_k = _operators(hollow_tet_mesh.to(device))
    mass_km1.requires_grad_()
    mass_k.requires_grad_()
    solver = SuperLU(mass_km1, backend="scipy")

    result = weak_down_laplacian(cbd_km1, mass_k, mass_km1=solver)
    result.sum().backward()

    assert mass_km1.grad is not None
    assert mass_k.grad is not None
    assert torch.isfinite(mass_km1.grad).all()
    assert torch.isfinite(mass_k.grad).all()

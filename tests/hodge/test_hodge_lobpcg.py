import pytest
import pytetwild
import pyvista as pv
import torch
from jaxtyping import Float
from torch import Tensor

from cochain.complex import SimplicialMesh
from cochain.hodge.eigen import (
    LaplacianLOBPCGPrecondConfig,
    LOBPCGConfig,
    mixed_weak_laplacian_lobpcg,
)
from cochain.hodge.laplacians import (
    MixedWeakLaplacianBlocks,
    codifferential,
    weak_down_laplacian,
    weak_up_laplacian,
)
from cochain.metric.tet import tet_hodge_stars, tet_masses
from cochain.sparse.linalg.eigen import canonicalize_eig_vec_signs
from cochain.sparse.linalg.solvers import SuperLU

# Since the eigensolver for mixed weak Hodge Laplacians always involve a sparse
# linear solve, we generally relax the assert_close() tolerance thresholds in
# this suite of tests to atol=1e-5.

itemize_backends = pytest.mark.parametrize(
    "backend",
    [
        pytest.param("scipy_splu", marks=[]),
        pytest.param(
            "cupy_splu", marks=[pytest.mark.gpu_only, pytest.mark.requires_cupy]
        ),
        pytest.param(
            "nvmath_direct_solver",
            marks=[pytest.mark.gpu_only, pytest.mark.requires_nvmath],
        ),
    ],
)


@pytest.fixture(scope="module")
def bunny_tet_mesh() -> SimplicialMesh:
    # Load the mesh from pyvista.
    bunny_pv_tri = (
        pv.examples.download_bunny_coarse()
        .clean()
        .triangulate()
        .decimate_pro(0.7, preserve_topology=True, splitting=False)
    )

    # Tetrahedralize the bunny tri mesh
    v_tet, t_tet = pytetwild.tetrahedralize(
        bunny_pv_tri.points,
        bunny_pv_tri.faces.reshape(-1, 4)[:, 1:],
        edge_length_fac=0.9,
    )

    # Scale the mesh size and recenter.
    bunny = SimplicialMesh.from_tet_mesh(
        vert_coords=15.0 * torch.from_numpy(v_tet).to(dtype=torch.float32),
        tets=torch.from_numpy(t_tet).to(dtype=torch.int64),
    )

    # Center the mesh
    bunny.vert_coords.sub_(bunny.vert_coords.mean(dim=0, keepdim=True))

    return bunny


def get_mixed_weak_down_2_laplacian(
    tet_mesh: SimplicialMesh,
) -> Float[MixedWeakLaplacianBlocks, "tri tri"]:
    return MixedWeakLaplacianBlocks(
        cbd_km1=tet_mesh.cbd[1],
        cbd_k=None,
        mass_km1=tet_masses.mass_1(tet_mesh),
        mass_k=tet_masses.mass_2(tet_mesh),
        mass_kp1=None,
    )


def get_schur_complement_weak_down_2_laplacian(
    tet_mesh: SimplicialMesh,
) -> Float[Tensor, "tri tri"]:
    mass_km1 = SuperLU(tet_masses.mass_1(tet_mesh), backend="scipy")
    return weak_down_laplacian(
        cbd_km1=tet_mesh.cbd[1],
        mass_k=tet_masses.mass_2(tet_mesh),
        mass_km1=mass_km1,
    )


def get_mixed_weak_1_laplacian(
    tet_mesh: SimplicialMesh,
) -> Float[MixedWeakLaplacianBlocks, "edge edge"]:
    return MixedWeakLaplacianBlocks(
        cbd_km1=tet_mesh.cbd[0],
        cbd_k=tet_mesh.cbd[1],
        mass_km1=tet_masses.mass_0(tet_mesh),
        mass_k=tet_masses.mass_1(tet_mesh),
        mass_kp1=tet_masses.mass_2(tet_mesh),
    )


def get_schur_complement_weak_1_laplacian(
    tet_mesh: SimplicialMesh,
) -> Float[Tensor, "edge edge"]:
    up_laplacian = weak_up_laplacian(
        cbd_k=tet_mesh.cbd[1], mass_kp1=tet_masses.mass_2(tet_mesh)
    ).to_dense()

    mass_km1 = SuperLU(tet_masses.mass_0(tet_mesh), backend="scipy")
    down_laplacian = weak_down_laplacian(
        cbd_km1=tet_mesh.cbd[0],
        mass_k=tet_masses.mass_1(tet_mesh),
        mass_km1=mass_km1,
    )

    laplacian = up_laplacian + down_laplacian

    return laplacian


def dense_gep(
    a: Float[Tensor, "m m"], m: Float[Tensor, "m m"]
) -> tuple[Float[Tensor, " k"], Float[Tensor, "m k"]]:
    # Since torch.linalg.eigh() does not support GEP, need to perform Cholesky
    # whitening on A.
    m_cho_inv = torch.linalg.inv(torch.linalg.cholesky(m))
    a_whitened = m_cho_inv @ a @ m_cho_inv.T

    eig_vals_true, eig_vecs_whitened = torch.linalg.eigh(a_whitened)
    eig_vecs_true = m_cho_inv.T @ eig_vecs_whitened

    return eig_vals_true, eig_vecs_true


def test_v0_shape_validation(bunny_tet_mesh, device):
    mesh = bunny_tet_mesh.to(device)

    laplacian = get_mixed_weak_down_2_laplacian(mesh)
    v0 = torch.randn(
        (laplacian.size(-1), 3), dtype=laplacian.dtype, device=laplacian.device
    )

    eig_vals, eig_vecs = mixed_weak_laplacian_lobpcg(
        laplacian, n=3, l=3, solver_type="scipy_splu", lobpcg_config=LOBPCGConfig(v0=v0)
    )
    assert eig_vals.shape == (3,)
    assert eig_vecs.shape == (laplacian.size(-1), 3)

    with pytest.raises(ValueError, match="v0 must have shape"):
        mixed_weak_laplacian_lobpcg(
            laplacian, n=3, l=3, lobpcg_config=LOBPCGConfig(v0=v0[:, :1])
        )

    with pytest.raises(ValueError, match="v0 must have shape"):
        mixed_weak_laplacian_lobpcg(
            laplacian, n=3, l=3, lobpcg_config=LOBPCGConfig(v0=v0[:5])
        )


@itemize_backends
def test_full_laplacian_forward(bunny_tet_mesh, backend, device):
    mesh = bunny_tet_mesh.to(device, torch.float64)

    mixed_laplacian = get_mixed_weak_1_laplacian(mesh)
    schur_complement = get_schur_complement_weak_1_laplacian(mesh)

    eig_vals_true, eig_vecs_true = dense_gep(
        schur_complement, mixed_laplacian.mass_k.to_dense()
    )

    l = 3

    # Test both largest=True and largest=False
    eig_vals_rev, eig_vecs_rev = mixed_weak_laplacian_lobpcg(
        mixed_laplacian,
        n=2 * l,
        l=l,
        solver_type=backend,
        lobpcg_config=LOBPCGConfig(largest=True),
    )
    eig_vals = torch.flip(eig_vals_rev, dims=(0,))
    eig_vecs = torch.flip(eig_vecs_rev, dims=(-1,))

    # If largest=True, lobpcg returns eigenvalues in descending order.
    torch.testing.assert_close(eig_vals, eig_vals_true[-l:])
    torch.testing.assert_close(
        canonicalize_eig_vec_signs(eig_vecs),
        canonicalize_eig_vec_signs(eig_vecs_true[:, -l:]),
        atol=1e-5,
        rtol=0.0,
    )

    eig_vals, eig_vecs = mixed_weak_laplacian_lobpcg(
        mixed_laplacian,
        n=2 * l,
        l=l,
        solver_type=backend,
        lobpcg_config=LOBPCGConfig(largest=False),
    )

    torch.testing.assert_close(eig_vals, eig_vals_true[:l])
    torch.testing.assert_close(
        canonicalize_eig_vec_signs(eig_vecs),
        canonicalize_eig_vec_signs(eig_vecs_true[:, :l]),
        atol=1e-5,
        rtol=0.0,
    )


def test_down_laplacian_forward(bunny_tet_mesh, device):
    mesh = bunny_tet_mesh.to(device, torch.float64)

    mixed_laplacian = get_mixed_weak_down_2_laplacian(mesh)
    schur_complement = get_schur_complement_weak_down_2_laplacian(mesh)

    eig_vals_true, eig_vecs_true = dense_gep(
        schur_complement, mixed_laplacian.mass_k.to_dense()
    )

    l = 3

    # Test only largest=True, since the down-component has a massive null space.
    eig_vals_rev, eig_vecs_rev = mixed_weak_laplacian_lobpcg(
        mixed_laplacian,
        n=2 * l,
        l=l,
        lobpcg_config=LOBPCGConfig(largest=True),
    )
    eig_vals = torch.flip(eig_vals_rev, dims=(0,))
    eig_vecs = torch.flip(eig_vecs_rev, dims=(-1,))

    # If largest=True, lobpcg returns eigenvalues in descending order.
    torch.testing.assert_close(eig_vals, eig_vals_true[-l:])
    torch.testing.assert_close(
        canonicalize_eig_vec_signs(eig_vecs),
        canonicalize_eig_vec_signs(eig_vecs_true[:, -l:]),
        atol=1e-5,
        rtol=0.0,
    )


@pytest.mark.gpu_only
@pytest.mark.requires_nvmath
def test_full_laplacian_forward_shifted_up_precond(bunny_tet_mesh, device):
    mesh = bunny_tet_mesh.to(device, torch.float64)

    mixed_laplacian = get_mixed_weak_1_laplacian(mesh)
    schur_complement = get_schur_complement_weak_1_laplacian(mesh)

    eig_vals_true, eig_vecs_true = dense_gep(
        schur_complement, mixed_laplacian.mass_k.to_dense()
    )

    l = 3

    # Test largest=False only
    eig_vals, eig_vecs = mixed_weak_laplacian_lobpcg(
        mixed_laplacian,
        n=2 * l,
        l=l,
        lobpcg_config=LOBPCGConfig(largest=False),
        precond_config=LaplacianLOBPCGPrecondConfig(
            method="shifted_up", star_k=tet_hodge_stars.star_1(mesh)
        ),
    )

    torch.testing.assert_close(eig_vals, eig_vals_true[:l])
    torch.testing.assert_close(
        canonicalize_eig_vec_signs(eig_vecs),
        canonicalize_eig_vec_signs(eig_vecs_true[:, :l]),
        atol=1e-5,
        rtol=0.0,
    )


@pytest.mark.gpu_only
@pytest.mark.requires_nvmath
def test_full_laplacian_forward_shifted_lumped_precond(bunny_tet_mesh, device):
    mesh = bunny_tet_mesh.to(device, torch.float64)

    mixed_laplacian = get_mixed_weak_1_laplacian(mesh)
    schur_complement = get_schur_complement_weak_1_laplacian(mesh)

    eig_vals_true, eig_vecs_true = dense_gep(
        schur_complement, mixed_laplacian.mass_k.to_dense()
    )

    l = 3

    # Test largest=False only
    eig_vals, eig_vecs = mixed_weak_laplacian_lobpcg(
        mixed_laplacian,
        n=2 * l,
        l=l,
        lobpcg_config=LOBPCGConfig(largest=False),
        precond_config=LaplacianLOBPCGPrecondConfig(
            method="shifted_lumped",
            star_km1=tet_hodge_stars.star_0(mesh),
            star_k=tet_hodge_stars.star_1(mesh),
        ),
    )

    torch.testing.assert_close(eig_vals, eig_vals_true[:l])
    torch.testing.assert_close(
        canonicalize_eig_vec_signs(eig_vecs),
        canonicalize_eig_vec_signs(eig_vecs_true[:, :l]),
        atol=1e-5,
        rtol=0.0,
    )


def test_full_laplacian_eig_vals_backward(bunny_tet_mesh, device):
    mesh = bunny_tet_mesh.to(device, torch.float64)
    mesh.requires_grad_()

    l = 3

    # dense path
    d_km1 = mesh.cbd[0]
    d_k = mesh.cbd[1]

    mass_km1 = tet_masses.mass_0(mesh).to_dense()
    mass_k = tet_masses.mass_1(mesh).to_dense()
    mass_kp1 = tet_masses.mass_2(mesh).to_dense()

    schur_complement = (
        mass_k @ d_km1 @ torch.linalg.solve(mass_km1, d_km1.T @ mass_k)
        + d_k.T @ mass_kp1 @ d_k
    )

    eig_vals_true_all, eig_vecs_true_all = dense_gep(schur_complement, mass_k)
    eig_vals_true = eig_vals_true_all[:l]

    eig_vals_rand = torch.randn_like(eig_vals_true)
    eig_vals_loss_true = torch.sum(eig_vals_true * eig_vals_rand)
    eig_vals_loss_true.backward()
    vert_coords_grad_true = mesh.grad.detach().clone()

    # sparse path
    mesh.grad = None
    mixed_laplacian = get_mixed_weak_1_laplacian(mesh)

    # Help the forward pass by using the dense path eigenvectors as initial guess.
    v0 = eig_vecs_true_all[:, : 2 * l].detach().clone()

    eig_vals, eig_vecs = mixed_weak_laplacian_lobpcg(
        mixed_laplacian,
        n=2 * l,
        l=l,
        lobpcg_config=LOBPCGConfig(largest=False, v0=v0),
    )

    eig_vals_loss = torch.sum(eig_vals * eig_vals_rand)
    eig_vals_loss.backward()
    vert_coords_grad = mesh.grad.detach().clone()

    torch.testing.assert_close(
        vert_coords_grad, vert_coords_grad_true, atol=1e-5, rtol=0.0
    )


def test_full_laplacian_eig_vecs_backward(bunny_tet_mesh, device):
    mesh = bunny_tet_mesh.to(device, torch.float64)
    l = 3

    # dense path
    d_km1 = mesh.cbd[0]
    d_k = mesh.cbd[1]

    mass_km1_sp = tet_masses.mass_0(mesh)
    mass_km1 = mass_km1_sp.to_dense()

    mass_k_sp = tet_masses.mass_1(mesh)
    mass_k = mass_k_sp.to_dense()

    mass_kp1_sp = tet_masses.mass_2(mesh)
    mass_kp1 = mass_kp1_sp.to_dense()

    schur_complement = (
        mass_k @ d_km1 @ torch.linalg.solve(mass_km1, d_km1.T @ mass_k)
        + d_k.T @ mass_kp1 @ d_k
    )

    # Treat M_k and S_k as independent leaf tensors.
    schur_complement.requires_grad_()
    mass_k.requires_grad_()

    eig_vals_true_all, eig_vecs_true_all = dense_gep(schur_complement, mass_k)
    eig_vecs_true = eig_vecs_true_all[:, :l]

    eig_vecs_rand = torch.randn_like(eig_vecs_true)
    eig_vecs_loss_true = torch.sum(eig_vecs_rand * eig_vecs_true, dim=0).abs().sum()
    eig_vecs_loss_true.backward()

    dLdS = schur_complement.grad.detach().clone()
    dLdM_k_rhs = mass_k.grad.detach().clone()

    with torch.no_grad():
        proj = eig_vecs_true @ eig_vecs_true.T @ mass_k
        dLdS_trunc = proj @ dLdS @ proj.T

        codiff_k = codifferential(
            cbd_km1=d_km1,
            mass_k=mass_k_sp,
            mass_km1=SuperLU(mass_km1_sp, backend="scipy"),
        )
        d_km1_delta_k = d_km1 @ codiff_k

        dLdM_kp1_trunc = d_k @ dLdS_trunc @ d_k.T
        dLdM_km1_trunc = -codiff_k @ dLdS_trunc @ codiff_k.T

        dLdM_k_lhs_trunc = dLdS_trunc @ d_km1_delta_k.T + d_km1_delta_k @ dLdS_trunc
        dLdM_k_rhs_trunc = proj @ dLdM_k_rhs @ proj.T
        dLdM_k_trunc = dLdM_k_rhs_trunc + dLdM_k_lhs_trunc

        dLdM_kp1_sp = dLdM_kp1_trunc[torch.unbind(mass_kp1_sp.pattern.idx_coo, dim=0)]
        dLdM_k_sp = dLdM_k_trunc[torch.unbind(mass_k_sp.pattern.idx_coo, dim=0)]
        dLdM_km1_sp = dLdM_km1_trunc[torch.unbind(mass_km1_sp.pattern.idx_coo, dim=0)]

    # sparse path
    laplacian = get_mixed_weak_1_laplacian(mesh)

    # Set the mass matrices as the leaf tensors for accumulating grad
    laplacian.mass_km1.requires_grad_()
    laplacian.mass_k.requires_grad_()
    laplacian.mass_kp1.requires_grad_()

    # Help the forward pass by using the dense path eigenvectors as initial guess.
    v0 = eig_vecs_true_all[:, : 2 * l].detach().clone()
    eig_vals, eig_vecs = mixed_weak_laplacian_lobpcg(
        laplacian,
        n=2 * l,
        l=l,
        eps=0,
        lobpcg_config=LOBPCGConfig(largest=False, v0=v0),
    )

    eig_vecs_loss = torch.sum(eig_vecs_rand * eig_vecs, dim=0).abs().sum()
    eig_vecs_loss.backward()

    dLdM_kp1 = laplacian.mass_kp1.grad.detach().clone()
    dLdM_k = laplacian.mass_k.grad.detach().clone()
    dLdM_km1 = laplacian.mass_km1.grad.detach().clone()

    torch.testing.assert_close(dLdM_km1, dLdM_km1_sp, atol=1e-5, rtol=0.0)
    torch.testing.assert_close(dLdM_k, dLdM_k_sp, atol=1e-5, rtol=0.0)
    torch.testing.assert_close(dLdM_kp1, dLdM_kp1_sp, atol=1e-5, rtol=0.0)


def test_full_laplacian_combined_backward(bunny_tet_mesh, device):
    mesh = bunny_tet_mesh.to(device, torch.float64)
    l = 3

    # dense path
    d_km1 = mesh.cbd[0]
    d_k = mesh.cbd[1]

    mass_km1_sp = tet_masses.mass_0(mesh)
    mass_km1 = mass_km1_sp.to_dense()

    mass_k_sp = tet_masses.mass_1(mesh)
    mass_k = mass_k_sp.to_dense()

    mass_kp1_sp = tet_masses.mass_2(mesh)
    mass_kp1 = mass_kp1_sp.to_dense()

    schur_complement = (
        mass_k @ d_km1 @ torch.linalg.solve(mass_km1, d_km1.T @ mass_k)
        + d_k.T @ mass_kp1 @ d_k
    )

    # Treat M_k and S_k as independent leaf tensors.
    schur_complement.requires_grad_()
    mass_k.requires_grad_()

    eig_vals_true_all, eig_vecs_true_all = dense_gep(schur_complement, mass_k)
    eig_vals_true = eig_vals_true_all[:l]
    eig_vecs_true = eig_vecs_true_all[:, :l]

    eig_vals_rand = torch.randn_like(eig_vals_true)
    eig_vecs_rand = torch.randn_like(eig_vecs_true)
    combined_loss_true = (
        torch.sum(eig_vals_rand * eig_vals_true)
        + torch.sum(eig_vecs_rand * eig_vecs_true, dim=0).abs().sum()
    )
    combined_loss_true.backward()

    dLdS = schur_complement.grad.detach().clone()
    dLdM_k_rhs = mass_k.grad.detach().clone()

    with torch.no_grad():
        proj = eig_vecs_true @ eig_vecs_true.T @ mass_k
        dLdS_trunc = proj @ dLdS @ proj.T

        codiff_k = codifferential(
            cbd_km1=d_km1,
            mass_k=mass_k_sp,
            mass_km1=SuperLU(mass_km1_sp, backend="scipy"),
        )
        d_km1_delta_k = d_km1 @ codiff_k

        dLdM_kp1_trunc = d_k @ dLdS_trunc @ d_k.T
        dLdM_km1_trunc = -codiff_k @ dLdS_trunc @ codiff_k.T

        dLdM_k_lhs_trunc = dLdS_trunc @ d_km1_delta_k.T + d_km1_delta_k @ dLdS_trunc
        dLdM_k_rhs_trunc = proj @ dLdM_k_rhs @ proj.T
        dLdM_k_trunc = dLdM_k_rhs_trunc + dLdM_k_lhs_trunc

        dLdM_kp1_sp = dLdM_kp1_trunc[torch.unbind(mass_kp1_sp.pattern.idx_coo, dim=0)]
        dLdM_k_sp = dLdM_k_trunc[torch.unbind(mass_k_sp.pattern.idx_coo, dim=0)]
        dLdM_km1_sp = dLdM_km1_trunc[torch.unbind(mass_km1_sp.pattern.idx_coo, dim=0)]

    # sparse path
    laplacian = get_mixed_weak_1_laplacian(mesh)

    # Set the mass matrices as the leaf tensors for accumulating grad
    laplacian.mass_km1.requires_grad_()
    laplacian.mass_k.requires_grad_()
    laplacian.mass_kp1.requires_grad_()

    # Help the forward pass by using the dense path eigenvectors as initial guess.
    v0 = eig_vecs_true_all[:, : 2 * l].detach().clone()
    eig_vals, eig_vecs = mixed_weak_laplacian_lobpcg(
        laplacian,
        n=2 * l,
        l=l,
        eps=0,
        lobpcg_config=LOBPCGConfig(largest=False, v0=v0),
    )

    combined_loss = (
        torch.sum(eig_vals_rand * eig_vals)
        + torch.sum(eig_vecs_rand * eig_vecs, dim=0).abs().sum()
    )
    combined_loss.backward()

    dLdM_kp1 = laplacian.mass_kp1.grad.detach().clone()
    dLdM_k = laplacian.mass_k.grad.detach().clone()
    dLdM_km1 = laplacian.mass_km1.grad.detach().clone()

    torch.testing.assert_close(dLdM_km1, dLdM_km1_sp, atol=1e-5, rtol=0.0)
    torch.testing.assert_close(dLdM_k, dLdM_k_sp, atol=1e-5, rtol=0.0)
    torch.testing.assert_close(dLdM_kp1, dLdM_kp1_sp, atol=1e-5, rtol=0.0)


def test_lorentzian_regularization_smoke(bunny_tet_mesh, device):
    mesh = bunny_tet_mesh.to(device, torch.float64)
    mesh.requires_grad_()

    mixed_laplacian = get_mixed_weak_down_2_laplacian(mesh)

    l = 3

    # Test both largest=True and largest=False
    eig_vals, eig_vecs = mixed_weak_laplacian_lobpcg(
        mixed_laplacian,
        n=2 * l,
        l=l,
        lobpcg_config=LOBPCGConfig(largest=False),
    )

    loss = eig_vals.sum() + eig_vecs.sum()
    loss.backward()

    assert mesh.grad is not None
    assert not torch.isnan(mesh.grad).any()
    assert not torch.all(mesh.grad == 0)

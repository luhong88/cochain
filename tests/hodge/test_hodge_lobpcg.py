import pytest
import torch
from jaxtyping import Float
from torch import Tensor

from cochain.complex import SimplicialMesh
from cochain.hodge.eigen import LOBPCGConfig, mixed_weak_laplacian_lobpcg
from cochain.hodge.laplacians import (
    MixedWeakLaplacianBlocks,
    weak_down_laplacian,
    weak_up_laplacian,
)
from cochain.metric.tet import tet_masses
from cochain.sparse.decoupled_tensor import SparseDecoupledTensor
from cochain.sparse.linalg.eigen import canonicalize_eig_vec_signs
from cochain.sparse.linalg.solvers import SuperLU


def get_mixed_weak_down_2_laplacian(
    tet_mesh: SimplicialMesh,
):
    return MixedWeakLaplacianBlocks(
        cbd_km1=tet_mesh.cbd[1],
        cbd_k=None,
        mass_km1=tet_masses.mass_1(tet_mesh),
        mass_k=tet_masses.mass_2(tet_mesh),
        mass_kp1=None,
    )


def get_schur_complement_weak_down_2_laplacian(tet_mesh: SimplicialMesh):
    mass_km1 = SuperLU(tet_masses.mass_1(tet_mesh), backend="scipy")
    return weak_down_laplacian(
        cbd_km1=tet_mesh.cbd[1],
        mass_k=tet_masses.mass_2(tet_mesh),
        mass_km1=mass_km1,
    )


def get_mixed_weak_1_laplacian(
    tet_mesh: SimplicialMesh,
):
    return MixedWeakLaplacianBlocks(
        cbd_km1=tet_mesh.cbd[0],
        cbd_k=tet_mesh.cbd[1],
        mass_km1=tet_masses.mass_0(tet_mesh),
        mass_k=tet_masses.mass_1(tet_mesh),
        mass_kp1=tet_masses.mass_2(tet_mesh),
    )


def get_schur_complement_weak_1_laplacian(tet_mesh: SimplicialMesh):
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


def test_v0_shape_validation(solid_torus_mesh, device):
    mesh = solid_torus_mesh.to(device)

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


def test_full_laplacian_forward(solid_torus_mesh, device):
    mesh = solid_torus_mesh.to(device, torch.float64)

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
        lobpcg_config=LOBPCGConfig(largest=True),
    )
    eig_vals = torch.flip(eig_vals_rev, dims=(0,))
    eig_vecs = torch.flip(eig_vecs_rev, dims=(-1,))

    # If largest=True, lobpcg returns eigenvalues in descending order.
    torch.testing.assert_close(eig_vals, eig_vals_true[-l:])
    torch.testing.assert_close(
        canonicalize_eig_vec_signs(eig_vecs),
        canonicalize_eig_vec_signs(eig_vecs_true[:, -l:]),
        atol=1e-6,
        rtol=1e-6,
    )

    eig_vals, eig_vecs = mixed_weak_laplacian_lobpcg(
        mixed_laplacian,
        n=2 * l,
        l=l,
        lobpcg_config=LOBPCGConfig(largest=False),
    )

    torch.testing.assert_close(eig_vals, eig_vals_true[:l])
    torch.testing.assert_close(
        canonicalize_eig_vec_signs(eig_vecs),
        canonicalize_eig_vec_signs(eig_vecs_true[:, :l]),
    )


def test_down_laplacian_forward(solid_torus_mesh, device):
    mesh = solid_torus_mesh.to(device, torch.float64)

    mixed_laplacian = get_mixed_weak_down_2_laplacian(mesh)
    schur_complement = get_schur_complement_weak_down_2_laplacian(mesh)

    eig_vals_true, eig_vecs_true = dense_gep(
        schur_complement, mixed_laplacian.mass_k.to_dense()
    )
    print(
        "[diagnostic] dense eigenvalues (first/last 8):",
        eig_vals_true[:8],
        eig_vals_true[-8:],
    )

    l = 3

    # Test both largest=True and largest=False
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
        atol=1e-6,
        rtol=1e-6,
    )

    eig_vals, eig_vecs = mixed_weak_laplacian_lobpcg(
        mixed_laplacian,
        n=2 * l,
        l=l,
        lobpcg_config=LOBPCGConfig(largest=False),
    )

    torch.testing.assert_close(eig_vals, eig_vals_true[:l])
    torch.testing.assert_close(
        canonicalize_eig_vec_signs(eig_vecs),
        canonicalize_eig_vec_signs(eig_vecs_true[:, :l]),
    )

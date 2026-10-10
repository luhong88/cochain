import pytest

from cochain.hodge.eigen import _preconditioners, lobpcg_
from cochain.hodge.laplacians import MixedWeakLaplacianBlocks
from cochain.metric.tet import tet_hodge_stars, tet_masses


@pytest.mark.gpu_only
def test_shifted_up_precond_import_error(two_tets_mesh, device, monkeypatch):
    monkeypatch.setattr(_preconditioners, "_HAS_NVMATH", False)

    tet_mesh = two_tets_mesh.to(device)
    l = 3

    laplacian = MixedWeakLaplacianBlocks(
        cbd_km1=tet_mesh.cbd[0],
        cbd_k=tet_mesh.cbd[1],
        mass_km1=tet_masses.mass_0(tet_mesh),
        mass_k=tet_masses.mass_1(tet_mesh),
        mass_kp1=tet_masses.mass_2(tet_mesh),
    )

    with pytest.raises(ImportError):
        eig_vals, eig_vecs = lobpcg_.mixed_weak_laplacian_lobpcg(
            laplacian,
            n=2 * l,
            l=l,
            lobpcg_config=lobpcg_.LOBPCGConfig(largest=False),
            precond_config=lobpcg_.LaplacianLOBPCGPrecondConfig(
                method="shifted_up", star_k=tet_hodge_stars.star_1(tet_mesh)
            ),
        )


@pytest.mark.gpu_only
def test_shifted_lumped_precond_import_error(two_tets_mesh, device, monkeypatch):
    monkeypatch.setattr(_preconditioners, "_HAS_NVMATH", False)

    tet_mesh = two_tets_mesh.to(device)
    l = 3

    laplacian = MixedWeakLaplacianBlocks(
        cbd_km1=tet_mesh.cbd[0],
        cbd_k=tet_mesh.cbd[1],
        mass_km1=tet_masses.mass_0(tet_mesh),
        mass_k=tet_masses.mass_1(tet_mesh),
        mass_kp1=tet_masses.mass_2(tet_mesh),
    )

    with pytest.raises(ImportError):
        eig_vals, eig_vecs = lobpcg_.mixed_weak_laplacian_lobpcg(
            laplacian,
            n=2 * l,
            l=l,
            lobpcg_config=lobpcg_.LOBPCGConfig(largest=False),
            precond_config=lobpcg_.LaplacianLOBPCGPrecondConfig(
                method="shifted_lumped",
                star_km1=tet_hodge_stars.star_0(tet_mesh),
                star_k=tet_hodge_stars.star_1(tet_mesh),
            ),
        )

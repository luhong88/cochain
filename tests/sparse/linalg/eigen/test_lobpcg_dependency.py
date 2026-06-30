import pytest

from cochain.sparse.decoupled_tensor import SparseDecoupledTensor
from cochain.sparse.linalg.eigen.lobpcg_ import (
    _lobpcg_operators,
    _lobpcg_preconditioners,
    lobpcg_,
)


def test_cupy_nvmath_independence(rand_sp_spd_6x6, device, monkeypatch):
    # Note that the LOBPCG module does not check for cupy and nvmath availability
    # at the top level.
    monkeypatch.setattr(_lobpcg_operators, "_HAS_NVMATH", False)

    monkeypatch.setattr(_lobpcg_preconditioners, "_HAS_CUPY", False)
    monkeypatch.setattr(_lobpcg_preconditioners, "_HAS_NVMATH", False)

    a_sdt = SparseDecoupledTensor.from_tensor(rand_sp_spd_6x6).to(device)

    eig_vals, eig_vecs = lobpcg_.lobpcg(
        a=a_sdt,
        m=None,
        k=2,
        lobpcg_config=lobpcg_.LOBPCGConfig(largest=True),
    )


@pytest.mark.gpu_only
def test_ilu_precond_import_error(rand_sp_spd_6x6, device, monkeypatch):
    monkeypatch.setattr(_lobpcg_preconditioners, "_HAS_CUPY", False)

    with pytest.raises(ImportError):
        a_sdt = SparseDecoupledTensor.from_tensor(rand_sp_spd_6x6).to(device)

        eig_vals_rev, eig_vecs_rev = lobpcg_.lobpcg(
            a=a_sdt,
            m=None,
            k=2,
            lobpcg_config=lobpcg_.LOBPCGConfig(largest=True),
            precond_config=lobpcg_.LOBPCGPrecondConfig(method="ilu"),
        )


@pytest.mark.gpu_only
def test_cholesky_precond_import_error(rand_sp_spd_6x6, device, monkeypatch):
    monkeypatch.setattr(_lobpcg_preconditioners, "_HAS_NVMATH", False)

    with pytest.raises(ImportError):
        a_sdt = SparseDecoupledTensor.from_tensor(rand_sp_spd_6x6).to(device)

        eig_vals_rev, eig_vecs_rev = lobpcg_.lobpcg(
            a=a_sdt,
            m=None,
            k=2,
            lobpcg_config=lobpcg_.LOBPCGConfig(largest=True),
            precond_config=lobpcg_.LOBPCGPrecondConfig(method="cholesky"),
        )


@pytest.mark.gpu_only
def test_shift_invert_import_error(rand_sp_spd_6x6, device, monkeypatch):
    monkeypatch.setattr(_lobpcg_operators, "_HAS_NVMATH", False)

    with pytest.raises(ImportError):
        a_sdt = SparseDecoupledTensor.from_tensor(rand_sp_spd_6x6).to(device)

        eig_val, eig_vec = lobpcg_.lobpcg(
            a=a_sdt,
            m=None,
            k=1,
            lobpcg_config=lobpcg_.LOBPCGConfig(sigma=18.5, largest=True, maxiter=10),
        )


@pytest.mark.gpu_only
def test_gep_shift_invert_import_error(rand_sp_gep_6x6, device, monkeypatch):
    monkeypatch.setattr(_lobpcg_operators, "_HAS_NVMATH", False)

    with pytest.raises(ImportError):
        a, m = rand_sp_gep_6x6

        a_sdt = SparseDecoupledTensor.from_tensor(a).to(device)
        m_sdt = SparseDecoupledTensor.from_tensor(m).to(device)

        eig_val, eig_vec = lobpcg_.lobpcg(
            a=a_sdt,
            m=m_sdt,
            k=1,
            lobpcg_config=lobpcg_.LOBPCGConfig(sigma=18.5, largest=True, maxiter=10),
        )

import pytest

from cochain.sparse.decoupled_tensor import SparseDecoupledTensor
from cochain.sparse.linalg.eigen.eigsh import cupy_eigsh_wrapper


@pytest.mark.gpu_only
def test_cupy_import_error(rand_sp_spd_6x6, device, monkeypatch):
    monkeypatch.setattr(cupy_eigsh_wrapper, "_HAS_CUPY", False)

    with pytest.raises(ImportError):
        a_sdt = SparseDecoupledTensor.from_tensor(rand_sp_spd_6x6).to(device)

        eig_vals, eig_vecs = cupy_eigsh_wrapper.cupy_eigsh(
            a_sdt,
            k=2,
            cp_config=cupy_eigsh_wrapper.CuPyEigshConfig(which="LM"),
        )


@pytest.mark.gpu_only
@pytest.mark.requires_cupy
def test_nvmath_independence(rand_sp_spd_6x6, device, monkeypatch):
    # cupy_eigsh() does not require nvmath-python outside of the SI mode.
    monkeypatch.setattr(cupy_eigsh_wrapper, "_HAS_NVMATH", False)

    a_sdt = SparseDecoupledTensor.from_tensor(rand_sp_spd_6x6).to(device)

    eig_vals, eig_vecs = cupy_eigsh_wrapper.cupy_eigsh(
        a_sdt,
        k=2,
        cp_config=cupy_eigsh_wrapper.CuPyEigshConfig(which="LM"),
    )


@pytest.mark.gpu_only
@pytest.mark.requires_cupy
def test_nvmath_import_error(rand_sp_spd_6x6, device, monkeypatch):
    monkeypatch.setattr(cupy_eigsh_wrapper, "_HAS_NVMATH", False)

    with pytest.raises(ImportError):
        a_sdt = SparseDecoupledTensor.from_tensor(rand_sp_spd_6x6).to(device)

        eig_val, eig_vec = cupy_eigsh_wrapper.cupy_eigsh(
            a_sdt,
            k=1,
            cp_config=cupy_eigsh_wrapper.CuPyEigshConfig(sigma=18.5, which="LM"),
        )

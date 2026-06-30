import pytest
import torch

from cochain.sparse.decoupled_tensor import SparseDecoupledTensor
from cochain.sparse.linalg.solvers import nvmath_wrapper


@pytest.mark.gpu_only
def test_nvmath_import_error(a, device, monkeypatch):
    monkeypatch.setattr(nvmath_wrapper, "_HAS_NVMATH", False)

    with pytest.raises(ImportError):
        a_sdt = SparseDecoupledTensor.from_tensor(a).to(device)
        a_dense = a_sdt.to_dense()

        n_dim = a_sdt.size(0)

        x_true = torch.randn(n_dim).to(device)
        b = a_dense @ x_true

        x = nvmath_wrapper.nvmath_direct_solver(a_sdt, b, sparse_system_type="general")

    with pytest.raises(ImportError):
        a_sdt = SparseDecoupledTensor.from_tensor(a).to(device)
        a_dense = a_sdt.to_dense()

        n_dim = a_sdt.size(0)
        x_true = torch.randn(n_dim).to(device)

        solver = nvmath_wrapper.NVMathDirectSolver(
            a_sdt, b, sparse_system_type="general"
        )
        x = solver(b, trans="N")

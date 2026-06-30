import pytest
import torch

from cochain.sparse.decoupled_tensor import SparseDecoupledTensor
from cochain.sparse.linalg.solvers import splu_wrapper


@pytest.mark.gpu_only
def test_cupy_import_error(a, device, monkeypatch):
    monkeypatch.setattr(splu_wrapper, "_HAS_CUPY", False)

    with pytest.raises(ImportError):
        a_sdt = SparseDecoupledTensor.from_tensor(a).to(device)
        a_dense = a_sdt.to_dense()

        n_dim = a_sdt.size(0)
        x_true = torch.randn(n_dim).to(device)
        b = a_dense @ x_true

        x = splu_wrapper.splu(a_sdt, b, backend="cupy")

    with pytest.raises(ImportError):
        a_sdt = SparseDecoupledTensor.from_tensor(a).to(device)
        a_dense = a_sdt.to_dense()

        n_dim = a_sdt.size(0)
        n_ch = 2

        x_true = torch.randn(n_dim, n_ch).to(device)

        solver = splu_wrapper.SuperLU(a_sdt, backend="cupy")

        x = solver(b, trans="N")

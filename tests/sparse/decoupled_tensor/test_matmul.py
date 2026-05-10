import pytest
import torch

from cochain.sparse.decoupled_tensor import SparseDecoupledTensor


@pytest.mark.parametrize("req_a", [True, False])
@pytest.mark.parametrize("req_b", [True, False])
def test_sp_dense_mm_backward(a, device, req_a, req_b):
    if not req_a and not req_b:
        return

    a_coo = a.to(device)

    a_dense = a_coo.to_dense()
    a_dense.requires_grad_(req_a)

    a_sdt = SparseDecoupledTensor.from_tensor(a_coo).detach().clone()
    a_sdt.requires_grad_(req_a)

    b_dense = torch.randn(a_coo.shape[::-1], dtype=a_coo.dtype, device=device)
    b_dense.requires_grad_(req_b)

    c_dense_true = a_dense @ b_dense

    loss_true = torch.sum(c_dense_true**2)
    loss_true.backward()

    if req_b:
        b_dense_grad_true = b_dense.grad.detach().clone()
        b_dense.grad = None

    if req_a:
        a_grad_true = a_dense.grad.detach().clone()[a_sdt.pattern.idx_coo.unbind(0)]

    c_dense = a_sdt @ b_dense
    loss = torch.sum(c_dense**2)
    loss.backward()

    if req_b:
        B_dense_grad = b_dense.grad.detach().clone()
        torch.testing.assert_close(B_dense_grad, b_dense_grad_true)
    else:
        assert b_dense.grad is None

    if req_a:
        a_grad = a_sdt.values.grad.detach().clone()
        torch.testing.assert_close(a_grad, a_grad_true)
    else:
        assert a_sdt.values.grad is None


@pytest.mark.parametrize("req_a", [True, False])
@pytest.mark.parametrize("req_b", [True, False])
def test_dense_sp_mm_backward(a, device, req_a, req_b):
    if not req_a and not req_b:
        return

    a_coo = a.to(device)

    a_dense = a_coo.to_dense()
    a_dense.requires_grad_(req_a)

    a_sdt = SparseDecoupledTensor.from_tensor(a_coo).detach().clone()
    a_sdt.requires_grad_(req_a)

    b_dense = torch.randn(a_coo.shape[::-1], dtype=a_coo.dtype, device=device)
    b_dense.requires_grad_(req_b)

    c_dense_true = b_dense @ a_dense
    loss_true = torch.sum(c_dense_true**2)
    loss_true.backward()

    if req_b:
        b_dense_grad_true = b_dense.grad.detach().clone()
        b_dense.grad = None

    if req_a:
        a_grad_true = a_dense.grad.detach().clone()[a_sdt.pattern.idx_coo.unbind(0)]

    c_dense = b_dense @ a_sdt
    loss = torch.sum(c_dense**2)
    loss.backward()

    if req_b:
        b_dense_grad = b_dense.grad.detach().clone()
        torch.testing.assert_close(b_dense_grad, b_dense_grad_true)
    else:
        assert b_dense.grad is None

    if req_a:
        a_grad = a_sdt.values.grad.detach().clone()
        torch.testing.assert_close(a_grad, a_grad_true)
    else:
        assert a_sdt.values.grad is None


@pytest.mark.parametrize("req_a", [True, False])
@pytest.mark.parametrize("req_b", [True, False])
def test_sp_sp_mm_backward(a, b, device, req_a, req_b):
    if not req_a and not req_b:
        return

    a_coo = a.to(device)
    a_dense = a_coo.to_dense()
    a_dense.requires_grad_(req_a)

    a_sdt = SparseDecoupledTensor.from_tensor(a_coo).detach().clone()
    a_sdt.requires_grad_(req_a)

    b_coo = b.to(device)
    b_dense = b_coo.to_dense()
    b_dense.requires_grad_(req_b)

    b_sdt = SparseDecoupledTensor.from_tensor(b_dense).detach().clone()
    b_sdt.requires_grad_(req_b)

    c_dense_true = a_dense @ b_dense
    loss_true = torch.sum(c_dense_true**2)
    loss_true.backward()

    if req_b:
        b_grad_true = b_dense.grad.detach().clone()[b_sdt.pattern.idx_coo.unbind(0)]

    if req_a:
        a_grad_true = a_dense.grad.detach().clone()[a_sdt.pattern.idx_coo.unbind(0)]

    c_sdt = a_sdt @ b_sdt
    loss = torch.sum(c_sdt.values**2)
    loss.backward()

    # Since the SpGEMM forward pass is custom, also test the forward pass correctness.
    torch.testing.assert_close(c_sdt.to_dense(), c_dense_true)

    if req_b:
        b_grad = b_sdt.values.grad.detach().clone()
        torch.testing.assert_close(b_grad, b_grad_true)
    else:
        assert b_sdt.values.grad is None

    if req_a:
        a_grad = a_sdt.values.grad.detach().clone()
        torch.testing.assert_close(a_grad, a_grad_true)
    else:
        assert a_sdt.values.grad is None


@pytest.mark.parametrize("req_a", [True, False])
@pytest.mark.parametrize("req_b", [True, False])
def test_sp_mv_backward(a, device, req_a, req_b):
    if not req_a and not req_b:
        return

    a_coo = a.to(device)
    a_dense = a_coo.to_dense()
    a_dense.requires_grad_(req_a)

    a_sdt = SparseDecoupledTensor.from_tensor(a_coo).detach().clone()
    a_sdt.requires_grad_(req_a)

    b_dense = torch.randn(a_coo.shape[-1], dtype=a_coo.dtype, device=device)
    b_dense.requires_grad_(req_b)

    C_dense_true = a_dense @ b_dense
    loss_true = torch.sum(C_dense_true**2)
    loss_true.backward()

    if req_b:
        b_dense_grad_true = b_dense.grad.detach().clone()
        b_dense.grad = None

    if req_a:
        a_grad_true = a_dense.grad.detach().clone()[a_sdt.pattern.idx_coo.unbind(0)]

    c_dense = a_sdt @ b_dense
    loss = torch.sum(c_dense**2)
    loss.backward()

    if req_b:
        b_dense_grad = b_dense.grad.detach().clone()
        torch.testing.assert_close(b_dense_grad, b_dense_grad_true)
    else:
        assert b_dense.grad is None

    if req_a:
        a_grad = a_sdt.values.grad.detach().clone()
        torch.testing.assert_close(a_grad, a_grad_true)
    else:
        assert a_sdt.values.grad is None


@pytest.mark.parametrize("req_a", [True, False])
@pytest.mark.parametrize("req_b", [True, False])
def test_sp_vm_backward(a, device, req_a, req_b):
    if not req_a and not req_b:
        return

    a_coo = a.to(device)
    a_dense = a_coo.to_dense()
    a_dense.requires_grad_(req_a)

    a_sdt = SparseDecoupledTensor.from_tensor(a_coo).detach().clone()
    a_sdt.requires_grad_(req_a)

    b_dense = torch.randn(a_coo.shape[0], dtype=a_coo.dtype, device=device)
    b_dense.requires_grad_(req_b)

    c_dense_true = b_dense @ a_dense
    loss_true = torch.sum(c_dense_true**2)
    loss_true.backward()

    if req_b:
        b_dense_grad_true = b_dense.grad.detach().clone()
        b_dense.grad = None

    if req_a:
        a_grad_true = a_dense.grad.detach().clone()[a_sdt.pattern.idx_coo.unbind(0)]

    c_dense = b_dense @ a_sdt
    loss = torch.sum(c_dense**2)
    loss.backward()

    if req_b:
        b_dense_grad = b_dense.grad.detach().clone()
        torch.testing.assert_close(b_dense_grad, b_dense_grad_true)
    else:
        assert b_dense.grad is None

    if req_a:
        a_grad = a_sdt.values.grad.detach().clone()
        torch.testing.assert_close(a_grad, a_grad_true)
    else:
        assert a_sdt.values.grad is None

import torch as t

from cochain.sparse.operator import SparseOperator


def test_sp_dense_mm_backward(A, device):
    A_tensor = A.to(device)
    A_dense = A_tensor.to_dense()
    A_dense.requires_grad_()

    A_operator = SparseOperator.from_tensor(A_tensor).detach().clone()
    A_operator.requires_grad_()

    B_dense = t.randn(A_tensor.shape[::-1], dtype=A_tensor.dtype, device=device)
    B_dense.requires_grad_()

    C_dense_true = A_dense @ B_dense
    loss_true = t.sum(C_dense_true**2)
    loss_true.backward()
    A_grad_true = A_dense.grad.detach().clone()[A_operator.sp_topo.idx_coo.unbind(0)]
    B_dense_grad_true = B_dense.grad.detach().clone()

    B_dense.grad = None
    C_dense = A_operator @ B_dense
    loss = t.sum(C_dense**2)
    loss.backward()
    B_dense_grad = B_dense.grad.detach().clone()
    A_grad = A_operator.val.grad.detach().clone()

    t.testing.assert_close(B_dense_grad, B_dense_grad_true)
    t.testing.assert_close(A_grad, A_grad_true)


def test_dense_sp_mm_backward(A, device):
    A_tensor = A.to(device)
    A_dense = A_tensor.to_dense()
    A_dense.requires_grad_()

    A_operator = SparseOperator.from_tensor(A_tensor).detach().clone()
    A_operator.requires_grad_()

    B_dense = t.randn(A_tensor.shape[::-1], dtype=A_tensor.dtype, device=device)
    B_dense.requires_grad_()

    C_dense_true = B_dense @ A_dense
    loss_true = t.sum(C_dense_true**2)
    loss_true.backward()
    A_grad_true = A_dense.grad.detach().clone()[A_operator.sp_topo.idx_coo.unbind(0)]
    B_dense_grad_true = B_dense.grad.detach().clone()

    B_dense.grad = None
    C_dense = B_dense @ A_operator
    loss = t.sum(C_dense**2)
    loss.backward()
    B_dense_grad = B_dense.grad.detach().clone()
    A_grad = A_operator.val.grad.detach().clone()

    t.testing.assert_close(B_dense_grad, B_dense_grad_true)
    t.testing.assert_close(A_grad, A_grad_true)


def test_sp_sp_mm_backward(A, device):
    A_tensor = A.to(device)
    A_dense = A_tensor.to_dense()
    A_dense.requires_grad_()

    A_operator = SparseOperator.from_tensor(A_tensor).detach().clone()
    A_operator.requires_grad_()

    B_dense = t.randn(A_tensor.shape[::-1], dtype=A_tensor.dtype, device=device)
    B_dense.requires_grad_()

    B_operator = SparseOperator.from_tensor(B_dense).detach().clone()
    B_operator.requires_grad_()

    C_dense_true = A_dense @ B_dense
    loss_true = t.sum(C_dense_true**2)
    loss_true.backward()
    A_grad_true = A_dense.grad.detach().clone()[A_operator.sp_topo.idx_coo.unbind(0)]
    B_grad_true = B_dense.grad.detach().clone()[B_operator.sp_topo.idx_coo.unbind(0)]

    C_operator = A_operator @ B_operator
    loss = t.sum(C_operator.val**2)
    loss.backward()
    B_grad = B_operator.val.grad.detach().clone()
    A_grad = A_operator.val.grad.detach().clone()

    t.testing.assert_close(B_grad, B_grad_true)
    t.testing.assert_close(A_grad, A_grad_true)


def test_sp_mv_backward(A, device):
    A_tensor = A.to(device)
    A_dense = A_tensor.to_dense()
    A_dense.requires_grad_()

    A_operator = SparseOperator.from_tensor(A_tensor).detach().clone()
    A_operator.requires_grad_()

    b_dense = t.randn(A_tensor.shape[-1], dtype=A_tensor.dtype, device=device)
    b_dense.requires_grad_()

    C_dense_true = A_dense @ b_dense
    loss_true = t.sum(C_dense_true**2)
    loss_true.backward()
    A_grad_true = A_dense.grad.detach().clone()[A_operator.sp_topo.idx_coo.unbind(0)]
    b_dense_grad_true = b_dense.grad.detach().clone()

    b_dense.grad = None
    C_dense = A_operator @ b_dense
    loss = t.sum(C_dense**2)
    loss.backward()
    b_dense_grad = b_dense.grad.detach().clone()
    A_grad = A_operator.val.grad.detach().clone()

    t.testing.assert_close(b_dense_grad, b_dense_grad_true)
    t.testing.assert_close(A_grad, A_grad_true)


def test_sp_vm_backward(A, device):
    A_tensor = A.to(device)
    A_dense = A_tensor.to_dense()
    A_dense.requires_grad_()

    A_operator = SparseOperator.from_tensor(A_tensor).detach().clone()
    A_operator.requires_grad_()

    b_dense = t.randn(A_tensor.shape[0], dtype=A_tensor.dtype, device=device)
    b_dense.requires_grad_()

    C_dense_true = b_dense @ A_dense
    loss_true = t.sum(C_dense_true**2)
    loss_true.backward()
    A_grad_true = A_dense.grad.detach().clone()[A_operator.sp_topo.idx_coo.unbind(0)]
    b_dense_grad_true = b_dense.grad.detach().clone()

    b_dense.grad = None
    C_dense = b_dense @ A_operator
    loss = t.sum(C_dense**2)
    loss.backward()
    b_dense_grad = b_dense.grad.detach().clone()
    A_grad = A_operator.val.grad.detach().clone()

    t.testing.assert_close(b_dense_grad, b_dense_grad_true)
    t.testing.assert_close(A_grad, A_grad_true)

import pytest
import torch as t
from jaxtyping import Float

from cochain.sparse.linalg.eigen import CuPyEigshConfig, cupy_eigsh
from cochain.sparse.linalg.eigen.utils import canonicalize_eig_vec_signs
from cochain.sparse.operators import SparseOperator

# TODO: test handling of degenerate eigenvalues


@pytest.mark.gpu_only
def test_standard_foward(rand_sp_spd_5x5: Float[t.Tensor, "5 5"], device):
    A_op = SparseOperator.from_tensor(rand_sp_spd_5x5).to(device)
    A_dense = rand_sp_spd_5x5.to_dense().to(device)

    eig_vals_true, eig_vecs_true = t.linalg.eigh(A_dense)

    k = 2

    # Test both the LM and SA modes
    eig_vals, eig_vecs = cupy_eigsh(A=A_op, k=k, cp_config=CuPyEigshConfig(which="LM"))

    # Both eigsolver returns eigenvalues in ascending orders
    t.testing.assert_close(eig_vals, eig_vals_true[-k:])
    t.testing.assert_close(
        canonicalize_eig_vec_signs(eig_vecs),
        canonicalize_eig_vec_signs(eig_vecs_true[:, -k:]),
    )

    eig_vals, eig_vecs = cupy_eigsh(A=A_op, k=k, cp_config=CuPyEigshConfig(which="SA"))

    t.testing.assert_close(eig_vals, eig_vals_true[:k])
    t.testing.assert_close(
        canonicalize_eig_vec_signs(eig_vecs),
        canonicalize_eig_vec_signs(eig_vecs_true[:, :k]),
    )


@pytest.mark.gpu_only
def test_standard_eig_vals_backward(rand_sp_spd_9x9: Float[t.Tensor, "9 9"], device):
    k = 3

    A_op = SparseOperator.from_tensor(rand_sp_spd_9x9).to(device)
    A_op.requires_grad_()

    A_dense = rand_sp_spd_9x9.to_dense().to(device)
    A_dense.requires_grad_()

    eig_vals_true_all, eig_vecs_true_all = t.linalg.eigh(A_dense)
    eig_vals_true = eig_vals_true_all[-k:]

    eig_vals, eig_vecs = cupy_eigsh(
        A=A_op, k=k, eps=0, cp_config=CuPyEigshConfig(which="LM")
    )

    # Compare eigenvalue gradient
    eig_vals_rand = t.randn_like(eig_vals_true)
    eig_vals_loss_true = t.sum(eig_vals_true * eig_vals_rand)
    eig_vals_loss = t.sum(eig_vals * eig_vals_rand)

    eig_vals_loss_true.backward()
    eig_vals_loss.backward()

    eig_vals_grad_true = A_dense.grad[t.unbind(A_op.sp_topo.idx_coo, dim=0)]
    eig_vals_grad = A_op.val.grad

    t.testing.assert_close(eig_vals_grad, eig_vals_grad_true)


@pytest.mark.gpu_only
def test_standard_eig_vecs_backward(rand_sp_spd_9x9: Float[t.Tensor, "9 9"], device):
    k = 3

    A_op = SparseOperator.from_tensor(rand_sp_spd_9x9).to(device)
    A_op.requires_grad_()

    A_dense = rand_sp_spd_9x9.to_dense().to(device)
    A_dense.requires_grad_()

    eig_vals_true_all, eig_vecs_true_all = t.linalg.eigh(A_dense)
    eig_vecs_true = eig_vecs_true_all[:, -k:]
    subspace_projector = eig_vecs_true @ eig_vecs_true.T

    # Turn off Lorentzian broadening for exact eigenvector gradient comparison.
    # SciPy eigsh returns the eigenvalues in ascending order.
    # By construction of the test fixture, all but its top three eigenvalues are
    # tiny, so the contribution from unresolved eigenvalues should be small, and
    # the custom backward (which ignores the unresolved eigenvectors) should agree
    # with the lobpcg backward (which accounts for the unresolved eigenvectors).
    eig_vals, eig_vecs = cupy_eigsh(
        A=A_op, k=k, eps=0, cp_config=CuPyEigshConfig(which="LM")
    )

    # Compare eigenvector gradient; here, we compute the Frobenius matrix inner
    # product between a random matrix and the eigenspace project matrix (V@V.T)
    # to make the loss invariant to eigenvector sign flips.
    eig_vecs_rand = t.randn_like(eig_vecs_true)
    eig_vecs_loss_true = t.sum(eig_vecs_rand * eig_vecs_true, dim=0).abs().sum()
    eig_vecs_loss = t.sum(eig_vecs_rand * eig_vecs, dim=0).abs().sum()

    eig_vecs_loss_true.backward()
    eig_vecs_loss.backward()

    eig_vecs_grad_true = (subspace_projector @ A_dense.grad @ subspace_projector)[
        t.unbind(A_op.sp_topo.idx_coo, dim=0)
    ]
    eig_vecs_grad = A_op.val.grad

    t.testing.assert_close(eig_vecs_grad, eig_vecs_grad_true)


@pytest.mark.gpu_only
def test_standard_combined_backward(rand_sp_spd_9x9: Float[t.Tensor, "9 9"], device):
    k = 3

    A_op = SparseOperator.from_tensor(rand_sp_spd_9x9).to(device)
    A_op.requires_grad_()

    A_dense = rand_sp_spd_9x9.to_dense().to(device)
    A_dense.requires_grad_()

    eig_vals_true_all, eig_vecs_true_all = t.linalg.eigh(A_dense)
    eig_vals_true = eig_vals_true_all[-k:]
    eig_vecs_true = eig_vecs_true_all[:, -k:]
    subspace_projector = eig_vecs_true @ eig_vecs_true.T

    eig_vals, eig_vecs = cupy_eigsh(
        A=A_op, k=k, eps=0, cp_config=CuPyEigshConfig(which="LM")
    )

    eig_vals_rand = t.randn_like(eig_vals_true)
    eig_vals_loss_true = t.sum(eig_vals_true * eig_vals_rand)
    eig_vals_loss = t.sum(eig_vals * eig_vals_rand)

    eig_vecs_rand = t.randn_like(eig_vecs_true)
    eig_vecs_loss_true = t.sum(eig_vecs_rand * eig_vecs_true, dim=0).abs().sum()
    eig_vecs_loss = t.sum(eig_vecs_rand * eig_vecs, dim=0).abs().sum()

    combined_loss_true = eig_vals_loss_true + eig_vecs_loss_true
    combined_loss = eig_vals_loss + eig_vecs_loss

    combined_loss_true.backward()
    combined_loss.backward()

    combined_grad_true = (subspace_projector @ A_dense.grad @ subspace_projector)[
        t.unbind(A_op.sp_topo.idx_coo, dim=0)
    ]
    combined_grad = A_op.val.grad

    t.testing.assert_close(combined_grad, combined_grad_true)


@pytest.mark.gpu_only
def test_shift_invert_foward(rand_sp_spd_5x5: Float[t.Tensor, "5 5"], device):
    A_op = SparseOperator.from_tensor(rand_sp_spd_5x5).to(device)
    A_dense = rand_sp_spd_5x5.to_dense().to(device)

    # Since t.linalg.eigh does not support shift-invert mode, need to manually
    # extract the target eigenvalue/eigenvector
    eig_vals_true, eig_vecs_true = t.linalg.eigh(A_dense)

    k = 1
    target_eig_val = 18.5

    target_idx = t.argmin(t.abs(eig_vals_true - target_eig_val), keepdim=True)
    eig_val_true = eig_vals_true[target_idx]
    eig_vec_true = eig_vecs_true[:, target_idx]

    eig_val, eig_vec = cupy_eigsh(
        A=A_op, k=k, cp_config=CuPyEigshConfig(sigma=target_eig_val, which="LM")
    )

    t.testing.assert_close(eig_val, eig_val_true)
    t.testing.assert_close(
        canonicalize_eig_vec_signs(eig_vec),
        canonicalize_eig_vec_signs(eig_vec_true),
    )

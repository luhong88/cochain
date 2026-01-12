import pytest
import torch as t
from jaxtyping import Float

from cochain.sparse.linalg.eigen import LOBPCGConfig, LOBPCGPrecondConfig, lobpcg
from cochain.sparse.linalg.eigen.utils import canonicalize_eig_vec_signs
from cochain.sparse.operators import SparseOperator

# TODO: test handling of degenerate eigenvalues
# TODO: test handling of batching


def dense_gep(
    A: Float[t.Tensor, "m m"], M: Float[t.Tensor, "m m"]
) -> tuple[Float[t.Tensor, " k"], Float[t.Tensor, "m k"]]:
    # Since t.linalg.eigh() does not support GEP, need to perform Cholesky
    # whitening on A.
    M_cho_inv = t.linalg.inv(t.linalg.cholesky(M))
    A_whitened = M_cho_inv @ A @ M_cho_inv.T

    eig_vals_true, eig_vecs_whitened = t.linalg.eigh(A_whitened)
    eig_vecs_true = M_cho_inv.T @ eig_vecs_whitened

    return eig_vals_true, eig_vecs_true


@pytest.mark.gpu_only
def test_standard_forward(rand_sp_spd_6x6: Float[t.Tensor, "6 6"], device):
    A_op = SparseOperator.from_tensor(rand_sp_spd_6x6).to(device)
    A_dense = rand_sp_spd_6x6.to_dense().to(device)

    eig_vals_true, eig_vecs_true = t.linalg.eigh(A_dense)

    k = 2

    # Test both largest=True and largest=False
    eig_vals_rev, eig_vecs_rev = lobpcg(
        A=A_op, M=None, k=k, lobpcg_config=LOBPCGConfig(largest=True)
    )
    eig_vals = t.flip(eig_vals_rev, dims=(0,))
    eig_vecs = t.flip(eig_vecs_rev, dims=(-1,))

    # If largest=True, lobpcg returns eigenvalues in descending order.
    t.testing.assert_close(eig_vals, eig_vals_true[-k:])
    t.testing.assert_close(
        canonicalize_eig_vec_signs(eig_vecs),
        canonicalize_eig_vec_signs(eig_vecs_true[:, -k:]),
    )

    eig_vals, eig_vecs = lobpcg(
        A=A_op, M=None, k=k, lobpcg_config=LOBPCGConfig(largest=False)
    )

    t.testing.assert_close(eig_vals, eig_vals_true[:k])
    t.testing.assert_close(
        canonicalize_eig_vec_signs(eig_vecs),
        canonicalize_eig_vec_signs(eig_vecs_true[:, :k]),
    )


@pytest.mark.gpu_only
@pytest.mark.parametrize(
    "preconditioner",
    ["identity", "jacobi", "ilu", "cholesky"],
)
def test_standard_forward_preconditioners(
    rand_sp_spd_6x6: Float[t.Tensor, "6 6"], preconditioner, device
):
    A_op = SparseOperator.from_tensor(rand_sp_spd_6x6).to(device)
    A_dense = rand_sp_spd_6x6.to_dense().to(device)

    eig_vals_true, eig_vecs_true = t.linalg.eigh(A_dense)

    k = 2

    eig_vals_rev, eig_vecs_rev = lobpcg(
        A=A_op,
        M=None,
        k=k,
        lobpcg_config=LOBPCGConfig(largest=True),
        precond_config=LOBPCGPrecondConfig(method=preconditioner),
    )
    eig_vals = t.flip(eig_vals_rev, dims=(0,))
    eig_vecs = t.flip(eig_vecs_rev, dims=(-1,))

    # If largest=True, lobpcg returns eigenvalues in descending order.
    t.testing.assert_close(eig_vals, eig_vals_true[-k:])
    t.testing.assert_close(
        canonicalize_eig_vec_signs(eig_vecs),
        canonicalize_eig_vec_signs(eig_vecs_true[:, -k:]),
    )


@pytest.mark.gpu_only
def test_batched_standard_forward(
    rand_sp_spd_6x6: Float[t.Tensor, "6 6"],
    rand_sp_spd_9x9: Float[t.Tensor, "9 9"],
    device,
):
    A1_dense = rand_sp_spd_6x6.to_dense().to(device)
    eig_vals_1_true, eig_vecs_1_true = t.linalg.eigh(A1_dense)

    A2_dense = rand_sp_spd_9x9.to_dense().to(device)
    eig_vals_2_true, eig_vecs_2_true = t.linalg.eigh(A2_dense)

    k = 2

    A1_op = SparseOperator.from_tensor(rand_sp_spd_6x6).to(device)
    A2_op = SparseOperator.from_tensor(rand_sp_spd_9x9).to(device)
    A_op = SparseOperator.pack_block_diag((A1_op, A2_op))

    eig_vals_rev, eig_vecs_rev = lobpcg(
        A=A_op,
        M=None,
        block_diag_batch=True,
        k=k,
        lobpcg_config=LOBPCGConfig(largest=True),
    )
    eig_vals = t.flip(eig_vals_rev, dims=(-1,))
    eig_vecs = t.flip(eig_vecs_rev, dims=(-1,))

    eig_vals_1, eig_vals_2 = eig_vals.unbind(0)
    eig_vecs_1 = eig_vecs[:6]
    eig_vecs_2 = eig_vecs[6:]

    t.testing.assert_close(eig_vals_1, eig_vals_1_true[-k:])
    t.testing.assert_close(eig_vals_2, eig_vals_2_true[-k:])

    t.testing.assert_close(
        canonicalize_eig_vec_signs(eig_vecs_1),
        canonicalize_eig_vec_signs(eig_vecs_1_true[:, -k:]),
    )
    t.testing.assert_close(
        canonicalize_eig_vec_signs(eig_vecs_2),
        canonicalize_eig_vec_signs(eig_vecs_2_true[:, -k:]),
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

    eig_vals_rev, eig_vecs_rev = lobpcg(
        A=A_op, M=None, k=k, eps=0, lobpcg_config=LOBPCGConfig(largest=True)
    )
    eig_vals = t.flip(eig_vals_rev, dims=(0,))

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
    eig_vals_rev, eig_vecs_rev = lobpcg(
        A=A_op, M=None, k=k, eps=0, lobpcg_config=LOBPCGConfig(largest=True)
    )
    eig_vecs = t.flip(eig_vecs_rev, dims=(-1,))

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

    eig_vals_rev, eig_vecs_rev = lobpcg(
        A=A_op, M=None, k=k, eps=0, lobpcg_config=LOBPCGConfig(largest=True)
    )
    eig_vals = t.flip(eig_vals_rev, dims=(0,))
    eig_vecs = t.flip(eig_vecs_rev, dims=(-1,))

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
def test_gep_forward(rand_sp_gep_6x6: Float[t.Tensor, "6 6"], device):
    A, M = rand_sp_gep_6x6

    A_dense = A.to_dense().to(device)
    M_dense = M.to_dense().to(device)

    eig_vals_true, eig_vecs_true = dense_gep(A_dense, M_dense)

    A_op = SparseOperator.from_tensor(A).to(device)
    M_op = SparseOperator.from_tensor(M).to(device)

    k = 2

    # Test both largest=True and largest=False
    eig_vals_rev, eig_vecs_rev = lobpcg(
        A=A_op, M=M_op, k=k, lobpcg_config=LOBPCGConfig(largest=True)
    )
    eig_vals = t.flip(eig_vals_rev, dims=(0,))
    eig_vecs = t.flip(eig_vecs_rev, dims=(-1,))

    # Both eigsolver returns eigenvalues in ascending orders
    t.testing.assert_close(eig_vals, eig_vals_true[-k:])
    t.testing.assert_close(
        canonicalize_eig_vec_signs(eig_vecs),
        canonicalize_eig_vec_signs(eig_vecs_true[:, -k:]),
    )

    eig_vals, eig_vecs = lobpcg(
        A=A_op, M=M_op, k=k, lobpcg_config=LOBPCGConfig(largest=False)
    )

    t.testing.assert_close(eig_vals, eig_vals_true[:k])
    t.testing.assert_close(
        canonicalize_eig_vec_signs(eig_vecs),
        canonicalize_eig_vec_signs(eig_vecs_true[:, :k]),
    )


@pytest.mark.gpu_only
def test_gep_eig_vals_backward(rand_sp_gep_9x9: Float[t.Tensor, "9 9"], device):
    k = 3

    A, M = rand_sp_gep_9x9

    A_dense = A.to_dense().to(device)
    A_dense.requires_grad_()

    M_dense = M.to_dense().to(device)
    M_dense.requires_grad_()

    eig_vals_true_all, eig_vecs_true_all = dense_gep(A_dense, M_dense)
    eig_vals_true = eig_vals_true_all[-k:]

    A_op = SparseOperator.from_tensor(A).to(device)
    A_op.requires_grad_()

    M_op = SparseOperator.from_tensor(M).to(device)
    M_op.requires_grad_()

    eig_vals_rev, eig_vecs_rev = lobpcg(
        A=A_op, M=M_op, k=k, eps=0, lobpcg_config=LOBPCGConfig(largest=True)
    )
    eig_vals = t.flip(eig_vals_rev, dims=(0,))

    eig_vals_rand = t.randn_like(eig_vals_true)
    eig_vals_loss_true = t.sum(eig_vals_true * eig_vals_rand)
    eig_vals_loss = t.sum(eig_vals * eig_vals_rand)

    eig_vals_loss_true.backward()
    eig_vals_loss.backward()

    A_grad_true = A_dense.grad[t.unbind(A_op.sp_topo.idx_coo, dim=0)]
    A_grad = A_op.val.grad

    t.testing.assert_close(A_grad, A_grad_true)

    M_grad_true = M_dense.grad[t.unbind(M_op.sp_topo.idx_coo, dim=0)]
    M_grad = M_op.val.grad

    t.testing.assert_close(M_grad, M_grad_true)


@pytest.mark.gpu_only
def test_gep_eig_vecs_backward(rand_sp_gep_9x9: Float[t.Tensor, "9 9"], device):
    k = 3

    A, M = rand_sp_gep_9x9

    A_dense = A.to_dense().to(device)
    A_dense.requires_grad_()

    M_dense = M.to_dense().to(device)
    M_dense.requires_grad_()

    eig_vals_true_all, eig_vecs_true_all = dense_gep(A_dense, M_dense)
    eig_vecs_true = eig_vecs_true_all[:, -k:]
    subspace_projector = eig_vecs_true @ eig_vecs_true.T @ M_dense

    A_op = SparseOperator.from_tensor(A).to(device)
    A_op.requires_grad_()

    M_op = SparseOperator.from_tensor(M).to(device)
    M_op.requires_grad_()

    eig_vals_rev, eig_vecs_rev = lobpcg(
        A=A_op, M=M_op, k=k, eps=0, lobpcg_config=LOBPCGConfig(largest=True)
    )
    eig_vecs = t.flip(eig_vecs_rev, dims=(-1,))

    eig_vecs_rand = t.randn_like(eig_vecs_true)
    eig_vecs_loss_true = t.sum(eig_vecs_rand * eig_vecs_true, dim=0).abs().sum()
    eig_vecs_loss = t.sum(eig_vecs_rand * eig_vecs, dim=0).abs().sum()

    eig_vecs_loss_true.backward()
    eig_vecs_loss.backward()

    A_grad_true = (subspace_projector @ A_dense.grad @ subspace_projector.T)[
        t.unbind(A_op.sp_topo.idx_coo, dim=0)
    ]
    A_grad = A_op.val.grad

    t.testing.assert_close(A_grad, A_grad_true)

    M_grad_true = (subspace_projector @ M_dense.grad @ subspace_projector.T)[
        t.unbind(M_op.sp_topo.idx_coo, dim=0)
    ]
    M_grad = M_op.val.grad

    t.testing.assert_close(M_grad, M_grad_true)


@pytest.mark.gpu_only
def test_shift_invert_forward(rand_sp_spd_6x6: Float[t.Tensor, "6 6"], device):
    A_op = SparseOperator.from_tensor(rand_sp_spd_6x6).to(device)
    A_dense = rand_sp_spd_6x6.to_dense().to(device)

    # Since t.linalg.eigh does not support shift-invert mode, need to manually
    # extract the target eigenvalue/eigenvector
    eig_vals_true, eig_vecs_true = t.linalg.eigh(A_dense)

    k = 1
    target_eig_val = 18.5

    target_idx = t.argmin(t.abs(eig_vals_true - target_eig_val), keepdim=True)
    eig_val_true = eig_vals_true[target_idx]
    eig_vec_true = eig_vecs_true[:, target_idx]

    # The algorithm will likely not be able to converge using the default atol/rtol
    # because the large condition number of the shift-inverted matrix causes the
    # sparse solver to lose precision.
    eig_val, eig_vec = lobpcg(
        A=A_op,
        M=None,
        k=k,
        lobpcg_config=LOBPCGConfig(sigma=target_eig_val, largest=True, maxiter=10),
    )

    t.testing.assert_close(eig_val, eig_val_true)
    t.testing.assert_close(
        canonicalize_eig_vec_signs(eig_vec),
        canonicalize_eig_vec_signs(eig_vec_true),
    )


@pytest.mark.gpu_only
def test_gep_shift_invert_forward(rand_sp_gep_6x6, device):
    A, M = rand_sp_gep_6x6

    A_dense = A.to_dense().to(device)
    M_dense = M.to_dense().to(device)

    eig_vals_true, eig_vecs_true = dense_gep(A_dense, M_dense)

    k = 1
    target_eig_val = 18.5

    target_idx = t.argmin(t.abs(eig_vals_true - target_eig_val), keepdim=True)
    eig_val_true = eig_vals_true[target_idx]
    eig_vec_true = eig_vecs_true[:, target_idx]

    A_op = SparseOperator.from_tensor(A).to(device)
    M_op = SparseOperator.from_tensor(M).to(device)

    # The algorithm will likely not be able to converge using the default atol/rtol
    # because the large condition number of the shift-inverted matrix causes the
    # sparse solver to lose precision.
    eig_val, eig_vec = lobpcg(
        A=A_op,
        M=M_op,
        k=k,
        lobpcg_config=LOBPCGConfig(sigma=target_eig_val, largest=True, maxiter=10),
    )

    t.testing.assert_close(eig_val, eig_val_true)
    t.testing.assert_close(
        canonicalize_eig_vec_signs(eig_vec),
        canonicalize_eig_vec_signs(eig_vec_true),
    )

import pytest
import torch
from jaxtyping import Float
from torch import Tensor

from cochain.sparse.decoupled_tensor import SparseDecoupledTensor
from cochain.sparse.linalg.eigen import (
    LOBPCGConfig,
    LOBPCGPrecondConfig,
    canonicalize_eig_vec_signs,
    lobpcg,
)

# TODO: test handling of degenerate eigenvalues
# TODO: test handling of batching

# TODO: clear up cupy/nvmath dependencies
pytest.importorskip("nvmath")


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


@pytest.mark.gpu_only
def test_standard_forward(rand_sp_spd_6x6: Float[Tensor, "6 6"], device):
    a_sdt = SparseDecoupledTensor.from_tensor(rand_sp_spd_6x6).to(device)
    a_dense = rand_sp_spd_6x6.to_dense().to(device)

    eig_vals_true, eig_vecs_true = torch.linalg.eigh(a_dense)

    k = 2

    # Test both largest=True and largest=False
    eig_vals_rev, eig_vecs_rev = lobpcg(
        a=a_sdt, m=None, k=k, lobpcg_config=LOBPCGConfig(largest=True)
    )
    eig_vals = torch.flip(eig_vals_rev, dims=(0,))
    eig_vecs = torch.flip(eig_vecs_rev, dims=(-1,))

    # If largest=True, lobpcg returns eigenvalues in descending order.
    torch.testing.assert_close(eig_vals, eig_vals_true[-k:])
    torch.testing.assert_close(
        canonicalize_eig_vec_signs(eig_vecs),
        canonicalize_eig_vec_signs(eig_vecs_true[:, -k:]),
    )

    eig_vals, eig_vecs = lobpcg(
        a=a_sdt, m=None, k=k, lobpcg_config=LOBPCGConfig(largest=False)
    )

    torch.testing.assert_close(eig_vals, eig_vals_true[:k])
    torch.testing.assert_close(
        canonicalize_eig_vec_signs(eig_vecs),
        canonicalize_eig_vec_signs(eig_vecs_true[:, :k]),
    )


@pytest.mark.gpu_only
@pytest.mark.parametrize(
    "preconditioner",
    ["identity", "jacobi", "ilu", "cholesky"],
)
def test_standard_forward_preconditioners(
    rand_sp_spd_6x6: Float[Tensor, "6 6"], preconditioner, device
):
    a_sdt = SparseDecoupledTensor.from_tensor(rand_sp_spd_6x6).to(device)
    a_dense = rand_sp_spd_6x6.to_dense().to(device)

    eig_vals_true, eig_vecs_true = torch.linalg.eigh(a_dense)

    k = 2

    eig_vals_rev, eig_vecs_rev = lobpcg(
        a=a_sdt,
        m=None,
        k=k,
        lobpcg_config=LOBPCGConfig(largest=True),
        precond_config=LOBPCGPrecondConfig(method=preconditioner),
    )
    eig_vals = torch.flip(eig_vals_rev, dims=(0,))
    eig_vecs = torch.flip(eig_vecs_rev, dims=(-1,))

    # If largest=True, lobpcg returns eigenvalues in descending order.
    torch.testing.assert_close(eig_vals, eig_vals_true[-k:])
    torch.testing.assert_close(
        canonicalize_eig_vec_signs(eig_vecs),
        canonicalize_eig_vec_signs(eig_vecs_true[:, -k:]),
    )


@pytest.mark.gpu_only
def test_standard_eig_vals_backward(rand_sp_spd_9x9: Float[Tensor, "9 9"], device):
    k = 3

    a_sdt = SparseDecoupledTensor.from_tensor(rand_sp_spd_9x9).to(device)
    a_sdt.requires_grad_()

    a_dense = rand_sp_spd_9x9.to_dense().to(device)
    a_dense.requires_grad_()

    eig_vals_true_all, eig_vecs_true_all = torch.linalg.eigh(a_dense)
    eig_vals_true = eig_vals_true_all[-k:]

    eig_vals_rev, eig_vecs_rev = lobpcg(
        a=a_sdt, m=None, k=k, eps=0, lobpcg_config=LOBPCGConfig(largest=True)
    )
    eig_vals = torch.flip(eig_vals_rev, dims=(0,))

    # Compare eigenvalue gradient
    eig_vals_rand = torch.randn_like(eig_vals_true)
    eig_vals_loss_true = torch.sum(eig_vals_true * eig_vals_rand)
    eig_vals_loss = torch.sum(eig_vals * eig_vals_rand)

    eig_vals_loss_true.backward()
    eig_vals_loss.backward()

    eig_vals_grad_true = a_dense.grad[torch.unbind(a_sdt.pattern.idx_coo, dim=0)]
    eig_vals_grad = a_sdt.grad

    torch.testing.assert_close(eig_vals_grad, eig_vals_grad_true)


@pytest.mark.gpu_only
def test_standard_eig_vecs_backward(rand_sp_spd_9x9: Float[Tensor, "9 9"], device):
    k = 3

    a_sdt = SparseDecoupledTensor.from_tensor(rand_sp_spd_9x9).to(device)
    a_sdt.requires_grad_()

    a_dense = rand_sp_spd_9x9.to_dense().to(device)
    a_dense.requires_grad_()

    eig_vals_true_all, eig_vecs_true_all = torch.linalg.eigh(a_dense)
    eig_vecs_true = eig_vecs_true_all[:, -k:]
    subspace_projector = eig_vecs_true @ eig_vecs_true.T

    # Turn off Lorentzian broadening for exact eigenvector gradient comparison.
    # SciPy eigsh returns the eigenvalues in ascending order.
    # By construction of the test fixture, all but its top three eigenvalues are
    # tiny, so the contribution from unresolved eigenvalues should be small, and
    # the custom backward (which ignores the unresolved eigenvectors) should agree
    # with the lobpcg backward (which accounts for the unresolved eigenvectors).
    eig_vals_rev, eig_vecs_rev = lobpcg(
        a=a_sdt, m=None, k=k, eps=0, lobpcg_config=LOBPCGConfig(largest=True)
    )
    eig_vecs = torch.flip(eig_vecs_rev, dims=(-1,))

    # Compare eigenvector gradient; here, we compute the Frobenius matrix inner
    # product between a random matrix and the eigenspace project matrix (V@V.T)
    # to make the loss invariant to eigenvector sign flips.
    eig_vecs_rand = torch.randn_like(eig_vecs_true)
    eig_vecs_loss_true = torch.sum(eig_vecs_rand * eig_vecs_true, dim=0).abs().sum()
    eig_vecs_loss = torch.sum(eig_vecs_rand * eig_vecs, dim=0).abs().sum()

    eig_vecs_loss_true.backward()
    eig_vecs_loss.backward()

    eig_vecs_grad_true = (subspace_projector @ a_dense.grad @ subspace_projector)[
        torch.unbind(a_sdt.pattern.idx_coo, dim=0)
    ]
    eig_vecs_grad = a_sdt.grad

    torch.testing.assert_close(eig_vecs_grad, eig_vecs_grad_true)


@pytest.mark.gpu_only
def test_standard_combined_backward(rand_sp_spd_9x9: Float[Tensor, "9 9"], device):
    k = 3

    a_sdt = SparseDecoupledTensor.from_tensor(rand_sp_spd_9x9).to(device)
    a_sdt.requires_grad_()

    a_dense = rand_sp_spd_9x9.to_dense().to(device)
    a_dense.requires_grad_()

    eig_vals_true_all, eig_vecs_true_all = torch.linalg.eigh(a_dense)
    eig_vals_true = eig_vals_true_all[-k:]
    eig_vecs_true = eig_vecs_true_all[:, -k:]
    subspace_projector = eig_vecs_true @ eig_vecs_true.T

    eig_vals_rev, eig_vecs_rev = lobpcg(
        a=a_sdt, m=None, k=k, eps=0, lobpcg_config=LOBPCGConfig(largest=True)
    )
    eig_vals = torch.flip(eig_vals_rev, dims=(0,))
    eig_vecs = torch.flip(eig_vecs_rev, dims=(-1,))

    eig_vals_rand = torch.randn_like(eig_vals_true)
    eig_vals_loss_true = torch.sum(eig_vals_true * eig_vals_rand)
    eig_vals_loss = torch.sum(eig_vals * eig_vals_rand)

    eig_vecs_rand = torch.randn_like(eig_vecs_true)
    eig_vecs_loss_true = torch.sum(eig_vecs_rand * eig_vecs_true, dim=0).abs().sum()
    eig_vecs_loss = torch.sum(eig_vecs_rand * eig_vecs, dim=0).abs().sum()

    combined_loss_true = eig_vals_loss_true + eig_vecs_loss_true
    combined_loss = eig_vals_loss + eig_vecs_loss

    combined_loss_true.backward()
    combined_loss.backward()

    combined_grad_true = (subspace_projector @ a_dense.grad @ subspace_projector)[
        torch.unbind(a_sdt.pattern.idx_coo, dim=0)
    ]
    combined_grad = a_sdt.grad

    torch.testing.assert_close(combined_grad, combined_grad_true)


@pytest.mark.gpu_only
def test_batched_standard_forward(
    rand_sp_spd_6x6: Float[Tensor, "6 6"],
    rand_sp_spd_9x9: Float[Tensor, "9 9"],
    device,
):
    a1_dense = rand_sp_spd_6x6.to_dense().to(device)
    eig_vals_1_true, eig_vecs_1_true = torch.linalg.eigh(a1_dense)

    a2_dense = rand_sp_spd_9x9.to_dense().to(device)
    eig_vals_2_true, eig_vecs_2_true = torch.linalg.eigh(a2_dense)

    k = 2

    a1_sdt = SparseDecoupledTensor.from_tensor(rand_sp_spd_6x6).to(device)
    a2_sdt = SparseDecoupledTensor.from_tensor(rand_sp_spd_9x9).to(device)
    a_sdt = SparseDecoupledTensor.pack_block_diag((a1_sdt, a2_sdt))

    eig_vals_rev, eig_vecs_rev = lobpcg(
        a=a_sdt,
        m=None,
        block_diag_batch=True,
        k=k,
        lobpcg_config=LOBPCGConfig(largest=True),
    )
    eig_vals = torch.flip(eig_vals_rev, dims=(-1,))
    eig_vecs = torch.flip(eig_vecs_rev, dims=(-1,))

    eig_vals_1, eig_vals_2 = eig_vals.unbind(0)
    eig_vecs_1 = eig_vecs[:6]
    eig_vecs_2 = eig_vecs[6:]

    torch.testing.assert_close(eig_vals_1, eig_vals_1_true[-k:])
    torch.testing.assert_close(eig_vals_2, eig_vals_2_true[-k:])

    torch.testing.assert_close(
        canonicalize_eig_vec_signs(eig_vecs_1),
        canonicalize_eig_vec_signs(eig_vecs_1_true[:, -k:]),
    )
    torch.testing.assert_close(
        canonicalize_eig_vec_signs(eig_vecs_2),
        canonicalize_eig_vec_signs(eig_vecs_2_true[:, -k:]),
    )


@pytest.mark.gpu_only
def test_gep_forward(
    rand_sp_gep_6x6: tuple[Float[Tensor, "6 6"], Float[Tensor, "6 6"]], device
):
    a, m = rand_sp_gep_6x6

    a_dense = a.to_dense().to(device)
    m_dense = m.to_dense().to(device)

    eig_vals_true, eig_vecs_true = dense_gep(a_dense, m_dense)

    a_sdt = SparseDecoupledTensor.from_tensor(a).to(device)
    m_sdt = SparseDecoupledTensor.from_tensor(m).to(device)

    k = 2

    # Test both largest=True and largest=False
    eig_vals_rev, eig_vecs_rev = lobpcg(
        a=a_sdt, m=m_sdt, k=k, lobpcg_config=LOBPCGConfig(largest=True)
    )
    eig_vals = torch.flip(eig_vals_rev, dims=(0,))
    eig_vecs = torch.flip(eig_vecs_rev, dims=(-1,))

    # Both eigsolver returns eigenvalues in ascending orders
    torch.testing.assert_close(eig_vals, eig_vals_true[-k:])
    torch.testing.assert_close(
        canonicalize_eig_vec_signs(eig_vecs),
        canonicalize_eig_vec_signs(eig_vecs_true[:, -k:]),
    )

    eig_vals, eig_vecs = lobpcg(
        a=a_sdt, m=m_sdt, k=k, lobpcg_config=LOBPCGConfig(largest=False)
    )

    torch.testing.assert_close(eig_vals, eig_vals_true[:k])
    torch.testing.assert_close(
        canonicalize_eig_vec_signs(eig_vecs),
        canonicalize_eig_vec_signs(eig_vecs_true[:, :k]),
    )


@pytest.mark.gpu_only
def test_gep_eig_vals_backward(
    rand_sp_gep_9x9: tuple[Float[Tensor, "9 9"], Float[Tensor, "9 9"]], device
):
    k = 3

    a, m = rand_sp_gep_9x9

    a_dense = a.to_dense().to(device)
    a_dense.requires_grad_()

    m_dense = m.to_dense().to(device)
    m_dense.requires_grad_()

    eig_vals_true_all, eig_vecs_true_all = dense_gep(a_dense, m_dense)
    eig_vals_true = eig_vals_true_all[-k:]

    a_sdt = SparseDecoupledTensor.from_tensor(a).to(device)
    a_sdt.requires_grad_()

    m_sdt = SparseDecoupledTensor.from_tensor(m).to(device)
    m_sdt.requires_grad_()

    eig_vals_rev, eig_vecs_rev = lobpcg(
        a=a_sdt, m=m_sdt, k=k, eps=0, lobpcg_config=LOBPCGConfig(largest=True)
    )
    eig_vals = torch.flip(eig_vals_rev, dims=(0,))

    eig_vals_rand = torch.randn_like(eig_vals_true)
    eig_vals_loss_true = torch.sum(eig_vals_true * eig_vals_rand)
    eig_vals_loss = torch.sum(eig_vals * eig_vals_rand)

    eig_vals_loss_true.backward()
    eig_vals_loss.backward()

    a_grad_true = a_dense.grad[torch.unbind(a_sdt.pattern.idx_coo, dim=0)]
    a_grad = a_sdt.grad

    torch.testing.assert_close(a_grad, a_grad_true)

    m_grad_true = m_dense.grad[torch.unbind(m_sdt.pattern.idx_coo, dim=0)]
    m_grad = m_sdt.grad

    torch.testing.assert_close(m_grad, m_grad_true)


@pytest.mark.gpu_only
def test_gep_eig_vecs_backward(
    rand_sp_gep_9x9: tuple[Float[Tensor, "9 9"], Float[Tensor, "9 9"]], device
):
    k = 3

    a, m = rand_sp_gep_9x9

    a_dense = a.to_dense().to(device)
    a_dense.requires_grad_()

    m_dense = m.to_dense().to(device)
    m_dense.requires_grad_()

    eig_vals_true_all, eig_vecs_true_all = dense_gep(a_dense, m_dense)
    eig_vecs_true = eig_vecs_true_all[:, -k:]
    subspace_projector = eig_vecs_true @ eig_vecs_true.T @ m_dense

    a_sdt = SparseDecoupledTensor.from_tensor(a).to(device)
    a_sdt.requires_grad_()

    m_sdt = SparseDecoupledTensor.from_tensor(m).to(device)
    m_sdt.requires_grad_()

    eig_vals_rev, eig_vecs_rev = lobpcg(
        a=a_sdt, m=m_sdt, k=k, eps=0, lobpcg_config=LOBPCGConfig(largest=True)
    )
    eig_vecs = torch.flip(eig_vecs_rev, dims=(-1,))

    eig_vecs_rand = torch.randn_like(eig_vecs_true)
    eig_vecs_loss_true = torch.sum(eig_vecs_rand * eig_vecs_true, dim=0).abs().sum()
    eig_vecs_loss = torch.sum(eig_vecs_rand * eig_vecs, dim=0).abs().sum()

    eig_vecs_loss_true.backward()
    eig_vecs_loss.backward()

    a_grad_true = (subspace_projector @ a_dense.grad @ subspace_projector.T)[
        torch.unbind(a_sdt.pattern.idx_coo, dim=0)
    ]
    a_grad = a_sdt.grad

    torch.testing.assert_close(a_grad, a_grad_true)

    m_grad_true = (subspace_projector @ m_dense.grad @ subspace_projector.T)[
        torch.unbind(m_sdt.pattern.idx_coo, dim=0)
    ]
    m_grad = m_sdt.grad

    torch.testing.assert_close(m_grad, m_grad_true)


@pytest.mark.gpu_only
def test_shift_invert_forward(rand_sp_spd_6x6: Float[Tensor, "6 6"], device):
    a_sdt = SparseDecoupledTensor.from_tensor(rand_sp_spd_6x6).to(device)
    a_dense = rand_sp_spd_6x6.to_dense().to(device)

    # Since torch.linalg.eigh does not support shift-invert mode, need to manually
    # extract the target eigenvalue/eigenvector
    eig_vals_true, eig_vecs_true = torch.linalg.eigh(a_dense)

    k = 1
    target_eig_val = 18.5

    target_idx = torch.argmin(torch.abs(eig_vals_true - target_eig_val), keepdim=True)
    eig_val_true = eig_vals_true[target_idx]
    eig_vec_true = eig_vecs_true[:, target_idx]

    # The algorithm will likely not be able to converge using the default tol
    # because the large condition number of the shift-inverted matrix causes the
    # sparse solver to lose precision.
    eig_val, eig_vec = lobpcg(
        a=a_sdt,
        m=None,
        k=k,
        lobpcg_config=LOBPCGConfig(sigma=target_eig_val, largest=True, maxiter=10),
    )

    torch.testing.assert_close(eig_val, eig_val_true)
    torch.testing.assert_close(
        canonicalize_eig_vec_signs(eig_vec),
        canonicalize_eig_vec_signs(eig_vec_true),
    )


@pytest.mark.gpu_only
def test_gep_shift_invert_forward(rand_sp_gep_6x6, device):
    a, m = rand_sp_gep_6x6

    a_dense = a.to_dense().to(device)
    m_dense = m.to_dense().to(device)

    eig_vals_true, eig_vecs_true = dense_gep(a_dense, m_dense)

    k = 1
    target_eig_val = 18.5

    target_idx = torch.argmin(torch.abs(eig_vals_true - target_eig_val), keepdim=True)
    eig_val_true = eig_vals_true[target_idx]
    eig_vec_true = eig_vecs_true[:, target_idx]

    a_sdt = SparseDecoupledTensor.from_tensor(a).to(device)
    m_sdt = SparseDecoupledTensor.from_tensor(m).to(device)

    # The algorithm will likely not be able to converge using the default tol
    # because the large condition number of the shift-inverted matrix causes the
    # sparse solver to lose precision.
    eig_val, eig_vec = lobpcg(
        a=a_sdt,
        m=m_sdt,
        k=k,
        lobpcg_config=LOBPCGConfig(sigma=target_eig_val, largest=True, maxiter=10),
    )

    torch.testing.assert_close(eig_val, eig_val_true)
    torch.testing.assert_close(
        canonicalize_eig_vec_signs(eig_vec),
        canonicalize_eig_vec_signs(eig_vec_true),
    )

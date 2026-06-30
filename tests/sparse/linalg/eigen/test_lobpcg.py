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

# TODO: clear up cupy/nvmath dependencies and skips
# TODO: relax GPU only marks

itemize_preconditioners = pytest.mark.parametrize(
    "preconditioner",
    [
        pytest.param("identity", marks=[pytest.mark.gpu_only]),
        pytest.param("jacobi", marks=[pytest.mark.gpu_only]),
        pytest.param("ilu", marks=[pytest.mark.gpu_only, pytest.mark.requires_cupy]),
        pytest.param(
            "cholesky", marks=[pytest.mark.gpu_only, pytest.mark.requires_nvmath]
        ),
    ],
)


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
def test_lobpcg_config_v0_expansion_and_batched_forward(
    rand_sp_spd_6x6: Float[Tensor, "6 6"],
    rand_sp_spd_9x9: Float[Tensor, "9 9"],
    device,
):
    k = 2

    # Create two random starting vectors matching the dimensions of the test matrices.
    v0_1 = torch.randn((6, k), dtype=torch.float64, device=device)
    v0_2 = torch.randn((9, k), dtype=torch.float64, device=device)

    # Initialize the config with a sequence of Tensors.
    config = LOBPCGConfig(v0=[v0_1, v0_2], largest=True)

    # Test expansion logic directly.
    expanded_configs = config.expand(n=2)
    assert len(expanded_configs) == 2
    torch.testing.assert_close(expanded_configs[0].v0, v0_1)
    torch.testing.assert_close(expanded_configs[1].v0, v0_2)

    # Ensure expansion fails if batch size doesn't match v0 list length.
    with pytest.raises(ValueError, match="Inconsistent v0 specification."):
        config.expand(n=3)

    # Setup batched sparse tensor.
    A1_op = SparseDecoupledTensor.from_tensor(rand_sp_spd_6x6).to(device)
    A2_op = SparseDecoupledTensor.from_tensor(rand_sp_spd_9x9).to(device)
    A_op = SparseDecoupledTensor.pack_block_diag((A1_op, A2_op))

    # Run batched LOBPCG forward pass to ensure the expanded config behaves properly
    # in the main loop without dimension mismatches.
    eig_vals, eig_vecs = lobpcg(
        a=A_op, m=None, block_diag_batch=True, k=k, lobpcg_config=config
    )

    assert eig_vals.size() == (2, k)
    assert eig_vecs.size() == (15, k)  # 6 + 9 = 15 total nodes.


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


@itemize_preconditioners
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
def test_batched_standard_backward(
    rand_sp_spd_6x6: Float[Tensor, "6 6"],
    rand_sp_spd_9x9: Float[Tensor, "9 9"],
    device,
):
    k = 3

    a1_sdt = SparseDecoupledTensor.from_tensor(rand_sp_spd_6x6).to(device)
    a2_sdt = SparseDecoupledTensor.from_tensor(rand_sp_spd_9x9).to(device)

    a1_dense = rand_sp_spd_6x6.to_dense().to(device)
    a1_dense.requires_grad_()

    a2_dense = rand_sp_spd_9x9.to_dense().to(device)
    a2_dense.requires_grad_()

    # Ground truth eigenpairs and subspace projectors.
    eig_vals_1_true_all, eig_vecs_1_true_all = torch.linalg.eigh(a1_dense)
    eig_vals_1_true = eig_vals_1_true_all[-k:]
    eig_vecs_1_true = eig_vecs_1_true_all[:, -k:]
    subspace_projector_1 = eig_vecs_1_true @ eig_vecs_1_true.T

    eig_vals_2_true_all, eig_vecs_2_true_all = torch.linalg.eigh(a2_dense)
    eig_vals_2_true = eig_vals_2_true_all[-k:]
    eig_vecs_2_true = eig_vecs_2_true_all[:, -k:]
    subspace_projector_2 = eig_vecs_2_true @ eig_vecs_2_true.T

    # Generate random projection vectors.
    eig_vals_1_rand = torch.randn_like(eig_vals_1_true)
    eig_vecs_1_rand = torch.randn_like(eig_vecs_1_true)

    eig_vals_2_rand = torch.randn_like(eig_vals_2_true)
    eig_vecs_2_rand = torch.randn_like(eig_vecs_2_true)

    # Compute ground truth invariant loss.
    loss_1_true = (
        torch.sum(eig_vals_1_true * eig_vals_1_rand)
        + torch.sum(eig_vecs_1_rand * eig_vecs_1_true, dim=0).abs().sum()
    )
    loss_2_true = (
        torch.sum(eig_vals_2_true * eig_vals_2_rand)
        + torch.sum(eig_vecs_2_rand * eig_vecs_2_true, dim=0).abs().sum()
    )
    (loss_1_true + loss_2_true).backward()

    # True gradients projected onto the subspace and extracted at sparse indices.
    grad_1_true = (subspace_projector_1 @ a1_dense.grad @ subspace_projector_1)[
        torch.unbind(a1_sdt.pattern.idx_coo, dim=0)
    ]
    grad_2_true = (subspace_projector_2 @ a2_dense.grad @ subspace_projector_2)[
        torch.unbind(a2_sdt.pattern.idx_coo, dim=0)
    ]

    # Setup batched sparse tensor.
    a_sdt = SparseDecoupledTensor.pack_block_diag((a1_sdt, a2_sdt))
    a_sdt.requires_grad_()

    # Compute batched eigensystem.
    eig_vals_rev, eig_vecs_rev = lobpcg(
        a=a_sdt,
        m=None,
        block_diag_batch=True,
        k=k,
        eps=0,
        lobpcg_config=LOBPCGConfig(largest=True),
    )

    # Re-align lobpcg's descending order with true ascending order for loss computation.
    eig_vals = torch.flip(eig_vals_rev, dims=(-1,))
    eig_vecs = torch.flip(eig_vecs_rev, dims=(-1,))

    # Unpack batched results.
    eig_vals_1, eig_vals_2 = eig_vals.unbind(0)
    eig_vecs_1 = eig_vecs[:6]
    eig_vecs_2 = eig_vecs[6:]

    # Compute batched invariant loss with the exact same random tensors.
    loss_1_batched = (
        torch.sum(eig_vals_1 * eig_vals_1_rand)
        + torch.sum(eig_vecs_1_rand * eig_vecs_1, dim=0).abs().sum()
    )
    loss_2_batched = (
        torch.sum(eig_vals_2 * eig_vals_2_rand)
        + torch.sum(eig_vecs_2_rand * eig_vecs_2, dim=0).abs().sum()
    )
    (loss_1_batched + loss_2_batched).backward()

    # Extract batched sparse gradients.
    grad_1_batched, grad_2_batched = torch.split(
        a_sdt.grad, [a1_sdt.values.size(0), a2_sdt.values.size(0)]
    )

    torch.testing.assert_close(grad_1_batched, grad_1_true)
    torch.testing.assert_close(grad_2_batched, grad_2_true)


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
@pytest.mark.requires_nvmath
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
@pytest.mark.requires_nvmath
def test_shift_invert_backward(rand_sp_spd_6x6: Float[Tensor, "6 6"], device):
    k = 1

    # We will compute the gradient twice: once in standard mode, and once in
    # shift-invert mode targeting the exact same eigenvalue. The gradients should match.
    a_std = SparseDecoupledTensor.from_tensor(rand_sp_spd_6x6).to(device)
    a_std.requires_grad_()

    a_sft = SparseDecoupledTensor.from_tensor(rand_sp_spd_6x6).to(device)
    a_sft.requires_grad_()

    # Standard Mode.
    eig_vals_std_rev, eig_vecs_std_rev = lobpcg(
        a=a_std, m=None, k=k, eps=0, lobpcg_config=LOBPCGConfig(largest=True)
    )
    eig_vals_std = torch.flip(eig_vals_std_rev, dims=(0,))
    eig_vecs_std = torch.flip(eig_vecs_std_rev, dims=(-1,))

    # Shift-Invert Mode.
    target_sigma = eig_vals_std.max().item() - 0.1
    eig_vals_sft_rev, eig_vecs_sft_rev = lobpcg(
        a=a_sft,
        m=None,
        k=k,
        eps=0,
        lobpcg_config=LOBPCGConfig(sigma=target_sigma, largest=True),
    )
    eig_vals_sft = torch.flip(eig_vals_sft_rev, dims=(0,))
    eig_vecs_sft = torch.flip(eig_vecs_sft_rev, dims=(-1,))

    eig_vals_rand = torch.randn_like(eig_vals_std)
    eig_vals_loss_std = torch.sum(eig_vals_rand * eig_vals_std)
    eig_vals_loss_sft = torch.sum(eig_vals_rand * eig_vals_sft)

    eig_vecs_rand = torch.randn_like(eig_vecs_std)
    eig_vecs_loss_std = torch.sum(eig_vecs_rand * eig_vecs_std, dim=0).abs().sum()
    eig_vecs_loss_sft = torch.sum(eig_vecs_rand * eig_vecs_sft, dim=0).abs().sum()

    combined_loss_std = eig_vals_loss_std + eig_vecs_loss_std
    combined_loss_sft = eig_vals_loss_sft + eig_vecs_loss_sft

    combined_loss_std.backward()
    combined_loss_sft.backward()

    torch.testing.assert_close(a_std.grad, a_sft.grad, rtol=1e-6, atol=1e-6)


@pytest.mark.gpu_only
@pytest.mark.requires_nvmath
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


@pytest.mark.gpu_only
def test_lorentzian_regularization_smoke(
    rand_sp_spd_degenerate_9x9: Float[Tensor, "9 9"], device
):
    k = 5
    a_sdt = SparseDecoupledTensor.from_tensor(rand_sp_spd_degenerate_9x9).to(device)
    a_sdt.requires_grad_()

    eig_vals, eig_vecs = lobpcg(
        a=a_sdt, m=None, k=k, eps="auto", lobpcg_config=LOBPCGConfig(largest=True)
    )

    loss = eig_vals.sum() + eig_vecs.sum()
    loss.backward()

    assert a_sdt.grad is not None
    assert not torch.isnan(a_sdt.grad).any()
    assert not torch.all(a_sdt.grad == 0)


@pytest.mark.gpu_only
def test_lorentzian_eig_vals_backward_exactness(
    rand_sp_spd_degenerate_9x9: Float[Tensor, "9 9"], device
):
    """Eigenvalue gradients should not be affected by Lorentzian regularization."""
    k = 5

    a_sdt = SparseDecoupledTensor.from_tensor(rand_sp_spd_degenerate_9x9).to(device)
    a_sdt.requires_grad_()

    a_dense = rand_sp_spd_degenerate_9x9.to_dense().to(device)
    a_dense.requires_grad_()

    eig_vals_true_all, _ = torch.linalg.eigh(a_dense)
    eig_vals_true = eig_vals_true_all[-k:]

    # Run sparse solver with regularization.
    eig_vals_rev, _ = lobpcg(
        a=a_sdt, m=None, k=k, eps=1e-6, lobpcg_config=LOBPCGConfig(largest=True)
    )
    eig_vals = torch.flip(eig_vals_rev, dims=(0,))

    # Compute loss on the eigenvalues. Note that we cannot just dot the eigenvalues
    # with a random tensor of the same shape here, because some of the resolved
    # eigenvalues are degenerate. As such, the basis eigenvectors chosen for
    # computing the gradient may be different between eigh() and lobpcg(). Therefore,
    # the coefficient on the degenerate eigenvalues must be identical.
    eig_vals_loss_true = torch.sum(eig_vals_true)
    eig_vals_loss = torch.sum(eig_vals)

    eig_vals_loss_true.backward()
    eig_vals_loss.backward()

    a_grad_true = a_dense.grad[torch.unbind(a_sdt.pattern.idx_coo, dim=0)]

    torch.testing.assert_close(eig_vals, eig_vals_true)
    torch.testing.assert_close(a_sdt.grad, a_grad_true)


@pytest.mark.gpu_only
def test_lorentzian_isolated_eig_vec_exactness(
    rand_sp_spd_degenerate_9x9: Float[Tensor, "9 9"], device
):
    """
    Test that the Lorentzian regularization effect is small for large spectral gaps.

    For an eigenvector separated by a large spectral gap, the Lorentzian broadening
    term ϵ is dwarfed by (Δλ)^2. The gradient should asymptotically approach the
    exact dense gradient.
    """
    k = 5

    a_sdt = SparseDecoupledTensor.from_tensor(rand_sp_spd_degenerate_9x9).to(device)
    a_sdt.requires_grad_()

    a_dense = rand_sp_spd_degenerate_9x9.to_dense().to(device)
    a_dense.requires_grad_()

    # Get ground truth.
    _, eig_vecs_true_all = torch.linalg.eigh(a_dense)
    eig_vecs_true_k = eig_vecs_true_all[:, -k:]

    # The top eigenvector 45.0 is isolated from 41.95.
    eig_vec_isolated_true = eig_vecs_true_k[:, -1:]

    # Subspace projector for the k resolved modes.
    subspace_projector = eig_vecs_true_k @ eig_vecs_true_k.T

    # Run sparse solver with regularization.
    _, eig_vecs_rev = lobpcg(
        a=a_sdt, m=None, k=k, eps=1e-6, lobpcg_config=LOBPCGConfig(largest=True)
    )
    eig_vecs = torch.flip(eig_vecs_rev, dims=(-1,))
    eig_vec_isolated = eig_vecs[:, -1:]

    # Compute loss on the isolated eigenvector.
    eig_vec_rand = torch.randn_like(eig_vec_isolated_true)
    eig_vec_loss_true = (
        torch.sum(eig_vec_rand * eig_vec_isolated_true, dim=0).abs().sum()
    )
    eig_vec_loss = torch.sum(eig_vec_rand * eig_vec_isolated, dim=0).abs().sum()

    eig_vec_loss_true.backward()
    eig_vec_loss.backward()

    # Project the dense gradient onto the k-dimensional subspace.
    a_grad_true = (subspace_projector @ a_dense.grad @ subspace_projector)[
        torch.unbind(a_sdt.pattern.idx_coo, dim=0)
    ]

    # We expect a tiny divergence caused by eps, so we relax atol/rtol slightly.
    # If eps=1e-6 and the gap is ~3.0, the perturbation is roughly on the
    # order of eps / gap^2 ≈ 1e-7.
    torch.testing.assert_close(a_sdt.grad, a_grad_true, rtol=1e-6, atol=1e-6)

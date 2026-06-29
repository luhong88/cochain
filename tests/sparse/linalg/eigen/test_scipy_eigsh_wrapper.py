import numpy as np
import pytest
import torch
from jaxtyping import Float
from torch import Tensor

from cochain.sparse.decoupled_tensor import SparseDecoupledTensor
from cochain.sparse.linalg.eigen import (
    SciPyEigshConfig,
    canonicalize_eig_vec_signs,
    scipy_eigsh,
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


def test_lorentzian_regularization_smoke(
    rand_sp_spd_degenerate_9x9: Float[Tensor, "9 9"], device
):
    # Note that the SciPy eigsh() function can sometimes have difficulty
    # resolving the multiplicity of degenerate eigenvalues; therefore, we will
    # only perform a smoke test on the degenerate SPD matrix.
    k = 5
    a_sdt = SparseDecoupledTensor.from_tensor(rand_sp_spd_degenerate_9x9).to(device)
    a_sdt.requires_grad_()

    eig_vals, eig_vecs = scipy_eigsh(
        a=a_sdt, m=None, k=k, eps=1e-3, config=SciPyEigshConfig(which="SM")
    )

    loss = eig_vals.sum() + eig_vecs.sum()
    loss.backward()

    assert a_sdt.grad is not None
    assert not torch.isnan(a_sdt.grad).any()
    assert not torch.all(a_sdt.grad == 0)


def test_eigsh_config_v0_expansion(device):
    # Create two random starting vectors (e.g., for a batch of 2).
    v0_1 = torch.randn(6, device=device)
    v0_2 = torch.randn(9, device=device)

    # Initialize the config with a sequence of Tensors.
    config = SciPyEigshConfig(v0=[v0_1, v0_2], which="LM")

    # Ensure post-init correctly converted them to numpy arrays.
    assert isinstance(config.v0, list)
    assert isinstance(config.v0[0], np.ndarray)
    assert isinstance(config.v0[1], np.ndarray)

    # Expand to a batch of 2.
    expanded_configs = config.expand(n=2)
    assert len(expanded_configs) == 2

    # Check that each expanded config got the correct slice of v0.
    assert isinstance(expanded_configs[0].v0, np.ndarray)
    np.testing.assert_array_equal(expanded_configs[0].v0, v0_1.cpu().numpy())

    assert isinstance(expanded_configs[1].v0, np.ndarray)
    np.testing.assert_array_equal(expanded_configs[1].v0, v0_2.cpu().numpy())

    # Ensure expansion fails if batch size doesn't match v0 list length.
    with pytest.raises(ValueError, match="Inconsistent v0 specification"):
        config.expand(n=3)


def test_hidden_eigenvector_grad_path(rand_sp_spd_6x6: Float[Tensor, "6 6"], device):
    k = 3
    a_sdt = SparseDecoupledTensor.from_tensor(rand_sp_spd_6x6).to(device)
    a_sdt.requires_grad_()

    # User requests NO eigenvectors.
    out = scipy_eigsh(
        a=a_sdt,
        m=None,
        k=k,
        return_eigenvectors=False,
        config=SciPyEigshConfig(which="LM"),
    )

    # Verify the API conforms to the expected return type (tensor only, no tuple).
    assert isinstance(out, torch.Tensor)

    # Even though eigenvectors weren't returned to the user, they should have
    # been computed under the hood and saved to the autograd context.
    loss = out.sum()
    loss.backward()

    # Verify gradient successfully propagated.
    assert a_sdt.grad is not None
    assert not torch.all(a_sdt.grad == 0)


def test_standard_forward(rand_sp_spd_6x6: Float[Tensor, "6 6"], device):
    a_sdt = SparseDecoupledTensor.from_tensor(rand_sp_spd_6x6).to(device)
    a_dense = rand_sp_spd_6x6.to_dense().to(device)

    eig_vals_true, eig_vecs_true = torch.linalg.eigh(a_dense)

    k = 3

    # Test both the LM and SM modes
    eig_vals, eig_vecs = scipy_eigsh(
        a=a_sdt, m=None, k=k, config=SciPyEigshConfig(which="LM")
    )

    # Both eigsolver returns eigenvalues in ascending orders
    torch.testing.assert_close(eig_vals, eig_vals_true[-k:])
    torch.testing.assert_close(
        canonicalize_eig_vec_signs(eig_vecs),
        canonicalize_eig_vec_signs(eig_vecs_true[:, -k:]),
    )

    eig_vals, eig_vecs = scipy_eigsh(
        a=a_sdt, m=None, k=k, config=SciPyEigshConfig(which="SM")
    )

    torch.testing.assert_close(eig_vals, eig_vals_true[:k])
    torch.testing.assert_close(
        canonicalize_eig_vec_signs(eig_vecs),
        canonicalize_eig_vec_signs(eig_vecs_true[:, :k]),
    )


def test_standard_eig_vals_backward(rand_sp_spd_9x9: Float[Tensor, "9 9"], device):
    k = 3

    a_sdt = SparseDecoupledTensor.from_tensor(rand_sp_spd_9x9).to(device)
    a_sdt.requires_grad_()

    a_dense = rand_sp_spd_9x9.to_dense().to(device)
    a_dense.requires_grad_()

    eig_vals_true_all, eig_vecs_true_all = torch.linalg.eigh(a_dense)
    eig_vals_true = eig_vals_true_all[-k:]

    eig_vals, eig_vecs = scipy_eigsh(
        a=a_sdt, m=None, k=k, eps=0, config=SciPyEigshConfig(which="LM")
    )

    # Compare eigenvalue gradient
    eig_vals_rand = torch.randn_like(eig_vals_true)
    eig_vals_loss_true = torch.sum(eig_vals_true * eig_vals_rand)
    eig_vals_loss = torch.sum(eig_vals * eig_vals_rand)

    eig_vals_loss_true.backward()
    eig_vals_loss.backward()

    eig_vals_grad_true = a_dense.grad[torch.unbind(a_sdt.pattern.idx_coo, dim=0)]
    eig_vals_grad = a_sdt.grad

    torch.testing.assert_close(eig_vals_grad, eig_vals_grad_true)


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
    eig_vals, eig_vecs = scipy_eigsh(
        a=a_sdt, m=None, k=k, eps=0, config=SciPyEigshConfig(which="LM")
    )

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

    eig_vals, eig_vecs = scipy_eigsh(
        a=a_sdt, m=None, k=k, eps=0, config=SciPyEigshConfig(which="LM")
    )

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

    eig_vals, eig_vecs = scipy_eigsh(
        a=a_sdt, block_diag_batch=True, k=k, config=SciPyEigshConfig(which="LM")
    )

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


def test_batched_standard_backward(
    rand_sp_spd_6x6: Float[Tensor, "6 6"],
    rand_sp_spd_9x9: Float[Tensor, "9 9"],
    device,
):
    k = 2

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
    eig_vals, eig_vecs = scipy_eigsh(
        a=a_sdt, block_diag_batch=True, k=k, eps=0, config=SciPyEigshConfig(which="LM")
    )

    # Unpack batched results
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


def test_gep_forward(
    rand_sp_gep_6x6: tuple[Float[Tensor, "6 6"], Float[Tensor, "6 6"]],
    device,
):
    a, m = rand_sp_gep_6x6

    a_dense = a.to_dense().to(device)
    m_dense = m.to_dense().to(device)

    eig_vals_true, eig_vecs_true = dense_gep(a_dense, m_dense)

    a_sdt = SparseDecoupledTensor.from_tensor(a).to(device)
    m_sdt = SparseDecoupledTensor.from_tensor(m).to(device)

    k = 3

    # Test both the LM and SM modes
    eig_vals, eig_vecs = scipy_eigsh(
        a=a_sdt, m=m_sdt, k=k, config=SciPyEigshConfig(which="LM")
    )

    # Both eigsolver returns eigenvalues in ascending orders
    torch.testing.assert_close(eig_vals, eig_vals_true[-k:])
    torch.testing.assert_close(
        canonicalize_eig_vec_signs(eig_vecs),
        canonicalize_eig_vec_signs(eig_vecs_true[:, -k:]),
    )

    eig_vals, eig_vecs = scipy_eigsh(
        a=a_sdt, m=m_sdt, k=k, config=SciPyEigshConfig(which="SM")
    )

    torch.testing.assert_close(eig_vals, eig_vals_true[:k])
    torch.testing.assert_close(
        canonicalize_eig_vec_signs(eig_vecs),
        canonicalize_eig_vec_signs(eig_vecs_true[:, :k]),
    )


def test_gep_eig_vals_backward(
    rand_sp_gep_9x9: tuple[Float[Tensor, "9 9"], Float[Tensor, "9 9"]],
    device,
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

    eig_vals, eig_vecs = scipy_eigsh(
        a=a_sdt, m=m_sdt, k=k, eps=0, config=SciPyEigshConfig(which="LM")
    )

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


def test_gep_eig_vecs_backward(
    rand_sp_gep_9x9: tuple[Float[Tensor, "9 9"], Float[Tensor, "9 9"]],
    device,
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

    eig_vals, eig_vecs = scipy_eigsh(
        a=a_sdt, m=m_sdt, k=k, eps=0, config=SciPyEigshConfig(which="LM")
    )

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

    eig_val, eig_vec = scipy_eigsh(
        a=a_sdt, m=None, k=k, config=SciPyEigshConfig(sigma=target_eig_val, which="LM")
    )

    torch.testing.assert_close(eig_val, eig_val_true)
    torch.testing.assert_close(
        canonicalize_eig_vec_signs(eig_vec),
        canonicalize_eig_vec_signs(eig_vec_true),
    )


def test_shift_invert_backward(rand_sp_spd_6x6: Float[Tensor, "6 6"], device):
    k = 1

    # We will compute the gradient twice: once in standard LM mode, and once in
    # shift-invert mode targeting the exact same eigenvalue. The gradients should match.
    a_std = SparseDecoupledTensor.from_tensor(rand_sp_spd_6x6).to(device)
    a_std.requires_grad_()

    a_sft = SparseDecoupledTensor.from_tensor(rand_sp_spd_6x6).to(device)
    a_sft.requires_grad_()

    # Standard Mode.
    eig_vals_std, eig_vecs_std = scipy_eigsh(
        a=a_std, m=None, k=k, eps=0, config=SciPyEigshConfig(which="LM")
    )

    # Shift-Invert Mode.
    target_sigma = eig_vals_std.item() - 0.1
    eig_vals_sft, eig_vecs_sft = scipy_eigsh(
        a=a_sft,
        m=None,
        k=k,
        eps=0,
        config=SciPyEigshConfig(sigma=target_sigma, which="LM"),
    )

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

    torch.testing.assert_close(a_std.grad, a_sft.grad)


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

    eig_val, eig_vec = scipy_eigsh(
        a=a_sdt, m=m_sdt, k=k, config=SciPyEigshConfig(sigma=target_eig_val, which="LM")
    )

    # Both eigsolver returns eigenvalues in ascending orders
    torch.testing.assert_close(eig_val, eig_val_true)
    torch.testing.assert_close(
        canonicalize_eig_vec_signs(eig_vec),
        canonicalize_eig_vec_signs(eig_vec_true),
    )

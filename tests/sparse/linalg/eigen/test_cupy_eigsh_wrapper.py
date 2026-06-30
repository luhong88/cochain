import cupy as cp
import pytest
import torch
from jaxtyping import Float
from torch import Tensor

from cochain.sparse.decoupled_tensor import SparseDecoupledTensor
from cochain.sparse.linalg.eigen import (
    CuPyEigshConfig,
    canonicalize_eig_vec_signs,
    cupy_eigsh,
)

pytest.importorskip("cupy")


@pytest.mark.gpu_only
def test_lorentzian_regularization_smoke(
    rand_sp_spd_degenerate_9x9: Float[Tensor, "9 9"], device
):
    # Note that the CuPy eigsh() function can sometimes have difficulty
    # resolving the multiplicity of degenerate eigenvalues; therefore, we will
    # only perform a smoke test on the degenerate SPD matrix.
    k = 5
    a_sdt = SparseDecoupledTensor.from_tensor(rand_sp_spd_degenerate_9x9).to(device)
    a_sdt.requires_grad_()

    eig_vals, eig_vecs = cupy_eigsh(
        a=a_sdt, k=k, eps="auto", cp_config=CuPyEigshConfig(which="LM")
    )

    loss = eig_vals.sum() + eig_vecs.sum()
    loss.backward()

    assert a_sdt.grad is not None
    assert not torch.isnan(a_sdt.grad).any()
    assert not torch.all(a_sdt.grad == 0)


@pytest.mark.gpu_only
def test_eigsh_config_v0_expansion(device):
    # Create two random starting vectors (e.g., for a batch of 2).
    v0_1 = torch.randn(6, device=device)
    v0_2 = torch.randn(9, device=device)

    # Initialize the config with a sequence of Tensors.
    config = CuPyEigshConfig(v0=[v0_1, v0_2], which="LM")

    # Ensure post-init correctly converted them to cupy arrays.
    assert isinstance(config.v0, list)
    assert isinstance(config.v0[0], cp.ndarray)
    assert isinstance(config.v0[1], cp.ndarray)

    # Expand to a batch of 2.
    expanded_configs = config.expand(n=2)
    assert len(expanded_configs) == 2

    # Check that each expanded config got the correct slice of v0.
    assert isinstance(expanded_configs[0].v0, cp.ndarray)
    cp.testing.assert_array_equal(expanded_configs[0].v0, v0_1.cpu().numpy())

    assert isinstance(expanded_configs[1].v0, cp.ndarray)
    cp.testing.assert_array_equal(expanded_configs[1].v0, v0_2.cpu().numpy())

    # Ensure expansion fails if batch size doesn't match v0 list length.
    with pytest.raises(ValueError, match="Inconsistent v0 specification"):
        config.expand(n=3)


@pytest.mark.gpu_only
def test_hidden_eigenvector_grad_path(rand_sp_spd_6x6: Float[Tensor, "6 6"], device):
    k = 3
    a_sdt = SparseDecoupledTensor.from_tensor(rand_sp_spd_6x6).to(device)
    a_sdt.requires_grad_()

    # User requests NO eigenvectors.
    out = cupy_eigsh(
        a=a_sdt,
        k=k,
        return_eigenvectors=False,
        cp_config=CuPyEigshConfig(which="LM"),
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


@pytest.mark.gpu_only
def test_standard_forward(rand_sp_spd_6x6: Float[Tensor, "6 6"], device):
    a_sdt = SparseDecoupledTensor.from_tensor(rand_sp_spd_6x6).to(device)
    a_dense = rand_sp_spd_6x6.to_dense().to(device)

    eig_vals_true, eig_vecs_true = torch.linalg.eigh(a_dense)

    k = 2

    # Test both the LM and SA modes
    eig_vals, eig_vecs = cupy_eigsh(a_sdt, k=k, cp_config=CuPyEigshConfig(which="LM"))

    # Both eigsolver returns eigenvalues in ascending orders
    torch.testing.assert_close(eig_vals, eig_vals_true[-k:])
    torch.testing.assert_close(
        canonicalize_eig_vec_signs(eig_vecs),
        canonicalize_eig_vec_signs(eig_vecs_true[:, -k:]),
    )

    eig_vals, eig_vecs = cupy_eigsh(a_sdt, k=k, cp_config=CuPyEigshConfig(which="SA"))

    torch.testing.assert_close(eig_vals, eig_vals_true[:k])
    torch.testing.assert_close(
        canonicalize_eig_vec_signs(eig_vecs),
        canonicalize_eig_vec_signs(eig_vecs_true[:, :k]),
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

    eig_vals, eig_vecs = cupy_eigsh(
        a_sdt, k=k, eps=0, cp_config=CuPyEigshConfig(which="LM")
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
    eig_vals, eig_vecs = cupy_eigsh(
        a_sdt, k=k, eps=0, cp_config=CuPyEigshConfig(which="LM")
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

    eig_vals, eig_vecs = cupy_eigsh(
        a_sdt, k=k, eps=0, cp_config=CuPyEigshConfig(which="LM")
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

    eig_vals, eig_vecs = cupy_eigsh(
        a_sdt, block_diag_batch=True, k=k, cp_config=CuPyEigshConfig(which="LM")
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


@pytest.mark.gpu_only
def test_batched_combined_backward(
    rand_sp_spd_6x6: Float[Tensor, "6 6"],
    rand_sp_spd_9x9: Float[Tensor, "9 9"],
    device,
):
    k = 2

    a1_sdt = SparseDecoupledTensor.from_tensor(rand_sp_spd_6x6).to(device)
    a2_sdt = SparseDecoupledTensor.from_tensor(rand_sp_spd_9x9).to(device)

    a1_dense = rand_sp_spd_6x6.to_dense().to(device)
    a2_dense = rand_sp_spd_9x9.to_dense().to(device)

    a1_dense.requires_grad_()
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
    eig_vals, eig_vecs = cupy_eigsh(
        a_sdt,
        block_diag_batch=True,
        k=k,
        eps=0,
        cp_config=CuPyEigshConfig(which="LM"),
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


@pytest.mark.requires_nvmath
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

    eig_val, eig_vec = cupy_eigsh(
        a_sdt, k=k, cp_config=CuPyEigshConfig(sigma=target_eig_val, which="LM")
    )

    torch.testing.assert_close(eig_val, eig_val_true)
    torch.testing.assert_close(
        canonicalize_eig_vec_signs(eig_vec),
        canonicalize_eig_vec_signs(eig_vec_true),
    )


@pytest.mark.requires_nvmath
@pytest.mark.gpu_only
def test_shift_invert_backward(rand_sp_spd_6x6: Float[Tensor, "6 6"], device):
    k = 1

    # We will compute the gradient twice: once in standard LM mode, and once in
    # shift-invert mode targeting the exact same eigenvalue. The gradients should match.
    a_std = SparseDecoupledTensor.from_tensor(rand_sp_spd_6x6).to(device)
    a_std.requires_grad_()

    a_sft = SparseDecoupledTensor.from_tensor(rand_sp_spd_6x6).to(device)
    a_sft.requires_grad_()

    # Standard Mode.
    eig_vals_std, eig_vecs_std = cupy_eigsh(
        a=a_std, k=k, eps=0, cp_config=CuPyEigshConfig(which="LM")
    )

    # Shift-Invert Mode.
    target_sigma = eig_vals_std.item() - 0.1
    eig_vals_sft, eig_vecs_sft = cupy_eigsh(
        a=a_sft,
        k=k,
        eps=0,
        cp_config=CuPyEigshConfig(sigma=target_sigma, which="LM"),
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

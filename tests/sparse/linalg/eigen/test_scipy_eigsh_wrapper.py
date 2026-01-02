import pytest
import scipy.linalg
import torch as t
from jaxtyping import Float

from cochain.sparse.linalg.eigen import SciPyEigshConfig, scipy_eigsh
from cochain.sparse.operators import SparseOperator


def test_standard_forward(rand_sp_spd_5x5: Float[t.Tensor, "5 5"], device):
    A_op = SparseOperator.from_tensor(rand_sp_spd_5x5).to(device)
    A_dense = rand_sp_spd_5x5.to_dense().to(device)

    eig_vals_true, eig_vecs_true = t.linalg.eigh(A_dense)

    k = 3
    eig_vals, eig_vecs = scipy_eigsh(
        A=A_op, M=None, k=k, config=SciPyEigshConfig(which="LM")
    )

    # Both eigsolver returns eigenvalues in ascending orders
    eig_vec_dot = t.sum(eig_vecs * eig_vecs_true[:, -k:], dim=0).abs()

    t.testing.assert_close(eig_vals, eig_vals_true[-k:])
    t.testing.assert_close(eig_vec_dot, t.ones_like(eig_vec_dot))

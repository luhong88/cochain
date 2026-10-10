from __future__ import annotations

__all__ = ["LOBPCGPrecondConfig", "LOBPCGConfig", "lobpcg"]

from dataclasses import asdict, dataclass, replace
from typing import Literal, Sequence

import torch
from jaxtyping import Float, Integer
from torch import Tensor

from ....decoupled_tensor import (
    BaseDecoupledTensor,
    DiagDecoupledTensor,
    SparseDecoupledTensor,
    SparsityPattern,
)
from ...solvers import DirectSolverConfig
from ..base._backward import dLdA_backward, dLdA_dLdM_backward
from ..base.utils import compute_lorentzian_eps_via_norm, matrix_inf_norm
from ._lobpcg_preconditioners import (
    ChoPrecond,
    IdentityPrecond,
    ILUPrecond,
    JacobiPrecond,
    LOBPCGPrecondConfig,
    LOBPCGPreconditioner,
)
from ._lobpcg_routines import lobpcg_forward

# In general, the generic `lobpcg()` function implemented here will only accept
# matrix-explicit `a`, even though the backend numerical routines/util functions
# can support `a` as a matrix-free LinearOp object. This limitation is due to the
# difficulty of implementing a custom autograd function for arbitrary linear operators.
# Therefore, the matrix-free backends are reserved exclusively for operator-specific
# LOBPCG frontends that they implement operator-specific autograd functions.


@dataclass
class LOBPCGConfig:
    """
    A dataclass encapsulating optional LOBPCG configuration parameters.

    Parameters
    ----------
    sigma
        Whether to find eigenvalues near sigma using shift-invert mode.
    v0
        Starting vectors for iteration. If the input matrix is block-diagonal,
        then `v0` should be a sequence of tensors, one for each batch element.
        In either case, the number of starting vectors need to match the `n`
        argument to `lobpcg()`.
    largest
        When `True`, solve for the largest eigenvalues; otherwise, solve for the
        smallest eigenvalues.
    tol
        Residual tolerance for stopping criterion; by default this is set to
        the square root of the machine epsilon at runtime.
    maxiter
        Maximum number of LOBPCG iterations allowed.
    generator
        A Pytorch random number generator.
    """

    sigma: float | int | None = None
    v0: Float[Tensor, "m n"] | Sequence[Float[Tensor, "coord n"]] | None = None
    largest: bool = False
    tol: float | Literal["auto"] = "auto"
    maxiter: int = 1000
    generator: torch.Generator | None = None

    def expand(self, n: int) -> list[LOBPCGConfig]:
        """
        Duplicate self for batched processing.

        Parameters
        ----------
        n
            How many times to duplicate self. If `v0` is a list of tensors, then
            `n` must be equal to the length of `v0`.

        Returns
        -------
        config_list
            A list of `LOBPCGConfig` duplicated from self. If `v0` is a list
            of tensors, then each duplicate is assigned one element from `v0`.
        """
        if isinstance(self.v0, Sequence):
            config_list = [replace(self, v0=v0) for v0 in self.v0]
            if len(config_list) != n:
                raise ValueError("Inconsistent v0 specification.")
        else:
            config_list = [self] * n

        return config_list


class LOBPCGAutogradFunction(torch.autograd.Function):
    @staticmethod
    def _dispatch_precond(
        a: Float[SparseDecoupledTensor, "m m"],
        lobpcg_config: LOBPCGConfig,
        precond_config: LOBPCGPrecondConfig,
    ) -> LOBPCGPreconditioner:
        if lobpcg_config.sigma is not None:
            # If doing shift-invert mode, always use the identity preconditioner and
            # ignore the user inputs.
            precond = IdentityPrecond()
        else:
            match precond_config.method:
                case "identity":
                    precond = IdentityPrecond()
                case "jacobi":
                    # a_op is not required to be int32-safe.
                    precond = JacobiPrecond(a_sdt=a)
                case "ilu":
                    # a_op is required to be int32-safe.
                    precond = ILUPrecond(
                        a_sdt=a,
                        diag_damp=precond_config.diag_damp,
                        spilu_kwargs=precond_config.spilu_kwargs,
                    )
                case "cholesky":
                    # a_op is required to be int32-safe.
                    precond = ChoPrecond(
                        a_sdt=a,
                        n=lobpcg_config.v0.size(-1),
                        diag_damp=precond_config.diag_damp,
                        nvmath_config=precond_config.nvmath_config,
                    )
                case _:
                    raise ValueError(
                        f"Unknown preconditioner '{precond_config.method}'."
                    )

        return precond

    @staticmethod
    def forward(
        a_val: Float[Tensor, " a_nz"],
        a_pattern: Integer[SparsityPattern, "m m"],
        m_val: Float[Tensor, " m_nz"] | None,
        m_pattern: Integer[SparsityPattern, "m m"] | None,
        k: int,
        eps: float | int,
        a_norm: float,
        m_norm: float,
        lobpcg_config: LOBPCGConfig,
        precond_config: LOBPCGPrecondConfig,
        nvmath_config: DirectSolverConfig,
    ) -> tuple[Float[Tensor, " k"], Float[Tensor, "m k"]]:
        a_op = SparseDecoupledTensor(a_pattern, a_val)
        precond = LOBPCGAutogradFunction._dispatch_precond(
            a_op, lobpcg_config, precond_config
        )

        if (m_val is None) and (m_pattern is None):
            m_op = None
        elif (m_val is not None) and (m_pattern is not None):
            m_op = SparseDecoupledTensor(m_pattern, m_val)
        else:
            raise ValueError()

        eig_vals, eig_vecs = lobpcg_forward(
            a_op=a_op,
            m_op=m_op,
            a_norm=a_norm,
            m_norm=m_norm,
            op_scale="auto",  # op_scale is only relevant for matrix-free linear operators
            precond=precond,
            nvmath_config=nvmath_config,
            **asdict(lobpcg_config),
        )

        if lobpcg_config.sigma is None:
            eig_vals_true = eig_vals[:k]
        else:
            # In shift-invert mode, λ_returned = 1 / (λ_true - σ)
            eig_vals_true = lobpcg_config.sigma + (1.0 / eig_vals[:k])

        return eig_vals_true, eig_vecs[:, :k]

    @staticmethod
    def setup_context(ctx, inputs, output):
        (
            a_val,
            a_pattern,
            m_val,
            m_pattern,
            k,
            eps,
            a_norm,
            m_norm,
            lobpcg_config,
            precond_config,
            nvmath_config,
        ) = inputs
        eig_vals, eig_vecs = output

        ctx.save_for_backward(eig_vals, eig_vecs)
        ctx.a_pattern = a_pattern
        ctx.m_pattern = m_pattern
        ctx.eps = eps

    @staticmethod
    def backward(
        ctx, dLdl: Float[Tensor, " k"], dLdv: Float[Tensor, "m k"] | None
    ) -> tuple[
        Float[Tensor, " a_nz"] | None,
        None,
        Float[Tensor, " a_nz"] | None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
    ]:
        needs_grad_A_val = ctx.needs_input_grad[0]
        needs_grad_M_val = ctx.needs_input_grad[2]

        if ctx.m_pattern is None:
            dLdA_val = dLdA_backward(ctx, dLdl, dLdv) if needs_grad_A_val else None
            dLdM_val = None
        else:
            dLdA_val, dLdM_val = dLdA_dLdM_backward(
                ctx, dLdl, dLdv, needs_grad_A_val, needs_grad_M_val
            )

        return dLdA_val, None, dLdM_val, None, None, None, None, None, None, None, None


def _lobpcg_no_batch(
    a: Float[SparseDecoupledTensor, "m m"],
    m: Float[SparseDecoupledTensor, "m m"] | None,
    k: int,
    eps: float | int,
    a_norm: float,
    m_norm: float,
    lobpcg_config: LOBPCGConfig,
    precond_config: LOBPCGPrecondConfig,
    nvmath_config: DirectSolverConfig,
) -> tuple[Float[Tensor, " k"], Float[Tensor, "m k"]]:
    eig_vals, eig_vecs = LOBPCGAutogradFunction.apply(
        a.values,
        a.pattern,
        None if m is None else m.values,
        None if m is None else m.pattern,
        k,
        eps,
        a_norm,
        m_norm,
        lobpcg_config,
        precond_config,
        nvmath_config,
    )

    return eig_vals, eig_vecs


def _lobpcg_batch(
    a_list: list[Float[SparseDecoupledTensor, "m m"]],
    m_batched: Float[SparseDecoupledTensor, "m m"] | None,
    k: int,
    eps: float | int,
    lobpcg_config_batched: LOBPCGConfig,
    precond_config: LOBPCGPrecondConfig,
    nvmath_config: DirectSolverConfig,
) -> tuple[Float[Tensor, " k"], Float[Tensor, "m k"]]:
    if m_batched is None:
        m_list = [None] * len(a_list)
    else:
        m_list = m_batched.unpack_block_diag()

    lobpcg_config_list = lobpcg_config_batched.expand(n=len(a_list))

    eig_val_list = []
    eig_vec_list = []
    for a, m, lobpcg_config in zip(a_list, m_list, lobpcg_config_list, strict=True):
        a_norm = matrix_inf_norm(a)
        m_norm = matrix_inf_norm(m)

        eig_val, eig_vec = _lobpcg_no_batch(
            a,
            m,
            k,
            eps,
            a_norm,
            m_norm,
            lobpcg_config,
            precond_config,
            nvmath_config,
        )

        eig_val_list.append(eig_val)
        eig_vec_list.append(eig_vec)

    eig_vals = torch.vstack(eig_val_list)

    if eig_vec_list[0] is None:
        eig_vecs = None
    else:
        eig_vecs = torch.vstack(eig_vec_list)

    return eig_vals, eig_vecs


def lobpcg(
    a: Float[SparseDecoupledTensor, "m m"],
    m: Float[BaseDecoupledTensor, "m m"] | None = None,
    block_diag_batch: bool = False,
    n: int | None = None,
    k: int = 6,
    eps: float | int | Literal["auto"] = "auto",
    lobpcg_config: LOBPCGConfig | None = None,
    nvmath_config: DirectSolverConfig | None = None,
    precond_config: LOBPCGPrecondConfig | None = None,
) -> tuple[Float[Tensor, "*b k"], Float[Tensor, "m k"]]:
    """
    Sparse differentiable eigensolver for SPD matrices using LOBPCG.

    This function implements a version of LOBPCG that's roughly equivalent to
    `torch.lobpcg(method='ortho')`; however, shift-invert mode for both standard
    and generalized eigenvalue problems is supported, and the function is differentiable
    with respect to the nonzero values of both `a` and `m`.

    Note that this function requires `nvmath-python` for the shift-invert mode.
    In addition, the incomplete LU preconditioner has `CuPy` dependency and the
    Cholesky preconditioner has `nvmath-python` dependency. Currently, due to the
    `nvmath-python` `DirectSolver` and `CuPy` `spilu()` limitations, the input
    `a` (and `m` in the shift-invert GEP mode) matrix need to be converted to
    CSR/CSC matrices with `int32` index dtype for the shift-invert mode and
    for the use of the incomplete LU and Cholesky preconditioners.

    Parameters
    ----------
    a : [m, m]
        A real, symmetric positive definite square matrix.
    m : [m, m]
        A real, symmetric positive definite square matrix that induces an inner
        product on the column space of `a`. If `m` is provided, solve a generalized
        eigenvalue problem. Note that, if `block_diag_batch=True`, then `m` cannot
        be a `DiagDecoupledTensor`.
    block_diag_batch
        Whether the input `a` matrix (and `m` if not `None`) is block-diagonal.
        If `a` and `m` are block-diagonal, then they must both have valid and
        matching `block_diag_config`, in which case each block/batch element
        will be solved sequentially. While it is possible to solve the entire block
        diagonal system in parallel without resorting to sequential processing,
        to do so robustly requires careful handling of the batch orthonormalization
        step that's currently not implemented. For use cases where a batched solver
        is preferred (e.g., for small meshes that have the same size), please refer
        to `torch.lobpcg()`.
    n
        The number of approximated eigenvalues/eigenvectors ("block size"), which
        should be in the range [`k`, `m`] (default value: `k`). In general, it is
        recommended to set the `n` argument somewhat higher than `k`, to make the
        convergence of the `k` desired eigenvalues faster and to account for
        possible degenerate eigenvalues.
    k
        The number of eigenvalues/eigenvectors to find. Note that, by default,
        this function finds the `k` smallest eigenvalues of `a`; this behavior
        can be changed in `lobpcg_config`.
    eps
        The strength of Lorentzian broadening/regularization, which removes
        singularities in backward gradient calculation when some of the
        eigenvalues are (near) degenerate. As a heuristic, the regularization starts
        to dominate the gradient calculation as the spectral gap approaches the
        square root of `eps`. Set to integer 0 to disable regularization; set to
        "auto" to select `eps` based on the input dtype and matrix inf-norm.
    lobpcg_config
        Additional optional LOBPCG configurations.
    nvmath_config
        Additional optional arguments for nvmath `DirectSolver()`; only relevant
        for the shift-invert mode. The config passed to this argument is
        independent of the `nvmath_config` attribute of the `LOBPCGPrecondConfig`
        class.
    precond_config
        Additional optional arguments for LOBPCG preconditioners. Note that the
        preconditioner config is ignored in the shift-invert mode.

    Returns
    -------
    eig_vals : [*b, k]
        A tensor of `k` eigenvalues. If `block_diag_batch` is `True`, then the tensor
        also has a leading batch dimension corresponding to the blocks in the
        input `a` matrix.
    eig_vecs : [m, k]
        A tensor of `k` orthonormal eigenvectors. If `block_diag_batch` is
        `False`, then each column represents an eigenvector; if `block_diag_batch`
        is `True`, then the eigenvectors for each block are stacked along
        the first dimension.

    Notes
    -----
    The autograd through eigenvectors do not account for contributions from the
    unresolved eigenvectors.

    This implementation accepts specific preconditioners, including: identity,
    Jacobi, incomplete LU, and Cholesky; the latter two support diagonal dampling/
    Tikhonov regularization. The preconditioner can be configured using a
    `LOBPCGPrecondConfig` and will be generated internally. Note that the Jacobi,
    incomplete LU, and Cholesky preconditioners are defined soly in terms of `a`.
    For generalized eigenvalue problems, this works reasonably well when searching
    for the smallest eigenvalues, which suppresses the effect of `m`, but performance
    will degrade for the largest eigenvalues; the shift-invert mode does not take
    preconditioners.

    This implementation employs a rank-adaptive, iterative, canonical/PCA
    orthonormalization strategy with soft-restart for constructing the trial subspace.
    Unlike the standard PyTorch LOBPCG implementation, this allows the solver to
    handle cases where the size of `a` (`m`) is smaller than 3x the block size `n`.
    However, if the dimension of the trial subspace (which is <= 3n) is equal to
    or larger than `m`, the Rayleigh-Ritz projection becomes a full similarity
    transformation. In this limit, the algorithm effectively performs an exact
    diagonalization similar to `torch.linalg.eigh()`, but less efficiently due
    to the subspace construction and projection steps.
    """
    # Note that we delegate the CuPy and nvmath-python dependency checks to
    # the operator and preconditioner constructors, rather than performing a
    # top-level check.

    if isinstance(m, DiagDecoupledTensor):
        if block_diag_batch:
            raise TypeError(
                "'m' cannot be a DiagDecoupledTensor when 'block_diag_batch' is True."
            )
        else:
            m = m.to_sdt()

    if lobpcg_config is None:
        lobpcg_config = LOBPCGConfig()
    if precond_config is None:
        precond_config = LOBPCGPrecondConfig()
    if nvmath_config is None:
        nvmath_config = DirectSolverConfig()

    if block_diag_batch:
        a_list = a.unpack_block_diag()
        block_sizes = [a_.size(0) for a_ in a_list]
    else:
        block_sizes = [a.size(0)]

    # Process raw LOBPCG config.
    if lobpcg_config.v0 is None:
        if n is None:
            n = k
    else:
        v0 = lobpcg_config.v0

        if block_diag_batch:
            if not isinstance(v0, Sequence) or len(v0) != len(block_sizes):
                raise ValueError("Batched v0 must contain one tensor per matrix block.")
            if n is None:
                n = v0[0].size(-1)
            for v0_, block_size in zip(v0, block_sizes):
                if v0_.ndim != 2 or v0_.shape != (block_size, n):
                    raise ValueError("Each batched v0 must have shape (block size, n).")

        else:
            if not isinstance(v0, Tensor):
                raise ValueError("Unbatched v0 must be a tensor.")
            if n is None:
                n = v0.size(-1)
            if v0.ndim != 2 or v0.shape != (a.size(0), n):
                raise ValueError("Unbatched v0 must have shape (matrix size, n).")

    if n < k or any(n > block_size for block_size in block_sizes):
        raise ValueError("n must be in the range [k, size of each matrix block].")

    if lobpcg_config.v0 is None:
        if block_diag_batch:
            v0 = [
                torch.randn(
                    (block_size, n),
                    generator=lobpcg_config.generator,
                    dtype=a.dtype,
                    device=a.device,
                )
                for block_size in block_sizes
            ]
        else:
            v0 = torch.randn(
                (a.size(0), n),
                generator=lobpcg_config.generator,
                dtype=a.dtype,
                device=a.device,
            )

    tol = (
        torch.finfo(a.dtype).eps ** 0.5
        if lobpcg_config.tol == "auto"
        else lobpcg_config.tol
    )

    processed_lobpcg_config = replace(lobpcg_config, v0=v0, tol=tol)

    if eps == "auto":
        eps = compute_lorentzian_eps_via_norm(a, m)

    if block_diag_batch:
        eig_vals, eig_vecs = _lobpcg_batch(
            a_list,
            m,
            k,
            eps,
            processed_lobpcg_config,
            precond_config,
            nvmath_config,
        )
    else:
        a_norm = matrix_inf_norm(a)
        m_norm = matrix_inf_norm(m)

        eig_vals, eig_vecs = _lobpcg_no_batch(
            a,
            m,
            k,
            eps,
            a_norm,
            m_norm,
            processed_lobpcg_config,
            precond_config,
            nvmath_config,
        )

    return eig_vals, eig_vecs

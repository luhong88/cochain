import nvmath.sparse.advanced as nvmath_sp
import torch as t
from cuda.core.experimental import Device
from jaxtyping import Float, Integer

from ..operators import SparseOperator, SparseTopology
from .nvmath_wrapper import DirectSolverConfig


def _batched_csr_eye(
    n: int, b: int, val_dtype: t.dtype, idx_dtype: t.dtype, device: t.device
) -> Float[t.Tensor, "*b n n"]:
    if b == 0:
        identity = t.sparse_csr_tensor(
            crow_indices=t.arange(n + 1, dtype=idx_dtype, device=device),
            col_indices=t.arange(n, dtype=idx_dtype, device=device),
            values=t.ones(n, dtype=val_dtype, device=device),
        )
    else:
        identity = t.sparse_csr_tensor(
            crow_indices=t.tile(
                t.arange(n + 1, dtype=idx_dtype, device=device), (b, 1)
            ),
            col_indices=t.tile(t.arange(n, dtype=idx_dtype, device=device), (b, 1)),
            values=t.tile(t.ones(n, dtype=val_dtype, device=device), (b, 1)),
        )

    return identity


class LOBPCGSparsePreconditioner:
    """
    Exact preconditioner for LOBPCG.

    For the generalized eigenvalue problem A@x = λ*M@x, this exact preconditioner
    is inv(A). If r is a residual vector, applying the preconditioner to r is
    equivalent to solving a sparse linear system A@w = r for w.
    """

    def __init__(
        self,
        A_val: Float[t.Tensor, " nnz"],
        A_sp_topo: Integer[SparseTopology, "m m"],
        n: int,
        regularization: float | int,
        config: DirectSolverConfig,
    ):
        A_csr = SparseOperator(A_val, A_sp_topo).to_sparse_csr(int32=True)

        b_dummy = t.zeros((A_csr.size(-1), n), dtype=A_csr.dtype, device=A_csr.device)

        if regularization < 0:
            raise ValueError("The regularization constant must be nonnegative.")

        # a good heuristic is reg = 1e-4 * mean(diag(A))
        if regularization > 0:
            eye = _batched_csr_eye(
                n=A_csr.size(-1),
                b=0,
                val_dtype=A_val.dtype,
                idx_dtype=t.int32,
                device=A_csr.device,
            )

            # If A_csr uses int32 indices, the op should also be int32.
            op = A_csr + regularization * eye

        else:
            op = A_csr

        from .nvmath_wrapper import sp_literal_to_matrix_type

        # Prepare nvmath DirectSolver.
        config.options.sparse_system_type = sp_literal_to_matrix_type["symmetric"]

        # Do not give DirectSolver constructor the current stream to prevent
        # possible stream mismatch in subsequent solver calls; instead, pass the
        # torch/cupy stream to individual methods to ensure sync.
        self.solver = nvmath_sp.DirectSolver(
            op, b_dummy, options=config.options, execution=config.execution
        )

        # force blocking operation to make it memory-safe to potentially call
        # free() immediately after solve().
        self.solver.options.blocking = True

        # Amortize planning and factorization costs upfront in __init__()
        t_stream = t.cuda.current_stream()

        for k, v in config.plan_kwargs.items():
            setattr(self.solver.plan_config, k, v)
        self.solver.plan(stream=t_stream)

        for k, v in config.factorization_kwargs.items():
            setattr(self.solver.factorization_config, k, v)
        self.solver.factorize(stream=t_stream)

        for k, v in config.solution_kwargs.items():
            setattr(self.solver.solution_config, k, v)

    def __matmul__(self, res: Float[t.Tensor, "m n"]) -> Float[t.Tensor, "m n"]:
        t.cuda.set_device(res.device)
        Device(res.device.index).set_current()

        stream = t.cuda.current_stream()

        res_col_major = res.transpose(-1, -2).contiguous().transpose(-1, -2)

        self.solver.reset_operands(b=res_col_major, stream=stream)

        return self.solver.solve(stream=stream)

    def __del__(self):
        # DirectSolver needs an explicit free() step to free up memory/resources.
        if hasattr(self, "solver"):
            if hasattr(self.solver, "free"):
                self.solver.free()
                self.solver = None


def _enforce_M_orthonormality(
    V: Float[t.Tensor, "m 3*n"], M_op: Float[SparseOperator, "m m"], rtol: float | None
) -> Float[t.Tensor, "m 3*n"]:
    """
    Convert the column vectors of V into M-orthonormal vectors using symmetric
    orthogonalization.

    Currently batched sparse-dense matrix operations are not well supported in
    torch; therefore, this function cannot support batch dimensions.
    """
    if rtol is None:
        rtol = M_op.size(-1) * t.finfo(M_op.dtype).eps

    # Compute the M-orthogonal gram matrix.
    G: Float[t.Tensor, "3*n 3*n"] = V.T @ (M_op @ V)

    # Perform an eigendecomposition of G = Q@Λ@Q.T.
    eig_vals, eig_vecs = t.linalg.eigh(G)

    # Clamp very small eigenvalues.
    eps = rtol * eig_vals.max()
    eig_vals_clamped = t.clip(eig_vals, min=eps)
    inv_eig_vals = 1.0 / t.sqrt(eig_vals_clamped)

    # Compute the whitening matrix W = Q@Λ^(-1/2)@Q.T as the inverse square root
    # of G.
    W = (eig_vecs * inv_eig_vals.view(1, -1)) @ eig_vecs.T

    # Find V_ortho = V@W, the M-orthonormal version of V. With some algebra,
    # one can check that V_ortho@M@V_ortho = I.
    V_ortho = eig_vecs @ W

    return V_ortho


def _lobpcg_one_iter(
    iter_: int,
    A_op: Float[SparseOperator, "m m"],
    M_op: Float[SparseOperator, "m m"],
    R: Float[t.Tensor, "m n"],
    X_current: Float[t.Tensor, "m n"],
    X_prev: Float[t.Tensor, "m n"],
    preconditioner: LOBPCGSparsePreconditioner,
    largest: bool,
    tol: float,
) -> tuple[Float[t.Tensor, " n"], Float[t.Tensor, "m n"]]:
    """
    Perform one iteration of LOBPCG
    """
    # Perform soft locking/deflation to lock in converged eigenvectors by zeroing
    # out the corresponding residual vectors.
    R_norm = t.linalg.norm(R, dim=0, keepdim=True)
    mask = (R_norm > tol).to(R_norm.dtype)
    R_masked = R * mask

    # Use the residual vectors R to compute the precondition directions W = inv(M)@R.
    W = preconditioner @ R_masked

    # Compute the momentum/conjugate directions P. During the first iteration,
    # X_current = X_prev so P = 0. Perform the same soft locking on the momentum.
    P = (X_current - X_prev) * mask

    # Assemble the new trial subspace and enforce M-orthonormality condition on
    # the subspace basis vectors. We perform the orthonomalization twice to
    # minimize numerical error. Since P = 0 during the first iteration, skip it.
    if iter_ == 0:
        V = t.hstack((X_current, W))
    else:
        V = t.hstack((X_current, W, P))

    V_ortho = _enforce_M_orthonormality(_enforce_M_orthonormality(V, M_op), M_op)

    # Rayleigh-Ritz projection
    # Let us approximate the eigenvectors using the subspace basis vectors; i.e.,
    # X_next = V@C, where C is a coefficient matrix. Then, the eigenvalue
    # equation A@X = M@X@Λ can be approximated as A@V@C = M@V@C@Λ. The best
    # approximation ensures that the error is orthogonal to the trial subspace,
    # i.e., V.T@(A@V@C - M@V@C@Λ) = 0. This is equivalent to solving a "reduced"
    # generalized eigenvalue problem A'@C = M'@C@Λ, where A' = V.T@A@V and
    # M' = V.T@M@V. Since V is M-orthonormal, M' should be close to the identity
    # matrix.
    A_reduced = V_ortho.T @ (A_op @ V_ortho)
    Lambda_next_all, X_next_reduced_all = t.linalg.eigh(A_reduced)

    # Lift the reduced eigenvectors back to the full space
    X_next_all = V_ortho @ X_next_reduced_all

    # Extract the n largest (or smallest) eigenvalue-eigenvector pairs.
    # Note that t.linalg.eigh() returns eigenvalues in ascending order
    n = X_current.size(-1)
    if largest:
        # if largest=True, sort eigenvalues in descending order
        X_next = t.flip(X_next_all[-n:], dim=-1)
        Lambda_next = t.flip(Lambda_next_all[-n:])
    else:
        # if largest=False, keep eigenvalues in ascending order
        X_next = X_next_all[:n]
        Lambda_next = Lambda_next[:n]

    return Lambda_next, X_next


def lobpcg_loop(
    A_op: Float[SparseOperator, "m m"],
    M_op: Float[SparseOperator, "m m"],
    X_0: Float[t.Tensor, "m n"],
    preconditioner: LOBPCGSparsePreconditioner,
    tol: float,
    niter: int,
) -> tuple[Float[t.Tensor, " n"], Float[t.Tensor, "m n"]]:
    X_current = _enforce_M_orthonormality(X_0, M_op)
    X_prev = X_current

    # Compute the eigenvalues using the Rayleigh quotient
    # X.T@A@X/X.T@M@X = X.T@A@X = X.T@M@X@Λ = Λ
    Lambda_current = t.diag(X_current.T @ (M_op @ X_current))

    for iter_ in range(niter):
        # Compute the residual vectors R = A@X - M@X@Λ
        R = A_op @ X_current - (M_op @ X_current) * Lambda_current.view(1, -1)
        R_norm = t.linalg.norm(R, dim=0)

        if (R_norm < tol).all():
            break
        else:
            Lambda_next, X_next = _lobpcg_one_iter(
                iter_, A_op, M_op, R, X_current, X_prev, preconditioner, tol
            )

            X_prev = X_current
            X_current = X_next
            Lambda_current = Lambda_next

    return Lambda_current, X_current

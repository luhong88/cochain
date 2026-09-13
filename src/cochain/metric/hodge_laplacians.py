__all__ = [
    "MixedWeakLaplacianBlocks",
    "codifferential",
    "weak_up_laplacian",
    "weak_down_laplacian",
]

from dataclasses import dataclass
from functools import cached_property
from typing import Any

import torch
from jaxtyping import Float
from torch import Tensor

from ..sparse.decoupled_tensor import BaseDecoupledTensor, SparseDecoupledTensor
from ..sparse.linalg.solvers import InvSparseOperator


def _inv_mass_matmul(
    rhs: Tensor | BaseDecoupledTensor,
    *,
    mass: Tensor | BaseDecoupledTensor | InvSparseOperator | None = None,
    inv_mass: Tensor | BaseDecoupledTensor | None = None,
    solver_kwargs: dict[str, Any] | None = None,
) -> Tensor | BaseDecoupledTensor:
    """
    Compute a matmul with an inverse mass matrix as the left operand.

    Given a mass matrix $M$ and a matrix $A$ of suitable size (represented by the
    `rhs` argument), this function computes $M^{-1}A$. If the inverse of $M$ is
    given explicitly via the `inv_mass` argument, then the matmul is computed
    directly; $M^{-1}$ and $A$ can be either dense or sparse. If no inverse of $M$
    is given explicitly, the matmul is performed via a linear solver. Specifically,
    if `mass` is already wrapped by a sparse solver as an `InvSparseOperator`
    object, then $A$ is converted to a dense tensor and passed as the rhs to the
    solver; otherwise, both $M$ and $A$ are converted to dense tensors and the
    matmul is computed using `torch.linalg.solve()`. The only path where the
    output of this function is sparse is when both `rhs` and `inv_mass` are
    `BaseDecoupledTensor`s.
    """
    if (mass is None) == (inv_mass is None):
        raise ValueError("Exactly one of 'mass' and 'inv_mass' must be provided.")

    if inv_mass is not None:
        return inv_mass @ rhs

    if isinstance(mass, InvSparseOperator):
        if solver_kwargs is None:
            solver_kwargs = {}

        return mass(rhs.to_dense(), **solver_kwargs)

    else:
        if solver_kwargs:
            raise ValueError(
                "'solver_kwargs' is only valid when 'mass' is an InvSparseOperator."
            )

        return torch.linalg.solve(mass.to_dense(), rhs.to_dense())


@dataclass(frozen=True)
class MixedWeakLaplacianBlocks:
    r"""
    Construct the mixed formulation representation for a weak $k$-Laplacian or its down component.

    Note that this class is only relevant for $k > 0$.

    Parameters
    ----------
    cbd_km1: [k_splx, km1_splx]
        The $(k-1)$-coboundary operator.
    cbd_k: [kp1_splx, k_splx]
        The $k$-coboundary operator.
    mass_km1: [km1_splx, km1_splx]
        The consistent mass matrix for discrete $(k-1)$-forms.
    mass_k: [k_splx, k_splx]
        The consistent mass matrix for discrete $k$-forms.
    mass_kp1: [kp1_splx, kp1_splx]
        The consistent mass matrix for discrete $(k+1)$-forms.

    Notes
    -----
    Consider a weak k-Laplacian

    $$S_k = d_k^T M_{k+1} d_k + M_k d_{k-1} M_{k-1}^{-1} d_{k-1}^T M_k$$

    A fundamental difficulty of representing $S_k$ as a sparse tensor is the presence
    of the matrix inverse $M_{k-1}^{-1}$ in the down component of $S_k$. Even if
    $M_{k-1}$ is sparse, its inverse is in general a dense matrix and its presence
    forces the down component of $S_k$ into a dense representation.

    To circumvent the need to densify $S_k$, let us consider the sparse linear system
    $S_k x = b$ for some $k$-cochain $x$ and rhs vector $b$. To work with this system,
    define an auxiliary $(k-1)$-cochain $y$ as the codifferential of $x$ (i.e., 
    $y = M_{k-1}^{-1} d_{k-1}^T M_k x$). This transforms the linear system into

    $$
    \begin{bmatrix}
        -M_{k-1}    & d_{k-1}^T M_k \\
        M_k d_{k-1} & d_k^T M_{k+1} d_k
    \end{bmatrix}
    \begin{bmatrix}
        y \\ x
    \end{bmatrix}
    =
    \begin{bmatrix}
    0 \\ b
    \end{bmatrix}
    $$

    This representation of the original $S_k x = b$ linear system is called the
    mixed formulation. This approach removes the need for matrix inverse ($M_{k-1}^{-1}$)
    required to construct $S_k$ explicitly; however, the block matrix is now symmetric
    indefinite compared to $S_k$, which is symmetric positive semidefinite. This 
    approach is called the "mixed" formulation because, instead of solving a system
    $S_k x = b$ for a $k$-cochain $x$, an auxiliary $(k-1)$-cochain $y$ is introduced
    and we solve for a concatenated, "mixed" cochain $[y, x]$.

    Note that, this approach also works for the down component of $S_k$
    ($M_k d_{k-1} M_{k-1}^{-1} d_{k-1}^T M_k$) alone, in which case the block 
    matrix simplifies to 

    $$
    \begin{bmatrix}
        -M_{k-1}    & d_{k-1}^T M_k \\
        M_k d_{k-1} & 0
    \end{bmatrix}
    $$

    This class offers util functions to facilitate three primary operations involving
    the weak $k$-Laplacians that makes use of this mixed formulation:

    * Solve the linear systems $S_k x = b$ (and $S_k x = M_k b$) for $x$: call the
    `get_full_system()` method to generate the mixed representation of $S_k$ (LHS)
    and $b$ (or $M_k b$) (RHS), which can be passed to a sparse linear solver to
    get the mixed cochain $[y, x]$; then, use `unpack_mixed_cochain()` to split
    the $x$ and $y$ components.

    * Perform the matrix-vector multiplication $S_k x = b$ to find $b$: this requires
    solving the coupled linear systems in the mixed formulation sequentially. first, call
    `get_codiff_system()` to generate the linear system $M_{k-1} y = d_{k-1}^T M_k x$
    and pass this system to a sparse linear solver to find the $(k-1)$-cochain
    $y$, then, call `get_forward_pass()` to compute $b = M_k d_{k-1} y + d_k^T M_{k+1} d_k x$.
    
    * Solve the generalized eigenvalue problem $S_k x = \lambda M_k x$: call the
    `get_gep()` method to generate the mixed representation of $S_k$ and $M_k$,
    which can be passed to a sparse eigensolver to find the eigenpairs.
    """

    cbd_km1: Float[SparseDecoupledTensor, "k_splx km1_splx"]
    cbd_k: Float[SparseDecoupledTensor, "kp1_splx k_splx"] | None
    mass_km1: Float[SparseDecoupledTensor, "km1_splx km1_splx"]
    mass_k: Float[SparseDecoupledTensor, "k_splx k_splx"]
    mass_kp1: Float[SparseDecoupledTensor, "kp1_splx kp1_splx"] | None

    def __post_init__(self):
        null_cbd_k = self.cbd_k is None
        null_mass_kp1 = self.mass_kp1 is None

        if null_cbd_k != null_mass_kp1:
            raise ValueError(
                "'cbd_k' and 'mass_kp1' must both be None or neither be None."
            )

        object.__setattr__(self, "down_only", null_cbd_k)

    @property
    def dtype(self) -> torch.dtype:
        return self.mass_k.dtype

    @property
    def device(self) -> torch.device:
        return self.mass_k.device

    @property
    def shape(self) -> torch.Size:
        """
        The shape of the weak k-Laplacian.

        Note that this is different from the shape of the mixed block system.
        """
        return self.mass_k.shape

    def size(self, dim: int | None = None) -> int | torch.Size:
        """
        Get the size of the weak k-Laplacian.

        Note that this is different from the size of the mixed block system.
        """
        if dim is None:
            return self.shape
        else:
            return self.shape[dim]

    @cached_property
    def _block_00(self) -> Float[SparseDecoupledTensor, "km1_splx km1_splx"]:
        return -self.mass_km1

    @cached_property
    def _block_10(self) -> Float[SparseDecoupledTensor, " k_splx km1_splx"]:
        return self.mass_k @ self.cbd_km1

    @property
    def _block_01(self) -> Float[SparseDecoupledTensor, "km1_splx k_splx"]:
        return self._block_10.T

    @cached_property
    def _block_11(self) -> Float[SparseDecoupledTensor, " k_splx k_splx"] | None:
        if self.down_only:
            return None
        else:
            return self.cbd_k.T @ self.mass_kp1 @ self.cbd_k

    @property
    def _n_km1_splx(self) -> int:
        return self.mass_km1.size(0)

    @cached_property
    def _mixed_k_laplacian(
        self,
    ) -> Float[SparseDecoupledTensor, "km1_splx+k_splx km1_splx+k_splx"]:
        """Compute the representation of the weak k-Laplacian in the mixed formulation."""
        return SparseDecoupledTensor.bmat(
            [[self._block_00, self._block_01], [self._block_10, self._block_11]]
        )

    def _get_metric(
        self, padded: bool = False
    ) -> (
        Float[SparseDecoupledTensor, "km1_splx+k_splx km1_splx+k_splx"]
        | Float[SparseDecoupledTensor, "km1_splx km1_splx"]
    ):
        """Get the M_k matrix in its original and mixed formulation representations."""
        if padded:
            zero = SparseDecoupledTensor.from_tensor(
                torch.sparse_coo_tensor(
                    indices=torch.empty((2, 0), dtype=self.mass_km1.pattern.dtype),
                    values=torch.empty((0,), dtype=self.dtype),
                    size=(self._n_km1_splx, self._n_km1_splx),
                    device=self.device,
                )
            )
            return SparseDecoupledTensor.pack_block_diag((zero, self.mass_k))

        else:
            return self.mass_k

    def _pad_k_cochain(
        self, b: Float[Tensor, " k_splx *ch"]
    ) -> Float[Tensor, " km1_splx+k_splx *ch"]:
        """Pad the RHS vector b with zeros into its mixed formulation representation."""
        b_pad = torch.zeros(
            (self._n_km1_splx, *b.shape[1:]), dtype=b.dtype, device=b.device
        )
        b_full = torch.cat((b_pad, b), dim=0)

        return b_full

    def unpack_mixed_cochain(
        self, x_full: Float[Tensor, " km1_splx+k_splx *ch"]
    ) -> tuple[Float[Tensor, " k_splx *ch"], Float[Tensor, " km1_splx *ch"]]:
        r"""
        Unpack the mixed cochain vector.

        For the mixed formulation representation of the linear system
        $S_k x = b$, Unpack the mixed cochain [y, x] into the two components,
        $x$, which satisfies the original $S_k x = b$, and $y$, which is the
        codifferential of $x$.

        Parameters
        ----------
        x_full : [km1_splx+k_splx, *ch]
            The mixed representation solution to $S_k x = b$.

        Returns
        -------
        x : [k_splx, *ch]
            The $k$-cochain as the solution to the original $S_k x = b$.
        y : [km1_splx, *ch]
            The codifferential of $x$ (i.e., $y = \delta x$)
        """
        x = x_full[self._n_km1_splx :]
        y = x_full[: self._n_km1_splx]

        return x, y

    def get_full_system(
        self, b: Float[Tensor, " k_splx *ch"], apply_mass_k: bool = False
    ) -> tuple[
        Float[SparseDecoupledTensor, "km1_splx+k_splx km1_splx+k_splx"],
        Float[Tensor, " km1_splx+k_splx *ch"],
    ]:
        r"""
        Generate the mixed formulation representation of $S_k x = b$.

        Parameters
        ----------
        b : [k_splx, *ch]
            The RHS vector b with arbitrary trailing independent channel dimensions.
        apply_mass_k
            If True, generate the mixed formulation representation of $S_k x = M_k b$
            by performing the $M_k b$ matrix-vector multiplication first. If False,
            skip this matrix-vector multiplication.

        Returns
        -------
        lhs : [km1_splx+k_splx, km1_splx+k_splx]
            The mixed formulation representation of $S_k$.
        rhs : [km1_splx+k_splx, *ch]
            The mixed formulation representation of $b$.
        """
        lhs = self._mixed_k_laplacian

        if apply_mass_k:
            rhs = self._pad_k_cochain(self._get_metric(padded=False) @ b)
        else:
            rhs = self._pad_k_cochain(b)

        return lhs, rhs

    # TODO: document singular metric issue
    def get_gep(
        self,
    ) -> tuple[
        Float[SparseDecoupledTensor, "km1_splx+k_splx km1_splx+k_splx"],
        Float[SparseDecoupledTensor, "km1_splx+k_splx km1_splx+k_splx"],
    ]:
        r"""
        Generate the mixed formulation representation of the weak k-Laplacian GEP.

        The generalized eigenvalue problem is defined as $S_k x = \lambda M_k x$.

        Returns
        -------
        mixed_k_laplacian : [km1_splx+k_splx, km1_splx+k_splx]
            The mixed formulation representation of $S_k$.
        metric : [km1_splx+k_splx, km1_splx+k_splx]
            The mixed formulation representation of $M_k$.
        """
        return self._mixed_k_laplacian, self._get_metric(padded=True)

    def get_codiff_system(
        self, x: Float[Tensor, " k_splx *ch"]
    ) -> tuple[
        Float[SparseDecoupledTensor, "km1_splx km1_splx"],
        Float[Tensor, " km1_splx *ch"],
    ]:
        r"""
        Generate the linear system required to solve for $y$.

        For a given mixed formulation representation of the linear system
        $S_k x = b$, generate the subsystem required to solve for $y$, the
        codifferential of $x$.

        Parameters
        ----------
        x : [k_splx, *ch]
            The vector $x$ in $S_k x = b$.

        Returns
        -------
        lhs : [km1_splx, km1_splx]
            The consistent 1-mass matrix, representing the LHS of the subsystem.
        rhs : [km1_splx, *ch]
            The RHS of the subsystem.
        """
        lhs = self.mass_km1
        rhs = self.cbd_km1.T @ self.mass_k @ x
        return lhs, rhs

    def get_forward_pass(
        self, x: Float[Tensor, " k_splx *ch"], y: Float[Tensor, " km1_splx *ch"]
    ) -> Float[Tensor, " k_splx *ch"]:
        r"""
        Evaluate $b = S_k x$ from $x$ and its codifferential $y$.

        Parameters
        ----------
        x : [k_splx, *ch]
            The vector $x$ in $S_k x = b$.
        y : [km1_splx, *ch]
            The codifferential of $x$, which can be computed by solving the
            linear system generated by `get_codiff_system()`.

        Returns
        -------
        b : [k_splx, *ch]
            The vector $b$ in $S_k x = b$.
        """
        if self.down_only:
            return self.mass_k @ self.cbd_km1 @ y

        else:
            return (
                self.mass_k @ self.cbd_km1 @ y
                + self.cbd_k.T @ self.mass_kp1 @ self.cbd_k @ x
            )


def codifferential(
    cbd_km1: Float[SparseDecoupledTensor, "k_splx km1_splx"],
    mass_k: Float[BaseDecoupledTensor, "k_splx k_splx"],
    *,
    mass_km1: Float[
        Tensor | BaseDecoupledTensor | InvSparseOperator, "km1_splx km1_splx"
    ]
    | None = None,
    inv_mass_km1: Float[BaseDecoupledTensor | Tensor, "km1_splx km1_splx"]
    | None = None,
    solver_kwargs: dict[str, Any] | None = None,
) -> Float[SparseDecoupledTensor | Tensor, "km1_splx k_splx"]:
    r"""
    Compute the codifferential on discrete k-forms.

    The k-codifferential is defined as

    $$\delta_k = M_{k-1}^{-1} d_{k-1}^T M_k$$

    where $M_k$ is the consistent $k$-mass matrix or the diagonal Hodge $k$-star,
    and $d_k$ is the $k$-coboundary operator/discrete exterior derivative.

    Parameters
    ----------
    cbd_km1 : [k_splx, km1_splx]
        The $(k-1)$-coboundary operator.
    mass_k : [k_splx, k_splx]
        The $k$-mass matrix.
    mass_km1 : [km1_splx, km1_splx]
        The $(k-1)$-mass matrix; either `mass_km1` or `inv_mass_km1` should be
        provided, but not both.
    inv_mass_km1 : [km1_splx, km1_splx]
        The inverse of the $(k-1)$-mass matrix; either `mass_km1` or `inv_mass_km1`
        should be provided, but not both.
    solver_kwargs
        Keyword arguments passed to the `mass_km1` sparse solver, if applicable.

    Returns
    -------
    codifferential : [km1_splx, k_splx]
        The $k$-codifferential operator. In general, the output tensor is dense
        unless `inv_mass_km1` is sparse.

    Notes
    -----
    In general, it is recommended to use `mass_k` and `mass_km1` (or `inv_mass_km1`)
    derived from the same theoretical framework to construct the codifferential;
    i.e., either both are consistent mass matrices, or both are diagonal Hodge
    stars constructed using the same kind of dual complex. The exception to this
    rule is that $M_0^{-1}$ is often adequately approximated by the inverse of the
    barycentric Hodge 0-star, regardless of the choice of $M_1$.

    The codifferential $\delta_k$ is also sometimes defined with a $(-1)^k$ or
    $(-1)^{n+k+1}$ factor (depending on the definition of the inner product) to
    satisfy the adjoint relation with the coboundary operator in the continuous
    setting. These sign corrections are not included in the current implementation;
    nevertheless, the linear algera still works out in the discrete setting such
    that $\delta_k$ and $d_{k-1}$ are adjoint under the inner product induced
    by the Hodge star operators (without the sign corrections).
    """
    return _inv_mass_matmul(
        rhs=cbd_km1.T @ mass_k,
        mass=mass_km1,
        inv_mass=inv_mass_km1,
        solver_kwargs=solver_kwargs,
    )


def weak_up_laplacian(
    cbd_k: Float[SparseDecoupledTensor, "kp1_splx k_splx"],
    mass_kp1: Float[BaseDecoupledTensor, "kp1_splx kp1_splx"],
) -> Float[SparseDecoupledTensor, "k_splx k_splx"]:
    r"""
    Compute the up component of the weak Hodge $k$-Laplacian.

    The up component of the weak $k$-Laplacian is defined as

    $$S_k^\text{up} = d_k^T M_{k+1} d_k$$

    where $d_k$ is the $k$-coboundary operator/discrete exterior derivative, and
    $M_{k+1}$ is the consistent $(k+1)$-mass matrix or the diagonal Hodge $(k+1)$-star.
    For $k = 1$, the up component is also known as the curl-curl component; for
    $k = 2$ on a tet mesh, the up component is also known as the grad-div component.

    Parameters
    ----------
    cbd_k : [kp1_splx, k_splx]
        The $k$-coboundary operator.
    mass_kp1 : [kp1_splx, kp1_splx]
        The $(k+1)$-mass matrix.

    Returns
    -------
    up_laplacian : [k_splx, k_splx]
        The up component of the weak $k$-Laplacian operator.

    Notes
    -----
    The weak up Laplacian operator for a well-defined mesh is symmetric positive
    semidefinite, and is related to the classical Laplacian by the relation
    $L_k = M_k^{-1} S_k$; in general, $L_k$ is self-adjoint w.r.t. the inner
    product induced by the mass matrices, but it is not symmetric.

    For $k = 0$, the weak DEC 0-Laplacian constructed using the circumcentric dual
    complex can be computed more directly via the cotan formula, which is implemented
    in the `stiffness_matrix()` function.

    To construct the full Hodge $k$-Laplacian from its up and down components,
    use the `SparseDecoupledTensor.assemble()` method if both components are sparse.
    """
    return cbd_k.T @ mass_kp1 @ cbd_k


def weak_down_laplacian(
    cbd_km1: Float[SparseDecoupledTensor, "k_splx km1_splx"],
    mass_k: Float[BaseDecoupledTensor, "k_splx k_splx"],
    *,
    mass_km1: Float[
        Tensor | BaseDecoupledTensor | InvSparseOperator, "km1_splx km1_splx"
    ]
    | None = None,
    inv_mass_km1: Float[BaseDecoupledTensor | Tensor, "km1_splx km1_splx"]
    | None = None,
    solver_kwargs: dict[str, Any] | None = None,
) -> Float[SparseDecoupledTensor | Tensor, "k_splx k_splx"]:
    r"""
    Compute the down component of the weak Hodge k-Laplacian.

    The down component of the weak $k$-Laplacian is defined as

    $$ S_k^\text{down} = M_k d_{k-1} M_{k-1}^{-1} d_{k-1}^T M_k$$

    where $d_k$ is the $k$-coboundary operator/discrete exterior derivative, and
    $M_k$ is the consistent $k$-mass matrix or the diagonal Hodge $k$-star. For
    $k = 1$, the down component is also known as the grad-div component; for
    $k = 2$, the down component is also known as the curl-curl component.

    Parameters
    ----------
    cbd_km1 : [k_splx, km1_splx]
        The $(k-1)$-coboundary operator.
    mass_k : [k_splx, k_splx]
        The $k$-mass matrix.
    mass_km1 : [km1_splx, km1_splx]
        The $(k-1)$-mass matrix; either `mass_km1` or `inv_mass_km1` should be
        provided, but not both.
    inv_mass_km1 : [km1_splx, km1_splx]
        The inverse of the $(k-1)$-mass matrix; either `mass_km1` or `inv_mass_km1`
        should be provided, but not both.
    solver_kwargs
        Keyword arguments passed to the `mass_km1` sparse solver, if applicable.

    Returns
    -------
    down_laplacian : [k_splx, k_splx]
        The down component of the weak $k$-Laplacian operator. In general, the
        output tensor is dense unless `inv_mass_km1` is sparse.

    Notes
    -----
    The weak down Laplacian operator for a well-defined mesh is symmetric positive
    semidefinite, and is related to the classical Laplacian by the relation
    $L_k = M_k^{-1} S_k$; in general, $L_k$ is self-adjoint w.r.t. the inner
    product induced by the mass matrices, but it is not symmetric.

    In general, it is recommended to use `mass_k` and `mass_km1` (or `inv_mass_km1`)
    derived from the same theoretical framework to construct the down Laplacian;
    i.e., either both are consistent mass matrices, or both are diagonal Hodge stars
    constructed using the same kind of dual complex. The exception to this rule is
    that $M_0^{-1}$ is often adequately approximated by the inverse of the barycentric
    Hodge 0-star, regardless of the choice of $M_1$; the combination of the inverse
    diagonal 0-star with the consistent 1-mass matrix results in a hybrid/mass-lumped
    down 1-Laplacian.

    To construct the full Hodge $k$-Laplacian from its up and down components,
    use the `SparseDecoupledTensor.assemble()` method if both components are sparse.
    To construct the mixed formulation block system, use the `MixedWeakLaplacianBlocks`
    class.
    """
    m_d = mass_k @ cbd_km1

    m_inv_d_m = _inv_mass_matmul(
        rhs=m_d.T,
        mass=mass_km1,
        inv_mass=inv_mass_km1,
        solver_kwargs=solver_kwargs,
    )

    return m_d @ m_inv_d_m

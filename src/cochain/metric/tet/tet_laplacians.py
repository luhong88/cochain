__all__ = [
    "MixedWeakLaplacianBlocks",
    "weak_laplacian_0",
    "weak_laplacian_1_grad_div",
    "weak_laplacian_1_curl_curl",
    "weak_laplacian_1",
    "weak_laplacian_2_curl_curl",
    "weak_laplacian_2_grad_div",
    "weak_laplacian_2",
    "weak_laplacian_3",
]

from dataclasses import dataclass
from functools import cached_property
from typing import Literal

import torch
from jaxtyping import Float
from torch import Tensor

from ...complex import SimplicialMesh
from ...sparse.decoupled_tensor import SparseDecoupledTensor
from .tet_hodge_stars import star_0, star_1, star_2
from .tet_masses import mass_1, mass_2, mass_3
from .tet_stiffness import stiffness_matrix

# The weak k-Laplacian is defined as,
#
# S_k = (
#   M_k @ d_{k-1} @ inv_M_{k-1} @ d_{k-1}.T @ M_k +
#   d_k.T @ M_{k+1} @ d_k
# )
#
# The weak k-Laplacian (also known as the stiffness matrix) is related to the
# k-Laplacian via
#
# L_k = inv_M_k @ S_k


@dataclass
class MixedWeakLaplacianBlocks:
    r"""
    Construct the mixed formulation representation for a weak k-Laplacian.

    Parameters
    ----------
    cbd_km1: [k_splx, km1_splx]
        The k-coboundary operator.
    cbd_k: [kp1_splx, k_splx]
        The (k+1)-coboundary operator.
    mass_km1: [km1_splx, km1_splx]
        The consistent mass matrix for discrete (k-1)-forms.
    mass_k: [k_splx, k_splx]
        The consistent mass matrix for discrete k-forms.
    mass_kp1: [kp1_splx, kp1_splx]
        The consistent mass matrix for discrete (k+1)-forms.

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

    Note that, this approach also works for the down component of $S_k$ ($M_k d_{k-1}
    M_{k-1}^{-1} d_{k-1}^T M_k$) alone, in which case the block matrix simplifies to 

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

        self.down_only = null_cbd_k

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

        Note that this is different from the shape of the weak k-Laplacian in
        the weak formulation.
        """
        return self.mass_k.shape

    def size(self, dim: int | None = None) -> int | torch.Size:
        """
        Get the size of the weak k-Laplacian.

        Note that this is different from the size of the weak k-Laplacian in
        the weak formulation.
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
    def _block_11(self) -> Float[SparseDecoupledTensor, " k_splx k_splx"]:
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
        metric : [km1_splx+k_splx, *ch]
            The mixed formulation representation of $M_k$.
        """
        return self._mixed_k_laplacian, self._get_metric(padded=True)

    def get_codiff_system(
        self, x: Float[Tensor, " k_splx *ch"]
    ) -> tuple[
        Float[SparseDecoupledTensor, "km1_splx km1_splx"],
        Float[SparseDecoupledTensor, " km1_splx *ch"],
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
        Solve $S_k x = b$ for $x$ via its codifferential $y$.

        Parameters
        ----------
        x : [k_splx, *ch]
            The vector $x$ in $S_k x = b$.
        y : [km1_splx, *ch]
            The codifferential of $x$, which can be computed by solving the
            linear system generated by `get_codiff_system()`.

        Returns
        -------
        [km1_splx, *ch]
            The vector $b$ in $S_k x = b$.
        """
        if self.down_only:
            return self.mass_k @ self.cbd_km1 @ y

        else:
            return (
                self.mass_k @ self.cbd_km1 @ y
                + self.cbd_k.T @ self.mass_kp1 @ self.cbd_k @ x
            )


def weak_laplacian_0(
    tet_mesh: SimplicialMesh, method: Literal["cotan", "consistent"]
) -> Float[SparseDecoupledTensor, "vert vert"]:
    r"""
    Compute the weak 0-Laplacian/stiffness matrix.

    The weak 0-Laplacian is defined as

    $$S_0 = d_0^T M_1 d_0$$

    where $M_k$ is the consistent mass matrix on discrete $k$-forms, and $d_k$ is
    the $k$-coboundary operator/discrete exterior derivative.

    Parameters
    ----------
    tet_mesh
        A tet mesh.
    method
        if `method` is "consistent", use the consistent 1-mass matrix for $M_1$ as
        per the definition of $S_0$; if `method` is "cotan", this function computes
        the cotan Laplacian, which is equivalent to using the circumcentric Hodge
        1-star matrix in place of the consistent 1-mass matrix.

    Returns
    -------
    [vert, vert]
        The weak Laplacian operator.

    Notes
    -----
    The weak Laplacian operator $S_k$ as defined in this function is symmetric
    positive definite, and is related to the strong Laplacian by the relation
    $L_k = M_k^{-1} S_k$; in general, $L_k$ is self-adjoint w.r.t. the inner
    product induced by the mass matrices, but it is not symmetric or sparse.
    """
    match method:
        case "cotan":
            return stiffness_matrix(tet_mesh)
        case "consistent":
            d0 = tet_mesh.cbd[0]
            d0_T = d0.T

            m1 = mass_1(tet_mesh)

            return d0_T @ m1 @ d0
        case _:
            raise ValueError()


def weak_laplacian_1_grad_div(
    tet_mesh: SimplicialMesh,
) -> Float[SparseDecoupledTensor, "edge edge"]:
    r"""
    Compute the grad-div component of the weak, hybrid 1-Laplacian/stiffness matrix.

    The grad-div component of the weak 1-Laplacian, also known as the "down" Laplacian,
    is defined as

    $$S_1^\text{down} = M_1 d_0 M_0^{-1} d_0^T M_1$$

    where $M_k$ is the consistent mass matrix on discrete $k$-forms, and $d_k$ is
    the $k$-coboundary operator/discrete exterior derivative. This function implements
    a hybrid/mass-lumped version of $S_1^\text{down}$ by replacing the inverse of the
    consistent 0-mass matrix with the inverse of the Hodge 0-star matrix.

    Parameters
    ----------
    tet_mesh
        A tet mesh.

    Returns
    -------
    [edge, edge]
        The weak, hybrid Laplacian operator.
    """
    d0 = tet_mesh.cbd[0]
    m1 = mass_1(tet_mesh)
    inv_m0 = star_0(tet_mesh).inv

    return m1 @ d0 @ inv_m0 @ d0.T @ m1


def weak_laplacian_1_curl_curl(
    tet_mesh: SimplicialMesh,
) -> Float[SparseDecoupledTensor, "edge edge"]:
    r"""
    Compute the curl-curl component of the weak 1-Laplacian/stiffness matrix.

    The curl-curl component of the weak 1-Laplacian, also known as the "up" Laplacian,
    is defined as

    $$S_1^\text{up} = d_1^T M_2 d_1$$

    where $M_k$ is the consistent mass matrix on discrete $k$-forms, and $d_k$ is
    the $k$-coboundary operator/discrete exterior derivative.

    Parameters
    ----------
    tet_mesh
        A tet mesh.

    Returns
    -------
    [edge, edge]
        The weak Laplacian operator.
    """
    d1 = tet_mesh.cbd[1]
    m2 = mass_2(tet_mesh)

    return d1.T @ m2 @ d1


def weak_laplacian_1(
    tet_mesh: SimplicialMesh,
) -> Float[SparseDecoupledTensor, "edge edge"]:
    r"""
    Compute the weak, hybrid 1-Laplacian/stiffness matrix.

    The weak 1-Laplacian is defined as

    $$S_1 = d_1^T M_2 d_1 + M_1 d_0 M_0^{-1} d_0^T @ M_1$$

    where $M_k$ is the consistent mass matrix on discrete $k$-forms, and $d_k$ is
    the $k$-coboundary operator/discrete exterior derivative. This function implements
    a hybrid/mass-lumped version of $S_1$ by replacing the inverse of the consistent
    0-mass matrix with the inverse of the Hodge 0-star matrix.

    Parameters
    ----------
    tet_mesh
        A tet mesh.

    Returns
    -------
    [edge, edge]
        The weak, hybrid Laplacian operator.

    Notes
    -----
    The weak Laplacian operator $S_k$ as defined in this function is symmetric
    positive definite, and is related to the strong Laplacian by the relation
    $L_k = M_k^{-1} S_k$; in general, $L_k$ is self-adjoint w.r.t. the inner
    product induced by the mass matrices, but it is not symmetric or sparse.
    """
    return SparseDecoupledTensor.assemble(
        weak_laplacian_1_grad_div(tet_mesh), weak_laplacian_1_curl_curl(tet_mesh)
    )


def weak_laplacian_2_curl_curl(
    tet_mesh: SimplicialMesh,
    method: Literal["dense", "inv_star", "mixed"],
) -> (
    Float[Tensor, "tri tri"]
    | Float[SparseDecoupledTensor, "tri tri"]
    | Float[MixedWeakLaplacianBlocks, " tri tri"]
):
    r"""
    Compute the curl-curl component of the weak 2-Laplacian/stiffness matrix.

    The curl-curl component of the weak 2-Laplacian, also known as the "down" 2-Laplacian,
    is defined as

    $$S_2^\text{down} = M_2 d_1 M_1^{-1} d_1^T M_2$$

    where $M_k$ is the consistent mass matrix on discrete $k$-forms, and $d_k$ is
    the $k$-coboundary operator/discrete exterior derivative.

    Parameters
    ----------
    tet_mesh
        A tet mesh.
    method
        If `method` is "dense", $M_1$ and $d_1 M_2$ are converted to dense matrices,
        and passed to `torch.linalg.solve()` to find $M_1^{-1} d_1 M_2$;
        the output $S_2^\text{down}$ is a dense matrix. If `method` is "inv_star",
        $M_1^{-1}$ is approximated by the inverse of the Hodge 1-star operator,
        and the output $S_2^\text{down}$ is a `SparseDecoupledTensor` representing
        the hybrid operator. If `method` is "mixed", $S_2^\text{down}$ is computed
        in the mixed formulation representation and the function returns a
        `MixedWeakLaplacianBlocks` object.

    Returns
    -------
    [tri, tri]
        The weak Laplacian operator.
    """
    d1 = tet_mesh.cbd[1]
    m1 = mass_1(tet_mesh)
    m2 = mass_2(tet_mesh)

    match method:
        case "dense":
            return (m2 @ d1) @ torch.linalg.solve(m1.to_dense(), (d1.T @ m2).to_dense())

        case "inv_star":
            m1 = mass_1(tet_mesh)
            inv_m1 = star_1(tet_mesh).inv
            m2 = mass_2(tet_mesh)

            return m2 @ d1 @ inv_m1 @ d1.T @ m2

        case "solver":
            return MixedWeakLaplacianBlocks(
                cbd_km1=d1,
                cbd_k=None,
                mass_km1=m1,
                mass_k=m2,
                mass_kp1=None,
            )

        case _:
            raise ValueError(f"Unknown 'method' argument ('{method}').")


def weak_laplacian_2_grad_div(
    tet_mesh: SimplicialMesh,
) -> Float[SparseDecoupledTensor, "tri tri"]:
    r"""
    Compute the grad-div component of the weak 2-Laplacian/stiffness matrix.

    The grad-div component of the weak 2-Laplacian, also known as the "up" 2-Laplacian,
    is defined as

    $$S_2^\text{up} = d_2^T M_3 d_2$$

    where $M_k$ is the consistent mass matrix on discrete $k$-forms, and $d_k$ is
    the $k$-coboundary operator/discrete exterior derivative.

    Parameters
    ----------
    tet_mesh
        A tet mesh.

    Returns
    -------
    [tri, tri]
        The weak Laplacian operator.
    """
    d2 = tet_mesh.cbd[2]
    m3 = mass_3(tet_mesh)

    return d2.T @ m3 @ d2


def weak_laplacian_2(
    tet_mesh: SimplicialMesh,
    method: Literal["dense", "inv_star", "mixed"],
) -> (
    Float[Tensor, "tri tri"]
    | Float[SparseDecoupledTensor, "tri tri"]
    | Float[MixedWeakLaplacianBlocks, " tri tri"]
):
    r"""
    Compute the weak 2-Laplacian/stiffness matrix.

    The weak 2-Laplacian is defined as

    $$S_2 = d_2^T M_3 d_2 + M_2 d_1 M_1^{-1} d_1^T M_2$$

    where $M_k$ is the consistent mass matrix on discrete $k$-forms, and $d_k$ is
    the $k$-coboundary operator/discrete exterior derivative.

    Parameters
    ----------
    tet_mesh
        A tet mesh.
    method
        If `method` is "dense", $M_1$ and $d_1 M_2$ are converted to dense matrices,
        and passed to `torch.linalg.solve()` to find $M_1^{-1} d_1 M_2$; the output
        $S_2$ is a dense matrix. If `method` is "inv_star", $M_1^{-1}$ is approximated
        by the inverse of the Hodge 1-star operator, and the output $S_2$ is a
        `SparseDecoupledTensor` representing the hybrid operator. If `method` is
        "mixed", $S_2$ is computed in the mixed formulation representation and
        the function returns a `MixedWeakLaplacianBlocks` object.

    Returns
    -------
    [tri, tri]
        The weak Laplacian operator.

    Notes
    -----
    The weak Laplacian operator $S_k$ as defined in this function is symmetric
    positive definite, and is related to the strong Laplacian by the relation
    $L_k = M_k^{-1} S_k$; in general, $L_k$ is self-adjoint w.r.t. the inner
    product induced by the mass matrices, but it is not symmetric or sparse.
    """
    match method:
        case "dense" | "inv_star":
            curl_curl = weak_laplacian_2_curl_curl(tet_mesh, method)
            div_grad = weak_laplacian_2_grad_div(tet_mesh)

            match curl_curl:
                case SparseDecoupledTensor():
                    return SparseDecoupledTensor.assemble(div_grad, curl_curl)
                case Tensor():
                    return div_grad + curl_curl.to_dense()
                case _:
                    raise TypeError()

        case "mixed":
            d1 = tet_mesh.cbd[1]
            d2 = tet_mesh.cbd[2]

            m1 = mass_1(tet_mesh)
            m2 = mass_2(tet_mesh)
            m3 = mass_3(tet_mesh)

            return MixedWeakLaplacianBlocks(
                cbd_km1=d1,
                cbd_k=d2,
                mass_km1=m1,
                mass_k=m2,
                mass_kp1=m3,
            )

        case _:
            raise ValueError(f"Unknown 'method' argument ('{method}').")


def weak_laplacian_3(
    tet_mesh: SimplicialMesh,
    method: Literal["dense", "inv_star", "mixed"],
) -> (
    Float[Tensor, "tet tet"]
    | Float[SparseDecoupledTensor, "tet tet"]
    | Float[MixedWeakLaplacianBlocks, " tet tet"]
):
    r"""
    Compute the weak 3-Laplacian/stiffness matrix.

    The weak 3-Laplacian is defined as

    $$S_3 = M_3 d_2 M_2^{-1} d_2^T M_3$$

    where $M_k$ is the consistent mass matrix on discrete $k$-forms, and $d_k$ is
    the $k$-coboundary operator/discrete exterior derivative.

    Parameters
    ----------
    tet_mesh
        A tet mesh.
    method
        If `method` is "dense", $M_2$ and $d_2 M_3$ are converted to dense matrices,
        and passed to `torch.linalg.solve()` to find $M_2^{-1} d_2 M_m$; the output
        $S_3$ is a dense matrix. If `method` is "inv_star", $M_2^{-1}$ is approximated
        by the inverse of the Hodge 2-star operator, and the output $S_3$ is a
        `SparseDecoupledTensor` representing the hybrid operator. If `method` is
        "mixed", $S_3$ is computed in the mixed formulation representation and
        the function returns a `MixedWeakLaplacianBlocks` object.

    Returns
    -------
    [tet, tet]
        The weak Laplacian operator.

    Notes
    -----
    The weak Laplacian operator $S_k$ as defined in this function is symmetric
    positive definite, and is related to the strong Laplacian by the relation
    $L_k = M_k^{-1} S_k$; in general, $L_k$ is self-adjoint w.r.t. the inner
    product induced by the mass matrices, but it is not symmetric or sparse.
    """
    d2 = tet_mesh.cbd[2]

    m2 = mass_2(tet_mesh)
    m3 = mass_3(tet_mesh)

    match method:
        case "dense":
            return (m3 @ d2) @ torch.linalg.solve(m2.to_dense(), (d2.T @ m3).to_dense())

        case "inv_star":
            m2 = mass_2(tet_mesh)
            inv_m2 = star_2(tet_mesh).inv
            m3 = mass_3(tet_mesh)

            return m3 @ d2 @ inv_m2 @ d2.T @ m3

        case "mixed":
            return MixedWeakLaplacianBlocks(
                cbd_km1=d2,
                cbd_k=None,
                mass_km1=m2,
                mass_k=m3,
                mass_kp1=None,
            )

        case _:
            raise ValueError(f"Unknown 'method' argument ('{method}').")

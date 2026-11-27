import math
import warnings
from dataclasses import dataclass
from functools import cached_property
from typing import Literal

import torch
from einops import einsum
from jaxtyping import Bool, Float
from torch import Tensor

from ..sparse.decoupled_tensor import BaseDecoupledTensor, SparseDecoupledTensor
from .laplacians import MixedWeakLaplacianBlocks

# TODO: revisit docstring to flesh out theory
# TODO: implement backward pass for KKT system linear solve
# TODO: docstring comment on difference with topo tree-cotree utils


# TODO: consider similar convergence report for LOBPCG
@dataclass(frozen=True)
class LoadCompatibility:
    r"""
    Primary load compatibility report.

    Parameters
    ----------
    kernel_proj : [b_k, *ch]
        The projection of the primary load onto the kernel basis.
    norm : [*ch,]
        The Euclidean norm of the column vectors of `kernel_proj`.
    tol : [*ch,]
        The effective compatibility tolerance threshold(s) for `norm`.
    compatible : [*ch,]
        Boolean mask indicating whether each channel's `norm` is at or below
        `tol`.

    Attributes
    ----------
    all_compatible : [,]
        Whether every channel's `norm` is at or below `tol`.
    """

    kernel_proj: Float[Tensor, " b_k *ch"]
    norm: Float[Tensor, "*ch"]
    tol: Float[Tensor, "*ch"] | float
    compatible: Bool[Tensor, "*ch"]

    @property
    def all_compatible(self) -> Bool[Tensor, ""]:
        """Whether all channels' `norm`s are at or below `tol`."""
        return torch.all(self.compatible)


def _check_primary_load_compatibility(
    b_k: int,
    kernel: Float[Tensor, "k_splx b_k"],
    primary_load: Float[Tensor, " k_splx *ch"],
    atol: Float[Tensor, "*ch"] | float = 0.0,
    rtol: Float[Tensor, "*ch"] | float | Literal["auto"] = "auto",
    incompatibility: Literal["raise", "warn", "project"] = "warn",
) -> LoadCompatibility:
    """Check compatibility of the primary load against the operator kernel basis."""
    if incompatibility not in ("raise", "warn", "project"):
        raise ValueError(f"Unknown 'incompatibility' argument '{incompatibility}'.")

    if b_k == 0:
        ch_dims = primary_load.shape[1:]

        load_compatibility = LoadCompatibility(
            kernel_proj=primary_load.new_zeros((b_k, *ch_dims)),
            norm=primary_load.new_zeros(ch_dims),
            tol=atol,
            compatible=primary_load.new_ones(ch_dims, dtype=torch.bool),
        )

    else:
        if rtol == "auto":
            rtol = torch.finfo(primary_load.dtype).eps ** 0.5

        # Define the tolerance threshold to be atol + rtol * ||H||_F * ||r_k||_2.
        kernel_fro_norm = torch.linalg.matrix_norm(kernel, ord="fro")
        primary_load_l2_norm = torch.linalg.vector_norm(primary_load, dim=0)
        tol = atol + rtol * kernel_fro_norm * primary_load_l2_norm

        # Check that the channels of ||H.T@r_k||_2 fall below the tolerance threshold.
        # Note that the Euclidean/L2 norm is used here because the incompatible
        # component of load r_k is given by M@H@H.T@r_k, and its inv(M)-norm is
        # ||H.T@r_k||_2 when H is M-orthonormal.
        kernel_proj = einsum(
            kernel,
            primary_load,
            "k_splx b_k, k_splx ... -> b_k ...",
        )
        kernel_proj_l2_norm = torch.linalg.vector_norm(kernel_proj, dim=0)
        compatible = kernel_proj_l2_norm <= tol

        load_compatibility = LoadCompatibility(
            kernel_proj=kernel_proj,
            norm=kernel_proj_l2_norm,
            tol=tol,
            compatible=compatible,
        )

        if not load_compatibility.all_compatible:
            msg = (
                "The linear system with the 'primary_load' as the RHS is inconsistent."
            )
            match incompatibility:
                case "raise":
                    raise ValueError(msg)
                case "warn":
                    warnings.warn(msg, UserWarning)
                case "project":
                    pass

    return load_compatibility


@dataclass(frozen=True)
class GaugeFixedSystem:
    r"""
    Construct a KKT gauge-fixed linear system for a symmetric positive semidefinite operator.

    Parameters
    ----------
    op : [k_splx, k_splx]
        A real, symmetric positive semidefinite square matrix.
    mass : [k_splx, k_splx]
        A real, symmetric positive definite mass matrix defining an inner product
        on the primal cochain space.
    kernel : [k_splx, b_k]
        A matrix whose column vectors form a basis set for the kernel of `op`
        that is orthonormal w.r.t. the inner product induced by `mass`. Note that
        the `GaugeFixedSystem` class always stores a detached view of the input
        `kernel` tensor.
    """

    op: Float[SparseDecoupledTensor, "k_splx k_splx"]
    mass: Float[BaseDecoupledTensor, "k_splx k_splx"]
    kernel: Float[Tensor, "k_splx b_k"]

    def __post_init__(self):
        object.__setattr__(self, "kernel", self.kernel.detach())

    @property
    def dtype(self) -> torch.dtype:
        return self.op.dtype

    @property
    def device(self) -> torch.device:
        return self.op.device

    @property
    def b_k(self) -> int:
        """The dimension of the kernel of `self.op`."""
        return self.kernel.size(-1)

    @property
    def shape(self) -> torch.Size:
        lhs_dim = self.op.size(-1) + self.b_k
        return torch.Size([lhs_dim, lhs_dim])

    def size(self, dim: int | None = None) -> int | torch.Size:
        if dim is None:
            return self.shape
        else:
            return self.shape[dim]

    @cached_property
    def lhs(self) -> Float[SparseDecoupledTensor, "k_splx+b_k k_splx+b_k"]:
        """
        Construct the LHS block matrix of the KKT system.

        Returns
        -------
        lhs : [k_splx+b_k, k_splx+b_k]
            The LHS block matrix of the KKT system.
        """
        if self.b_k == 0:
            return self.op
        else:
            m_h = self.mass @ self.kernel
            return SparseDecoupledTensor.bmat([[self.op, m_h], [m_h.T, None]])

    def build_rhs(
        self,
        primary_load: Float[Tensor, " k_splx *ch"],
        gauge_constraint: Float[Tensor, " b_k *ch"] | None = None,
        atol: Float[Tensor, "*ch"] | float = 0.0,
        rtol: Float[Tensor, "*ch"] | float | Literal["auto"] = "auto",
        incompatibility: Literal["raise", "warn", "project"] = "warn",
    ) -> tuple[Float[Tensor, " k_splx+b_k *ch"], LoadCompatibility]:
        r"""
        Construct the RHS load vector of the KKT system.

        Parameters
        ----------
        primary_load : [k_splx, *ch]
            The weak primary load vector.
        gauge_constraint : [b_k, *ch]
            Optional target for the gauge constraint; defaults to zeros, which
            selects the minimum-norm representative). This argument is ignored if
            `b_k` is zero.
        atol : [*ch,]
            Absolute tolerance for checking load compatibility.
        rtol : [*ch,]
            Relative tolerance for checking load compatibility, scaled by the
            norms of the kernel basis and primary load vectors; if `'auto'`, uses
            the square root of the `primary_load` dtype machine epsilon.
        incompatibility
            Policy when `primary_load` is incompatible with `self.op`: raise a
            `ValueError` if `'raise'`, emit a `UserWarning` if `'warn'`, or
            proceed silently if `'project'` (the KKT system then solves the
            range-projected linear system).

        Returns
        -------
        rhs : [k_splx+b_k, *ch]
            The assembled KKT RHS vector.
        load_compatibility
            A `LoadCompatibility` report quantifying the incompatible kernel
            component of `primary_load`.
        """
        load_compatibility = _check_primary_load_compatibility(
            b_k=self.b_k,
            kernel=self.kernel,
            primary_load=primary_load,
            atol=atol,
            rtol=rtol,
            incompatibility=incompatibility,
        )

        if self.b_k == 0:
            rhs = primary_load

        else:
            if gauge_constraint is None:
                b_k = self.kernel.size(-1)
                ch_dims = primary_load.shape[1:]
                gauge_constraint = primary_load.new_zeros((b_k, *ch_dims))

            rhs = torch.cat([primary_load, gauge_constraint], dim=0)

        return rhs, load_compatibility

    def unpack_sol(
        self, sol: Float[Tensor, " k_splx+b_k *ch"]
    ) -> tuple[Float[Tensor, " k_splx *ch"], Float[Tensor, " b_k *ch"]]:
        r"""
        Unpack a solution of the KKT system.

        Parameters
        ----------
        sol : [k_splx+b_k, *ch]
            A solution vector of the KKT system.

        Returns
        -------
        primary_sol : [k_splx, *ch]
            The primal solution satisfying the gauge constraint target; if the
            primary load is incompatible, this is the solution to the range-projected
            linear system.
        kernel_proj : [b_k, *ch]
            The KKT Lagrange multiplier.
        """
        n_k_splx = self.op.size(-1)
        return sol[:n_k_splx], sol[n_k_splx:]


@dataclass(frozen=True)
class MixedGaugeFixedSystem:
    r"""
    Construct a KKT gauge-fixed mixed system for a `MixedWeakLaplacianBlocks` object.

    Parameters
    ----------
    mixed_block : [k_splx, k_splx]
        A mixed formulation representation of a weak $k$-Laplacian.
    kernel : [k_splx, b_k]
        A matrix whose columns form an $M_k$-orthonormal basis for the
        kernel of the weak $k$-Laplacian. Note that `MixedGaugeFixedSystem` always
        stores a detached view of `kernel`.
    """

    mixed_block: Float[MixedWeakLaplacianBlocks, "k_splx k_splx"]
    kernel: Float[Tensor, "k_splx b_k"]

    def __post_init__(self):
        object.__setattr__(self, "kernel", self.kernel.detach())

    @property
    def dtype(self) -> torch.dtype:
        return self.mixed_block.dtype

    @property
    def device(self) -> torch.device:
        return self.mixed_block.device

    @property
    def b_k(self) -> int:
        """The dimension of the kernel of `self.mixed_block`."""
        return self.kernel.size(-1)

    @property
    def shape(self) -> torch.Size:
        lhs_dim = (
            self.mixed_block.mass_k.size(-1)
            + self.mixed_block.mass_km1.size(-1)
            + self.b_k
        )
        return torch.Size([lhs_dim, lhs_dim])

    def size(self, dim: int | None = None) -> int | torch.Size:
        if dim is None:
            return self.shape
        else:
            return self.shape[dim]

    @cached_property
    def lhs(
        self,
    ) -> Float[SparseDecoupledTensor, "km1_splx+k_splx+b_k km1_splx+k_splx+b_k"]:
        """
        Construct the LHS block matrix of the mixed KKT system.

        Returns
        -------
        lhs : [km1_splx+k_splx+b_k, km1_splx+k_splx+b_k]
            The LHS block matrix of the mixed KKT system.
        """
        if self.b_k == 0:
            return self.mixed_block._mixed_k_laplacian
        else:
            m_h = self.mixed_block.mass_k @ self.kernel
            return SparseDecoupledTensor.bmat(
                [
                    [self.mixed_block._block_00, self.mixed_block._block_01, None],
                    [self.mixed_block._block_10, self.mixed_block._block_11, m_h],
                    [None, m_h.T, None],
                ]
            )

    def build_rhs(
        self,
        primary_load: Float[Tensor, " k_splx *ch"],
        aux_load: Float[Tensor, " km1_splx *ch"] | None = None,
        gauge_constraint: Float[Tensor, " b_k *ch"] | None = None,
        atol: Float[Tensor, "*ch"] | float = 0.0,
        rtol: Float[Tensor, "*ch"] | float | Literal["auto"] = "auto",
        incompatibility: Literal["raise", "warn", "project"] = "warn",
    ) -> tuple[Float[Tensor, " km1_splx+k_splx+b_k *ch"], LoadCompatibility]:
        r"""
        Construct the RHS load vector of the mixed KKT system.

        Parameters
        ----------
        primary_load : [k_splx, *ch]
            The weak primary degree-$k$ load vector.
        aux_load : [km1_splx, *ch]
            Optional weak auxiliary degree-$(k-1)$ load vector; defaults to zeros.
        gauge_constraint : [b_k, *ch]
            Optional target for the gauge constraint; defaults to zeros, which
            select the minimum-norm representative. This argument is ignored if
            `b_k` is zero.
        atol : [*ch,]
            Absolute tolerance for checking load compatibility.
        rtol : [*ch,]
            Relative tolerance for checking load compatibility, scaled by the
            norms of the kernel basis and primary load vectors. If `'auto'`, uses
            the square root of the `primary_load` dtype machine epsilon.
        incompatibility
            Policy when `primary_load` is incompatible with `self.mixed_block`:
            raise a `ValueError` if `'raise'`, emit a `UserWarning` if `'warn'`,
            or proceed silently if `'project'` (the KKT system then solves the
            range-projected primary equation).

        Returns
        -------
        rhs : [km1_splx+k_splx+b_k, *ch]
            The assembled mixed KKT RHS vector.
        load_compatibility
            A `LoadCompatibility` report quantifying the incompatible kernel
            component of `primary_load`.
        """
        load_compatibility = _check_primary_load_compatibility(
            b_k=self.b_k,
            kernel=self.kernel,
            primary_load=primary_load,
            atol=atol,
            rtol=rtol,
            incompatibility=incompatibility,
        )

        if aux_load is None:
            n_km1_splx = self.mixed_block.mass_km1.size(-1)
            ch_dims = primary_load.shape[1:]
            scaled_aux_load = primary_load.new_zeros((n_km1_splx, *ch_dims))
        else:
            scaled_aux_load = aux_load * math.sqrt(self.mixed_block.down_scale)

        if self.b_k == 0:
            rhs = torch.cat([scaled_aux_load, primary_load], dim=0)

        else:
            if gauge_constraint is None:
                b_k = self.kernel.size(-1)
                ch_dims = primary_load.shape[1:]
                gauge_constraint = primary_load.new_zeros((b_k, *ch_dims))

            rhs = torch.cat([scaled_aux_load, primary_load, gauge_constraint], dim=0)

        return rhs, load_compatibility

    def unpack_sol(
        self, sol: Float[Tensor, " km1_splx+k_splx+b_k *ch"]
    ) -> tuple[
        Float[Tensor, " k_splx *ch"],
        Float[Tensor, " km1_splx *ch"],
        Float[Tensor, " b_k *ch"],
    ]:
        r"""
        Unpack a solution of the mixed KKT system.

        Parameters
        ----------
        sol : [km1_splx+k_splx+b_k, *ch]
            A solution vector of the mixed KKT system.

        Returns
        -------
        primary_sol : [k_splx, *ch]
            The primal degree-$k$ solution satisfying the gauge constraint
            target; solves the range-projected primary equation if `primary_load`
            is incompatible.
        aux_sol : [km1_splx, *ch]
            The auxiliary degree-$(k-1)$ solution ; equal to the
            codifferential of the `primary_sol` when the auxiliary load is zero.
        kernel_proj : [b_k, *ch]
            The KKT Lagrange multiplier.
        """
        n_km1_splx = self.mixed_block.mass_km1.size(-1)
        n_k_splx = self.mixed_block.mass_k.size(-1)
        return (
            sol[n_km1_splx : n_km1_splx + n_k_splx],
            sol[:n_km1_splx] / math.sqrt(self.mixed_block.down_scale),
            sol[n_km1_splx + n_k_splx :],
        )

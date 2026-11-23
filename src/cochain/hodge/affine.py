from dataclasses import dataclass
from typing import Literal

import torch
from einops import einsum
from jaxtyping import Float
from torch import Tensor

from ..boundary import BoundarySelection
from ..sparse.decoupled_tensor import BaseDecoupledTensor, SparseDecoupledTensor
from .gauge import GaugeFixedSystem, LoadCompatibility, MixedGaugeFixedSystem

# TODO: revisit docstring to flesh out theory


@dataclass(frozen=True)
class HodgeBoundaryData:
    """
    Boundary data for Hodge linear systems.

    Parameters
    ----------
    primary_essential_bd_coef : [constrained_k_splx, *ch]
        Prescribed essential boundary coefficients of the primary $k$-cochain on
        the tangential boundary subcomplex.
    aux_essential_bd_coef : [constrained_km1_splx, *ch]
        Prescribed essential boundary coefficients of the auxiliary $(k-1)$-cochain
        on the tangential boundary subcomplex.
    primary_natural_bd_load : [k_splx, *ch]
        Natural boundary load in the degree-$k$ primary equation.
    aux_natural_bd_load : [km1_splx, *ch]
        Natural boundary load in the degree-$k-1$ auxiliary equation.
    primary_lift : [k_splx, *ch]
        A chosen coefficient lift of `primary_essential_bd_coef` from the tangential
        boundary subcomplex to the full complex.
    aux_lift : [km1_splx, *ch]
        A chosen coefficient lift of `aux_essential_bd_coef` from the tangential
        boundary subcomplex to the full complex.
    """

    primary_essential_bd_coef: Float[Tensor, " constrained_k_splx *ch"] | None = None
    aux_essential_bd_coef: Float[Tensor, " constrained_km1_splx *ch"] | None = None
    primary_natural_bd_load: Float[Tensor, " k_splx *ch"] | None = None
    aux_natural_bd_load: Float[Tensor, " km1_splx *ch"] | None = None
    primary_lift: Float[Tensor, " k_splx *ch"] | None = None
    aux_lift: Float[Tensor, " km1_splx *ch"] | None = None

    @staticmethod
    def resolve_lift(
        k: int,
        bc_sel: BoundarySelection,
        essential_bd_coef: Float[Tensor, " constrained_k_splx *ch"] | None,
        lift: Float[Tensor, " k_splx *ch"] | None,
    ) -> Float[Tensor, " k_splx *ch"] | None:
        """
        Resolve a coefficient lift based on the boundary data specification.

        Parameters
        ----------
        k
            The degree of the boundary data.
        bc_sel
            The specification of the tangential boundary subcomplex.
        essential_bd_coef : [constrained_k_splx, *ch]
            Prescribed essential boundary coefficients on the tangential boundary
            subcomplex.
        lift : [k_splx, *ch]
            A chosen coefficient lift of `essential_bd_coef` from the tangential
            boundary subcomplex to the full complex.

        Returns
        -------
        resolved_lift : [k_splx, *ch]
            The resolved coefficient lift.
        """
        match essential_bd_coef, lift:
            case Tensor(), Tensor():
                # If both the essential boundary coefficients and a lift are
                # provided, check that they agree on the tangential boundary
                # subcomplex, and then return the lift as is.
                lift_constrained = bc_sel.extract_constrained(k, lift)
                if (lift_constrained.shape != essential_bd_coef.shape) or (
                    not torch.allclose(lift_constrained, essential_bd_coef)
                ):
                    raise ValueError(
                        "The essential boundary coefficient is inconsistent with the "
                        "lift on the constrained simplices."
                    )
                return lift
            case Tensor(), None:
                # If only the essential boundary coefficients are provided, perform
                # a zero-interior lift of the boundary coefficients to the full
                # complex.
                return bc_sel.embed_constrained(k, essential_bd_coef)
            case None, Tensor():
                # If only the lift is provided, return the lift as is.
                return lift
            case None, None:
                # If neither the essential boundary coefficients or a lift is
                # specified, return None, which represents the homogeneous boundary
                # condition.
                return None


@dataclass(frozen=True)
class AffineLoad:
    """
    Prepared affine right-hand side load information.

    Parameters
    ----------
    rhs : [sys_dim, *ch]
        The RHS affine load vector. The size of the `sys_dim` depends on the
        type of Hodge system in question. For an `AffineGaugeFixedSystem` of
        degree $k$, the `sys_dim` is `k_splx+b_k`; for an `AffineMixedGaugeFixedSystem`
        of degree $k$, the `sys_dim` is `km1_splx+k_splx+b_k`.
    compatibility
        The primary load vector incompatibility report.
    primary_lift : [k_splx, *ch]
        A chosen coefficient lift of the primary essential boundary coefficients.
    aux_lift : [km1_splx, *ch]
        A chosen coefficient lift of the auxiliary essential boundary coefficients.
    """

    rhs: Float[Tensor, " sys_dim *ch"]
    compatibility: LoadCompatibility
    primary_lift: Float[Tensor, " k_splx *ch"] | None
    aux_lift: Float[Tensor, " km1_splx *ch"] | None = None

    @property
    def dtype(self) -> torch.dtype:
        return self.rhs.dtype

    @property
    def shape(self) -> torch.Size:
        return self.rhs.shape

    def size(self, dim: int | None = None) -> int | torch.Size:
        if dim is None:
            return self.shape
        else:
            return self.shape[dim]


@dataclass(frozen=True)
class AffineGaugeFixedSystem:
    """
    Construct a KKT gauge-fixed Hodge up Laplacian system with inhomogeneous boundary data.

    Parameters
    ----------
    k
        The degree of the original linear system.
    bc_sel
        Specification of the tangential boundary subcomplex.
    gauge_sys : [retained_k_splx+b_k, retained_k_splx+b_k]
        The KKT gauge fixing system.
    cbd_k : [kp1_splx, k_splx]
        The $k$-th coboundary operator.
    mass_k : [k_splx, k_splx]
        The $k$-mass matrix.
    mass_kp1 : [kp1_splx, kp1_splx]
        The $(k+1)$-mass matrix.
    """

    k: int
    bc_sel: BoundarySelection
    gauge_sys: Float[GaugeFixedSystem, "retained_k_splx+b_k retained_k_splx+b_k"]
    cbd_k: Float[SparseDecoupledTensor, "kp1_splx k_splx"]
    mass_k: Float[BaseDecoupledTensor, "k_splx k_splx"]
    mass_kp1: Float[BaseDecoupledTensor, "kp1_splx kp1_splx"]

    @property
    def lhs(
        self,
    ) -> Float[SparseDecoupledTensor, "retained_k_splx+b_k retained_k_splx+b_k"]:
        """The LHS block matrix of the KKT system."""
        return self.gauge_sys.lhs

    def build_rhs(
        self,
        volume_load: Float[Tensor, " k_splx *ch"] | None = None,
        bc_data: HodgeBoundaryData | None = None,
        *,
        atol: Float[Tensor, "*ch"] | float = 0.0,
        rtol: Float[Tensor, "*ch"] | float | Literal["auto"] = "auto",
        incompatibility: Literal["raise", "warn", "project"] = "warn",
    ) -> Float[AffineLoad, " retained_k_splx+b_k *ch"]:
        """
        Construct the RHS load vector of the KKT system.

        Parameters
        ----------
        volume_load : [k_splx, *ch]
            The volume load vector.
        bc_data
            Boundary data for the system.
        atol : [*ch,]
            Absolute tolerance for checking linear system consistency.
        rtol : [*ch,]
            Relative tolerance for checking linear system consistency.
        incompatibility
            What to do if the primary load produces an inconsistent linear system:
            raise a `ValueError` if `'raise'`, print a warning if `'warn'`, or
            ignore the inconsistency if `'project'`.

        Returns
        -------
        affine_load : [retained_k_splx+b_k, *ch]
            The prepared affine right-hand side load information.
        """
        if bc_data is None:
            bc_data = HodgeBoundaryData()

        if (
            (bc_data.aux_essential_bd_coef is not None)
            or (bc_data.aux_natural_bd_load is not None)
            or (bc_data.aux_lift is not None)
        ):
            raise ValueError(
                "Auxiliary boundary data is not supported for an AffineGaugeFixedSystem."
            )

        resolved_primary_lift = bc_data.resolve_lift(
            self.k,
            self.bc_sel,
            bc_data.primary_essential_bd_coef,
            bc_data.primary_lift,
        )

        # Construct the primary load.
        primary_load_terms = []
        if volume_load is not None:
            primary_load_terms.append(volume_load)
        if bc_data.primary_natural_bd_load is not None:
            primary_load_terms.append(bc_data.primary_natural_bd_load)
        if resolved_primary_lift is not None:
            primary_load_terms.append(
                -(self.cbd_k.T @ (self.mass_kp1 @ (self.cbd_k @ resolved_primary_lift)))
            )

        if len(primary_load_terms) > 0:
            primary_load = self.bc_sel.extract_retained(
                self.k, sum(primary_load_terms[1:], primary_load_terms[0])
            )
        else:
            primary_load = torch.zeros(
                self.mass_k.size(-1),
                dtype=self.gauge_sys.dtype,
                device=self.gauge_sys.device,
            )

        # Construct the gauge constraint.
        if (resolved_primary_lift is not None) and (self.gauge_sys.b_k > 0):
            gauge_constraint = -einsum(
                self.gauge_sys.kernel,
                self.bc_sel.extract_retained(
                    self.k, self.mass_k @ resolved_primary_lift
                ),
                "retained_k_splx b_k, retained_k_splx ... -> b_k ...",
            )
        else:
            gauge_constraint = None

        # Construct the RHS.
        rhs, load_compatibility = self.gauge_sys.build_rhs(
            primary_load, gauge_constraint, atol, rtol, incompatibility
        )
        affine_load = AffineLoad(rhs, load_compatibility, resolved_primary_lift, None)

        return affine_load

    def reconstruct_sol(
        self,
        sol: Float[Tensor, " retained_k_splx+b_k *ch"],
        affine_load: Float[AffineLoad, " retained_k_splx+b_k *ch"],
    ) -> tuple[Float[Tensor, " k_splx *ch"], Float[Tensor, " b_k *ch"]]:
        """
        Reconstruct the full solution to the KKT system.

        Parameters
        ----------
        sol : [retained_k_splx+b_k, *ch]
            A solution to the shifted KKT system.
        affine_load : [retained_k_splx+b_k, *ch]
            The prepared affine right-hand side load information.

        Returns
        -------
        primary_sol : [k_splx, *ch]
            The solution to the original linear system; if the primary load is
            incompatible, this is the solution to the range-projected linear
            system.
        kernel_proj : [b_k, *ch]
            The KKT Lagrange multiplier; when the kernel basis is orthonormal
            with respect to the mass, this is also the projection of the primary
            load onto the kernel basis.
        """
        primary_sol, kernel_proj = self.gauge_sys.unpack_sol(sol)

        if affine_load.primary_lift is None:
            full_primary_sol = self.bc_sel.embed_retained(self.k, primary_sol)
        else:
            full_primary_sol = (
                self.bc_sel.embed_retained(self.k, primary_sol)
                + affine_load.primary_lift
            )

        return full_primary_sol, kernel_proj


@dataclass(frozen=True)
class AffineMixedGaugeFixedSystem:
    """
    Construct a KKT gauge-fixed Hodge system with inhomogeneous boundary data.

    Parameters
    ----------
    k
        The degree of the original linear system.
    bc_sel
        Specification of the tangential boundary subcomplex.
    gauge_sys : [retained_km1_splx+retained_k_splx+b_k, retained_km1_splx+retained_k_splx+b_k]
        The KKT gauge-fixed Hodge system.
    cbd_km1: [k_splx, km1_splx]
        The $(k-1)$-coboundary operator.
    cbd_k: [kp1_splx, k_splx]
        The $k$-coboundary operator.
    mass_km1: [km1_splx, km1_splx]
        The consistent mass matrix for discrete $(k-1)$-forms.
    mass_k: [k_splx, k_splx]
        The consistent mass matrix or diagonal Hodge star for discrete $k$-forms.
    mass_kp1: [kp1_splx, kp1_splx]
        The consistent mass matrix or diagonal Hodge star for discrete $(k+1)$-forms.
    """

    k: int
    bc_sel: BoundarySelection
    gauge_sys: Float[
        MixedGaugeFixedSystem,
        "retained_km1_splx+retained_k_splx+b_k retained_km1_splx+retained_k_splx+b_k",
    ]
    cbd_km1: Float[SparseDecoupledTensor, "k_splx km1_splx"]
    cbd_k: Float[SparseDecoupledTensor, "kp1_splx k_splx"] | None
    mass_km1: Float[SparseDecoupledTensor, "km1_splx km1_splx"]
    mass_k: Float[BaseDecoupledTensor, "k_splx k_splx"]
    mass_kp1: Float[BaseDecoupledTensor, "kp1_splx kp1_splx"] | None

    def __post_init__(self):
        null_cbd_k = self.cbd_k is None
        null_mass_kp1 = self.mass_kp1 is None

        if null_cbd_k != null_mass_kp1:
            raise ValueError(
                "'cbd_k' and 'mass_kp1' must both be None or neither be None."
            )

        object.__setattr__(self, "down_only", null_cbd_k)

        if self.down_only != self.gauge_sys.mixed_block.down_only:
            raise ValueError(
                "The input 'cbd_k' and 'mass_kp1' arguments are inconsistent with "
                "the underlying mixed Hodge Laplacian block system."
            )

    @property
    def lhs(
        self,
    ) -> Float[
        SparseDecoupledTensor,
        "retained_km1_splx+retained_k_splx+b_k retained_km1_splx+retained_k_splx+b_k",
    ]:
        """The LHS block matrix of the KKT system."""
        return self.gauge_sys.lhs

    def build_rhs(
        self,
        volume_load: Float[Tensor, " k_splx *ch"] | None = None,
        bc_data: HodgeBoundaryData | None = None,
        *,
        atol: Float[Tensor, "*ch"] | float = 0.0,
        rtol: Float[Tensor, "*ch"] | float | Literal["auto"] = "auto",
        incompatibility: Literal["raise", "warn", "project"] = "warn",
    ) -> Float[AffineLoad, " retained_km1_splx+retained_k_splx+b_k *ch"]:
        """
        Construct the RHS load vector of the KKT system.

        Parameters
        ----------
        volume_load : [k_splx, *ch]
            The volume load vector.
        bc_data
            Boundary data for the system.
        atol : [*ch,]
            Absolute tolerance for checking linear system consistency.
        rtol : [*ch,]
            Relative tolerance for checking linear system consistency.
        incompatibility
            What to do if the primary load produces an inconsistent linear system:
            raise a `ValueError` if `'raise'`, print a warning if `'warn'`, or
            ignore the inconsistency if `'project'`.

        Returns
        -------
        affine_load : [retained_km1_splx+retained_k_splx+b_k, *ch]
            The prepared affine right-hand side load information.
        """
        if bc_data is None:
            bc_data = HodgeBoundaryData()

        if self.down_only and (bc_data.primary_natural_bd_load is not None):
            raise ValueError(
                "A non-trivial Primary natural boundary load is incompatible with "
                "a weak down Hodge Laplacian system."
            )

        resolved_primary_lift = bc_data.resolve_lift(
            self.k,
            self.bc_sel,
            bc_data.primary_essential_bd_coef,
            bc_data.primary_lift,
        )
        resolved_aux_lift = bc_data.resolve_lift(
            self.k - 1,
            self.bc_sel,
            bc_data.aux_essential_bd_coef,
            bc_data.aux_lift,
        )

        # Construct the primary load.
        primary_load_terms = []
        if volume_load is not None:
            primary_load_terms.append(volume_load)
        if bc_data.primary_natural_bd_load is not None:
            primary_load_terms.append(bc_data.primary_natural_bd_load)
        if resolved_aux_lift is not None:
            primary_load_terms.append(
                -self.gauge_sys.mixed_block.down_scale
                * (self.mass_k @ (self.cbd_km1 @ resolved_aux_lift))
            )
        if (not self.down_only) and (resolved_primary_lift is not None):
            primary_load_terms.append(
                -self.gauge_sys.mixed_block.up_scale
                * (
                    self.cbd_k.T
                    @ (self.mass_kp1 @ (self.cbd_k @ resolved_primary_lift))
                )
            )

        if len(primary_load_terms) > 0:
            primary_load = self.bc_sel.extract_retained(
                self.k, sum(primary_load_terms[1:], primary_load_terms[0])
            )
        else:
            ch_dims = ()
            for attr in bc_data.__dict__.values():
                if isinstance(attr, Tensor):
                    ch_dims = attr.shape[1:]

            primary_load = torch.zeros(
                (self.mass_k.size(-1), *ch_dims),
                dtype=self.gauge_sys.dtype,
                device=self.gauge_sys.device,
            )

        # Construct the auxiliary load.
        aux_load_terms = []
        if bc_data.aux_natural_bd_load is not None:
            aux_load_terms.append(bc_data.aux_natural_bd_load)
        if resolved_aux_lift is not None:
            aux_load_terms.append(self.mass_km1 @ resolved_aux_lift)
        if resolved_primary_lift is not None:
            aux_load_terms.append(
                -(self.cbd_km1.T @ (self.mass_k @ resolved_primary_lift))
            )

        if len(aux_load_terms) > 0:
            aux_load = self.bc_sel.extract_retained(
                self.k - 1, sum(aux_load_terms[1:], aux_load_terms[0])
            )
        else:
            aux_load = None

        # Construct the gauge constraint
        if (resolved_primary_lift is not None) and (self.gauge_sys.b_k > 0):
            gauge_constraint = -einsum(
                self.gauge_sys.kernel,
                self.bc_sel.extract_retained(
                    self.k, self.mass_k @ resolved_primary_lift
                ),
                "retained_k_splx b_k, retained_k_splx ... -> b_k ...",
            )
        else:
            gauge_constraint = None

        # Assemble the RHS.
        rhs, load_compatibility = self.gauge_sys.build_rhs(
            primary_load, aux_load, gauge_constraint, atol, rtol, incompatibility
        )
        affine_load = AffineLoad(
            rhs, load_compatibility, resolved_primary_lift, resolved_aux_lift
        )

        return affine_load

    def reconstruct_sol(
        self,
        sol: Float[Tensor, " retained_km1_splx+retained_k_splx+b_k *ch"],
        affine_load: Float[AffineLoad, " retained_km1_splx+retained_k_splx+b_k *ch"],
    ) -> tuple[
        Float[Tensor, " k_splx *ch"],
        Float[Tensor, " km1_splx *ch"],
        Float[Tensor, " b_k *ch"],
    ]:
        """
        Reconstruct the full solution to the KKT system.

        Parameters
        ----------
        sol : [retained_km1_splx+retained_k_splx+b_k, *ch]
            A solution to the shifted KKT system.
        affine_load : [retained_km1_splx+retained_k_splx+b_k, *ch]
            The prepared affine right-hand side load information.

        Returns
        -------
        primary_sol : [k_splx, *ch]
            The solution to the original linear system; if the primary load is
            incompatible, this is the solution to the range-projected linear
            system.
        aux_sol : [km1_splx, *ch]
            The codifferential of the `primary_sol` when the auxiliary load
            vector is zero, or the reconstructed full auxiliary cochain.
        kernel_proj : [b_k, *ch]
            The KKT Lagrange multiplier; when the kernel basis is $M_k$-orthonormal,
            this is also the projection of the primary load onto the kernel basis.
        """
        primary_sol, aux_sol, kernel_proj = self.gauge_sys.unpack_sol(sol)

        if affine_load.primary_lift is None:
            full_primary_sol = self.bc_sel.embed_retained(self.k, primary_sol)
        else:
            full_primary_sol = (
                self.bc_sel.embed_retained(self.k, primary_sol)
                + affine_load.primary_lift
            )

        if affine_load.aux_lift is None:
            full_aux_sol = self.bc_sel.embed_retained(self.k - 1, aux_sol)
        else:
            full_aux_sol = (
                self.bc_sel.embed_retained(self.k - 1, aux_sol) + affine_load.aux_lift
            )

        return full_primary_sol, full_aux_sol, kernel_proj

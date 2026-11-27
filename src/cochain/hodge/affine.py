from dataclasses import dataclass
from typing import Literal

import torch
from einops import einsum
from jaxtyping import Float
from torch import Tensor

from ..boundary import BoundarySelection
from ..sparse.decoupled_tensor import (
    BaseDecoupledTensor,
    DiagDecoupledTensor,
    SparseDecoupledTensor,
)
from ..utils.parsing import dtype_allclose
from .gauge import GaugeFixedSystem, LoadCompatibility, MixedGaugeFixedSystem

# TODO: revisit docstring to flesh out theory


@dataclass(frozen=True)
class HodgeBoundaryData:
    r"""
    Coefficient-level inhomogeneous boundary data for Hodge linear systems.

    All the fields of this dataclass default to `None`, which represents homogeneous
    (zero) boundary data.

    Parameters
    ----------
    primary_essential_bd_coef : [constrained_k_splx, *ch]
        Prescribed essential boundary coefficients of the primary $k$-cochain on
        the tangential boundary subcomplex.
    aux_essential_bd_coef : [constrained_km1_splx, *ch]
        Prescribed essential boundary coefficients of the auxiliary $(k-1)$-cochain
        on the tangential boundary subcomplex.
    primary_natural_bd_load : [k_splx, *ch]
        Preassembled weak natural boundary load in the degree-$k$ primary equation.
    aux_natural_bd_load : [km1_splx, *ch]
        Preassembled weak natural boundary load in the degree-$(k-1)$ auxiliary equation.
    primary_lift : [k_splx, *ch]
        Optional full-complex coefficient lift matching `primary_essential_bd_coef`
        on the tangential boundary subcomplex. If omitted when `primary_essential_bd_coef`
        is provided, a zero-interior lift is constructed automatically.
    aux_lift : [km1_splx, *ch]
        Optional full-complex coefficient lift matching `aux_essential_bd_coef`
        on the tangential boundary subcomplex. If omitted when `aux_essential_bd_coef`
        is provided, a zero-interior lift is constructed automatically.
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
        r"""
        Resolve a full-complex coefficient lift from essential boundary data.

        Parameters
        ----------
        k
            Cochain degree of the boundary data.
        bc_sel
            Specification of the tangential boundary subcomplex.
        essential_bd_coef : [constrained_k_splx, *ch]
            Prescribed essential boundary coefficients on the tangential boundary
            subcomplex.
        lift : [k_splx, *ch]
            Optional full-complex coefficient lift of `essential_bd_coef`. If
            both `essential_bd_coef` and `lift` are provided, their agreement on
            constrained simplices is validated.

        Returns
        -------
        resolved_lift : [k_splx, *ch]
            The resolved full-complex coefficient lift, or `None` if both inputs
            are `None`, which represents the homogeneous essential boundary condition.
        """
        match essential_bd_coef, lift:
            case Tensor(), Tensor():
                # If both the essential boundary coefficients and a lift are
                # provided, check that they agree on the tangential boundary
                # subcomplex, and then return the lift as is.
                lift_constrained = bc_sel.extract_constrained(k, lift)
                if (lift_constrained.shape != essential_bd_coef.shape) or (
                    not dtype_allclose(lift_constrained, essential_bd_coef)
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
    r"""
    Prepared affine right-hand side bundle for a shifted KKT Hodge solve.

    Parameters
    ----------
    rhs : [sys_dim, *ch]
        The assembled RHS vector for the reduced KKT system. `sys_dim` is
        `retained_k_splx + b_k` for `AffineGaugeFixedSystem` and
        `retained_km1_splx + retained_k_splx + b_k` for
        `AffineMixedGaugeFixedSystem`.
    compatibility
        Compatibility report for the complete shifted primary load.
    primary_lift : [k_splx, *ch]
        The resolved full-complex primary lift used to form `rhs`, or `None` for
        homogeneous primary essential boundary conditions.
    aux_lift : [km1_splx, *ch]
        The resolved full-complex auxiliary lift used to form `rhs`, or `None`
        when unused or homogeneous.
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
class AffineUpGaugeFixedSystem:
    r"""
    Construct a KKT gauge-fixed weak up-Laplacian system with inhomogeneous boundary data.

    Parameters
    ----------
    k
        Cochain degree of the primary unknown.
    bc_sel
        Specification of the tangential boundary subcomplex.
    gauge_sys : [retained_k_splx+b_k, retained_k_splx+b_k]
        The restricted KKT gauge-fixed system.
    full_op : [k_splx, k_splx]
        The full-complex weak up-Laplacian operator.
    mass_k : [k_splx, k_splx]
        The full-complex $k$-mass matrix.
    """

    k: int
    bc_sel: BoundarySelection
    gauge_sys: Float[GaugeFixedSystem, "retained_k_splx+b_k retained_k_splx+b_k"]
    full_op: Float[SparseDecoupledTensor, "k_splx k_splx"]
    mass_k: Float[BaseDecoupledTensor, "k_splx k_splx"]

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
        gauge_constraint: Float[Tensor, " b_k *ch"] | None = None,
        *,
        atol: Float[Tensor, "*ch"] | float = 0.0,
        rtol: Float[Tensor, "*ch"] | float | Literal["auto"] = "auto",
        incompatibility: Literal["raise", "warn", "project"] = "warn",
    ) -> Float[AffineLoad, " retained_k_splx+b_k *ch"]:
        r"""
        Construct the shifted KKT RHS bundle from volume and boundary data.

        Parameters
        ----------
        volume_load : [k_splx, *ch]
            Optional full-complex weak volume/domain load; defaults to zero.
        bc_data
            Optional boundary data for the system; defaults to homogeneous boundary
            conditions.
        gauge_constraint : [b_k, *ch]
            Optional target for the gauge constraint; defaults to zeros, which
            select the minimum-norm representative. This argument is ignored if
            `b_k` is zero.
        atol : [*ch,]
            Absolute tolerance for checking compatibility of the shifted primary
            load.
        rtol : [*ch,]
            Relative tolerance for checking compatibility of the shifted primary
            load; if `'auto'`, uses the square root of the `primary_load` dtype
            machine epsilon.
        incompatibility
            Policy when the shifted primary load is incompatible with `self.gauge_sys`:
            raise a `ValueError` if `'raise'`, emit a `UserWarning` if `'warn'`,
            or proceed silently if `'project'`.

        Returns
        -------
        affine_load : [retained_k_splx+b_k, *ch]
            Prepared `AffineLoad` containing the shifted KKT RHS, compatibility
            report, and resolved primary lift.
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
            # Note that gauge_sys.op is already restricted.
            primary_load_terms.append(-self.full_op @ resolved_primary_lift)

        if len(primary_load_terms) > 0:
            primary_load = self.bc_sel.extract_retained(
                self.k, sum(primary_load_terms[1:], primary_load_terms[0])
            )
        else:
            primary_load = torch.zeros(
                self.bc_sel.n_retained_splx[self.k],
                dtype=self.gauge_sys.dtype,
                device=self.gauge_sys.device,
            )

        # Construct the gauge constraint.
        if (resolved_primary_lift is not None) and (self.gauge_sys.b_k > 0):
            lift_proj = -einsum(
                self.gauge_sys.kernel,
                self.bc_sel.extract_retained(
                    self.k, self.mass_k @ resolved_primary_lift
                ),
                "retained_k_splx b_k, retained_k_splx ... -> b_k ...",
            )
        else:
            lift_proj = None

        if self.gauge_sys.b_k == 0:
            gauge_constraint = None

        match gauge_constraint, lift_proj:
            case Tensor(), Tensor():
                gauge_constraint = gauge_constraint + lift_proj
            case Tensor(), None:
                pass
            case None, Tensor():
                gauge_constraint = lift_proj
            case None, None:
                pass

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
        r"""
        Reconstruct the full-complex primary solution from the reduced KKT solution.

        Parameters
        ----------
        sol : [retained_k_splx+b_k, *ch]
            Solution vector of the reduced shifted KKT system.
        affine_load : [retained_k_splx+b_k, *ch]
            The `AffineLoad` returned by `build_rhs()` for the solved system.

        Returns
        -------
        primary_sol : [k_splx, *ch]
            The reconstructed full-complex primary solution satisfying the prescribed
            essential boundary values and total-field gauge condition; solves the
            range-projected equation if the shifted primary load was incompatible.
        kernel_proj : [b_k, *ch]
            The KKT Lagrange multiplier.
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


def _build_weak_laplacian_rhs(
    k: int,
    b_k: int,
    cbd_km1: Float[SparseDecoupledTensor, "k_splx km1_splx"],
    cbd_k: Float[SparseDecoupledTensor, "kp1_splx k_splx"] | None,
    mass_km1: Float[BaseDecoupledTensor, "km1_splx km1_splx"],
    mass_k: Float[BaseDecoupledTensor, "k_splx k_splx"],
    mass_kp1: Float[BaseDecoupledTensor, "kp1_splx kp1_splx"] | None,
    down_scale: float,
    up_scale: float,
    kernel: Float[Tensor, " b_k *ch"],
    bc_sel: BoundarySelection,
    volume_load: Float[Tensor, " k_splx *ch"] | None,
    bc_data: HodgeBoundaryData | None,
    gauge_constraint: Float[Tensor, " b_k *ch"] | None,
) -> tuple[
    Float[Tensor, " k_splx *ch"],
    Float[Tensor, " km1_splx *ch"],
    Float[Tensor, " retained_k_splx *ch"],
    Float[Tensor, " retained_km1_splx *ch"],
    Float[Tensor, " b_k *ch"],
]:
    if bc_data is None:
        bc_data = HodgeBoundaryData()

    resolved_primary_lift = bc_data.resolve_lift(
        k,
        bc_sel,
        bc_data.primary_essential_bd_coef,
        bc_data.primary_lift,
    )
    resolved_aux_lift = bc_data.resolve_lift(
        k - 1,
        bc_sel,
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
            -down_scale * (mass_k @ (cbd_km1 @ resolved_aux_lift))
        )
    if (
        (cbd_k is not None)
        and (mass_kp1 is not None)
        and (resolved_primary_lift is not None)
    ):
        primary_load_terms.append(
            -up_scale * (cbd_k.T @ (mass_kp1 @ (cbd_k @ resolved_primary_lift)))
        )

    if len(primary_load_terms) > 0:
        primary_load = bc_sel.extract_retained(
            k, sum(primary_load_terms[1:], primary_load_terms[0])
        )
    else:
        ch_dims = ()
        for attr in bc_data.__dict__.values():
            if isinstance(attr, Tensor):
                ch_dims = attr.shape[1:]
                break

        primary_load = torch.zeros(
            (bc_sel.n_retained_splx[k], *ch_dims),
            dtype=mass_k.dtype,
            device=mass_k.device,
        )

    # Construct the auxiliary load.
    aux_load_terms = []
    if bc_data.aux_natural_bd_load is not None:
        aux_load_terms.append(bc_data.aux_natural_bd_load)
    if resolved_aux_lift is not None:
        aux_load_terms.append(mass_km1 @ resolved_aux_lift)
    if resolved_primary_lift is not None:
        aux_load_terms.append(-(cbd_km1.T @ (mass_k @ resolved_primary_lift)))

    if len(aux_load_terms) > 0:
        aux_load = bc_sel.extract_retained(
            k - 1, sum(aux_load_terms[1:], aux_load_terms[0])
        )
    else:
        aux_load = None

    # Construct the gauge constraint
    if (resolved_primary_lift is not None) and (b_k > 0):
        lift_proj = -einsum(
            kernel,
            bc_sel.extract_retained(k, mass_k @ resolved_primary_lift),
            "retained_k_splx b_k, retained_k_splx ... -> b_k ...",
        )
    else:
        lift_proj = None

    if b_k == 0:
        gauge_constraint = None

    match gauge_constraint, lift_proj:
        case Tensor(), Tensor():
            gauge_constraint = gauge_constraint + lift_proj
        case Tensor(), None:
            pass
        case None, Tensor():
            gauge_constraint = lift_proj
        case None, None:
            pass

    return (
        resolved_primary_lift,
        resolved_aux_lift,
        primary_load,
        aux_load,
        gauge_constraint,
    )


@dataclass(frozen=True)
class AffineSchurGaugeFixedSystem:
    r"""
    Construct a KKT gauge-fixed Schur-complement Hodge system with inhomogeneous boundary data.

    Parameters
    ----------
    k
        Cochain degree of the primary unknown ($k \ge 1$).
    bc_sel
        Specification of the tangential boundary subcomplex.
    gauge_sys : [retained_k_splx+b_k, retained_k_splx+b_k]
        The restricted mixed KKT gauge-fixed Schur-complement Hodge system.
    cbd_km1 : [k_splx, km1_splx]
        The full-complex $(k-1)$-coboundary operator.
    cbd_km1_retained : [retained_k_splx, retained_km1_splx]
        The restricted $(k-1)$-coboundary operator.
    cbd_k : [kp1_splx, k_splx]
        The full-complex $k$-coboundary operator.
    mass_km1 : [km1_splx, km1_splx]
        The full-complex consistent mass matrix for discrete $(k-1)$-forms.
    mass_km1_restricted : [retained_km1_splx, retained_km1_splx]
        The restricted $(k-1)$-mass matrix.
    mass_k : [k_splx, k_splx]
        The full-complex consistent mass matrix or diagonal Hodge star for discrete
        $k$-forms.
    mass_k_restricted : [retained_k_splx, retained_k_splx]
        The restricted $k$-mass matrix/diagonal Hodge star.
    mass_kp1 : [kp1_splx, kp1_splx]
        The full-complex consistent mass matrix or diagonal Hodge star for discrete
        $(k+1)$-forms.
    """

    k: int
    bc_sel: BoundarySelection
    gauge_sys: Float[GaugeFixedSystem, "retained_k_splx+b_k retained_k_splx+b_k"]
    cbd_km1: Float[SparseDecoupledTensor, "k_splx km1_splx"]
    cbd_km1_retained: Float[SparseDecoupledTensor, "retained_k_splx retained_km1_splx"]
    cbd_k: Float[SparseDecoupledTensor, "kp1_splx k_splx"] | None
    mass_km1: Float[DiagDecoupledTensor, "km1_splx km1_splx"]
    mass_km1_retained: Float[DiagDecoupledTensor, "retained_km1_splx retained_km1_splx"]
    mass_k: Float[BaseDecoupledTensor, "k_splx k_splx"]
    mass_k_restricted: Float[BaseDecoupledTensor, "retained_k_splx retained_k_splx"]
    mass_kp1: Float[BaseDecoupledTensor, "kp1_splx kp1_splx"] | None

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
        gauge_constraint: Float[Tensor, " b_k *ch"] | None = None,
        *,
        atol: Float[Tensor, "*ch"] | float = 0.0,
        rtol: Float[Tensor, "*ch"] | float | Literal["auto"] = "auto",
        incompatibility: Literal["raise", "warn", "project"] = "warn",
    ) -> Float[AffineLoad, " retained_k_splx+b_k *ch"]:
        r"""
        Construct the shifted mixed KKT RHS bundle from volume and boundary data.

        Parameters
        ----------
        volume_load : [k_splx, *ch]
            Optional full-complex weak volume/domain load. Defaults to zero.
        bc_data
            Optional boundary data for the system; defaults to homogeneous boundary
            condition.
        gauge_constraint : [b_k, *ch]
            Optional target for the gauge constraint; defaults to zeros, which
            select the minimum-norm representative. This argument is ignored if
            `b_k` is zero.
        atol : [*ch,]
            Absolute tolerance for checking compatibility of the shifted primary
            load.
        rtol : [*ch,]
            Relative tolerance for checking compatibility of the shifted primary
            load; if `'auto'`, uses the square root of the `primary_load` dtype
            machine epsilon.
        incompatibility
            Policy when the shifted primary load is incompatible with `self.gauge_sys`:
            raise a `ValueError` if `'raise'`, emit a `UserWarning` if `'warn'`,
            or proceed silently if `'project'`.

        Returns
        -------
        affine_load : [retained_k_splx+b_k, *ch]
            Prepared `AffineLoad` containing the shifted Schur-complement KKT RHS,
            compatibility report, and resolved primary and auxiliary lifts.
        """
        (
            resolved_primary_lift,
            resolved_aux_lift,
            primary_load,
            aux_load,
            gauge_constraint,
        ) = _build_weak_laplacian_rhs(
            k=self.k,
            b_k=self.gauge_sys.b_k,
            cbd_km1=self.cbd_km1,
            cbd_k=self.cbd_k,
            mass_km1=self.mass_km1,
            mass_k=self.mass_k,
            mass_kp1=self.mass_kp1,
            down_scale=1.0,
            up_scale=1.0,
            kernel=self.gauge_sys.kernel,
            bc_sel=self.bc_sel,
            volume_load=volume_load,
            bc_data=bc_data,
            gauge_constraint=gauge_constraint,
        )

        combined_primary_load = primary_load + self.mass_k_restricted @ (
            self.cbd_km1_retained @ (self.mass_km1_retained.inv @ aux_load)
        )

        # Assemble the RHS.
        rhs, load_compatibility = self.gauge_sys.build_rhs(
            combined_primary_load,
            gauge_constraint,
            atol,
            rtol,
            incompatibility,
        )
        affine_load = AffineLoad(
            rhs, load_compatibility, resolved_primary_lift, resolved_aux_lift
        )

        return affine_load

    def reconstruct_sol(
        self,
        sol: Float[Tensor, " retained_k_splx+b_k *ch"],
        affine_load: Float[AffineLoad, " retained_k_splx+b_k *ch"],
    ) -> tuple[
        Float[Tensor, " k_splx *ch"],
        Float[Tensor, " km1_splx *ch"],
        Float[Tensor, " b_k *ch"],
    ]:
        r"""
        Reconstruct the full-complex primary solution from the reduced KKT solution.

        Parameters
        ----------
        sol : [retained_k_splx+b_k, *ch]
            Solution vector of the reduced shifted KKT system.
        affine_load : [retained_k_splx+b_k, *ch]
            The `AffineLoad` returned by `build_rhs()` for the solved system.

        Returns
        -------
        primary_sol : [k_splx, *ch]
            The reconstructed full-complex primary solution satisfying the prescribed
            essential boundary values and total-field gauge condition; solves the
            range-projected equation if the shifted primary load was incompatible.
        kernel_proj : [b_k, *ch]
            The KKT Lagrange multiplier.
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
    r"""
    Construct a KKT gauge-fixed mixed Hodge system with inhomogeneous boundary data.

    Parameters
    ----------
    k
        Cochain degree of the primary unknown ($k \ge 1$).
    bc_sel
        Specification of the tangential boundary subcomplex.
    gauge_sys : [retained_km1_splx+retained_k_splx+b_k, retained_km1_splx+retained_k_splx+b_k]
        The restricted mixed KKT gauge-fixed Hodge system.
    cbd_km1 : [k_splx, km1_splx]
        The full-complex $(k-1)$-coboundary operator.
    cbd_k : [kp1_splx, k_splx]
        The full-complex $k$-coboundary operator.
    mass_km1 : [km1_splx, km1_splx]
        The full-complex consistent mass matrix for discrete $(k-1)$-forms.
    mass_k : [k_splx, k_splx]
        The full-complex consistent mass matrix or diagonal Hodge star for discrete
        $k$-forms.
    mass_kp1 : [kp1_splx, kp1_splx]
        The full-complex consistent mass matrix or diagonal Hodge star for discrete
        $(k+1)$-forms.
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

        down_only = null_cbd_k
        if down_only != self.gauge_sys.mixed_block.down_only:
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
        gauge_constraint: Float[Tensor, " b_k *ch"] | None = None,
        *,
        atol: Float[Tensor, "*ch"] | float = 0.0,
        rtol: Float[Tensor, "*ch"] | float | Literal["auto"] = "auto",
        incompatibility: Literal["raise", "warn", "project"] = "warn",
    ) -> Float[AffineLoad, " retained_km1_splx+retained_k_splx+b_k *ch"]:
        r"""
        Construct the shifted mixed KKT RHS bundle from volume and boundary data.

        Parameters
        ----------
        volume_load : [k_splx, *ch]
            Optional full-complex weak volume/domain load. Defaults to zero.
        bc_data
            Optional boundary data for the system; defaults to homogeneous boundary
            condition.
        gauge_constraint : [b_k, *ch]
            Optional target for the gauge constraint; defaults to zeros, which
            select the minimum-norm representative. This argument is ignored if
            `b_k` is zero.
        atol : [*ch,]
            Absolute tolerance for checking compatibility of the shifted primary
            load.
        rtol : [*ch,]
            Relative tolerance for checking compatibility of the shifted primary
            load; if `'auto'`, uses the square root of the `primary_load` dtype
            machine epsilon.
        incompatibility
            Policy when the shifted primary load is incompatible with `self.gauge_sys`:
            raise a `ValueError` if `'raise'`, emit a `UserWarning` if `'warn'`,
            or proceed silently if `'project'`.

        Returns
        -------
        affine_load : [retained_km1_splx+retained_k_splx+b_k, *ch]
            Prepared `AffineLoad` containing the shifted mixed KKT RHS,
            compatibility report, and resolved primary and auxiliary lifts.
        """
        if self.down_only and (bc_data.primary_natural_bd_load is not None):
            raise ValueError(
                "A non-trivial primary natural boundary load is incompatible with "
                "a weak down Hodge Laplacian system."
            )

        (
            resolved_primary_lift,
            resolved_aux_lift,
            primary_load,
            aux_load,
            gauge_constraint,
        ) = _build_weak_laplacian_rhs(
            k=self.k,
            b_k=self.gauge_sys.b_k,
            cbd_km1=self.cbd_km1,
            cbd_k=self.cbd_k,
            mass_km1=self.mass_km1,
            mass_k=self.mass_k,
            mass_kp1=self.mass_kp1,
            down_scale=self.gauge_sys.mixed_block.down_scale,
            up_scale=self.gauge_sys.mixed_block.up_scale,
            kernel=self.gauge_sys.kernel,
            bc_sel=self.bc_sel,
            volume_load=volume_load,
            bc_data=bc_data,
            gauge_constraint=gauge_constraint,
        )

        # Assemble the RHS.
        rhs, load_compatibility = self.gauge_sys.build_rhs(
            primary_load,
            aux_load,
            gauge_constraint,
            atol,
            rtol,
            incompatibility,
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
        r"""
        Reconstruct the full-complex primary and auxiliary solutions from the reduced KKT solution.

        Parameters
        ----------
        sol : [retained_km1_splx+retained_k_splx+b_k, *ch]
            Solution vector of the reduced shifted mixed KKT system.
        affine_load : [retained_km1_splx+retained_k_splx+b_k, *ch]
            The `AffineLoad` returned by `build_rhs()` for the solved system.

        Returns
        -------
        primary_sol : [k_splx, *ch]
            The reconstructed full-complex primary solution satisfying the prescribed
            primary essential boundary values and total-field gauge condition; solves
            the range-projected primary equation if the shifted primary load was
            incompatible.
        aux_sol : [km1_splx, *ch]
            The reconstructed full-complex auxiliary solution satisfying the
            prescribed auxiliary essential boundary values and weak codifferential
            relation.
        kernel_proj : [b_k, *ch]
            The KKT Lagrange multiplier.
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

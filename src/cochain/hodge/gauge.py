import warnings
from dataclasses import dataclass
from typing import Literal

import torch
from jaxtyping import Bool, Float
from torch import Tensor

from ..sparse.decoupled_tensor import BaseDecoupledTensor, SparseDecoupledTensor

# TODO: revisit docstring to flesh out theory


@dataclass(frozen=True)
class LoadCompatibility:
    """
    Primary load vector incompatibility report.

    Parameters
    ----------
    kernel_proj : [b_k, *ch]
        The projection of the primary load onto the kernel basis.
    norm : [*ch,]
        The Euclidean norm of the column vectors of `kernel_proj`.
    tol : [*ch,]
        The tolerance thresholds for the column vectors of `kernel_proj`.
    compatible : [*ch,]
        Whether the column vector norms of `kernel_proj` fall below the `tol`.

    Attributes
    ----------
    all_compatible : [,]
        Whether all column vectors of `kernel_proj` fall below the `tol.
    """

    kernel_proj: Float[Tensor, " b_k *ch"]
    norm: Float[Tensor, "*ch"]
    tol: Float[Tensor, "*ch"] | float
    compatible: Bool[Tensor, "*ch"]

    @property
    def all_compatible(self) -> Bool[Tensor, ""]:
        """Whether all column vectors of `kernel_proj` fall below the `tol."""
        torch.all(self.compatible)


@dataclass(frozen=True)
class GaugeFixedSystem:
    """
    Construct a KKT gauge fixing system for a symmetric positive semidefinite operator.

    Parameters
    ----------
    op : [k_splx, k_splx]
        A real, symmetric positive definite square matrix.
    mass : [k_splx, k_splx]
        A real, symmetric positive definite square matrix that induces an inner
        product on the column space of `op`.
    kernel : [k_splx, b_k]
        A matrix whose column vectors form a basis set for the kernel of `op`
        that is orthonormal w.r.t. the inner product induced by `mass`.
    """

    op: Float[SparseDecoupledTensor, "k_splx k_splx"]
    mass: Float[BaseDecoupledTensor, "k_splx k_splx"]
    kernel: Float[Tensor, "k_splx b_k"]

    def __post_init__(self):
        b_k = self.kernel.size(-1)
        if b_k == 0:
            raise ValueError(
                "The GaugeFixedSystem is inapplicable when the kernel is trivial."
            )

    @property
    def lhs(self) -> Float[SparseDecoupledTensor, "k_splx+b_k k_splx+b_k"]:
        """
        Construct the LHS block matrix of the KKT system.

        Returns
        -------
        lhs : [k_splx+b_k, k_splx+b_k]
            The LHS block matrix of the KKT system.
        """
        m_k = self.mass @ self.kernel
        return SparseDecoupledTensor.bmat([[self.op, m_k], [m_k.T, None]])

    def build_rhs(
        self,
        primary_load: Float[Tensor, " k_splx *ch"],
        gauge_constraint: Float[Tensor, " b_k *ch"] | None = None,
        atol: Float[Tensor, "*ch"] | float = 0.0,
        rtol: Float[Tensor, "*ch"] | float | Literal["auto"] = "auto",
        incompatibility: Literal["raise", "warn", "project"] = "warn",
    ) -> tuple[Float[Tensor, " k_splx+b_k"], LoadCompatibility]:
        """
        Construct the RHS load vector of the KKT system.

        Parameters
        ----------
        primary_load : [k_splx, *ch]
            The primary load vector.
        gauge_constraint : [b_k, *ch]
            The gauge-constraint target for the kernel basis.
        atol : [*ch,]
            Absolute tolerance for checking linear system consistency.
        rtol : [*ch,]
            Relative tolerance for checking linear system consistency; if set to
            "auto", use the square root of the `primary_load` dtype machine epsilon.
        incompatibility
            What to do if the input `primary_load` produces an inconsistent
            linear system in conjunction with the `self.op` operator: raise a
            `ValueError` if `'raise'`, print a warning if `'warn'`, or ignore
            the inconsistency if `'project'`.

        Returns
        -------
        rhs : [k_splx+b_k, *ch]
            The RHS load vector.
        load_compatibility
            A `LoadCompatibility` object that documents how much of the
            `primary_load` is inside the kernel of the operator.
        """
        # Check whether the operator + the primary_load vector constitute a
        # consistent linear system.
        if rtol == "auto":
            rtol = torch.sqrt(torch.finfo(primary_load.dtype).eps)

        # Define the tolerance threshold to be atol + rtol * ||H||_F * ||r_k||_2.
        kernel_F_norm = torch.sqrt(torch.trace(self.kernel.T @ self.kernel))
        primary_load_l2_norm = torch.linalg.norm(primary_load, dim=0)
        tol = atol + rtol * kernel_F_norm * primary_load_l2_norm

        # Check that the channels of ||H.T@r_k||_2 fall below the tolerance threshold.
        # Note that the Euclidean/L2 norm is used here because the incompatible
        # component of load r_k is given by M@H@H.T@r_k, and its inv(M)-norm is
        # ||H.T@r_k||_2 when H is M-orthonormal.
        kernel_proj = self.kernel @ primary_load
        kernel_proj_l2_norm = torch.linalg.norm(kernel_proj, dim=0)
        compatible = kernel_proj_l2_norm <= tol

        load_compatibility = LoadCompatibility(
            kernel_proj=kernel_proj,
            norm=kernel_proj_l2_norm,
            tol=tol,
            compatible=compatible,
        )

        if not load_compatibility.all_compatible:
            msg = (
                "The linear system with 'self.op' as the LHS and "
                "'primary_load' as the RHS is inconsistent."
            )
            match incompatibility:
                case "raise":
                    raise ValueError(msg)
                case "warn":
                    warnings.warn(msg, UserWarning)
                case "project":
                    pass

        # Assemble the full RHS load vector.
        if gauge_constraint is None:
            b_k = self.kernel.size(-1)
            ch_dims = primary_load.shape[1:]
            rhs = torch.vstack(
                [
                    primary_load,
                    torch.zeros(
                        (b_k, *ch_dims),
                        dtype=primary_load.dtype,
                        device=primary_load.device,
                    ),
                ]
            )
        else:
            rhs = torch.vstack([primary_load, gauge_constraint])

        return rhs, load_compatibility

    def unpack_sol(
        self, sol: Float[Tensor, " k_splx+b_k *ch"]
    ) -> tuple[Float[Tensor, " k_splx *ch"], Float[Tensor, " b_k *ch"]]:
        """
        Unpack a solution to the KKT system.

        Parameters
        ----------
        sol : [k_splx+bk, *ch]
            A solution to the KKT system.

        Returns
        -------
        primary_sol : [k_splx, *ch]
            The solution to the original linear system.
        kernel_proj : [b_k, *ch]
            The projection of the primary load onto the kernel basis.
        """
        n_k_splx = self.op.size(-1)
        return sol[:n_k_splx], sol[n_k_splx:]

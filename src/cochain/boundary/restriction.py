from __future__ import annotations

__all__ = ["BoundaryRestrictionCache"]

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TypeVar

from jaxtyping import Float

from ..complex import SimplicialMesh
from ..sparse.decoupled_tensor import (
    BaseDecoupledTensor,
    DiagDecoupledTensor,
    SparseDecoupledTensor,
    SparsityPattern,
    SubmatPlan,
)
from .selection import BoundarySelection

T = TypeVar("T", bound=BaseDecoupledTensor)


@dataclass(frozen=True, eq=False)
class BoundaryRestrictionCache:
    """
    A convenience class for managing the construction of restricted operators.

    For a given `SimplicialMesh` and `BoundarySelection` specification, this class
    holds the restricted coboundary operators, and acts as a cache for managing
    the `SubmatPlan`s of other restricted operators (e.g., the consistent mass
    matrices and the diagonal Hodge star operators).

    Parameters
    ----------
    restricted_cbd
        A tuple of the first three restricted coboundary operators for the mesh
        subjected to the given boundary condition.
    bc_sel
        A `BoundarySelection` object representing the boundary condition.
    cache
        A dictionary that caches the `SubmatPlan`s for any operators that have
        been passed to `restrict_op()`.
    """

    restricted_cbd: tuple[
        Float[SparseDecoupledTensor, "retained_edge retained_vert"],
        Float[SparseDecoupledTensor, "retained_tri retained_edge"],
        Float[SparseDecoupledTensor, "retained_tet retained_tri"],
    ]
    bc_sel: BoundarySelection
    cache: dict[tuple[tuple[int, ...], SparsityPattern | str], SubmatPlan]

    @classmethod
    def from_mesh(
        cls, mesh: SimplicialMesh, bc_sel: BoundarySelection
    ) -> BoundaryRestrictionCache:
        """
        Construct a cache from a mesh with a given boundary condition.

        Parameters
        ----------
        mesh
            A simplicial mesh.
        bc_sel
            A boundary condition defined on the input `mesh` object.

        Returns
        -------
        cache
            A `BoundaryRestrictionCache` object for managing the construction of
            restricted operators for the input mesh and the given boundary condition.
        """
        # Check that the bc_sel is compatible with the mesh.
        if tuple(mask.numel() for mask in bc_sel.constrained_mask) != mesh.n_splx:
            raise ValueError("The input 'mesh' and 'bc_sel' has incompatible shapes.")

        if mesh.device != bc_sel.device:
            raise RuntimeError(
                "The input 'mesh' and 'bc_sel' are not on the same device."
            )

        restricted_cbd = tuple(
            cbd.submatrix(
                bc_sel.retained_mask[deg + 1], bc_sel.retained_mask[deg]
            ).tensor
            for deg, cbd in enumerate(mesh.cbd)
        )
        return cls(restricted_cbd, bc_sel, {})

    def restrict_op(self, deg: int | tuple[int, ...], op: T) -> T:
        """
        Enforce boundary restriction on an operator to extract its submatrix.

        Operators that have been passed to this method will have their `SubmatPlan`
        cached.

        Parameters
        ----------
        deg
            The degrees of the (co)chain groups over which the domain and image
            of the operator is defined. For example, for a k-coboundary operator,
            `deg = (k+1, k)`; for a consistent k-mass matrix, `deg = (k, k)`.
            If the degrees of the domain and image are the same, `deg` can be
            provided simply as an integer (which is preferred).
        op
            The operator over which to enforce boundary restriction. The operator
            must be either as a `SparseDecoupledTensor` or a `DiagDecoupledTensor`.

        Returns
        -------
        restricted_op
            A boundary-restricted version of the input operator.

        Notes
        -----
        Operators such as the Hodge stars that are represented as `DiagDecoupledTensor`s
        are not associated with `SparsityPattern`s that can be used as unique keys
        to identify the operators in the cache. Here, we simply identify such an
        operator with the string "diag" (alongside the given degrees) in the cache.
        This is sufficient because all diagonal operators of the same degrees are
        subject to the same `SubMatPlan` (e.g., a barycentric and circumcentric
        Hodge k-star may have different numerical values on the diagonal, they are
        restricted in the same way under a given boundary condition specification).
        """
        match deg:
            case int():
                deg = (deg,)
            case Sequence():
                deg = tuple(deg)
            case _:
                raise TypeError("Unknown 'deg' argument datatype.")

        if isinstance(op, DiagDecoupledTensor):
            if len(set(deg)) > 1:
                raise ValueError(
                    "Specifying different domain/image degrees for a DiagDecoupledTensor is not supported."
                )

        match op:
            case DiagDecoupledTensor():
                key = (deg, "diag")
            case SparseDecoupledTensor():
                key = (deg, op.pattern)
            case _:
                raise TypeError("Unrecognized 'op' argument datatype.")

        submat_plan = self.cache.get(key, None)

        res = op.submatrix(
            *[self.bc_sel.retained_mask[k] for k in deg], submat_plan=submat_plan
        )

        if submat_plan is None:
            self.cache[key] = res.plan

        return res.tensor

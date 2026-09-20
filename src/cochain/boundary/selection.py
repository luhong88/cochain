__all__ = ["BoundarySelection"]

from dataclasses import dataclass
from functools import cached_property

import torch
from jaxtyping import Bool, Float, Integer
from torch import Tensor

from ..complex import SimplicialMesh
from ..topology.boundaries import detect_mesh_boundaries


@dataclass(frozen=True, eq=False)
class BoundarySelection:
    """
    An immutable dataclass for storing boundary condition selections.

    This class supports the specification of relative, absolute, and mixed
    boundary conditions on a simplicial mesh, and provides automatic downward
    closure of boundary simplex selections and utilities for cochain manipulation
    with respect to the retained/constrained degrees of freedom.

    Parameters
    ----------
    constrained_mask
        a tuple of boolean masks, where `constrained_mask[k]` marks the k-simplices
        representing the tangential degrees of freedom constrained by the boundary
        condition specification.

    Attributes
    ----------
    constrained_idx
        A tuple of integer tensors, where `constrained_idx[k]` contains the indices
        of the k-simplices marked by `constrained_mask[k]`.
    retained_mask
        A tuple of boolean masks, where `retained_mask[k]` marks the k-simplices
        that are retained/unconstrained by the boundary condition specification.
        More specifically, `retained_mask[k]=~constrained_mask[k]`.
    retained_idx
        A tuple of integer tensors, where `retained_idx[k]` contains the indices
        of the k-simplices marked by `retained_mask[k]`.
    """

    constrained_mask: tuple[
        Bool[Tensor, " vert"],
        Bool[Tensor, " edge"],
        Bool[Tensor, " tri"],
        Bool[Tensor, " tet"],
    ]

    def __post_init__(self):
        # Enforce dtype and ownership.
        object.__setattr__(
            self,
            "constrained_mask",
            tuple(
                mask.to(dtype=torch.bool, copy=True) for mask in self.constrained_mask
            ),
        )

    @cached_property
    def constrained_idx(
        self,
    ) -> tuple[
        Integer[Tensor, " constrained_vert"],
        Integer[Tensor, " constrained_edge"],
        Integer[Tensor, " constrained_tri"],
        Integer[Tensor, " constrained_tet"],
    ]:
        return tuple(torch.argwhere(mask).flatten() for mask in self.constrained_mask)

    @cached_property
    def retained_mask(
        self,
    ) -> tuple[
        Bool[Tensor, " vert"],
        Bool[Tensor, " edge"],
        Bool[Tensor, " tri"],
        Bool[Tensor, " tet"],
    ]:
        return tuple(~mask for mask in self.constrained_mask)

    @cached_property
    def retained_idx(
        self,
    ) -> tuple[
        Integer[Tensor, " retained_vert"],
        Integer[Tensor, " retained_edge"],
        Integer[Tensor, " retained_tri"],
        Integer[Tensor, " retained_tet"],
    ]:
        return tuple(torch.argwhere(mask).flatten() for mask in self.retained_mask)

    @classmethod
    def from_absolute_bc(cls, mesh: SimplicialMesh):
        """Create a `BoundarySelection` for a mesh with absolute boundary condition."""
        constrained_masks = tuple(
            torch.zeros(
                n_splx,
                dtype=torch.bool,
                device=mesh.device,
            )
            for n_splx in mesh.n_splx
        )
        return cls(constrained_masks)

    @classmethod
    def from_relative_bc(cls, mesh: SimplicialMesh):
        """Create a `BoundarySelection` for a mesh with relative boundary condition."""
        constrained_masks = detect_mesh_boundaries(mesh.cbd)
        return cls(constrained_masks)

    @classmethod
    def from_tangential_bd_mask(
        cls, mesh: SimplicialMesh, tangent_mask_km1: Bool[Tensor, " km1_splx"]
    ):
        """Create a `BoundarySelection` for a mesh with mixed boundary condition."""
        # Check that the input mask is indeed a subset of the (k-1)-dim bd mask.
        # In logic, (p -> q) is equivalent to (~p | q).
        if not torch.all(~tangent_mask_km1 | mesh.bd_mask[mesh.dim - 1]):
            raise ValueError(
                "The input 'tangent_mask_km1' is not a subset of the "
                "(k-1)-dimensional boundary simplices."
            )

        constrained_masks = detect_mesh_boundaries(
            mesh.cbd, bd_mask_km1=tangent_mask_km1
        )
        return cls(constrained_masks)

    def extract_retained(
        self, k: int, k_cochain: Float[Tensor, " k_splx *ch"]
    ) -> Float[Tensor, " retained_k_splx *ch"]:
        """
        Extract the retained degrees of freedom from a cochain.

        Parameters
        ----------
        k
            The degree of the input cochain.
        k_cochain : [k_splx, *ch]
            The input cochain, with optional trailing channel dimensions.

        Returns
        -------
        retained_cochain : [retained_k_splx, *ch]
            An output cochain that only contains coefficients on the retained
            degrees of freedom.
        """
        return k_cochain[self.retained_mask[k]]

    def embed_retained(
        self, k: int, k_cochain: Float[Tensor, " retained_k_splx *ch"]
    ) -> Float[Tensor, " k_splx *ch"]:
        """
        Lift the coefficients on the retained degrees of freedom to a full cochain.

        Parameters
        ----------
        k
            The degree of the input cochain.
        k_cochain : [retained_k_splx, *ch]
            The input cochain, with optional trailing channel dimensions.

        Returns
        -------
        full_cochain : [k_splx, *ch]
            An output cochain with zero coefficients on constrained k-simplices.
        """
        full_cochain = torch.zeros(
            (self.constrained_mask[k].size(0), *k_cochain.shape[1:]),
            dtype=k_cochain.dtype,
            device=k_cochain.device,
        )
        full_cochain[self.retained_idx[k]] = k_cochain
        return full_cochain

    def exxtract_constrained(
        self, k: int, k_cochain: Float[Tensor, " k_splx *ch"]
    ) -> Float[Tensor, " constrained_k_splx *ch"]:
        """
        Extract the constrained degrees of freedom from a cochain.

        Parameters
        ----------
        k
            The degree of the input cochain.
        k_cochain : [k_splx, *ch]
            The input cochain, with optional trailing channel dimensions.

        Returns
        -------
        constrained_cochain : [retained_k_splx, *ch]
            An output cochain that only contains coefficients on the constrained
            degrees of freedom.
        """
        return k_cochain[self.constrained_mask[k]]

    def embed_constrained(
        self, k: int, k_cochain: Float[Tensor, " constrained_k_splx *ch"]
    ) -> Float[Tensor, " k_splx *ch"]:
        """
        Lift the coefficients on the constrained degrees of freedom to a full cochain.

        Parameters
        ----------
        k
            The degree of the input cochain.
        k_cochain : [constrained_k_splx, *ch]
            The input cochain, with optional trailing channel dimensions.

        Returns
        -------
        full_cochain : [k_splx, *ch]
            An output cochain with zero coefficients on retained k-simplices.
        """
        full_cochain = torch.zeros(
            (self.constrained_mask[k].size(0), *k_cochain.shape[1:]),
            dtype=k_cochain.dtype,
            device=k_cochain.device,
        )
        full_cochain[self.constrained_idx[k]] = k_cochain
        return full_cochain

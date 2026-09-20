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
    Select the tangential boundary subcomplex of a simplicial mesh.

    The factory methods construct absolute, relative, or mixed boundary condition
    selections and ensure downward closure of the selected codim-one boundary simplices.
    Cochain helpers gather from and scatter into the resulting retained and constrained
    spaces.

    Parameters
    ----------
    constrained_mask
        A tuple of boolean masks. `constrained_mask[k]` marks the k-simplices
        in the tangential boundary subcomplex.

    Attributes
    ----------
    constrained_idx
        Indices selected by `constrained_mask` in global simplex order.
    retained_mask
        Complements of `constrained_mask` in each degree.
    retained_idx
        Indices selected by `retained_mask` in global simplex order.

    Notes
    -----
    In general, a k-cochain in the tangential boundary subcomplex is a constrained
    k-simplex, and a k-cochain in the complement is a retained k-simplex; similar
    terminologies are used to describe the coefficients of a k-cochain on the
    constrained/retained k-simplices.
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
        Restrict a full cochain to its retained coefficients.

        Parameters
        ----------
        k
            Cochain degree.
        k_cochain : [k_splx, *ch]
            Full cochain, with optional trailing dimensions.

        Returns
        -------
        retained_cochain : [retained_k_splx, *ch]
            Coefficients on retained k-simplices, in global simplex order.
        """
        return k_cochain[self.retained_mask[k]]

    def embed_retained(
        self, k: int, k_cochain: Float[Tensor, " retained_k_splx *ch"]
    ) -> Float[Tensor, " k_splx *ch"]:
        """
        Prolong retained coefficients to a full cochain by zero extension.

        Parameters
        ----------
        k
            Cochain degree.
        k_cochain : [retained_k_splx, *ch]
            Retained coefficients, with optional trailing dimensions.

        Returns
        -------
        full_cochain : [k_splx, *ch]
            Full cochain with zeros on constrained k-simplices.
        """
        full_cochain = torch.zeros(
            (self.constrained_mask[k].size(0), *k_cochain.shape[1:]),
            dtype=k_cochain.dtype,
            device=k_cochain.device,
        )
        full_cochain[self.retained_idx[k]] = k_cochain
        return full_cochain

    def extract_constrained(
        self, k: int, k_cochain: Float[Tensor, " k_splx *ch"]
    ) -> Float[Tensor, " constrained_k_splx *ch"]:
        """
        Gather a full cochain's constrained coefficients.

        Parameters
        ----------
        k
            Cochain degree.
        k_cochain : [k_splx, *ch]
            Full cochain, with optional trailing dimensions.

        Returns
        -------
        constrained_cochain : [constrained_k_splx, *ch]
            Coefficients on constrained k-simplices, in global simplex order.
        """
        return k_cochain[self.constrained_mask[k]]

    def embed_constrained(
        self, k: int, k_cochain: Float[Tensor, " constrained_k_splx *ch"]
    ) -> Float[Tensor, " k_splx *ch"]:
        """
        Scatter constrained coefficients into a full cochain by zero extension.

        Parameters
        ----------
        k
            Cochain degree.
        k_cochain : [constrained_k_splx, *ch]
            Constrained coefficients, with optional trailing dimensions.

        Returns
        -------
        full_cochain : [k_splx, *ch]
            Full cochain with zeros on retained k-simplices.
        """
        full_cochain = torch.zeros(
            (self.constrained_mask[k].size(0), *k_cochain.shape[1:]),
            dtype=k_cochain.dtype,
            device=k_cochain.device,
        )
        full_cochain[self.constrained_idx[k]] = k_cochain
        return full_cochain

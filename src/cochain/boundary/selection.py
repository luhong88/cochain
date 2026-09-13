__all__ = ["BoundarySelection"]

from dataclasses import dataclass
from functools import cached_property

import torch
from jaxtyping import Bool, Float, Integer
from torch import Tensor

from ..complex import SimplicialMesh


@dataclass(frozen=True, eq=False)
class BoundarySelection:
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
        constrained_masks = detect_mesh_boundaries(mesh.cbd)
        return cls(constrained_masks)

    @classmethod
    def from_tangential_bd_mask(
        cls, mesh: SimplicialMesh, tangent_mask_km1: Bool[Tensor, " km1_splx"]
    ):
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

    def restrict_full_cochain(
        self, k: int, k_cochain: Float[Tensor, " k_splx *ch"]
    ) -> Float[Tensor, " retained_k_splx *ch"]:
        return k_cochain[self.retained_mask[k]]

    def prolong_constrained_cochain(
        self, k: int, k_cochain: Float[Tensor, " retained_k_splx *ch"]
    ) -> Float[Tensor, " k_splx *ch"]:
        full_cochain = torch.zeros(
            (self.constrained_mask[k].size(0), *k_cochain.shape[1:]),
            dtype=k_cochain.dtype,
            device=k_cochain.device,
        )
        full_cochain[self.retained_idx[k]] = k_cochain
        return full_cochain

    def gather_constrained_cochain(
        self, k: int, k_cochain: Float[Tensor, " k_splx *ch"]
    ) -> Float[Tensor, " constrained_k_splx *ch"]:
        return k_cochain[self.constrained_mask[k]]

    def scatter_constrained_cochain(
        self, k: int, k_cochain: Float[Tensor, " constrained_k_splx *ch"]
    ) -> Float[Tensor, " k_splx *ch"]:
        full_cochain = torch.zeros(
            (self.constrained_mask[k].size(0), *k_cochain.shape[1:]),
            dtype=k_cochain.dtype,
            device=k_cochain.device,
        )
        full_cochain[self.constrained_idx[k]] = k_cochain
        return full_cochain

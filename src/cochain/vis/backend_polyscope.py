from collections import ChainMap
from dataclasses import dataclass
from typing import Any

import numpy as np
import polyscope as ps
import torch
from einops import rearrange
from jaxtyping import Float, Integer, Real
from torch import Tensor

from ..complex import SimplicialMesh
from ..utils.parsing import to_np
from .backend_base import VisBackend


@dataclass
class PolyscopeBackend(VisBackend):
    name: str
    mesh: SimplicialMesh

    def __post_init__(self):
        self.tri_edge_perm_map = self._compute_ps_edge_map(dim=2)

        self.ps_skel_1: ps.CurveNetwork | None = None
        self.ps_skel_2: ps.SurfaceMesh | None = None
        self.ps_mesh: ps.Structure

    def init(self, mesh_kwargs: dict[str, Any] | None = None):
        ps.init()

        if mesh_kwargs is None:
            mesh_kwargs = {}

        default_kwargs = {"edge_width": 1.0}
        kwargs = ChainMap(mesh_kwargs, default_kwargs)

        match self.mesh.dim:
            case 2:
                self.ps_mesh: ps.SurfaceMesh = ps.register_surface_mesh(
                    name=self.name,
                    vertices=to_np(self.mesh.vert_coords),
                    triangles=to_np(self.mesh.tris),
                    **kwargs,
                )
                self.ps_mesh.set_edge_permutation(perm=self.tri_edge_perm_map)

            case 3:
                self.ps_mesh: ps.VolumeMesh = ps.register_volume_mesh(
                    name=self.name,
                    vertices=to_np(self.mesh.vert_coords),
                    tets=to_np(self.mesh.tets),
                    **kwargs,
                )

            case _:
                raise ValueError(f"Unsupported mesh dimension: {self.mesh.dim}.")

    def add_k_cochain(
        self, k: int, name: str, cochain: Float[Tensor, " k_splx"], **kwargs
    ):
        self._add_scalar_quantity(
            name=name,
            values=to_np(cochain),
            degree=k,
            scalar_kwargs=kwargs,
        )

    def add_vector_field(
        self, k: int, name: str, vec_field: Float[Tensor, "k_splx coord=3"], **kwargs
    ):
        self._add_vector_quantity(
            name=name,
            values=to_np(vec_field),
            degree=k,
            vector_kwargs=kwargs,
        )

    def show(self):
        ps.show()

    def _compute_ps_edge_map(self, dim: int) -> Integer[np.ndarray, " edge"]:
        """
        Compute the Polyscope edge permutation map.

        Computes the permutation tensor required by Polyscope to map its internally
        generated edge indices to lex-ordered edge indices.

        Parameters
        ----------
        dim
            The dimension of the top-level simplices whose edge faces are considered.

        Returns
        -------
        [edge,]
            A tensor where the i-th element is the canonical edge index corresponding
            to Polyscope's i-th dynamically generated edge.
        """
        # Get the top level tris/tets
        splx = self.mesh.splx[dim]

        dtype = splx.dtype
        device = splx.device

        # Emulate Polyscope's cyclic traversal of the top-level simplex.
        # Extract vertex A (current) and vertex B (next) for every edge in every top
        # level simplex.
        vert_A = splx
        vert_B = torch.roll(splx, shifts=-1, dims=1)

        # Stack to create the oriented edges and flatten to match Polyscope's
        # sequential "first-come, first-served" loop.
        oriented_edges = torch.stack((vert_A, vert_B), dim=-1)
        flattened_edges = rearrange(
            oriented_edges, "splx edge vert -> (splx edge) vert"
        )

        # Find the unique, global, lex-sorted canonical edges.
        sorted_edges = flattened_edges.sort(dim=-1).values
        unique_canon_edges, all_edge_idx = sorted_edges.unique(
            dim=0, return_inverse=True
        )

        n_total_edges = flattened_edges.size(0)
        n_unique_edges = unique_canon_edges.size(0)

        # For each unique, canonical edge, compute its "discovery time", which is its
        # lowest index on the list of all local edges (i.e., all_edge_idx).
        disc_times = torch.full(
            (n_unique_edges,), n_total_edges + 1, dtype=dtype, device=device
        )
        traversal_steps = torch.arange(n_total_edges, dtype=dtype, device=device)

        disc_times.scatter_reduce_(
            dim=0,
            index=all_edge_idx,
            src=traversal_steps,
            reduce="amin",
            include_self=True,
        )

        # Rank the canonical edges by discovery time to get the permutation map.
        ps_perm = torch.argsort(disc_times)

        return ps_perm

    def _register_1_skeleton(self):
        if self.ps_skel_1 is None:
            self.ps_skel_1 = ps.register_curve_network(
                name=f"{self.name}_skel_1",
                nodes=to_np(self.mesh.vert_coords),
                edges=to_np(self.mesh.edges),
                radius=0.0,
            )

    def _register_2_skeleton(self):
        if self.ps_skel_2 is None:
            self.ps_skel_2 = ps.register_surface_mesh(
                name=f"{self.name}_skel_2",
                vertices=to_np(self.mesh.vert_coords),
                triangles=to_np(self.mesh.tris),
                edge_width=0.0,
            )
            self.ps_skel_2.set_edge_permutation(perm=self.tri_edge_perm_map)

    def _add_scalar_quantity(
        self,
        name: str,
        values: Real[np.ndarray, " splx"],
        degree: int,
        scalar_kwargs: dict[str, Any],
    ):
        match degree, self.mesh.dim:
            case (0, _):
                self.ps_mesh.add_scalar_quantity(
                    name, values, defined_on="vertices", **scalar_kwargs
                )

            case (1, 3):
                self._register_2_skeleton()
                self.ps_skel_2.add_scalar_quantity(
                    name, values, defined_on="edges", **scalar_kwargs
                )
            case (1, 2):
                self.ps_mesh.add_scalar_quantity(
                    name, values, defined_on="edges", **scalar_kwargs
                )

            case (2, 3):
                self._register_2_skeleton()
                self.ps_skel_2.add_scalar_quantity(
                    name, values, defined_on="faces", **scalar_kwargs
                )
            case (2, 2):
                self.ps_mesh.add_scalar_quantity(
                    name, values, defined_on="faces", **scalar_kwargs
                )

            case (3, 3):
                self.ps_mesh.add_scalar_quantity(
                    name, values, defined_on="cells", **scalar_kwargs
                )
            case (3, 2):
                raise ValueError(
                    "Degree 3 scalar quantities are not supported for tri meshes."
                )

    def _add_vector_quantity(
        self,
        name: str,
        values: Real[np.ndarray, "splx coord=3"],
        degree: int,
        vector_kwargs: dict[str, Any],
    ):
        match degree, self.mesh.dim:
            case (0, _):
                self.ps_mesh.add_vector_quantity(
                    name, values, defined_on="vertices", **vector_kwargs
                )

            case (1, _):
                self._register_1_skeleton()
                self.ps_skel_1.add_vector_quantity(
                    name, values, defined_on="edges", **vector_kwargs
                )

            case (2, 3):
                self._register_2_skeleton()
                self.ps_skel_2.add_vector_quantity(
                    name, values, defined_on="faces", **vector_kwargs
                )
            case (2, 2):
                self.ps_mesh.add_vector_quantity(
                    name, values, defined_on="faces", **vector_kwargs
                )

            case (3, 3):
                self.ps_mesh.add_vector_quantity(
                    name, values, defined_on="cells", **vector_kwargs
                )
            case (3, 2):
                raise ValueError(
                    "Degree 3 vector quantities are not supported for tri meshes."
                )

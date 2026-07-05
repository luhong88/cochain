from __future__ import annotations

from collections import ChainMap
from typing import Any

import numpy as np
import torch
from einops import rearrange
from jaxtyping import Float, Integer, Real
from torch import Tensor

from ..complex import SimplicialMesh
from ..utils.parsing import to_np

try:
    import polyscope as ps

    _HAS_POLYSCOPE = True

except ImportError:
    _HAS_POLYSCOPE = False


class PolyscopeViewer:
    """
    A class for visualization of SimplicialMesh objects with Polyscope.

    Parameters
    ----------
    name
        The name of the mesh.
    mesh
        The mesh to visualize.
    kwargs
        Additional keyword arguments to pass to Polyscope `register_surface_mesh()`
        and `register_volume_mesh()` functions.

    Attributes
    ----------
    ps_mesh
        The Polyscope surface/volume mesh object derived from the input SimplicialMesh.
    ps_skel_1
        An optional Polyscope curve network object derived from the edge faces
        of the input mesh (i.e., the 1-skeleton of the mesh).
    ps_skel_2
        An optional Polyscope surface mesh object derived from the tri faces of
        the input mesh (i.e., the 2-skeleton of the mesh).

    Notes
    -----
    Polyscope needs to be initialized with `ps.init()` before constructing this
    class. Use `ps.show()` to display the mesh and attached quantities.
    """

    def __init__(self, name: str, mesh: SimplicialMesh, **kwargs):
        if not _HAS_POLYSCOPE:
            raise ImportError("Polyscope backend required.")

        self.name = name
        self.mesh = mesh
        self._tri_edge_perm_map = to_np(self._compute_ps_edge_map(dim=2))

        default_kwargs = {"edge_width": 1.0}
        updated_mesh_kwargs = ChainMap(kwargs, default_kwargs)

        match self.mesh.dim:
            case 2:
                self.ps_mesh: ps.SurfaceMesh = ps.register_surface_mesh(
                    name=self.name,
                    vertices=to_np(self.mesh.vert_coords),
                    faces=to_np(self.mesh.tris),
                    **updated_mesh_kwargs,
                )
                self.ps_mesh.set_edge_permutation(perm=self._tri_edge_perm_map)

            case 3:
                self.ps_mesh: ps.VolumeMesh = ps.register_volume_mesh(
                    name=self.name,
                    vertices=to_np(self.mesh.vert_coords),
                    tets=to_np(self.mesh.tets),
                    **updated_mesh_kwargs,
                )

            case _:
                raise ValueError(f"Unsupported mesh dimension: {self.mesh.dim}.")

        self.ps_skel_1: ps.CurveNetwork | None = None
        self.ps_skel_2: ps.SurfaceMesh | None = None

    def add_k_cochain(
        self, k: int, name: str, cochain: Float[Tensor, " k_splx"], **kwargs
    ):
        """
        Add a cochain to the Polyscope mesh visualization.

        Parameters
        ----------
        k
            The degree of the cochain.
        name
            The name of the cochain.
        cochain : [k_splx,]
            The cochain tensor.
        kwargs
            Additional keyword arguments to pass to Polyscope `add_scalar_quantity()`
            function.

        Notes
        -----
        To visualize categorical/boolean data, pass `datatype='categorical'` as
        an additional keyword argument to Polyscope.

        For tet/volume meshes, Polyscope does not allow for visualization of
        scalar quantities attached to the edge and tri faces of the tets. To
        bypass this limitation, we create a 2-skeleton surface mesh and attach
        the 1- and 2-cochains to the 2-skeleton instead.
        """
        self._add_scalar_quantity(
            name=name,
            values=to_np(cochain),
            degree=k,
            scalar_kwargs=kwargs,
        )

    def add_vector_field(
        self, k: int, name: str, vec_field: Float[Tensor, "k_splx coord=3"], **kwargs
    ):
        """
        Add a vector-like field to the Polyscope mesh visualization.

        Parameters
        ----------
        k
            The dimension of the simplices to which the vectors are attached to.
        name
            The name of the vector field.
        vec_field : [k_splx, coord]
            The vector field tensor.
        kwargs
            Additional keyword arguments to pass to Polyscope `add_vector_quantity()`
            function.

        Notes
        -----
        For both tet/volume meshes and tri/surface meshes, Polyscope does not allow
        for visualization of vector quantities attached to the edge faces of the
        top-level simplices. To bypass this limitation, we create a 1-skeleton
        curve network and attach the vectors to the curve network instead.

        For tet/volume meshes, Polyscope in addition does not allow for visualization
        of vector quantities attached to the tri faces of the tets. To bypass this
        limitation, we create a 2-skeleton surface mesh and attach the vectors to
        the 2-skeleton instead.
        """
        self._add_vector_quantity(
            name=name,
            values=to_np(vec_field),
            degree=k,
            vector_kwargs=kwargs,
        )

    @torch.no_grad()
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
        ps_perm : [edge,]
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
        # Note that the 2-skeleton is disabled by default to avoid obscuring
        # the visualization of the tet mesh.
        if self.ps_skel_2 is None:
            self.ps_skel_2 = ps.register_surface_mesh(
                name=f"{self.name}_skel_2",
                vertices=to_np(self.mesh.vert_coords),
                faces=to_np(self.mesh.tris),
                enabled=False,
                edge_width=0.0,
            )
            self.ps_skel_2.set_edge_permutation(perm=self._tri_edge_perm_map)

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

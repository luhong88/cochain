import numpy as np
import polyscope as ps

from .backend_base import VisBackend


class PolyscopeBackend(VisBackend):
    def __init__(self):
        self.mesh = None
        self.edges = None
        self.vert_coords = None
        self.is_tet_mesh = False
        ps.init()

    def initialize(self, vert_coords, simplices, is_tet_mesh):
        self.vert_coords = vert_coords
        self.is_tet_mesh = is_tet_mesh
        if is_tet_mesh:
            self.mesh = ps.register_volume_mesh("base_mesh", vert_coords, simplices)
        else:
            self.mesh = ps.register_surface_mesh("base_mesh", vert_coords, simplices)

    def _ensure_curve_network(self, mesh_obj):
        if not ps.has_curve_network("base_edges"):
            if mesh_obj is not None:
                edges = mesh_obj.edges.detach().cpu().numpy()
                self.edges = edges
                ps.register_curve_network("base_edges", self.vert_coords, edges)

    def add_scalar_quantity(self, name, data, degree, mesh_obj=None):
        top_degree = 3 if self.is_tet_mesh else 2
        if degree == 0:
            self.mesh.add_scalar_quantity(name, data, defined_on="vertices")
        elif degree == 1:
            self._ensure_curve_network(mesh_obj)
            net = ps.get_curve_network("base_edges")
            net.add_scalar_quantity(name, data, defined_on="edges")
        elif degree == top_degree:
            defined_on = "cells" if self.is_tet_mesh else "faces"
            self.mesh.add_scalar_quantity(name, data, defined_on=defined_on)

    def add_mask(self, name, mask_array, degree, isolate, mesh_obj=None):
        top_degree = 3 if self.is_tet_mesh else 2

        if not isolate:
            if degree == 0:
                self.mesh.add_scalar_quantity(
                    name, mask_array, defined_on="vertices", datatype="categorical"
                )
            elif degree == 1:
                self._ensure_curve_network(mesh_obj)
                net = ps.get_curve_network("base_edges")
                net.add_scalar_quantity(
                    name, mask_array, defined_on="edges", datatype="categorical"
                )
            elif degree == top_degree:
                defined_on = "cells" if self.is_tet_mesh else "faces"
                self.mesh.add_scalar_quantity(
                    name, mask_array, defined_on=defined_on, datatype="categorical"
                )
        else:
            # isolate=True: Slicing simplices to create a new mesh
            if degree == 0:
                ps.register_point_cloud(
                    f"{name}_isolated", self.vert_coords[mask_array]
                )
            elif degree == 1:
                if mesh_obj is not None:
                    edges = mesh_obj.edges.detach().cpu().numpy()
                    ps.register_curve_network(
                        f"{name}_isolated", self.vert_coords, edges[mask_array]
                    )
            elif degree == top_degree:
                if mesh_obj is not None:
                    simplices = mesh_obj.tets if self.is_tet_mesh else mesh_obj.tris
                    simplices = simplices.detach().cpu().numpy()
                    if self.is_tet_mesh:
                        ps.register_volume_mesh(
                            f"{name}_isolated", self.vert_coords, simplices[mask_array]
                        )
                    else:
                        ps.register_surface_mesh(
                            f"{name}_isolated", self.vert_coords, simplices[mask_array]
                        )

    def add_vector_field(self, name, vectors, localization, normalize):
        if normalize:
            norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
            norms = np.where(norms == 0, 1.0, norms)
            vectors = vectors / norms
        self.mesh.add_vector_quantity(name, vectors, defined_on=localization)

    def show(self):
        ps.show()

    def register_trajectory(self, name, data_sequence):
        pass

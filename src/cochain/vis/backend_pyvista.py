import numpy as np
import pyvista as pv
import torch

from ..cochain.music_ops import local_sharp
from .backend_base import VisBackend


class PyVistaBackend(VisBackend):
    def __init__(self):
        self.mesh = None
        self.plotter = pv.Plotter()
        self.is_tet_mesh = False

    def initialize(self, vert_coords, simplices, is_tet_mesh):
        self.is_tet_mesh = is_tet_mesh
        if is_tet_mesh:
            # For tetrahedra, PyVista needs cell types. VTK_TETRA is 10.
            # PyVista UnstructuredGrid expects [n_points, p1, p2, p3, p4] format
            # where n_points is the number of points in the cell (4 for tet).
            n_tets = simplices.shape[0]
            cells = np.empty((n_tets, 5), dtype=simplices.dtype)
            cells[:, 0] = 4
            cells[:, 1:] = simplices
            cell_type = np.full(n_tets, pv.CellType.TETRA, dtype=np.uint8)
            self.mesh = pv.UnstructuredGrid(cells.ravel(), cell_type, vert_coords)
        else:
            n_tris = simplices.shape[0]
            faces = np.empty((n_tris, 4), dtype=simplices.dtype)
            faces[:, 0] = 3
            faces[:, 1:] = simplices
            self.mesh = pv.PolyData(vert_coords, faces.ravel())

        self.plotter.add_mesh(self.mesh, color="white", show_edges=True, opacity=0.8)

    def add_scalar_quantity(self, name, data, degree, mesh_obj=None):
        top_degree = 3 if self.is_tet_mesh else 2
        if degree == 0:
            self.mesh.point_data[name] = data
            self.mesh.set_active_scalars(name)
        elif degree == top_degree:
            self.mesh.cell_data[name] = data
            self.mesh.set_active_scalars(name)
        elif degree == 1:
            if mesh_obj is None:
                raise ValueError("mesh_obj must be provided for degree 1 in PyVista")

            # cochain_1 is expected to be a PyTorch tensor on the correct device.
            # We recreate it as a tensor if data is numpy.
            cochain_1 = torch.from_numpy(data).to(mesh_obj.device, dtype=mesh_obj.dtype)

            # call local_sharp
            sharp_vecs = local_sharp(cochain_1, mesh_obj, mode="element")
            sharp_vecs_np = sharp_vecs.detach().cpu().numpy()

            # Attach to cell_data
            self.mesh.cell_data[f"{name}_sharp"] = sharp_vecs_np

            # Use PyVista's glyph filter to draw arrows
            centers = self.mesh.cell_centers()
            centers.point_data[name] = sharp_vecs_np
            # Glyph requires vectors to be in the point_data
            centers.set_active_vectors(name)
            arrows = centers.glyph(orient=name, scale=name, factor=1.0)
            self.plotter.add_mesh(arrows, color="blue", name=f"{name}_arrows")
        else:
            raise NotImplementedError(
                f"Degree {degree} not supported for scalar quantity in PyVista"
            )

    def add_mask(self, name, mask_array, degree, isolate, mesh_obj=None):
        top_degree = 3 if self.is_tet_mesh else 2
        if not isolate:
            if degree == 0:
                self.mesh.point_data[name] = mask_array
                self.mesh.set_active_scalars(name)
            elif degree == top_degree:
                self.mesh.cell_data[name] = mask_array
                self.mesh.set_active_scalars(name)
            else:
                pass  # Cannot trivially visualize edge mask without sharp or extraction in pyvista
        else:
            if degree == top_degree:
                sub_mesh = self.mesh.extract_cells(mask_array)
                self.plotter.add_mesh(sub_mesh, color="red", name=f"{name}_isolated")
            elif degree == 0:
                sub_mesh = self.mesh.extract_points(mask_array)
                self.plotter.add_mesh(sub_mesh, color="red", name=f"{name}_isolated")

    def add_vector_field(self, name, vectors, localization, normalize):
        if localization == "vertices":
            self.mesh.point_data[name] = vectors
            self.mesh.set_active_vectors(name)
            if normalize:
                arrows = self.mesh.glyph(orient=name, scale=False, factor=0.1)
            else:
                arrows = self.mesh.glyph(orient=name, scale=name, factor=1.0)
        elif localization in ("faces", "cells"):
            centers = self.mesh.cell_centers()
            centers.point_data[name] = vectors
            centers.set_active_vectors(name)
            if normalize:
                arrows = centers.glyph(orient=name, scale=False, factor=0.1)
            else:
                arrows = centers.glyph(orient=name, scale=name, factor=1.0)
        else:
            raise ValueError(f"Unknown localization {localization}")

        self.plotter.add_mesh(arrows, color="green", name=f"{name}_vectors")

    def show(self):
        self.plotter.show()

    def register_trajectory(self, name, data_sequence):
        pass

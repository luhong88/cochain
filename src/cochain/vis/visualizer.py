import numpy as np
import torch

from .backend_polyscope import PolyscopeBackend
from .backend_pyvista import PyVistaBackend


def _sanitize(data):
    if torch.is_tensor(data):
        data = data.detach().cpu().numpy()

    if isinstance(data, np.ndarray):
        if data.dtype in (np.float16, np.float32, np.float64):
            return data.astype(np.float64)
        elif data.dtype == bool or data.dtype == np.bool_:
            return data.astype(np.bool_)
        elif np.issubdtype(data.dtype, np.integer):
            return data.astype(np.int64)

    return data


class Visualizer:
    def __init__(self, mesh_obj, backend="pyvista"):
        self.mesh_obj = mesh_obj
        if backend == "pyvista":
            self.backend = PyVistaBackend()
        elif backend == "polyscope":
            self.backend = PolyscopeBackend()
        else:
            raise ValueError(f"Unknown backend: {backend}")

        self.is_tet_mesh = mesh_obj.dim == 3
        vert_coords = _sanitize(mesh_obj.vert_coords)
        if self.is_tet_mesh:
            simplices = _sanitize(mesh_obj.tets)
        else:
            simplices = _sanitize(mesh_obj.tris)

        self.backend.initialize(vert_coords, simplices, self.is_tet_mesh)

    def add_cochain(self, name, tensor, degree):
        data = _sanitize(tensor)
        self.backend.add_scalar_quantity(name, data, degree, mesh_obj=self.mesh_obj)

    def add_mask(self, name, mask_tensor, degree, isolate=False):
        mask_array = _sanitize(mask_tensor)
        self.backend.add_mask(name, mask_array, degree, isolate, mesh_obj=self.mesh_obj)

    def add_vector_field(
        self, name, vector_tensor, localization="vertices", normalize=False
    ):
        vectors = _sanitize(vector_tensor)
        self.backend.add_vector_field(name, vectors, localization, normalize)

    def show(self):
        self.backend.show()

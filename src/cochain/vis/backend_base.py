from abc import ABC, abstractmethod


class VisBackend(ABC):
    @abstractmethod
    def initialize(self, vert_coords, simplices, is_tet_mesh): ...

    @abstractmethod
    def add_scalar_quantity(self, name, data, degree, mesh_obj=None): ...

    @abstractmethod
    def add_mask(self, name, mask_array, degree, isolate, mesh_obj=None): ...

    @abstractmethod
    def add_vector_field(self, name, vectors, localization, normalize): ...

    @abstractmethod
    def show(self): ...

    @abstractmethod
    def register_trajectory(self, name, data_sequence): ...

import copy
from collections.abc import Sequence
from dataclasses import dataclass
from functools import cached_property
from typing import Any

import torch
from jaxtyping import Bool, Float, Int64, Integer
from torch import Tensor

from .sparse.decoupled_tensor import (
    BaseDecoupledTensor,
    SparseDecoupledTensor,
    SparsityPattern,
)
from .topology import boundaries, coboundaries
from .utils.faces import GlobalFaces, enumerate_global_faces
from .utils.parsing import parse_to
from .utils.search import splx_search


def _is_tensor_like(obj: Any) -> bool:
    return torch.is_tensor(obj) | isinstance(obj, BaseDecoupledTensor)


@dataclass
class SimplicialMesh:
    """
    A dataclass containing the topology and vertex coordinates of a simplicial mesh.

    A valid simplicial mesh satisfies the following requirements:

    * The mesh is a pure simplicial k-complex (i.e., every l-simplex in the mesh
      for l < k must be the face/subset of a k-simplex).
    * The mesh vertices are 0-indexed and the indices are strictly consecutive.
    * The mesh is an immersion in the 3D Euclidean space; Specifically, this
      means that all vertex coordinates must be three-dimensional, and degenerate
      edges, triangles, or tetrahedra with zero length/area/volume are not allowed;
      however, self-intersection is allowed. However, no assumptions are made on
      whether the mesh is manifold or orientable.

    Parameters
    ----------
    cbd
        A tuple of the first three coboundary operators/discrete exterior derivatives
        for the mesh.
    splx
        A tuple of connectivity tensors specifying the canonical 1-, 2-, and 3-simplices.
        This parameter is modified post initialization to also include the 0-simplices,
        such that `splx[k]` returns the tensor of canonical k-simplices in the mesh.
    vert_coords : [vert, coord=3]
        The vertex coordinates of the mesh.

    Attributes
    ----------
    dtype
        The dtype of `vert_coords`.
    device
        The device of `vert_coords`.
    requires_grad
        Whether gradients need to be computed for `vert_coords`.
    grad
        The gradient computed for `vert_coords`.
    verts : [vert, 1]
        The connectivity tensor for the 0-simplices/vertices.
    edges : [edge, 2]
        The connectivity tensor for the canonical 1-simplices/edges.
    tris : [tri, 3]
        The connectivity tensor for the canonical 2-simplices/triangles.
    tets : [tet, 4]
        The connectivity tensor for the canonical 3-simplices/tetrahedra.
    n_verts
        The number of vertices.
    n_edges
        The number of canonical edges.
    n_tris
        The number of canonical triangles.
    n_tets
        The number of canonical tetrahedra.
    n_splx
        A tuple where `n_splx[k]` is the number of canonical k-simplices.
    dim
        The dimension of the mesh.
    vert_faces
        A `NamedTuple` describing the 0-faces of the top-level simplices.
    edge_faces
        A `NamedTuple` describing the 1-faces of the top-level simplices.
    tri_faces
        A `NamedTuple` describing the 2-faces of the top-level simplices.
    tet_faces
        A `NamedTuple` describing the 3-faces of the top-level simplices.
    dual_cbd
        A tuple of the first three dual coboundary operators for the mesh.
    bd_mask
        A tuple of boundary masks for the mesh.
    bd_vert_mask
        A boolean mask marking the boundary 0-simplices among the canonical 0-simplices.
    bd_edge_mask
        A boolean mask marking the boundary 1-simplices among the canonical 1-simplices.
    bd_tri_mask
        A boolean mask marking the boundary 2-simplices among the canonical 2-simplices.
    bd_tet_mask
        A boolean mask marking the boundary 3-simplices among the canonical 3-simplices.

    Notes
    -----
    In this package, we adopt the following convention on the definition of canonical
    k-simplices (as stored in `splx[k]`, e.g.). If k is the dimension of the mesh,
    then there is no restrictions on the ordering of the k-simplices or the ordering
    of vertices within a k-simplex (which can be used to carry information on
    the geometric orientation of the top-level simplices). For all other k, both
    the k-simplices and the ordering of vertices within the k-simplices must follow
    the lexicographic ordering (i.e., the "lex order").
    """

    cbd: tuple[
        Float[SparseDecoupledTensor, "edge vert"],
        Float[SparseDecoupledTensor, "tri edge"],
        Float[SparseDecoupledTensor, "tet tri"],
    ]
    splx: tuple[
        Integer[Tensor, "edge 2"],
        Integer[Tensor, "tri 3"],
        Integer[Tensor, "tet 4"],
    ]
    vert_coords: Float[Tensor, "vert coord=3"] | None

    # TODO: add a check for empty input mesh.
    def __post_init__(self):
        # The list of vertices contains only redundant information, but we
        # materialize it anyways so that simplices[0] gives the list of 0-simplices.
        verts = torch.arange(self.n_verts, device=self.edges.device).view(-1, 1)
        self.splx = (verts,) + self.splx

        # Check for dim consistency in coboundary operators.
        assert self.cbd[0].shape[0] == self.cbd[1].shape[1]
        assert self.cbd[1].shape[0] == self.cbd[2].shape[1]

        # Check for dim consistency between the coboundary operators and the simplex
        # definitions.
        assert self.edges.shape[0] == self.cbd[0].shape[0]
        assert self.tris.shape[0] == self.cbd[1].shape[0]
        assert self.tets.shape[0] == self.cbd[2].shape[0]

        # Check for device consistency
        for tensor in self.cbd + self.splx:
            assert tensor.device == self.device

        # Check for dtype consistency
        for tensor in self.cbd:
            assert tensor.dtype == self.dtype

        int_dtype = self.splx[0].dtype
        for tensor in self.splx:
            assert tensor.dtype == int_dtype

        # A cache of sparsity patterns for computing/coalescing sparse operators
        # whose sparsity pattern is dictated by mesh topology.
        self._coalesced_patterns: dict[str, SparsityPattern] = {}

    def _apply(self, func):
        """Apply a function (recursively) to all tensor-like attributes."""
        for key, value in self.__dict__.items():
            if _is_tensor_like(value):
                setattr(self, key, func(value))

            elif isinstance(value, Sequence) and not isinstance(value, str):
                processed_elements = [
                    func(v) if _is_tensor_like(v) else v for v in value
                ]

                # Reconstruct the sequence while preserving its original type.
                if isinstance(value, tuple) and hasattr(value, "_fields"):
                    # NamedTuples require positional argument unpacking.
                    setattr(self, key, type(value)(*processed_elements))
                else:
                    # Standard sequences (list, tuple) take an iterable.
                    setattr(self, key, type(value)(processed_elements))

        return self

    # TODO: also implement .cuda() and .cpu()
    def to(self, *args, **kwargs) -> "SimplicialMesh":
        """
        Move and/or casts the tensor-like attributes of the mesh.

        This function accepts the same arguments as `torch.Tensor.to()`.

        Notes
        -----
        Note that device casts apply to all tensors, but dtype casts follows a more
        specific set of rules:
        * If casting to `int` dtypes, only integer tensors get typecasted. In particular,
          it is not possible to modify the sparsity pattern indices of a
          `SparseDecoupledTensor` this way.
        * If casting to `float` dtypes, only float tensors and the value of the
          `SparseDecoupledTensor` get typecasted.
        * Casting to `bool` dtype has no effect.
        * Casting to `complex` dtypes is not permitted.
        """
        # Parse input arguments.
        input_device, input_dtype, copy_flag, non_blocking, memory_format = parse_to(
            *args, **kwargs
        )

        # Reject complex dtypes.
        if (input_dtype is not None) and input_dtype.is_complex:
            raise TypeError(
                f"Complex dtype '{input_dtype}' is not permitted for float "
                "tensors in SimplicialMesh."
            )

        # Extract remaining kwargs.
        other_kwargs = {
            "copy": copy_flag,
            "non_blocking": non_blocking,
            "memory_format": memory_format,
        }

        # Define the type-safe casting function.
        def custom_cast(t: Tensor | BaseDecoupledTensor):
            # Target the core .values dtype for BaseDecoupledTensors, and standard
            # .dtype for native tensors.
            target_dtype = input_dtype
            if hasattr(t, "values") and not callable(getattr(t, "values")):
                current_dtype = t.values.dtype
            else:
                current_dtype = getattr(t, "dtype", None)

            if (target_dtype is not None) and (current_dtype is not None):
                is_target_float = target_dtype.is_floating_point
                is_target_int = (not is_target_float) and (target_dtype != torch.bool)

                is_curent_float = current_dtype.is_floating_point
                is_current_int = (not is_curent_float) and (current_dtype != torch.bool)

                # Prevent int/bool from being cast to float, and float from being cast to int
                if is_target_float and not is_curent_float:
                    target_dtype = None
                elif is_target_int and not is_current_int:
                    target_dtype = None
                elif target_dtype == torch.bool and current_dtype != torch.bool:
                    target_dtype = None

            # Build the kwarg dictionary for this specific tensor/operator.
            cast_kwargs = dict(other_kwargs)
            if input_device is not None:
                cast_kwargs["device"] = input_device
            if target_dtype is not None:
                cast_kwargs["dtype"] = target_dtype

            return t.to(**cast_kwargs)

        # Create a shallow copy to bypass __post_init__() and leave the original
        # instance intact.
        new_mesh = copy.copy(self)

        # 2. Apply the custom cast recursively.
        # This will update the references on new_mesh via setattr, leaving `self` intact.
        new_mesh._apply(custom_cast)

        # 3. Handle the coalesced_patterns cache dict.
        new_mesh._coalesced_patterns = {}
        for k, v in self._coalesced_patterns.items():
            new_mesh._coalesced_patterns[k] = v.to(*args, **kwargs)

        return new_mesh

    def clone(self) -> "SimplicialMesh":
        """Return a copy of the mesh, cloning only float dtype tensors."""
        new_mesh = copy.copy(self)

        # Only clone floating-point tensors (like vert_coords and sparse values)
        # Integer tensors (splx, cached faces) are passed by reference to save compute.
        def _clone_floats(t):
            if (
                hasattr(t, "clone")
                and getattr(t, "dtype", None)
                and t.dtype.is_floating_point
            ):
                return t.clone()
            return t

        new_mesh._apply(_clone_floats)
        return new_mesh

    def detach(self) -> "SimplicialMesh":
        """Return a copy of the mesh with detached float dtype tensors."""
        new_mesh = copy.copy(self)

        def _detach_floats(t):
            if (
                hasattr(t, "detach")
                and getattr(t, "dtype", None)
                and t.dtype.is_floating_point
            ):
                return t.detach()
            return t

        new_mesh._apply(_detach_floats)

        # Explicitly wipe any residual gradients from the detached copy.
        if getattr(new_mesh.vert_coords, "grad", None) is not None:
            new_mesh.vert_coords.grad = None

        return new_mesh

    @property
    def dtype(self) -> torch.dtype:
        """The dtype of `vert_coords`."""
        return self.vert_coords.dtype

    @property
    def device(self) -> torch.device:
        """The device of `vert_coords`."""
        return self.vert_coords.device

    @property
    def requires_grad(self) -> bool:
        """Whether gradients need to be computed for `vert_coords`."""
        return self.vert_coords.requires_grad

    def requires_grad_(self, mode: bool = True):
        """Change if autograd should record operations on `vert_coords`."""
        self.vert_coords.requires_grad_(mode)

    @property
    def grad(self) -> Tensor | None:
        """The gradient computed for `vert_coords`."""
        return self.vert_coords.grad

    @grad.setter
    def grad(self, value):
        self.vert_coords.grad = value

    @property
    def verts(self) -> Integer[Tensor, "vert 1"]:
        """
        The connectivity tensor for the 0-simplices/vertices.

        This tensor simply consists of a list of integers from 0 to `n_verts`-1.
        """
        return self.splx[0]

    @property
    def edges(self) -> Integer[Tensor, "edge 2"]:
        """The connectivity tensor for the canonical 1-simplices/edges."""
        return self.splx[1]

    @property
    def tris(self) -> Integer[Tensor, "tri 3"]:
        """The connectivity tensor for the canonical 2-simplices/triangles."""
        return self.splx[2]

    @property
    def tets(self) -> Integer[Tensor, "tet 4"]:
        """The connectivity tensor for the canonical 3-simplices/tetrahedra."""
        return self.splx[3]

    @property
    def n_verts(self) -> int:
        """The number of vertices."""
        return self.cbd[0].shape[1]

    @property
    def n_edges(self) -> int:
        """The number of canonical edges."""
        return self.cbd[0].shape[0]

    @property
    def n_tris(self) -> int:
        """The number of canonical triangles."""
        return self.cbd[1].shape[0]

    @property
    def n_tets(self) -> int:
        """The number of canonical tetrahedra."""
        return self.cbd[2].shape[0]

    @property
    def n_splx(self) -> tuple[int, int, int, int]:
        """A tuple where `n_splx[k]` is the number of canonical k-simplices."""
        return (self.n_verts, self.n_edges, self.n_tris, self.n_tets)

    @property
    def dim(self) -> int:
        """The dimension of the mesh."""
        return max(
            1 * (self.n_edges != 0), 2 * (self.n_tris != 0), 3 * (self.n_tets != 0)
        )

    @cached_property
    def vert_faces(self) -> GlobalFaces:
        """
        A `NamedTuple` describing the 0-faces of the top-level simplices.

        The `GlobalFaces` named tuple contains the following attributes:

        idx : [top_splx, 0_face]
            The indices of the 0-faces of the top-level simplices on the list of
            canonical 0-simplices; note that identity between a 0-face and canonical
            0-simplex is determined up to vertex permutation.
        parity : [top_splx, 0_face]
            The permutation sign/parity of the 0-faces relative to the canonical
            0-simplices.
        """
        return enumerate_global_faces(
            self.splx[self.dim],
            self.verts,
            is_k_lex_sorted=(0 < self.dim),
            float_dtype=self.dtype,
        )

    @cached_property
    def edge_faces(self) -> GlobalFaces:
        """
        A `NamedTuple` describing the 1-faces of the top-level simplices.

        The `GlobalFaces` named tuple contains the following attributes:

        idx : [top_splx, 1_face]
            The indices of the 1-faces of the top-level simplices on the list of
            canonical 1-simplices; note that identity between a 1-face and canonical
            1-simplex is determined up to vertex permutation.
        parity : [top_splx, 1_face]
            The permutation sign/parity of the 1-faces relative to the canonical
            1-simplices.
        """
        return enumerate_global_faces(
            self.splx[self.dim],
            self.edges,
            is_k_lex_sorted=(1 < self.dim),
            float_dtype=self.dtype,
        )

    @cached_property
    def tri_faces(self) -> GlobalFaces:
        """
        A `NamedTuple` describing the 2-faces of the top-level simplices.

        The `GlobalFaces` named tuple contains the following attributes:

        idx : [top_splx, 2_face]
            The indices of the 2-faces of the top-level simplices on the list of
            canonical 2-simplices; note that identity between a 2-face and canonical
            2-simplex is determined up to vertex permutation.
        parity : [top_splx, 2_face]
            The permutation sign/parity of the 2-faces relative to the canonical
            2-simplices.
        """
        return enumerate_global_faces(
            self.splx[self.dim],
            self.tris,
            is_k_lex_sorted=(2 < self.dim),
            float_dtype=self.dtype,
        )

    @cached_property
    def tet_faces(self) -> GlobalFaces:
        """
        A `NamedTuple` describing the 3-faces of the top-level simplices.

        The `GlobalFaces` named tuple contains the following attributes:

        idx : [top_splx, 3_face]
            The indices of the 3-faces of the top-level simplices on the list of
            canonical 3-simplices; note that identity between a 3-face and canonical
            3-simplex is determined up to vertex permutation.
        parity : [top_splx, 3_face]
            The permutation sign/parity of the 3-faces relative to the canonical
            3-simplices.
        """
        return enumerate_global_faces(
            self.splx[self.dim],
            self.tets,
            is_k_lex_sorted=(3 < self.dim),
            float_dtype=self.dtype,
        )

    def faces(self, face_dim: int) -> GlobalFaces:
        """
        Get the k-faces of the top-level simplices.

        This is a convenience function for accessing the `vert_faces`, `edge_faces`,
        `tri_faces`, and `tet_faces` attributes of the mesh using the face dimension.

        Parameters
        ----------
        face_dim
            The dimension of the faces.

        Returns
        -------
        k_faces
            A `NamedTuple` describing the k-faces of the top-level simplices.
        """
        match face_dim:
            case 0:
                return self.vert_faces
            case 1:
                return self.edge_faces
            case 2:
                return self.tri_faces
            case 3:
                return self.tet_faces
            case _:
                raise ValueError(f"Invalid face_dim {face_dim}.")

    @property
    def dual_cbd(
        self,
    ) -> tuple[
        Float[SparseDecoupledTensor, "dual_edge dual_vert"],
        Float[SparseDecoupledTensor, "dual_tri dual_edge"],
        Float[SparseDecoupledTensor, "dual_tet dual_tri"],
    ]:
        """A tuple of the first three dual coboundary operators for the mesh."""
        # the k-th coboundary operator d_k* on the dual complex is given by
        # (-1)^k * d_{n-k-1}.T, where n is the dimension of the simplicial complex.
        return tuple(self.cbd[self.dim - k - 1].T * ((-1.0) ** k) for k in range(3))

    @cached_property
    def bd_mask(
        self,
    ) -> tuple[
        Bool[Tensor, " vert"],
        Bool[Tensor, " edge"],
        Bool[Tensor, " tri"],
        Bool[Tensor, " tet"],
    ]:
        """
        A tuple of boundary masks for the mesh.

        `bd_mask[k]` returns a boolean mask marking the boundary k-simplices
        among the canonical k-simplices.
        """
        return boundaries.detect_mesh_boundaries(self.cbd)

    @property
    def bd_vert_mask(self) -> Bool[Tensor, " vert"]:
        """A boolean mask marking the boundary 0-simplices among the canonical 0-simplices."""
        return self.bd_mask[0]

    @property
    def bd_edge_mask(self) -> Bool[Tensor, " edge"]:
        """A boolean mask marking the boundary 1-simplices among the canonical 1-simplices."""
        return self.bd_mask[1]

    @property
    def bd_tri_mask(self) -> Bool[Tensor, " tri"]:
        """A boolean mask marking the boundary 2-simplices among the canonical 2-simplices."""
        return self.bd_mask[2]

    @property
    def bd_tet_mask(self) -> Bool[Tensor, " tet"]:
        """A boolean mask marking the boundary 3-simplices among the canonical 3-simplices."""
        return self.bd_mask[3]

    @classmethod
    def from_tri_mesh(
        cls,
        vert_coords: Float[Tensor, "vert coord=3"],
        tris: Integer[Tensor, "tri 3"],
    ):
        """
        Construct a simplicial 2-mesh from the 2-simplex connectivity tensor.

        Parameters
        ----------
        vert_coords : [vert, coord=3]
            The vertex coordinates of the mesh.
        tris : [tri, 3]
            The connectivity tensor for the canonical 2-simplices/triangles.

        Returns
        -------
        tri_mesh
            A `SimplicialMesh` object representing the simplicial 2-mesh.
        """
        tris = tris.long()

        unique_canon_edges, cbd_0, cbd_1 = coboundaries.cbd_from_tri_mesh(
            tris, dtype=vert_coords.dtype
        )

        cbd_2 = SparseDecoupledTensor.from_tensor(
            torch.sparse_coo_tensor(
                indices=torch.empty((2, 0), dtype=torch.int64),
                values=torch.empty((0,), dtype=vert_coords.dtype),
                size=(0, tris.shape[0]),
                device=cbd_0.device,
            )
        )

        tets = torch.empty((0, 4), dtype=torch.int64, device=tris.device)

        return cls(
            cbd=(cbd_0, cbd_1, cbd_2),
            splx=(unique_canon_edges, tris, tets),
            vert_coords=vert_coords,
        )

    @classmethod
    def from_tet_mesh(
        cls,
        vert_coords: Float[Tensor, "vert 3"],
        tets: Integer[Tensor, "tet 4"],
    ):
        """
        Construct a simplicial 3-mesh from the 3-simplex connectivity tensor.

        Parameters
        ----------
        vert_coords : [vert, coord=3]
            The vertex coordinates of the mesh.
        tets : [tet, 4]
            The connectivity tensor for the canonical 3-simplices/triangles.

        Returns
        -------
        tet_mesh
            A `SimplicialMesh` object representing the simplicial 3-mesh.
        """
        tets = tets.long()

        (
            unique_canon_edges,
            unique_canon_tris,
            cbd_0,
            cbd_1,
            cbd_2,
        ) = coboundaries.cbd_from_tet_mesh(tets, dtype=vert_coords.dtype)

        return cls(
            cbd=(cbd_0, cbd_1, cbd_2),
            splx=(unique_canon_edges, unique_canon_tris, tets),
            vert_coords=vert_coords,
        )

    def _sparse_coalesced_matrix(
        self,
        operator: str,
        indices: Int64[Tensor, "2 nz"],
        values: Float[Tensor, " nz"],
        size: tuple[int, int] | torch.Size,
    ) -> SparseDecoupledTensor:
        """
        Construct coalsced sparse matrices for mesh operators.

        There are some sparse operators in topology and DEC where the specific
        values of the nonzero elements are metric-dependent, but the sparsity
        pattern of the operator is strictly dictated by the topology/connectivity
        of the simplicial mesh. In addition, such operators are often constructed
        first locally by computing relevant quantities on each top-level simplex,
        before the local quantities are scattered to generate a global operator.
        In such cases, it is beneficial to cache the sparsity patterns of these
        operators and the local-to-global index mapping required to coalesce the
        local terms.

        This class acts as a wrapper/replacement for `torch.sparse_coo_tensor()`.
        When called during the final construction of the global operator, this
        function checks whether the sparsity pattern of the operator is already
        pre-computed and cached in the self `SimplicialMesh` object. If not, it
        calls `torch.sparse_coo_tensor()` to construct the coalesced global
        operator and computes the local-to-global index mapping, which is used
        to perform a much faster 1D scatter-add for subsequent operator constructions.
        """
        pattern = self._coalesced_patterns.get(operator, None)

        if pattern is None:
            # If the sparsity pattern for an operator has not been cached yet,
            # then rely on torch to generate the coalesced indices, but manually
            # compute the

            # from_tensor() handles the colesce() call.
            sdt = SparseDecoupledTensor.from_tensor(
                torch.sparse_coo_tensor(indices=indices, values=values, size=size)
            )

            # Co-opt the simplex search function to match the uncoalesced (row, col)
            # index pairs to the coalsced pairs. Both sort_key_vert and sort_query_vert
            # must be False so that the search treates (i, j) and (j, i) as
            # distinct index pairs whenever i != j.
            coalesce_idx_map = splx_search(
                key_splx=sdt.pattern.idx_coo.T,
                query_splx=indices.T,
                sort_key_splx=False,
                sort_key_vert=False,
                sort_query_vert=False,
                method="lex_sort",
            )

            object.__setattr__(sdt.pattern, "_coalesce_idx_map", coalesce_idx_map)
            self._coalesced_patterns[operator] = sdt.pattern

            return sdt

        else:
            coalesced_values = torch.zeros(
                pattern._nnz(), dtype=values.dtype, device=values.device
            )
            coalesced_values.scatter_add_(
                dim=0, index=pattern._coalesce_idx_map, src=values
            )

            sdt = SparseDecoupledTensor(pattern, coalesced_values)

            return sdt


@dataclass
class MeshBatch(SimplicialMesh):
    """
    A batch of `SimplicialMesh`, represented as a single, disconnected complex.

    This class is a subclass of the `SimplicialMesh` class, and enables
    block-diagonal batching of meshes, so called because the operators derived
    from a `MeshBatch` tend to have a block-diagonal structure, where each
    block represents one mesh in the batch.

    Parameters
    ----------
    n_meshes
        The number of meshes in the batch.
    ptrs
        A tuple of pointer tensors; specifically, `ptrs[k][i] = [j]` indicates
        that the `i`th simplex in the batched list of canonical `k`-simplices
        come from the `j`th mesh in the batch. This is useful for unpacking
        the block-diagonal batching.
    """

    n_meshs: int
    ptrs: tuple[
        Integer[Tensor, " vert"],
        Integer[Tensor, " edge"],
        Integer[Tensor, " tri"],
        Integer[Tensor, " tet"],
    ]


def collate_fn(mesh_batch: Sequence[SimplicialMesh]) -> MeshBatch:
    """
    Collate a list of `SimplicialMesh` bojects into a `MeshBatch` object.

    Parameters
    ----------
    mesh_batch
        A list of `SimplicialMesh` objects. Note that this function allows batching
        of meshes with different dimensions.

    Returns
    -------
    batched_mesh
        A `MeshBatch` object representing the batch of meshes.
    """
    # Assume that all simplices and their tensors are on the same device and
    # all float tensors have the same dtype.
    device = mesh_batch[0].device
    idx_dtype = mesh_batch[0].verts.dtype

    # Generate a cumsum n_mesh list for each simplex dimension
    n_splx_batch = torch.tensor(
        [
            [0] + [mesh.n_verts for mesh in mesh_batch],
            [0] + [mesh.n_edges for mesh in mesh_batch],
            [0] + [mesh.n_tris for mesh in mesh_batch],
            [0] + [mesh.n_tets for mesh in mesh_batch],
        ],
        dtype=idx_dtype,
        device=device,
    )

    n_verts_cumsum_batch = torch.cumsum(n_splx_batch[0], dim=-1, dtype=idx_dtype)

    # Generate the ptr tensor for each simplex dimension
    ptrs = [
        torch.repeat_interleave(
            torch.arange(len(mesh_batch), dtype=idx_dtype, device=device),
            repeats=n_splx_batch[dim, 1:],
        )
        for dim in range(4)
    ]

    # Collate the coboundary operators into sparse block-diagonal forms.
    cbd_batch = [
        SparseDecoupledTensor.pack_block_diag([mesh.cbd[idx] for mesh in mesh_batch])
        for idx in range(3)
    ]

    # Collate the simplices; increment the vertex indices for each sc in the batch.
    splx_batch = [
        torch.vstack(
            [
                mesh.splx[dim] + n_verts_cumsum_batch[mesh_idx]
                for mesh_idx, mesh in enumerate(mesh_batch)
            ]
        )
        for dim in [1, 2, 3]
    ]

    # Collate the vertex coordinates
    vert_coords_batch = torch.vstack([sc.vert_coords for sc in mesh_batch])

    return MeshBatch(
        cbd=tuple(cbd_batch),
        splx=tuple(splx_batch),
        vert_coords=vert_coords_batch,
        n_meshs=len(mesh_batch),
        ptrs=tuple(ptrs),
    )

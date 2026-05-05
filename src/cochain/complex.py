import copy
import inspect
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
    A dataclass containing the topology and vertex coordinates of a mesh.

    A valid mesh satisfies the following requirements:

    * The mesh is a pure simplicial $k$-complex (i.e., every $l$-simplex in the
      mesh for $l < k$ must be the face/subset of a $k$-simplex).
    * The mesh vertices are 0-indexed and the indices are strictly consecutive.
    * The mesh is an immersion in the 3D Euclidean space; Specifically, this
      means that all vertex coordinates must be three-dimensional, and degenerate
      edges, triangles, or tetrahedra with zero length/area/volume are not allowed;
      however, self-intersection is allowed.

    In general, no assumptions are made on whether the mesh is manifold or orientable.
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
    vert_coords: Float[Tensor, "vert 3"] | None

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
        self.coalesced_patterns: dict[str, SparsityPattern] = {}

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
        Move and/or casts the tensor-like attributes in the mesh.

        Note that device casts apply to all tensors, but dtype casts follows a
        more specific set of rules:
        * If casting to int dtypes, only integer tensors get typecasted. In particular,
          it is not possible to modify the sparsity pattern indices of a
          `SparseDecoupledTensor` this way.
        * If casting to float dtypes, only float tensors and the value of the
          `SparseDecoupledTensor` get typecasted.
        * Casting to bool dtype has no effect.
        * Casting to complex dtypes is not permitted.
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
        new_mesh.coalesced_patterns = {}
        for k, v in self.coalesced_patterns.items():
            new_mesh.coalesced_patterns[k] = v.to(*args, **kwargs)

        return new_mesh

    @property
    def dtype(self) -> torch.dtype:
        return self.vert_coords.dtype

    @property
    def device(self) -> torch.device:
        return self.vert_coords.device

    @property
    def requires_grad(self) -> bool:
        return self.vert_coords.requires_grad

    def requires_grad_(self, mode: bool = True):
        self.vert_coords.requires_grad_(mode)

    @property
    def grad(self) -> Tensor | None:
        return self.vert_coords.grad

    @property
    def verts(self) -> Integer[Tensor, "vert 1"]:
        return self.splx[0]

    @property
    def edges(self) -> Integer[Tensor, "edge 2"]:
        return self.splx[1]

    @property
    def tris(self) -> Integer[Tensor, "tri 3"]:
        return self.splx[2]

    @property
    def tets(self) -> Integer[Tensor, "tet 4"]:
        return self.splx[3]

    @property
    def n_verts(self) -> int:
        return self.cbd[0].shape[1]

    @property
    def n_edges(self) -> int:
        return self.cbd[0].shape[0]

    @property
    def n_tris(self) -> int:
        return self.cbd[1].shape[0]

    @property
    def n_tets(self) -> int:
        return self.cbd[2].shape[0]

    @property
    def n_splx(self) -> tuple[int, int, int, int]:
        return (self.n_verts, self.n_edges, self.n_tris, self.n_tets)

    @property
    def dim(self) -> int:
        return max(
            1 * (self.n_edges != 0), 2 * (self.n_tris != 0), 3 * (self.n_tets != 0)
        )

    @cached_property
    def edge_faces(self) -> GlobalFaces:
        return enumerate_global_faces(self.splx[self.dim], self.edges)

    @cached_property
    def tri_faces(self) -> GlobalFaces:
        return enumerate_global_faces(self.splx[self.dim], self.tris)

    @property
    def dual_cbd(
        self,
    ) -> tuple[
        Float[SparseDecoupledTensor, "dual_edge dual_vert"],
        Float[SparseDecoupledTensor, "dual_tri dual_edge"],
        Float[SparseDecoupledTensor, "dual_tet dual_tri"],
    ]:
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
        return boundaries.detect_mesh_boundaries(self.cbd)

    @property
    def bd_vert_mask(self) -> Bool[Tensor, " vert"]:
        return self.bd_mask[0]

    @property
    def bd_edge_mask(self) -> Bool[Tensor, " edge"]:
        return self.bd_mask[1]

    @property
    def bd_tri_mask(self) -> Bool[Tensor, " tri"]:
        return self.bd_mask[2]

    @property
    def bd_tet_mask(self) -> Bool[Tensor, " tet"]:
        return self.bd_mask[3]

    # TODO: write test for this method
    def is_pure(self) -> bool:
        # A simplicial complex is pure if every k-simplex is a face of at least
        # one (k+1)-simplex, unless k is the top level.
        cbd_ops = [self.cbd[dim].to_sparse_coo() for dim in [2, 1, 0]]

        for cbd in cbd_ops:
            if cbd._nnz() == 0:
                continue
            else:
                face_relation_count = cbd.abs().sum(dim=0)
                if face_relation_count._nnz() > face_relation_count.size(-1):
                    return False

        return True

    # TODO: check for immersion
    @classmethod
    def from_tri_mesh(
        cls,
        vert_coords: Float[Tensor, "vert 3"],
        tris: Integer[Tensor, "tri 3"],
    ):
        """
        Construct a special geometric simplicial 2-complex as a t riangulated 2D
        mesh immersed in 3D Euclidean space; note that this function does not assume
        that the complex is a 2-manifold.

        Since no orientation is assigned to the edges using this constructor, we
        will assign a "canonical" orientation to each edge ij such that i < j.
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
        Construct a special geometric simplicial 3-complex as a triangulated 3D
        mesh immersed in 3D Euclidean space.

        Since no orientation is assigned to the triangles and edges using this
        constructor, we will assign a "canonical" orientation to each edge ij such
        that i < j and a "canonical" orientation to edge triangle ijk such that
        i < j < k.
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
        indices: Int64[Tensor, "2 nz"],
        values: Float[Tensor, " nz"],
        size: torch.Size,
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
        # Get the name of the function/operator that's calling this method.
        op_class: str = inspect.stack()[1].function

        pattern = self.coalesced_patterns.get(op_class, None)

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
                key_splx=sdt.idx_coo.T,
                query_splx=indices.T,
                sort_key_splx=False,
                sort_key_vert=False,
                sort_query_vert=False,
                method="lex_sort",
            )

            object.__setattr__(sdt.pattern, "_coalesce_idx_map", coalesce_idx_map)
            self.coalesced_patterns[op_class] = sdt.pattern

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
    A batch of SimplicialMesh, represented as a single, disconnected complex.
    """

    # the "ptr" tensor for each dimension maps a simplex back to the original
    # simplicial complex; i.e., edges_ptr[i] = k indicates that the i-th edge
    # in the batch comes from the k-th simplicial complex.
    n_meshs: int
    ptrs: tuple[
        Integer[Tensor, " vert"],
        Integer[Tensor, " edge"],
        Integer[Tensor, " tri"],
        Integer[Tensor, " tet"],
    ]


def collate_fn(mesh_batch: Sequence[SimplicialMesh]) -> MeshBatch:
    """
    This function takes in a list of `SimplicialMesh` objects and collate them
    into a single batched complex.

    Note that this function allows batching of meshes with different dimensions.
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

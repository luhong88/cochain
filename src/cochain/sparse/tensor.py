from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

import torch as t
from jaxtyping import Float, Integer

from ._matmul import dense_sp_mm, sp_dense_mm, sp_mv, sp_sp_mm, sp_vm
from ._sp_topo import SparseTopology


# TODO: check handling of contiguous memory
@dataclass
class SparseOperator:
    val: Float[t.Tensor, " nnz *d"]
    sp_topo: Integer[SparseTopology, "*b r c"]

    def __post_init__(self):
        if self.val.device != self.sp_topo.device:
            raise RuntimeError("'val' and 'sp_topo' must be on the same device.")

        if self.val.size(0) != self.sp_topo._nnz():
            raise ValueError("nnz mismatch between 'val' and 'sp_topo'.")

    @property
    def shape(self) -> t.Size:
        return self.sp_topo.shape + self.val.shape[1:]

    @property
    def device(self) -> t.device:
        return self.val.device

    @property
    def dtype(self) -> t.dtype:
        return self.val.dtype

    @property
    def n_dense_dim(self) -> int:
        return self.val.ndim - 1

    @property
    def n_sp_dim(self) -> int:
        return self.sp_topo.n_sp_dim

    @property
    def n_batch_dim(self) -> int:
        return self.sp_topo.n_batch_dim

    @property
    def n_dim(self) -> int:
        return self.n_batch_dim + self.n_sp_dim + self.n_dense_dim

    @property
    def T(self) -> SparseOperator:
        val_trans = self.val[self.sp_topo.coo_to_csc_perm]
        sp_topo_trans = self.sp_topo.T
        return SparseOperator(val_trans, sp_topo_trans)

    def _nnz(self) -> int:
        return self.sp_topo._nnz()

    def __matmul__(self, other):
        """
        Implement self @ other
        """
        match other:
            case SparseOperator():
                return sp_sp_mm(self.val, self.sp_topo, other.val, other.sp_topo)

            case t.Tensor():
                match other.ndim:
                    case 1:
                        return sp_mv(self.val, self.sp_topo, other)
                    case 2:
                        return sp_dense_mm(self.val, self.sp_topo, other)
                    case _:
                        raise NotImplementedError()

            case _:
                raise ValueError()

    def __rmatmul__(self, other):
        """
        Implement other @ self
        """
        match other:
            case SparseOperator():
                return sp_sp_mm(
                    other.val,
                    other.sp_topo,
                    self.val,
                    self.sp_topo,
                )

            case t.Tensor():
                match other.ndim:
                    case 1:
                        return sp_vm(self.val, self.sp_topo, other)
                    case 2:
                        return dense_sp_mm(self.val, self.sp_topo, other)
                    case _:
                        raise NotImplementedError()

            case _:
                raise ValueError()

    def to_sparse_coo(self):
        return t.sparse_coo_tensor(
            self.sp_topo.idx_coo,
            self.val,
            self.sp_topo.shape,
            dtype=self.dtype,
            device=self.device,
        )

    def to_sparse_csr(self, int32: bool = False):
        if int32:
            idx_crow = self.sp_topo.idx_crow_int32
            idx_col = self.sp_topo.idx_col_int32
        else:
            idx_crow = self.sp_topo.idx_crow
            idx_col = self.sp_topo.idx_col

        return t.sparse_csr_tensor(
            idx_crow,
            idx_col,
            self.val,
            self.sp_topo.shape,
            dtype=self.dtype,
            device=self.device,
        )

    def to_sparse_csc(self, int32: bool = False):
        if int32:
            idx_ccol = self.sp_topo.idx_ccol_int32
            idx_row = self.sp_topo.idx_row_int32
        else:
            idx_ccol = self.sp_topo.idx_ccol
            idx_row = self.sp_topo.idx_row

        return t.sparse_csc_tensor(
            idx_ccol,
            idx_row,
            self.val[self.sp_topo.coo_to_csc_perm],
            self.sp_topo.shape,
            dtype=self.dtype,
            device=self.device,
        )

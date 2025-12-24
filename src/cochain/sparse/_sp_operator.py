from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch as t
from jaxtyping import Float, Integer

from ._base_operator import BaseOperator, is_scalar, validate_matmul_args
from ._matmul import dense_sp_mm, sp_dense_mm, sp_mv, sp_sp_mm, sp_vm
from ._sp_topo import SparseTopology


def _check_topo_equality(self: SparseOperator, other: SparseOperator, msg: str):
    # Enforce equal topology requirement with three increasingly more expensive
    # checks: 1) same underlying sp_topo object, 2) same sp_topo shape, and 3)
    # same sp_topo.idx_coo elements.
    if self.sp_topo is other.sp_topo:
        pass

    elif self.sp_topo.shape == other.sp_topo.shape and t.equal(
        self.sp_topo.idx_coo, other.sp_topo.idx_coo
    ):
        pass

    else:
        raise ValueError(msg)


@dataclass
class SparseOperator(BaseOperator):
    sp_topo: Integer[SparseTopology, "*b r c"]
    val: Float[t.Tensor, " nnz *d"]

    def __post_init__(self):
        if self.val.device != self.sp_topo.device:
            raise RuntimeError("'val' and 'sp_topo' must be on the same device.")

        if self.val.size(0) != self.sp_topo._nnz():
            raise ValueError("nnz mismatch between 'val' and 'sp_topo'.")

        if not t.isfinite(self.val).all():
            raise ValueError("SparseOperator values contain NaN or Inf.")

        # Enforce contiguous memory layout.
        self.val = self.val.contiguous()

    @classmethod
    def from_tensor(cls, tensor: t.Tensor) -> SparseOperator:
        coalesced_tensor = tensor.to_sparse_coo().coalesce()

        return cls(
            sp_topo=SparseTopology(coalesced_tensor.indices(), coalesced_tensor.shape),
            val=coalesced_tensor.values(),
        )

    @classmethod
    def assemble(cls, *operators: BaseOperator) -> SparseOperator:
        """
        Efficiently sum multiple SparseOperators and/or DiagOperators with different
        topologies.
        """
        if not operators:
            raise ValueError("No operators to assemble.")

        coo_tensors = [op.to_sparse_coo() for op in operators]

        all_idx = t.hstack([coo.indices() for coo in coo_tensors])
        all_val = t.hstack([coo.values() for coo in coo_tensors])

        return t.sparse_coo_tensor(
            all_idx, all_val, size=coo_tensors[0].size()
        ).coalesce()

    @property
    def shape(self) -> t.Size:
        return self.sp_topo.shape + self.val.shape[1:]

    @property
    def n_dense_dim(self) -> int:
        return self.val.ndim - 1

    @property
    def n_sp_dim(self) -> int:
        """
        For sparse csr/csc tensors, the leading batch dimension(s) do not count
        towards sparse_dim(); for sparse coo tensors, no such distinction is
        made. Here we follow the sparse csr/csc convention.
        """
        return self.sp_topo.n_sp_dim

    @property
    def n_batch_dim(self) -> int:
        return self.sp_topo.n_batch_dim

    @property
    def T(self) -> SparseOperator:
        """
        Note that the transpose preserves the batch and dense dimensions and only
        operates on the sparse dimensions.
        """
        val_trans = self.val[self.sp_topo.coo_to_csc_perm]
        sp_topo_trans = self.sp_topo.T
        return SparseOperator(sp_topo_trans, val_trans)

    def apply(self, fn: Callable, **kwargs) -> SparseOperator:
        """
        Apply a sparsity-preserving function on the values of SparseOperator.
        """
        new_val = fn(self.val, **kwargs)

        if new_val.size(0) != self.val.size(0):
            raise RuntimeError("Function changed the nnz dim of the SparseOperator.")

        return SparseOperator(self.sp_topo, new_val)

    def detach(self) -> SparseOperator:
        """
        Create a new SparseOperator with the same `sp_topo` but with the `val` detached.
        """
        return SparseOperator(self.sp_topo, self.val.detach())

    def clone(
        self, memory_format: t.memory_format = t.contiguous_format
    ) -> SparseOperator:
        """
        Create a new SparseOperator with the same `sp_topo` but with the `val`
        cloned (in the contiguous format by default).
        """
        return SparseOperator(self.sp_topo, self.val.clone(memory_format=memory_format))

    def _nnz(self) -> int:
        """
        For batched sparse csr/csc tensors, the _nnz() method returns the number
        of nonzero elements per batch item; for sparse coo tensors, the _nnz()
        method returns the total number of nonzero elements, regardless of batch
        dimensions. Here we follow the sparse coo convention.
        """
        return self.sp_topo._nnz()

    def __matmul__(self, other):
        """
        Implement self @ other
        """
        validate_matmul_args(self, other)

        match other:
            case SparseOperator():
                idx_crow, idx_col, val, shape = sp_sp_mm(
                    self.val, self.sp_topo, other.val, other.sp_topo
                )
                sp_sp = t.sparse_csr_tensor(idx_crow, idx_col, val, shape)
                return self.from_tensor(sp_sp)

            case t.Tensor():
                match other.ndim:
                    case 1:
                        return sp_mv(self.val, self.sp_topo, other)
                    case 2:
                        return sp_dense_mm(self.val, self.sp_topo, other)

            case _:
                return NotImplemented

    def __rmatmul__(self, other):
        """
        Implement other @ self
        """
        validate_matmul_args(self, other)

        match other:
            # Do not check for case SparseOperator(), which is handled by __matmul__
            case t.Tensor():
                match other.ndim:
                    case 1:
                        return sp_vm(other, self.val, self.sp_topo)
                    case 2:
                        return dense_sp_mm(other, self.val, self.sp_topo)

            case _:
                return NotImplemented

    def __neg__(self) -> SparseOperator:
        return SparseOperator(self.sp_topo, -self.val)

    def __mul__(self, other) -> SparseOperator:
        """
        Scalar multiplication.
        """
        return (
            SparseOperator(self.sp_topo, self.val * other)
            if is_scalar()
            else NotImplemented
        )

    def __truediv__(self, other) -> SparseOperator:
        """
        Scalar division.
        """
        return (
            SparseOperator(self.sp_topo, self.val / other)
            if is_scalar()
            else NotImplemented
        )

    def __add__(self, other) -> SparseOperator:
        """
        Elementwise-addition of two SparseOperators that share the same topology/
        sparsity pattern.
        """
        match other:
            case SparseOperator():
                _check_topo_equality(
                    self,
                    other,
                    msg="SparseOperator __add__ only supports operators with identical topologies.",
                )
                return SparseOperator(self.sp_topo, self.val + other.val)
            case _:
                return NotImplemented

    def __sub__(self, other) -> SparseOperator:
        match other:
            case SparseOperator():
                _check_topo_equality(
                    self,
                    other,
                    msg="SparseOperator __sub__ only supports operators with identical topologies.",
                )
                return SparseOperator(self.sp_topo, self.val - other.val)
            case _:
                return NotImplemented

    def to_sparse_coo(self) -> Float[t.Tensor, "*b r c *d"]:
        return t.sparse_coo_tensor(
            self.sp_topo.idx_coo,
            self.val,
            self.shape,
            dtype=self.dtype,
            device=self.device,
        ).coalesce()

    def to_sparse_csr(self, int32: bool = False) -> Float[t.Tensor, "*b r c *d"]:
        if int32:
            idx_crow = self.sp_topo.idx_crow_int32
            idx_col = self.sp_topo.idx_col_int32
        else:
            idx_crow = self.sp_topo.idx_crow
            idx_col = self.sp_topo.idx_col

        if self.n_batch_dim == 0:
            val = self.val
        else:
            val = self.val.view(self.size(0), -1).contiguous()

        return t.sparse_csr_tensor(
            idx_crow,
            idx_col,
            val,
            self.shape,
            dtype=self.dtype,
            device=self.device,
        )

    def to_sparse_csc(self, int32: bool = False) -> Float[t.Tensor, "*b r c *d"]:
        if int32:
            idx_ccol = self.sp_topo.idx_ccol_int32
            idx_row_csc = self.sp_topo.idx_row_csc_int32
        else:
            idx_ccol = self.sp_topo.idx_ccol
            idx_row_csc = self.sp_topo.idx_row_csc

        if self.n_batch_dim == 0:
            val = self.val[self.sp_topo.coo_to_csc_perm]
        else:
            val = (
                self.val[self.sp_topo.coo_to_csc_perm]
                .view(self.size(0), -1)
                .contiguous()
            )

        return t.sparse_csc_tensor(
            idx_ccol,
            idx_row_csc,
            val,
            self.shape,
            dtype=self.dtype,
            device=self.device,
        )

    def to_dense(self) -> Float[t.Tensor, "*b r c *d"]:
        return self.to_sparse_coo().to_dense()

    def to(self, *args, **kwargs) -> SparseOperator:
        new_val = self.val.to(*args, **kwargs)

        # The topology object ignores dtype
        new_sp_topo = self.sp_topo.to(
            device=new_val.device,
            copy=kwargs.get("copy", False),
            non_blocking=kwargs.get("non_blocking", False),
        )

        return SparseOperator(new_sp_topo, new_val)

    @property
    def zeros_like(self) -> SparseOperator:
        return SparseOperator(self.sp_topo, t.zeros_like(self.val))

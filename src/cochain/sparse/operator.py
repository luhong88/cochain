from __future__ import annotations

from dataclasses import dataclass

import torch as t
from jaxtyping import Float, Integer

from ._matmul import dense_sp_mm, sp_dense_mm, sp_mv, sp_sp_mm, sp_vm
from ._sp_topo import SparseTopology


def _validate_coo_idx_shape(
    coo_idx: Integer[t.LongTensor, "sp nnz"], shape: tuple[int, ...] | t.Size
):
    match len(shape):
        case 1:
            raise ValueError(
                "The 'idx_coo' tensor must have at least two sparse dimensions."
            )

        case 2:
            if coo_idx.size(0) != 2:
                raise ValueError(
                    "For a 2D sparse coo tensor, 'idx_coo' must be of shape (2, nnz)."
                )

        case 3:
            if coo_idx.size(0) != 3:
                raise ValueError(
                    "For a sparse coo tensor with a batch dimension, 'idx_coo' "
                    + "must be of shape (3, nnz)."
                )

            nnz = coo_idx.size(-1)
            n_batch = shape[0]
            batch_idx = coo_idx[0]

            # For batched sparse tensors, enforce the condition that the tensor
            # has equal nnz along the batch dimension, which is required for
            # conversion to sparse csr format.

            # If the input tensor has equal nnz along the batch dimension, then
            # the nnz per tensor in the batch is given by nnz // batch.
            if nnz % n_batch != 0:
                raise ValueError(
                    f"Total nnz ({nnz}) is not divisible by batch size ({n_batch})."
                )

            nnz_per_batch = nnz // n_batch

            # It is possible for a tensor to have non-equal nnz along the batch
            # dimension but still satisfies nnz % batch = 0 (e.g., if the first
            # tensor has 6 nnz, while the second has 2). This second check rules
            # out this possibility.
            if batch_idx is not None:
                batch_counts = t.bincount(batch_idx, minlength=n_batch)
                if not (batch_counts == nnz_per_batch).all():
                    raise ValueError(
                        "The equal nnz per batch item condition is not met."
                    )

        case _:
            raise NotImplementedError(
                "More than one batch dimensions is not supported."
            )


def _validate_matmul_args(self: SparseOperator, other: SparseOperator | t.Tensor):
    if self.n_batch_dim > 0:
        raise NotImplementedError(
            "__matmul__ with batched SparseOperator is not supported."
        )

    if self.n_dense_dim > 0:
        raise NotImplementedError(
            "__matmul__ with sparse hybrid SparseOperator is not supported."
        )

    match other:
        case SparseOperator():
            if other.n_batch_dim > 0:
                raise NotImplementedError(
                    "__matmul__ with batched SparseOperator is not supported."
                )

            if other.n_dense_dim > 0:
                raise NotImplementedError(
                    "__matmul__ with sparse hybrid SparseOperator is not supported."
                )

        case t.Tensor():
            if (other.ndim < 1) or (other.ndim > 2):
                raise NotImplementedError(
                    f"__matmul__ with tensor of shape {other.shape} is not supported."
                )

        case _:
            raise TypeError(
                f"__matmul__ between SparseOperator and {type(other)} is not supported."
            )


@dataclass
class SparseOperator:
    val: Float[t.Tensor, " nnz *d"]
    sp_topo: Integer[SparseTopology, "*b r c"]

    def __post_init__(self):
        if self.val.device != self.sp_topo.device:
            raise RuntimeError("'val' and 'sp_topo' must be on the same device.")

        if self.val.size(0) != self.sp_topo._nnz():
            raise ValueError("nnz mismatch between 'val' and 'sp_topo'.")

        if not t.isfinite(self.val).all():
            raise ValueError("SparseOperator values contain NaN or Inf.")

        # Enforce contiguous memory layout.
        self.val = self.val.contiguous()

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
        _validate_matmul_args(self, other)

        match other:
            case SparseOperator():
                return sp_sp_mm(self.val, self.sp_topo, other.val, other.sp_topo)

            case t.Tensor():
                match other.ndim:
                    case 1:
                        return sp_mv(self.val, self.sp_topo, other)
                    case 2:
                        return sp_dense_mm(self.val, self.sp_topo, other)

    def __rmatmul__(self, other):
        """
        Implement other @ self
        """
        _validate_matmul_args(self, other)

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

    def to_sparse_coo(self) -> Float[t.Tensor, "*b r c *d"]:
        return t.sparse_coo_tensor(
            self.sp_topo.idx_coo,
            self.val,
            self.sp_topo.shape,
            dtype=self.dtype,
            device=self.device,
        )

    def to_sparse_csr(self, int32: bool = False) -> Float[t.Tensor, "*b r c *d"]:
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

    def to_sparse_csc(self, int32: bool = False) -> Float[t.Tensor, "*b r c *d"]:
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

        return SparseOperator(new_val, new_sp_topo)

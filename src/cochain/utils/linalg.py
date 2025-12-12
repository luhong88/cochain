import torch as t
from jaxtyping import Float


def diag_sp_mm(
    diag: Float[t.Tensor, "r"], sp: Float[t.Tensor, "r c"]
) -> Float[t.Tensor, "r c"]:
    """
    Performs the matrix multiplication `D@A` where `D` is diagonal and `A` is a
    2D sparse tensor in COO format. Effectively, `D@A` scales the `i`th row of `A`
    by the `i`th element of `D`.
    """
    rows = sp.indices()[0]
    scaled_vals = sp.values() * diag[rows]
    return t.sparse_coo_tensor(sp.indices(), scaled_vals, sp.size()).coalesce()


def sp_diag_mm(
    sp: Float[t.Tensor, "r c"], diag: Float[t.Tensor, "c"]
) -> Float[t.Tensor, "r c"]:
    """
    Performs the matrix multiplication `A@D` where `D` is diagonal and `A` is a
    2D sparse tensor in COO format. Effectively, `A@D` scales the `i`th col of `A`
    by the `i`th element of `D`.
    """
    cols = sp.indices()[1]
    scaled_vals = sp.values() * diag[cols]
    return t.sparse_coo_tensor(sp.indices(), scaled_vals, sp.size()).coalesce()

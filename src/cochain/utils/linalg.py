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


def sp_diag(sp_m: Float[t.Tensor, "dim1 dim2"]) -> Float[t.Tensor, "min_dim"]:
    """
    Extract the diagonal elements of a 2D sparse tensor.
    """
    sp_m = sp_m.coalesce()

    idx = sp_m.indices()
    val = sp_m.values()

    mask = idx[0] == idx[1]

    diag_vals = val[mask]
    diag_indices = idx[0][mask]

    diag = t.zeros(min(sp_m.shape), dtype=sp_m.dtype, device=sp_m.device)

    diag[diag_indices] = diag_vals

    return diag

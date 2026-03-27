import torch as t
from jaxtyping import Float, Integer


def _vertex_perm_parity(
    verts: Integer[t.LongTensor, "vert 1"], dtype: t.dtype
) -> Float[t.Tensor, " vert"]:
    return t.ones(verts.size(0), dtype=dtype, device=verts.device)


def _edge_perm_parity(
    edges: Integer[t.LongTensor, "edge 2"], dtype: t.dtype
) -> Float[t.Tensor, " edge"]:
    return (edges[:, 1] - edges[:, 0]).sign().to(dtype=dtype)


def _tri_perm_parity(
    tris: Integer[t.LongTensor, "tri 3"], dtype: t.dtype
) -> Float[t.Tensor, " tri"]:
    i, j, k = tris.unbind(-1)

    parity = t.ones(tris.size(0), dtype=dtype, device=tris.device)
    for idx1, idx2 in [[i, j], [i, k], [j, k]]:
        parity.mul_(t.sign(idx2 - idx1))
    return parity


def _tet_perm_parity(
    tets: Integer[t.LongTensor, "tet 4"], dtype: t.dtype
) -> Float[t.Tensor, " tet"]:
    i, j, k, l = tets.unbind(-1)

    parity = t.ones(tets.size(0), dtype=dtype, device=tets.device)
    for idx1, idx2 in [[i, j], [i, k], [i, l], [j, k], [j, l], [k, l]]:
        parity.mul_(t.sign(idx2 - idx1))

    return parity


def compute_lex_rel_orient(
    splx: Integer[t.LongTensor, "*b splx vert"], dtype: t.dtype = t.float32
) -> Float[t.Tensor, "*b splx"]:
    if splx.size(-2) == 0:
        return splx[..., 0]

    splx_flat = splx.flatten(end_dim=-2)

    splx_dim = splx_flat.size(-1) - 1

    match splx_dim:
        case 0:
            perm_parity = _vertex_perm_parity(splx_flat, dtype)
        case 1:
            perm_parity = _edge_perm_parity(splx_flat, dtype)
        case 2:
            perm_parity = _tri_perm_parity(splx_flat, dtype)
        case 3:
            perm_parity = _tet_perm_parity(splx_flat, dtype)
        case _:
            raise NotImplementedError()

    perm_parity_shaped = perm_parity.view(*splx.shape[:-1])

    return perm_parity_shaped

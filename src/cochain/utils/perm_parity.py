import torch
from jaxtyping import Float, Integer
from torch import LongTensor, Tensor


def _vertex_perm_parity(
    verts: Integer[LongTensor, "vert 1"], dtype: torch.dtype
) -> Float[Tensor, " vert"]:
    return torch.ones(verts.size(0), dtype=dtype, device=verts.device)


def _edge_perm_parity(
    edges: Integer[LongTensor, "edge 2"], dtype: torch.dtype
) -> Float[Tensor, " edge"]:
    return (edges[:, 1] - edges[:, 0]).sign().to(dtype=dtype)


def _tri_perm_parity(
    tris: Integer[LongTensor, "tri 3"], dtype: torch.dtype
) -> Float[Tensor, " tri"]:
    i, j, k = tris.unbind(-1)

    parity = torch.ones(tris.size(0), dtype=dtype, device=tris.device)
    for idx1, idx2 in [[i, j], [i, k], [j, k]]:
        parity.mul_(torch.sign(idx2 - idx1))
    return parity


def _tet_perm_parity(
    tets: Integer[LongTensor, "tet 4"], dtype: torch.dtype
) -> Float[Tensor, " tet"]:
    i, j, k, l = tets.unbind(-1)

    parity = torch.ones(tets.size(0), dtype=dtype, device=tets.device)
    for idx1, idx2 in [[i, j], [i, k], [i, l], [j, k], [j, l], [k, l]]:
        parity.mul_(torch.sign(idx2 - idx1))

    return parity


def compute_lex_rel_orient(
    splx: Integer[LongTensor, "*b splx vert"], dtype: torch.dtype = torch.float32
) -> Float[Tensor, "*b splx"]:
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

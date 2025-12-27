import torch as t
from jaxtyping import Integer


def _vertex_perm_parity(
    verts: Integer[t.LongTensor, "vert 1"],
) -> Integer[t.LongTensor, " vert"]:
    return t.ones(verts.size(0), dtype=verts.dtype, device=verts.device)


def _edge_perm_parity(
    edges: Integer[t.LongTensor, "edge 2"],
) -> Integer[t.LongTensor, " edge"]:
    return (edges[:, 1] - edges[:, 0]).sign().to(dtype=edges.dtype)


def _tri_perm_parity(
    tris: Integer[t.LongTensor, "tri 3"],
) -> Integer[t.LongTensor, " tri"]:
    i, j, k = tris.unbind(-1)

    parity = t.ones(tris.size(0), dtype=tris.dtype, device=tris.device)
    for idx1, idx2 in [[i, j], [i, k], [j, k]]:
        parity.mul_(t.sign(idx2 - idx1))
    return parity


def _tet_perm_parity(
    tets: Integer[t.LongTensor, "tet 4"],
) -> Integer[t.LongTensor, " tet"]:
    i, j, k, l = tets.unbind(-1)

    parity = t.ones(tets.size(0), dtype=tets.dtype, device=tets.device)
    for idx1, idx2 in [[i, j], [i, k], [i, l], [j, k], [j, l], [k, l]]:
        parity.mul_(t.sign(idx2 - idx1))

    return parity


def compute_lex_rel_orient(
    simps: Integer[t.LongTensor, "simp vert"],
) -> Integer[t.LongTensor, " simp"]:
    simp_dim = simps.size(-1) - 1

    match simp_dim:
        case 0:
            return _vertex_perm_parity(simps)
        case 1:
            return _edge_perm_parity(simps)
        case 2:
            return _tri_perm_parity(simps)
        case 3:
            return _tet_perm_parity(simps)
        case _:
            raise NotImplementedError()

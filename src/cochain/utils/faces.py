import itertools

import torch as t
from jaxtyping import Integer


def enumerate_faces(
    simp_dim: int, face_dim: int, device: t.device
) -> Integer[t.LongTensor, "face vert"]:
    """
    For a simplex of dimension `simp_dim`, enumerate all faces of dimension
    `face_dim` (up to vertex index permutation) in lex order.
    """
    if face_dim > simp_dim:
        raise ValueError()

    return t.tensor(
        list(itertools.combinations(list(range(simp_dim + 1)), face_dim + 1)),
        device=device,
    )


# TODO: depreciate in favor of enumerate_faces()
def enumerate_unique_faces(
    simp_dim: int, face_dim: int, device: t.device
) -> Integer[t.LongTensor, "face vert"]:
    if face_dim > simp_dim:
        raise ValueError()

    match simp_dim:
        case 2:
            match face_dim:
                case 0:
                    return t.tensor([[0], [1], [2]], dtype=t.long, device=device)
                case 1:
                    return t.tensor(
                        [[0, 1], [0, 2], [1, 2]], dtype=t.long, device=device
                    )
                case 2:
                    return t.tensor([[0, 1, 2]], dtype=t.long, device=device)

        case 3:
            match face_dim:
                case 0:
                    return t.tensor([[0], [1], [2], [3]], dtype=t.long, device=device)
                case 1:
                    # TODO: check reason for non-lex ordering of vertices
                    return t.tensor(
                        [[0, 1], [0, 2], [1, 2], [1, 3], [2, 3], [0, 3]],
                        dtype=t.long,
                        device=device,
                    )
                case 2:
                    # For each tet and each vertex, find the outward-facing triangle
                    # opposite to the vertex (note that the way the triangles are
                    # indexed here satisfies the right-hand rule for positively
                    # oriented tets).
                    return t.tensor(
                        [[1, 2, 3], [0, 3, 2], [0, 1, 3], [0, 2, 1]],
                        dtype=t.long,
                        device=device,
                    )
                case 3:
                    return t.tensor([[0, 1, 2, 3]], dtype=t.long, device=device)
                case _:
                    return ValueError()

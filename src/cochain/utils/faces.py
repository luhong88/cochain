import itertools

import torch as t
from jaxtyping import Integer


def enumerate_faces(
    simp_dim: int, face_dim: int, device: t.device
) -> Integer[t.Tensor, "face vert"]:
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

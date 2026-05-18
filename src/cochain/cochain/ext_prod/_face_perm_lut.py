import itertools
from dataclasses import dataclass

import torch
from jaxtyping import Float, Integer
from torch import Tensor

from ...utils.perm_parity import compute_lex_rel_orient


@dataclass
class FacePermLUT:
    """
    A "lookup table" for face permutation information.

    This class contains face permutation information required for computing the
    anti-symmetrized cup product between a k-cochain and an l-cochain.

    Parameters
    ----------
    k
        The order of the k-cochain.
    l
        The order of the l-cochain.
    unique_front : [uf_face, vert]
        A list of all unique k-subcomplexes (up to vertex permutation) in a
        (k+l)-simplex.
    unique_back : [ub_face, verta]
        A list of all unique l-subcomplexes (up to vertex permutation) in a
        (k+l)-simplex.
    front_idx : [face,]
        A list of indices referring to the indices of k-front faces in the
        `unique_front` tensor. Together with the `back_idx`, forms the
        (k-subcomplex, l-subcomplex) index pairs that enumerate all valid k-front
        face/k-back face splits of a (k+l)-simplex (up to vertex permutation within
        the fron/back subcomplexes).
    back_idx : [face,]
        See `front_idx`.
    sign : [1, face, 1]
        The parity of the permutation required to rearrange the (k+l)-simplex
        vertex indices to the front/back split order. Note that this correction is
        distinct from the sign correction required to map a geometric simplex to
        a canonical simplex (with lex sorted indices).
    """

    k: int
    l: int
    unique_front: Integer[Tensor, "uf_face vert"]
    unique_back: Integer[Tensor, "ub_face vert"]
    front_idx: Integer[Tensor, " face"]
    back_idx: Integer[Tensor, " face"]
    sign: Float[Tensor, "1 face 1"]


def compute_face_perm_lut(k: int, l: int) -> FacePermLUT:
    m = k + l

    all_perms = torch.tensor(list(itertools.permutations(range(m + 1))))

    f_faces = all_perms[:, : k + 1]
    b_faces = all_perms[:, k:]

    # Consider a 2-simplex i<j<k and the permutation jik, which is split into a
    # 1-front face ji and a 1-backface ik. We need to account for the top-level
    # permutation parity (jik is related to ijk by an odd permutation), as well
    # as the within-face permutation (ji is related to ij by an odd permutation,
    # and ik is related to ik by an even permutation). This is why three terms
    # are needed to fully account for the permutation parity of the face split.
    sign = (
        compute_lex_rel_orient(all_perms)
        * compute_lex_rel_orient(f_faces)
        * compute_lex_rel_orient(b_faces)
    )

    f_faces_sorted = f_faces.sort(dim=-1).values
    b_faces_sorted = b_faces.sort(dim=-1).values

    unique_front, front_idx = f_faces_sorted.unique(dim=0, return_inverse=True)
    unique_back, back_idx = b_faces_sorted.unique(dim=0, return_inverse=True)

    lut = FacePermLUT(
        k=k,
        l=l,
        unique_front=unique_front,
        unique_back=unique_back,
        front_idx=front_idx,
        back_idx=back_idx,
        sign=sign.view(1, -1, 1),
    )

    return lut

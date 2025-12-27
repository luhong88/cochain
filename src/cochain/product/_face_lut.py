from dataclasses import dataclass

import torch as t
from jaxtyping import Float, Integer


# TODO: explain why it is not necessary to enumerate all permutation of vertex
# indices.
@dataclass
class FaceLUT:
    """
    This class contains face permutation information required for computing the
    anti-symmetrized cup product between a k-cochain and an l-cochain.

    unique_front: a list of all unique k-subcomplexes (up to vertex permutation)
    in a (k+l)-simplex.

    unique_back: a lsit of all unique l-subcomplexes (up to vertex permutation)
    in a (k+l)-simplex.

    front_idx/back_idx: indices of (k-subcomplex, l-subcomplex) pairs that form
    a valid k-front face/k-back face split of a (k+l)-simplex (up to vertex
    permutation within the subcomplexes).

    sign: the parity of the permutation required to rearrange the (k+l)-simplex
    vertex indices to the front/back split order.
    """

    k: int
    l: int
    unique_front: Integer[t.Tensor, "1 uf_face vert"]
    unique_back: Integer[t.Tensor, "1 ub_face vert"]
    front_idx: Integer[t.Tensor, " face"]
    back_idx: Integer[t.Tensor, " face"]
    sign: Float[t.Tensor, "1 face 1"]

    def __post_init__(self):
        for attr in [
            "unique_front",
            "unique_back",
            "front_idx",
            "back_idx",
            "sign",
        ]:
            setattr(self, attr, t.tensor(getattr(self, attr)))

    def flip(self):
        """
        Because the anti-symmetrized cup product is graded-commutative, it is not
        necessary to write out the front/back face splits explicitly when `k` and
        `l` are flipped. When this happens, the front and back face information
        switches position, and the permutation sign is multiplied by a factor of
        (-1)^(k + l) due to graded commutativity.
        """
        return FaceLUT(
            self.l,
            self.k,
            self.unique_back,
            self.unique_front,
            self.back_idx,
            self.front_idx,
            ((-1) ** (self.k * self.l)) * self.sign,
        )

    def to(self, device: str | t.device):
        for attr, value in self.__dict__.items():
            if t.is_tensor(value):
                setattr(self, attr, value.to(device))
        return self


# 0-simplex: {i}
# --------------------
# front/back perm sign
# --------------------
# [i]/[i]    i     1
# --------------------
s00 = FaceLUT(
    k=0,
    l=0,
    unique_front=[[0]],
    unique_back=[[0]],
    front_idx=[0],
    back_idx=[0],
    sign=[[[1.0]]],
)

# 1-simplex: {i < j}
# --------------------
# front/back perm sign
# --------------------
# [i]/[ij]   ij    1
# [j]/[ij]   ji   -1
# --------------------
# [ij]/[i]   ji   -1
# [ij]/[j]   ij    1
# --------------------
s01 = FaceLUT(
    k=0,
    l=1,
    unique_front=[[0], [1]],
    unique_back=[[0, 1]],
    front_idx=[0, 1],
    back_idx=[0, 0],
    sign=[[[1.0], [-1.0]]],
)
s10 = s01.flip()

# 2-simplex: {i < j < k}
# --------------------
# front/back perm sign
# --------------------
# [i]/[ijk]  ijk   1
# [j]/[ijk]  jik  -1
# [k]/[ijk]  kij   1
# --------------------
# [ij][ik]   jik  -1
# [ij][jk]   ijk   1
# [ik][ij]   kij   1
# [ik][jk]   ikj  -1
# [jk][ij]   kji  -1
# [jk][ik]   jki   1
# --------------------
# [ijk]/[i]  jki   1
# [ijk]/[j]  ikj  -1
# [ijk]/[k]  ijk   1
# --------------------
s02 = FaceLUT(
    k=0,
    l=2,
    unique_front=[[0], [1], [2]],
    unique_back=[[0, 1, 2]],
    front_idx=[0, 1, 2],
    back_idx=[0, 0, 0],
    sign=[[[1.0], [-1.0], [1.0]]],
)
s11 = FaceLUT(
    k=1,
    l=1,
    # 0: ij, 1: ik, 2: jk
    unique_front=[[0, 1], [0, 2], [1, 2]],
    # 0: ij, 1: ik, 2: jk
    unique_back=[[0, 1], [0, 2], [1, 2]],
    front_idx=[0, 0, 1, 1, 2, 2],
    back_idx=[1, 2, 0, 2, 0, 1],
    sign=[[[-1.0], [1.0], [1.0], [-1.0], [-1.0], [1.0]]],
)
s20 = s02.flip()

# 3-simplex: {i < j < k < l}
# --------------------
# front/back perm sign
# --------------------
# [i]/[ijkl] ijkl  1
# [j]/[ijkl] jikl -1
# [k]/[ijkl] kijl  1
# [l]/[ijkl] lijk -1
# --------------------
# [ij]/[ikl] jikl -1
# [ij]/[jkl] ijkl  1
# [ik]/[ijl] kijl  1
# [ik]/[jkl] ikjl -1
# [il]/[ijk] lijk -1
# [il]/[jkl] iljk  1
# [jk]/[ijl] kjil -1
# [jk]/[ikl] jkil  1
# [jl]/[ijk] ljik  1
# [jl]/[ikl] jlik -1
# [kl]/[ijk] lkij -1
# [kl]/[ijl] klij  1
# --------------------
# [ijk]/[il] jkil  1
# [ijk]/[jl] ikjl -1
# [ijk]/[kl] ijkl  1
# [ijl]/[ik] jlik -1
# [ijl]/[jk] iljk  1
# [ijl]/[kl] ijlk -1
# [ikl]/[ij] klij  1
# [ikl]/[jk] ilkj -1
# [ikl]/[jl] iklj  1
# [jkl]/[ij] klji -1
# [jkl]/[ik] jlki  1
# [jkl]/[il] jkli -1
# --------------------
# [ijkl]/[i] jkli -1
# [ijkl]/[j] iklj  1
# [ijkl]/[k] ijlk -1
# [ijkl]/[l] ijkl  1
# --------------------
s03 = FaceLUT(
    k=0,
    l=3,
    unique_front=[[0], [1], [2], [3]],
    unique_back=[[0, 1, 2, 3]],
    front_idx=[0, 1, 2, 3],
    back_idx=[0, 0, 0, 0],
    sign=[[[1.0], [-1.0], [1.0], [-1.0]]],
)
s12 = FaceLUT(
    k=1,
    l=2,
    # 0: ij, 1: ik, 2: il, 3: jk, 4: jl, 5: kl
    unique_front=[[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]],
    # 0: ijk, 1: ijl, 2: ikl, 3: jkl
    unique_back=[[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]],
    front_idx=[0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5],
    back_idx=[2, 3, 1, 3, 0, 3, 1, 2, 0, 2, 0, 1],
    sign=[
        [
            [-1.0],
            [1.0],
            [1.0],
            [-1.0],
            [-1.0],
            [1.0],
            [-1.0],
            [1.0],
            [1.0],
            [-1.0],
            [-1.0],
            [1.0],
        ]
    ],
)
s21 = s12.flip()
s30 = s03.flip()


face_lut: dict[int, FaceLUT] = {
    (perm.k, perm.l): perm
    for perm in [s00, s01, s10, s02, s11, s20, s03, s12, s21, s30]
}

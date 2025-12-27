from typing import Literal

import torch as t
from jaxtyping import Float, Integer

from ..complex import SimplicialComplex
from ..utils.search import simplex_search
from ._face_lut import face_lut


class CupProduct(t.nn.Module):
    def __init__(
        self,
        k: int,
        l: int,
        mesh: SimplicialComplex,
    ):
        """
        Compute the cup product between a `k`-cochain `ξ` and an `l`-cochain `η`,
        which produces a (k+l)-cochain.

        This operator satisfies the following properties:
        * Associativity: `(ξ ⋀ η) ⋀ μ = ξ ⋀ (η ⋀ μ)`.
        * Leibniz rule: `d(ξ ⋀ η) = dξ ⋀ η + (-1)^k ξ ⋀ dη`, where `d` is the
        coboundary operator.

        Note that this operator does not satisfy the graded commutativity property;
        i.e., in general, it is not true that `ξ ⋀ η = (-1)^(k*l)*(η ⋀ ξ)`, unless
        k = l = 0.
        """
        super().__init__()

        m = k + l

        simp_map = {
            dim: simp
            for dim, simp in enumerate([mesh.verts, mesh.edges, mesh.tris, mesh.tets])
        }

        # The cup product is purely topological and should be invariant to the
        # geometric orientation of each simplex in the mesh. To enforce this, we
        # compute the product on lex sorted simplices. The SimplicialComplex
        # class stores k-simplices this way, except for the top level n-simplices.
        # Therefore, if k = n, l = n, or k + l = n, then a sort is needed.
        if m == mesh.dim:
            m_simps_sorted = simp_map[m].sort(dim=-1).values
        else:
            m_simps_sorted = simp_map[m]

        f_face_idx = simplex_search(
            key_simps=simp_map[k],
            query_simps=m_simps_sorted[:, : k + 1],
            sort_key_simp=True if k == mesh.dim else False,
            sort_key_vert=True if k == mesh.dim else False,
            sort_query_vert=False,
        )

        self.f_face_idx: Integer[t.LongTensor, " m_simp"]
        self.register_buffer("f_face_idx", f_face_idx)

        b_face_idx = simplex_search(
            key_simps=simp_map[l],
            query_simps=m_simps_sorted[:, k:],
            sort_key_simp=True if l == mesh.dim else False,
            sort_key_vert=True if l == mesh.dim else False,
            sort_query_vert=False,
        )

        self.b_face_idx: Integer[t.LongTensor, " m_simp"]
        self.register_buffer("b_face_idx", b_face_idx)

    def forward(
        self,
        k_cochain: Float[t.Tensor, " k_simp *ch_in"],
        l_cochain: Float[t.Tensor, " l_simp *ch_in"],
        pairing: Literal["scalar", "dot", "cross", "outer"] = "scalar",
    ) -> Float[t.Tensor, " m_simp *ch_out"]:
        k_cochain_at_f_face = k_cochain[self.f_face_idx]
        l_cochain_at_b_face = l_cochain[self.b_face_idx]

        # If pairing='scalar', *ch_in can match to an arbitrary number of channel
        # dimensions; for other pairing method, *ch_in need to match to one dimension.
        match pairing:
            case "scalar":
                prod = k_cochain_at_f_face * l_cochain_at_b_face

            case "dot":
                prod = t.sum(
                    k_cochain_at_f_face * l_cochain_at_b_face,
                    dim=-1,
                    keepdim=True,
                )

            case "cross":
                prod = t.cross(k_cochain_at_f_face, l_cochain_at_b_face, dim=-1)

            case "outer":
                prod = t.einsum("nk,nl->nkl", k_cochain_at_f_face, l_cochain_at_b_face)

            case _:
                raise ValueError()

        return prod


class AntisymmetricCupProduct(t.nn.Module):
    def __init__(
        self,
        k: int,
        l: int,
        mesh: SimplicialComplex,
    ):
        """
        Compute the anti-symmetrized cup product between a `k`-cochain `ξ` and an
        `l`-cochain `η`, which produces a (k+l)-cochain. This differs from the
        regular cup product in that it averages over all permutations of k-front
        and k-back face splits, thus is invariant to simplex vertex permutation.

        This operator satisfies the the graded commutativity property; i.e.,
        `ξ ⋀ η = (-1)^(k*l)*(η ⋀ ξ)`. However, unlike the regular cup product,
        it does not ingeneral satisfies the associativity rule or the Leibniz
        rule.
        """
        super().__init__()

        m = k + l

        simp_map = {
            dim: simp
            for dim, simp in enumerate([mesh.verts, mesh.edges, mesh.tris, mesh.tets])
        }

        perm = face_lut[(k, l)]

        self.perm_sign: Float[t.Tensor, "1 face 1"]
        self.register_buffer("perm_sign", perm.sign)

        uf_face: Integer[t.LongTensor, " m_simp uf_face k+1"] = simp_map[m][
            :, perm.unique_front
        ]
        uf_face_flat = uf_face.view(-1, k + 1)
        uf_face_idx: Integer[t.LongTensor, " m_simp uf_face"] = simplex_search(
            key_simps=simp_map[k],
            query_simps=uf_face_flat,
            sort_key_simp=True if k == mesh.dim else False,
            sort_key_vert=True if k == mesh.dim else False,
            sort_query_vert=True,
        )
        f_face_idx = uf_face_idx[:, perm.front_idx]

        self.f_face_idx: Integer[t.LongTensor, "m_simp face"]
        self.register_buffer("f_face_idx", f_face_idx)

        ub_face: Integer[t.LongTensor, " m_simp ub_face l+1"] = (
            simp_map[m][:, perm.unique_back].sort(dim=-1).values
        )
        ub_face_flat = ub_face.view(-1, l + 1)
        ub_face_idx: Integer[t.LongTensor, " (k+l)_simp ub_face"] = simplex_search(
            key_simps=simp_map[l],
            query_simps=ub_face_flat,
            sort_key_simp=True if l == mesh.dim else False,
            sort_key_vert=True if l == mesh.dim else False,
            sort_query_vert=True,
        )
        b_face_idx = ub_face_idx[:, perm.front_idx]

        self.b_face_idx: Integer[t.LongTensor, "m_simp face"]
        self.register_buffer("b_face_idx", b_face_idx)

    def forward(
        self,
        k_cochain: Float[t.Tensor, " k_simp *ch_in"],
        l_cochain: Float[t.Tensor, " l_simp *ch_in"],
        pairing: Literal["scalar", "dot", "cross", "outer"] = "scalar",
    ) -> Float[t.Tensor, " m_simp *ch_out"]:
        k_cochain_at_f_face: Float[t.Tensor, "m_simp face *ch_in"] = k_cochain[
            self.f_face_idx
        ]
        l_cochain_at_b_face: Float[t.Tensor, "m_simp face *ch_in"] = l_cochain[
            self.b_face_idx
        ]

        # If pairing='scalar', *ch_in can match to an arbitrary number of channel
        # dimensions; for other pairing method, *ch_in need to match to one dimension.
        match pairing:
            case "scalar":
                prod = t.mean(
                    self.perm_sign * k_cochain_at_f_face * l_cochain_at_b_face, dim=1
                )
            case "dot":
                prod = t.mean(
                    t.sum(
                        self.perm_sign * k_cochain_at_f_face * l_cochain_at_b_face,
                        dim=-1,
                        keepdim=True,
                    ),
                    dim=1,
                )
            case "cross":
                prod = t.mean(
                    self.perm_sign
                    * t.cross(k_cochain_at_f_face, l_cochain_at_b_face, dim=-1),
                    dim=1,
                )
            case "outer":
                prod = t.einsum(
                    "nf,nfk,nfl->nkl",
                    self.perm_sign.flatten(start_dim=1),
                    k_cochain_at_f_face,
                    l_cochain_at_b_face,
                ) / self.perm_sign.size(1)
            case _:
                raise ValueError()

        return prod

from typing import Literal

import torch as t
from jaxtyping import Float, Integer

from ..complex import SimplicialComplex
from ..utils.perm_parity import compute_lex_rel_orient
from ..utils.search import simplex_search
from ._face_lut import compute_face_lut


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

        # Note that, in algebraic topology, the orientation on the chain groups
        # is imposed globally by the lex order of the vertex indices. In the
        # SimplicialComplex class, this orientation is imposed on all but the top
        # level n-simplices, where the vertex index ordering carries information
        # on geometric orientation. In general, such "geometric" n-simplices are
        # not necessarily vectors in the n-th chain group defined using the
        # canonically oriented n-simplices as bases. Therefore, a vector space
        # isomorphism/coordinate transform is required to map the "geometric"
        # n-simplices to the canonical n-simplices prior to the application of
        # cup product, and this mapping incurs a permutation sign correction (i.e.,
        # a geometric n-simplex is related to the corresponding canonical n-simplex
        # by the parity of the permutation required to put its vertices in lex
        # order). For the cup product between a k- and l-cochain, this results in
        # three potential sign corrections:
        #
        # 1. Convert the (k+l)-simplices to the canonical (k+l)-simplices, which
        #    incurs a sign correction at the (k+l)-simplex level.
        # 2. Identify the k-front and k-back faces of the canonical (k+l)-simplices,
        #    and then convert them to the geometric (k+l)-simplices, which incurs
        #    two sign corrections on the front/back face level.
        # 3. Look up the k- and l-cochain values at the geometric front/back faces,
        #    and find their product.

        # Compute (k+l)-simplex sign correction
        if m == mesh.dim:
            m_simps_sorted = simp_map[m].sort(dim=-1).values
            m_simp_parity = compute_lex_rel_orient(simp_map[m]).to(
                dtype=mesh.vert_coords.dtype
            )
        else:
            m_simps_sorted = simp_map[m]
            m_simp_parity = t.ones(1, dtype=mesh.vert_coords.dtype).expand(
                simp_map[m].size(0)
            )

        self.m_simp_parity: Float[t.Tensor, " m_simp"]
        self.register_buffer("m_simp_parity", m_simp_parity)

        # Identify the k-front faces of (k+l)-simplices and their sign correction
        f_face_idx = simplex_search(
            key_simps=simp_map[k],
            query_simps=m_simps_sorted[:, : k + 1],
            sort_key_simp=True if k == mesh.dim else False,
            sort_key_vert=True if k == mesh.dim else False,
            sort_query_vert=False,
        )

        self.f_face_idx: Integer[t.LongTensor, " m_simp"]
        self.register_buffer("f_face_idx", f_face_idx)

        if k == mesh.dim:
            f_face_parity = compute_lex_rel_orient(simp_map[k][self.f_face_idx])
        else:
            f_face_parity = t.ones(1, dtype=mesh.vert_coords.dtype).expand(
                self.f_face_idx.size(0)
            )

        self.f_face_parity: Integer[t.LongTensor, " m_simp"]
        self.register_buffer("f_face_parity", f_face_parity)

        # Identify the k-back faces of (k+l)-simplices and their sign correction
        b_face_idx = simplex_search(
            key_simps=simp_map[l],
            query_simps=m_simps_sorted[:, k:],
            sort_key_simp=True if l == mesh.dim else False,
            sort_key_vert=True if l == mesh.dim else False,
            sort_query_vert=False,
        )

        self.b_face_idx: Integer[t.LongTensor, " m_simp"]
        self.register_buffer("b_face_idx", b_face_idx)

        if l == mesh.dim:
            b_face_parity = compute_lex_rel_orient(simp_map[l][self.b_face_idx])
        else:
            b_face_parity = t.ones(1, dtype=mesh.vert_coords.dtype).expand(
                self.b_face_idx.size(0)
            )

        self.b_face_parity: Integer[t.LongTensor, " m_simp"]
        self.register_buffer("b_face_parity", b_face_parity)

    def forward(
        self,
        k_cochain: Float[t.Tensor, " k_simp *ch_in"],
        l_cochain: Float[t.Tensor, " l_simp *ch_in"],
        pairing: Literal["scalar", "dot", "cross", "outer"] = "scalar",
    ) -> Float[t.Tensor, " m_simp *ch_out"]:
        k_cochain_at_f_face = t.einsum(
            "n,n...->n...", self.f_face_parity, k_cochain[self.f_face_idx]
        )
        l_cochain_at_b_face = t.einsum(
            "n,n...->n...", self.b_face_parity, l_cochain[self.b_face_idx]
        )

        # If pairing='scalar', *ch_in can match to an arbitrary number of channel
        # dimensions; for other pairing method, *ch_in need to match to one dimension.
        match pairing:
            case "scalar":
                prod = t.einsum(
                    "n,n...->n...",
                    self.m_simp_parity,
                    k_cochain_at_f_face * l_cochain_at_b_face,
                )

            case "dot":
                prod = self.m_simp_parity.view(-1, 1) * t.sum(
                    k_cochain_at_f_face * l_cochain_at_b_face,
                    dim=-1,
                    keepdim=True,
                )

            case "cross":
                prod = self.m_simp_parity.view(-1, 1) * t.cross(
                    k_cochain_at_f_face, l_cochain_at_b_face, dim=-1
                )

            case "outer":
                prod = t.einsum(
                    "n,nk,nl->nkl",
                    self.m_simp_parity,
                    k_cochain_at_f_face,
                    l_cochain_at_b_face,
                )

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
        `l`-cochain `η`, which produces a `(k+l)`-cochain. This differs from the
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

        perm = compute_face_lut(k, l)

        self.perm_sign: Float[t.Tensor, "1 face 1"]
        self.register_buffer("perm_sign", perm.sign.to(mesh.vert_coords.dtype))

        # Compute (k+l)-simplex sign correction.
        if m == mesh.dim:
            m_simps_sorted = simp_map[m].sort(dim=-1).values
            m_simp_parity = compute_lex_rel_orient(simp_map[m]).to(
                dtype=mesh.vert_coords.dtype
            )
        else:
            m_simps_sorted = simp_map[m]
            m_simp_parity = t.ones(1, dtype=mesh.vert_coords.dtype).expand(
                simp_map[m].size(0)
            )

        self.m_simp_parity: Float[t.Tensor, " m_simp"]
        self.register_buffer("m_simp_parity", m_simp_parity)

        # Identify permutations of the  k-front faces of (k+l)-simplices and their
        # sign correction.
        uf_face: Integer[t.LongTensor, " m_simp uf_face k+1"] = m_simps_sorted[
            :, perm.unique_front
        ]
        uf_face_flat = uf_face.view(-1, k + 1)
        uf_face_idx: Integer[t.LongTensor, " m_simp*uf_face"] = simplex_search(
            key_simps=simp_map[k],
            query_simps=uf_face_flat,
            sort_key_simp=True if k == mesh.dim else False,
            sort_key_vert=True if k == mesh.dim else False,
            sort_query_vert=False,
        )
        f_face_idx = uf_face_idx.view(*uf_face.shape[:-1])[:, perm.front_idx]

        self.f_face_idx: Integer[t.LongTensor, "m_simp face"]
        self.register_buffer("f_face_idx", f_face_idx)

        if k == mesh.dim:
            f_face_parity = (
                compute_lex_rel_orient(simp_map[k][uf_face_idx])
                .to(dtype=mesh.vert_coords.dtype)
                .view(*uf_face.shape[:-1])[:, perm.front_idx]
            )
        else:
            f_face_parity = t.ones(1, dtype=mesh.vert_coords.dtype).expand(
                self.f_face_idx.shape
            )

        self.f_face_parity: Integer[t.LongTensor, " m_simp face"]
        self.register_buffer("f_face_parity", f_face_parity)

        # Identify permutations of the  k-back faces of (k+l)-simplices and their
        # sign correction.
        ub_face: Integer[t.LongTensor, " m_simp ub_face l+1"] = (
            m_simps_sorted[:, perm.unique_back].sort(dim=-1).values
        )
        ub_face_flat = ub_face.view(-1, l + 1)
        ub_face_idx: Integer[t.LongTensor, " m_simp*ub_face"] = simplex_search(
            key_simps=simp_map[l],
            query_simps=ub_face_flat,
            sort_key_simp=True if l == mesh.dim else False,
            sort_key_vert=True if l == mesh.dim else False,
            sort_query_vert=False,
        )
        b_face_idx = ub_face_idx.view(*ub_face.shape[:-1])[:, perm.back_idx]

        self.b_face_idx: Integer[t.LongTensor, "m_simp face"]
        self.register_buffer("b_face_idx", b_face_idx)

        if l == mesh.dim:
            b_face_parity = (
                compute_lex_rel_orient(simp_map[l][ub_face_idx])
                .to(dtype=mesh.vert_coords.dtype)
                .view(*ub_face.shape[:-1])[:, perm.back_idx]
            )
        else:
            b_face_parity = t.ones(1, dtype=mesh.vert_coords.dtype).expand(
                self.b_face_idx.shape
            )

        self.b_face_parity: Integer[t.LongTensor, " m_simp face"]
        self.register_buffer("b_face_parity", b_face_parity)

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

        combined_perm_sign: Float[t.Tensor, "m_simp face 1"] = (
            self.m_simp_parity.view(-1, 1, 1)
            * self.f_face_parity.unsqueeze(-1)
            * self.b_face_parity.unsqueeze(-1)
            * self.perm_sign
        )

        # If pairing='scalar', *ch_in can match to an arbitrary number of channel
        # dimensions; for other pairing method, *ch_in need to match to one dimension.
        match pairing:
            case "scalar":
                prod = t.einsum(
                    "nf,nf...->n...",
                    combined_perm_sign.squeeze(-1),
                    k_cochain_at_f_face * l_cochain_at_b_face,
                ) / combined_perm_sign.size(1)

            case "dot":
                prod = t.mean(
                    t.sum(
                        combined_perm_sign * k_cochain_at_f_face * l_cochain_at_b_face,
                        dim=-1,
                        keepdim=True,
                    ),
                    dim=1,
                )

            case "cross":
                prod = t.mean(
                    combined_perm_sign
                    * t.cross(k_cochain_at_f_face, l_cochain_at_b_face, dim=-1),
                    dim=1,
                )

            case "outer":
                prod = t.einsum(
                    "nf,nfk,nfl->nkl",
                    combined_perm_sign.flatten(start_dim=1),
                    k_cochain_at_f_face,
                    l_cochain_at_b_face,
                ) / combined_perm_sign.size(1)

            case _:
                raise ValueError()

        return prod

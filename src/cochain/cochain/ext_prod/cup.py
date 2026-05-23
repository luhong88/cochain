__all__ = ["CupProduct", "AntisymmetricCupProduct"]

from typing import Literal

import torch
from jaxtyping import Float, Integer
from torch import Tensor

from ...complex import SimplicialMesh
from ...utils.perm_parity import compute_lex_rel_orient
from ...utils.search import splx_search
from ._face_perm_lut import compute_face_perm_lut


class CupProduct(torch.nn.Module):
    r"""
    Compute the cup product between a k-cochain and an l-cochain.

    Parameters
    ----------
    k
        The order of the k-cochain.
    l
        The order of the l-cochain.
    mesh
        A simplicial mesh over which the cochains are defined.

    Notes
    -----
    Let $\xi$, $\eta$, and $\mu$ be cochains defined on a simplicial complex. The
    cup product satisfies the following properties:

    * Associativity: $(\xi \wedge \eta) \wedge mu = \xi \wedge (\eta \wedge \mu)$.
    * Leibniz rule: $d(\xi \wedge \eta) = d\xi \wedge \eta + (-1)^k \xi \wedge d\eta$,
      where $d$ is the coboundary operator.

    However, this operator does not satisfy the graded commutativity property;
    i.e., in general, it is not true that $\xi \wedge \eta = (-1)^{kl} \eta \wedge \xi$,
    unless $k = l = 0$.

    The cup product is purely topological and is independent of the mesh geometry.
    """

    def __init__(
        self,
        k: int,
        l: int,
        mesh: SimplicialMesh,
    ):
        super().__init__()

        m = k + l
        device = mesh.device

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
        # * Convert the (k+l)-simplices to the canonical (k+l)-simplices, which
        #   incurs a sign correction at the (k+l)-simplex level.
        # * Identify the k-front and k-back faces of the canonical (k+l)-simplices,
        #   and then convert them to the geometric (k+l)-simplices, which incurs
        #   two sign corrections on the front/back face level.

        # Compute (k+l)-simplex sign correction
        if m == mesh.dim:
            m_splx_sorted = mesh.splx[m].sort(dim=-1).values
            m_splx_parity = compute_lex_rel_orient(mesh.splx[m]).to(dtype=mesh.dtype)
        else:
            m_splx_sorted = mesh.splx[m]
            m_splx_parity = torch.ones(1, dtype=mesh.dtype, device=device).expand(
                mesh.splx[m].size(0)
            )

        self.m_splx_parity: Float[Tensor, " m_splx"]
        self.register_buffer("m_splx_parity", m_splx_parity)

        # Identify the k-front faces of (k+l)-simplices and their sign correction
        f_face_idx = splx_search(
            key_splx=mesh.splx[k],
            query_splx=m_splx_sorted[:, : k + 1],
            sort_key_splx=True if k == mesh.dim else False,
            sort_key_vert=True if k == mesh.dim else False,
            sort_query_vert=False,
        )

        self.f_face_idx: Integer[Tensor, " m_splx"]
        self.register_buffer("f_face_idx", f_face_idx)

        if k == mesh.dim:
            f_face_parity = compute_lex_rel_orient(mesh.splx[k][self.f_face_idx])
        else:
            f_face_parity = torch.ones(1, dtype=mesh.dtype, device=device).expand(
                self.f_face_idx.size(0)
            )

        self.f_face_parity: Integer[Tensor, " m_splx"]
        self.register_buffer("f_face_parity", f_face_parity)

        # Identify the k-back faces of (k+l)-simplices and their sign correction
        b_face_idx = splx_search(
            key_splx=mesh.splx[l],
            query_splx=m_splx_sorted[:, k:],
            sort_key_splx=True if l == mesh.dim else False,
            sort_key_vert=True if l == mesh.dim else False,
            sort_query_vert=False,
        )

        self.b_face_idx: Integer[Tensor, " m_splx"]
        self.register_buffer("b_face_idx", b_face_idx)

        if l == mesh.dim:
            b_face_parity = compute_lex_rel_orient(mesh.splx[l][self.b_face_idx])
        else:
            b_face_parity = torch.ones(1, dtype=mesh.dtype, device=device).expand(
                self.b_face_idx.size(0)
            )

        self.b_face_parity: Integer[Tensor, " m_splx"]
        self.register_buffer("b_face_parity", b_face_parity)

    def forward(
        self,
        k_cochain: Float[Tensor, " k_splx *ch_in"],
        l_cochain: Float[Tensor, " l_splx *ch_in"],
        pairing: Literal["scalar", "dot", "cross", "outer"] = "scalar",
    ) -> Float[Tensor, " m_splx *ch_out"]:
        """
        Execute on the cup product between a k-cochain and an l-cochain.

        Parameters
        ----------
        k_cochain : [k_splx, *ch_in]
            The k-cochain. The trailing channel/batch dimensions should match
            those of the `l_cochain`.
        l_cochain : [l_splx, *ch_in]
            The l-cochain. The trailing channel/batch dimensions should match
            those of the `k_cochain`.
        pairing
            How to pair the channel dimensions of the input cochains. If
            `pairing=scalar`, then the cup product is performed elementwise.
            If `pairing` is "dot", "cross", or "outer", then `*ch_in` must match
            to one dimension, and the cup product performs a dot, cross, or outer
            product, respectively, along the channel dimension between the k-cochain
            at the k-front faces and the l-cochain at the k-back faces.

        Returns
        -------
        [m_splx, *ch_out]
            The cup product between the k-cochain and the l-cochain. If `pairing`
            is "scalar", then `*ch_out` matches the input `*ch_in`; if `pairing`
            is "dot", then `*ch_out` is trivial; if `pairing` is "cross", then
            `*ch_out` matches the single `ch_in` dimension; if `pairing` is
            "outer", then `*ch_out` matches to `(ch_in, ch_in)`.
        """
        k_cochain_at_f_face = torch.einsum(
            "n,n...->n...", self.f_face_parity, k_cochain[self.f_face_idx]
        )
        l_cochain_at_b_face = torch.einsum(
            "n,n...->n...", self.b_face_parity, l_cochain[self.b_face_idx]
        )

        match pairing:
            case "scalar":
                prod = torch.einsum(
                    "n,n...->n...",
                    self.m_splx_parity,
                    k_cochain_at_f_face * l_cochain_at_b_face,
                )

            case "dot":
                prod = self.m_splx_parity.view(-1, 1) * torch.sum(
                    k_cochain_at_f_face * l_cochain_at_b_face,
                    dim=-1,
                    keepdim=True,
                )

            case "cross":
                prod = self.m_splx_parity.view(-1, 1) * torch.cross(
                    k_cochain_at_f_face, l_cochain_at_b_face, dim=-1
                )

            case "outer":
                prod = torch.einsum(
                    "n,nk,nl->nkl",
                    self.m_splx_parity,
                    k_cochain_at_f_face,
                    l_cochain_at_b_face,
                )

            case _:
                raise ValueError(f"Unknown pairing method '{pairing}'.")

        return prod


class AntisymmetricCupProduct(torch.nn.Module):
    r"""
    Compute the anti-symmetrized cup product between a k-cochain and an l-cochain.

    Parameters
    ----------
    k
        The order of the k-cochain.
    l
        The order of the l-cochain.
    mesh
        A simplicial mesh over which the cochains are defined.

    Notes
    -----
    The anti-symmetrized cup product differs from the regular cup product in that
    it averages over all permutations of k-front and k-back face splits, thus
    is invariant to simplex vertex permutation.

    As a result of this permutation invariance, this operator satisfies the the
    graded commutativity property; i.e., $\xi \wedge \eta = (-1)^{kl} \eta \wedge \xi$.
    However, unlike the regular cup product, it does not in general satisfies the
    associativity rule or the Leibniz rule.

    The antisymmetrized cup product is purely topological and is independent of
    the mesh geometry.
    """

    def __init__(
        self,
        k: int,
        l: int,
        mesh: SimplicialMesh,
    ):
        super().__init__()

        m = k + l
        device = mesh.device

        perm = compute_face_perm_lut(k, l, device=device)

        # Compared to the regular cup product, the antisymmetrized cup product
        # requires an additional sign correction handled by the FacePermLUT class
        # that accounts for the permutation required to rearrange a canonical
        # m-simplex into a given k-front/back face split.
        self.perm_sign: Float[Tensor, "1 face 1"]
        self.register_buffer("perm_sign", perm.sign.to(mesh.dtype))

        # Compute (k+l)-simplex sign correction.
        if m == mesh.dim:
            m_splx_sorted = mesh.splx[m].sort(dim=-1).values
            m_splx_parity = compute_lex_rel_orient(mesh.splx[m]).to(dtype=mesh.dtype)
        else:
            m_splx_sorted = mesh.splx[m]
            m_splx_parity = torch.ones(1, dtype=mesh.dtype, device=device).expand(
                mesh.splx[m].size(0)
            )

        self.m_splx_parity: Float[Tensor, " m_splx"]
        self.register_buffer("m_splx_parity", m_splx_parity)

        # Identify permutations of the k-front faces of (k+l)-simplices and their
        # sign correction.
        uf_face: Integer[Tensor, "m_splx uf_face k+1"] = m_splx_sorted[
            :, perm.unique_front
        ]
        uf_face_idx: Integer[Tensor, "m_splx uf_face"] = splx_search(
            key_splx=mesh.splx[k],
            query_splx=uf_face,
            sort_key_splx=True if k == mesh.dim else False,
            sort_key_vert=True if k == mesh.dim else False,
            sort_query_vert=False,
        )
        f_face_idx = uf_face_idx[:, perm.front_idx]

        self.f_face_idx: Integer[Tensor, "m_splx face"]
        self.register_buffer("f_face_idx", f_face_idx)

        if k == mesh.dim:
            f_face_parity = (
                compute_lex_rel_orient(mesh.splx[k][uf_face_idx])
                .to(dtype=mesh.dtype)
                .view(*uf_face.shape[:-1])[:, perm.front_idx]
            )
        else:
            f_face_parity = torch.ones(1, dtype=mesh.dtype, device=device).expand(
                self.f_face_idx.shape
            )

        self.f_face_parity: Float[Tensor, "m_splx face"]
        self.register_buffer("f_face_parity", f_face_parity)

        # Identify permutations of the k-back faces of (k+l)-simplices and their
        # sign correction.
        ub_face: Integer[Tensor, "m_splx ub_face l+1"] = (
            m_splx_sorted[:, perm.unique_back].sort(dim=-1).values
        )
        ub_face_idx: Integer[Tensor, "m_splx ub_face"] = splx_search(
            key_splx=mesh.splx[l],
            query_splx=ub_face,
            sort_key_splx=True if l == mesh.dim else False,
            sort_key_vert=True if l == mesh.dim else False,
            sort_query_vert=False,
        )
        b_face_idx = ub_face_idx[:, perm.back_idx]

        self.b_face_idx: Integer[Tensor, "m_splx face"]
        self.register_buffer("b_face_idx", b_face_idx)

        if l == mesh.dim:
            b_face_parity = (
                compute_lex_rel_orient(mesh.splx[l][ub_face_idx])
                .to(dtype=mesh.dtype)
                .view(*ub_face.shape[:-1])[:, perm.back_idx]
            )
        else:
            b_face_parity = torch.ones(1, dtype=mesh.dtype, device=device).expand(
                self.b_face_idx.shape
            )

        self.b_face_parity: Float[Tensor, " m_splx face"]
        self.register_buffer("b_face_parity", b_face_parity)

    def forward(
        self,
        k_cochain: Float[Tensor, " k_splx *ch_in"],
        l_cochain: Float[Tensor, " l_splx *ch_in"],
        pairing: Literal["scalar", "dot", "cross", "outer"] = "scalar",
    ) -> Float[Tensor, " m_splx *ch_out"]:
        """
        Execute on the anti-symmetrized cup product between a k-cochain and an l-cochain.

        Parameters
        ----------
        k_cochain : [k_splx, *ch_in]
            The k-cochain. The trailing channel/batch dimensions should match
            those of the `l_cochain`.
        l_cochain : [l_splx, *ch_in]
            The l-cochain. The trailing channel/batch dimensions should match
            those of the `k_cochain`.
        pairing
            How to pair the channel dimensions of the input cochains. If
            `pairing=scalar`, then the cup product is performed elementwise.
            If `pairing` is "dot", "cross", or "outer", then `*ch_in` must match
            to one dimension, and the cup product performs a dot, cross, or outer
            product, respectively, along the channel dimension between the k-cochain
            at the k-front faces and the l-cochain at the k-back faces.

        Returns
        -------
        [m_splx, *ch_out]
            The cup product between the k-cochain and the l-cochain. If `pairing`
            is "scalar", then `*ch_out` matches the input `*ch_in`; if `pairing`
            is "dot", then `*ch_out` is trivial; if `pairing` is "cross", then
            `*ch_out` matches the single `ch_in` dimension; if `pairing` is
            "outer", then `*ch_out` matches to `(ch_in, ch_in)`.
        """
        k_cochain_at_f_face: Float[Tensor, "m_splx face *ch_in"] = k_cochain[
            self.f_face_idx
        ]
        l_cochain_at_b_face: Float[Tensor, "m_splx face *ch_in"] = l_cochain[
            self.b_face_idx
        ]

        combined_perm_sign: Float[Tensor, "m_splx face 1"] = (
            self.m_splx_parity.view(-1, 1, 1)
            * self.f_face_parity.unsqueeze(-1)
            * self.b_face_parity.unsqueeze(-1)
            * self.perm_sign
        )

        n_face_splits = combined_perm_sign.size(1)

        match pairing:
            case "scalar":
                prod = (
                    torch.einsum(
                        "nf,nf...->n...",
                        combined_perm_sign.squeeze(-1),
                        k_cochain_at_f_face * l_cochain_at_b_face,
                    )
                    / n_face_splits
                )

            case "dot":
                prod = torch.mean(
                    torch.sum(
                        combined_perm_sign * k_cochain_at_f_face * l_cochain_at_b_face,
                        dim=-1,
                        keepdim=True,
                    ),
                    dim=1,
                )

            case "cross":
                prod = torch.mean(
                    combined_perm_sign
                    * torch.cross(k_cochain_at_f_face, l_cochain_at_b_face, dim=-1),
                    dim=1,
                )

            case "outer":
                prod = (
                    torch.einsum(
                        "nf,nfk,nfl->nkl",
                        combined_perm_sign.squeeze(-1),
                        k_cochain_at_f_face,
                        l_cochain_at_b_face,
                    )
                    / n_face_splits
                )

            case _:
                raise ValueError(f"Unknown pairing method '{pairing}'.")

        return prod

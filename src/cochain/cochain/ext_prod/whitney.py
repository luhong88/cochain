__all__ = ["WhitneyWedgeL2Projector"]

from typing import Literal

import torch
from jaxtyping import Float, Integer
from torch import Tensor

from ...complex import SimplicialMesh
from ._whitney_3_form import compute_3_form_triple_prod_tensor
from ._whitney_m_form import compute_triple_prod_tensor


class WhitneyWedgeL2Projector(torch.nn.Module):
    r"""
    Compute the load vector required to perform the Galerkin wedge product.

    To compute the wedge product between a $k$-cochain and an $l$-cochain, first
    use this class to compute the load vector $b$, then, solve the linear system
    $M w = b$ to find the wedge product $(k+l)$-cochain $w$; here, $M$ is the
    $(k+l)$-mass matrix.

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
    The Galerkin wedge product satisfies graded commutativity, but not associativity
    or Leibniz rule. However, compared the the cup product and anti-symmetric
    cup product, the Galerkin wedge product takes into account metric information.
    This wedge product is also known as the $L^2$-projected wedge product.
    """

    def __init__(self, k: int, l: int, mesh: SimplicialMesh):
        super().__init__()

        m = k + l

        # Identify the k-faces, l-faces, and m-faces of the top level simplices
        # and register their indices and sign corrections.
        k_faces = mesh.faces(k)
        l_faces = mesh.faces(l)
        m_faces = mesh.faces(m)

        self.k_face_idx: Integer[Tensor, "top_splx k_face"]
        self.k_face_parity: Float[Tensor, "top_splx k_face"]
        self.register_buffer("k_face_idx", k_faces.idx)
        self.register_buffer("k_face_parity", k_faces.parity)

        self.l_face_idx: Integer[Tensor, "top_splx l_face"]
        self.l_face_parity: Float[Tensor, "top_splx l_face"]
        self.register_buffer("l_face_idx", l_faces.idx)
        self.register_buffer("l_face_parity", l_faces.parity)

        self.m_face_idx: Integer[Tensor, "top_splx m_face"]
        self.m_face_parity: Float[Tensor, "top_splx m_face"]
        self.register_buffer("m_face_idx", m_faces.idx)
        self.register_buffer("m_face_parity", m_faces.parity)

        self.n_m_splx = mesh.splx[m].size(0)

        # Compute the triple tensor product. When k + l = 3, a special optimized
        # version of the method is applied that is more memory efficient.
        if m == 3:
            triple_prod = compute_3_form_triple_prod_tensor(k, l, mesh)
        else:
            triple_prod = compute_triple_prod_tensor(k, l, mesh)

        self.triple_prod: Float[Tensor, "top_splx k_face l_face m_face"]
        self.register_buffer("triple_prod", triple_prod)

    def forward(
        self,
        k_cochain: Float[Tensor, " k_splx *ch_in"],
        l_cochain: Float[Tensor, " l_splx *ch_in"],
        pairing: Literal["scalar", "dot", "cross"] = "scalar",
    ) -> Float[Tensor, " m_splx *ch_out"]:
        """
        Execute on the wedge product between a k-cochain and an l-cochain.

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
            If `pairing` is "dot" or "cross", then `*ch_in` must match to one dimension,
            and the cup product performs a dot or cross product, respectively,
            along the channel dimension between the k-cochain at the k-front faces
            and the l-cochain at the k-back faces.

        Returns
        -------
        [m_splx, *ch_out]
            The cup product between the k-cochain and the l-cochain. If `pairing`
            is "scalar", then `*ch_out` matches the input `*ch_in`; if `pairing`
            is "dot", then `*ch_out` is trivial; if `pairing` is "cross", then
            `*ch_out` matches the single `ch_in` dimension.
        """
        k_cochain_at_k_face = torch.einsum(
            "tf,tf...->tf...", self.k_face_parity, k_cochain[self.k_face_idx]
        )
        l_cochain_at_l_face = torch.einsum(
            "tf,tf...->tf...", self.l_face_parity, l_cochain[self.l_face_idx]
        )

        match pairing:
            case "scalar":
                m_cochain_at_m_face = torch.einsum(
                    "tuvw,tu...,tv...,tw->tw...",
                    self.triple_prod,
                    k_cochain_at_k_face,
                    l_cochain_at_l_face,
                    self.m_face_parity,
                )

            case "dot":
                m_cochain_at_m_face = torch.einsum(
                    "tuvw,tuc,tvc,tw->tw",
                    self.triple_prod,
                    k_cochain_at_k_face,
                    l_cochain_at_l_face,
                    self.m_face_parity,
                ).unsqueeze(-1)

            case "cross":
                epsilon = torch.tensor(
                    [
                        [[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]],
                        [[0.0, 0.0, -1.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
                        [[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                    ],
                    device=self.triple_prod.device,
                    dtype=self.triple_prod.dtype,
                )

                m_cochain_at_m_face = torch.einsum(
                    "tuvw,tuc,tvd,tw,cde->twe",
                    self.triple_prod,
                    k_cochain_at_k_face,
                    l_cochain_at_l_face,
                    self.m_face_parity,
                    epsilon,
                )

            case _:
                raise ValueError(f"Unknown pairing method '{pairing}'.")

        ch_out_dims = m_cochain_at_m_face.shape[2:]
        load = torch.zeros(
            (self.n_m_splx,) + ch_out_dims,
            device=self.triple_prod.device,
            dtype=self.triple_prod.dtype,
        )
        load.index_add_(
            0,
            self.m_face_idx.flatten(),
            m_cochain_at_m_face.flatten(end_dim=1),
        )

        return load

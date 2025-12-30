from typing import Literal

import torch as t
from jaxtyping import Float, Integer

from ..complex import SimplicialComplex
from ._whitney_3_form import triple_tensor_prod_3_form
from ._whitney_m_form import triple_tensor_prod
from ._whitney_utils import find_top_simp_faces


class WhitneyWedgeL2Projector(t.nn.Module):
    def __init__(self, k: int, l: int, mesh: SimplicialComplex):
        """
        Compute the load vector required to compute the L^2-projected wedge product
        (or the Galerkin wedge product).

        To compute the wedge product between a k-form and an l-form, first use
        this class to compute the load vector `b`, then, solve the linear system
        `M@w=b` to find the wedge product (k+l)-form `w`; here, `M` is the (k+l)-
        form mass matrix.

        The Galerkin wedge product satisfies graded commutativity, but not associativity
        or Leibniz rule. However, compared the the cup product and anti-symmetric
        cup product, the Galerkin wedge product takes into account metric information.

        """
        super().__init__()

        simp_map = {
            dim: simp
            for dim, simp in enumerate([mesh.verts, mesh.edges, mesh.tris, mesh.tets])
        }

        m = k + l

        # Identify the k-faces of the top level simplices and their sign corrections.
        k_face_idx, k_face_parity = find_top_simp_faces(k, mesh.dim, mesh, simp_map)

        self.k_face_idx: Integer[t.LongTensor, "top_simp k_face"]
        self.register_buffer("k_face_idx", k_face_idx)

        self.k_face_parity: Float[t.Tensor, "top_simp k_face"]
        self.register_buffer("k_face_parity", k_face_parity)

        # Identify the l-faces of the top level simplices and their sign corrections.
        l_face_idx, l_face_parity = find_top_simp_faces(l, mesh.dim, mesh, simp_map)

        self.l_face_idx: Integer[t.LongTensor, "top_simp l_face"]
        self.register_buffer("l_face_idx", l_face_idx)

        self.l_face_parity: Float[t.Tensor, "top_simp l_face"]
        self.register_buffer("l_face_parity", l_face_parity)

        # Identify the (k+l)-faces of the top level simplices and their sign corrections.
        m_face_idx, m_face_parity = find_top_simp_faces(m, mesh.dim, mesh, simp_map)

        self.m_face_idx: Integer[t.LongTensor, "top_simp m_face"]
        self.register_buffer("m_face_idx", m_face_idx)

        self.m_face_parity: Float[t.Tensor, "top_simp m_face"]
        self.register_buffer("m_face_parity", m_face_parity)

        self.n_m_simp = simp_map[m].size(0)

        # Compute the triple tensor product. When k + l = 3, a special optimized
        # version of the method is applied that is more memory efficient.
        if m == 3:
            triple_prod = triple_tensor_prod_3_form(k, l, mesh)
        else:
            triple_prod = triple_tensor_prod(k, l, mesh)

        self.triple_prod: Float[t.Tensor, "top_simp k_face l_face m_face"]
        self.register_buffer("triple_prod", triple_prod)

    def forward(
        self,
        k_cochain: Float[t.Tensor, " k_simp *ch_in"],
        l_cochain: Float[t.Tensor, " l_simp *ch_in"],
        pairing: Literal["scalar", "dot", "cross"] = "scalar",
    ) -> Float[t.Tensor, " m_simp *ch_out"]:
        k_cochain_at_k_face = t.einsum(
            "tf,tf...->tf...", self.k_face_parity, k_cochain[self.k_face_idx]
        )
        l_cochain_at_l_face = t.einsum(
            "tf,tf...->tf...", self.l_face_parity, l_cochain[self.l_face_idx]
        )

        # If pairing='scalar', *ch_in can match to an arbitrary number of channel
        # dimensions; for other pairing method, *ch_in need to match to one dimension.
        match pairing:
            case "scalar":
                m_cochain_at_m_face = t.einsum(
                    "tuvw,tu...,tv...,tw->tw...",
                    self.triple_prod,
                    k_cochain_at_k_face,
                    l_cochain_at_l_face,
                    self.m_face_parity,
                )

            case "dot":
                m_cochain_at_m_face = t.einsum(
                    "tuvw,tuc,tvc,tw->tw",
                    self.triple_prod,
                    k_cochain_at_k_face,
                    l_cochain_at_l_face,
                    self.m_face_parity,
                ).unsqueeze(-1)

            case "cross":
                epsilon = t.tensor(
                    [
                        [[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]],
                        [[0.0, 0.0, -1.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
                        [[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                    ],
                    device=self.triple_prod.device,
                    dtype=self.triple_prod.dtype,
                )

                m_cochain_at_m_face = t.einsum(
                    "tuvw,tuc,tvd,tw,cde->twe",
                    self.triple_prod,
                    k_cochain_at_k_face,
                    l_cochain_at_l_face,
                    self.m_face_parity,
                    epsilon,
                )

            case _:
                raise NotImplementedError()

        ch_out_dims = m_cochain_at_m_face.shape[2:]
        load = t.zeros(
            (self.n_m_simp,) + ch_out_dims,
            device=self.triple_prod.device,
            dtype=self.triple_prod.dtype,
        )
        load.index_add_(
            0,
            self.m_face_idx.flatten(),
            m_cochain_at_m_face.flatten(end_dim=1),
        )

        return load

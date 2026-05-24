__all__ = ["WhitneyWedgeL2Projector"]

from typing import Literal

import torch
from einops import einsum
from jaxtyping import Float, Integer
from torch import Tensor

from ...complex import SimplicialMesh
from ._triple_prod_3_form import compute_3_form_triple_prod_tensor
from ._triple_prod_m_form import compute_triple_prod_tensor


class WhitneyWedgeL2Projector(torch.nn.Module):
    r"""
    Compute the load vector required to perform the Galerkin wedge product.

    To compute the wedge product between a $k$-cochain and an $l$-cochain, first
    use this class to compute the load vector $b$, then, solve the linear system
    $M \mu = b$ to find the wedge product $(k+l)$-cochain $\mu$; here, $M$ is the
    $(k+l)$-mass matrix. This wedge product is also known as the $L^2$-projected
    wedge product.

    Instances of this class must be re-initialized whenever the mesh geometry
    is modified.

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
    Let $\xi$ be a discrete $k$-form and $\eta$ be a discrete $l$-form. First, we
    expand $\xi$ in the basis of Whitney $k$-forms (using the Einstein notation),

    $$\xi = \xi_r W_k^r$$

    and $\eta$ in the basis of Whitney $l$-forms,

    $$\eta = \eta_s W_l^s$$

    Here, $r$ iterates over the $k$-simplices and $s$ iterates over the $l$-simplices;
    $W_k^r$ is the Whitney $k$-form basis function defined on the $k$-simplex $r$,
    and $W_l^s$ is the Whitney $l$-form basis function defined on the $l$-simplex $s$.

    Let $m = k + l$. In general, $\xi \wedge \eta$ is not a Whitney $m$-form, since
    it involves higher-order products of barycentric weights from  $\xi$ and $\eta$
    that cannot be represented by Whitney $m$-forms with first-order barycentric
    weight coefficients only. Therefore, we instead find a discrete $m$-form $\mu$
    that best approximates $\xi \wedge \eta$. Using the Galerkin projection approach,
    this is equivalent to asserting that the error $\epsilon = \xi \wedge \eta - \mu$
    is orthogonal to the space of test functions (i.e., Whitney $m$-forms):

    $$
    \int_\Omega \left<\xi_r\eta_s W_k^r \wedge W_l^s, W_m^t\right> dV =
    \int_\Omega \left<\mu_u W_m^u, W_m^t\right> dV
    $$

    for all basis functions $W_m^t$. Let us define the triple product tensor

    $$
    T^{rst} = \int_\Omega \left<W_k^r \wedge W_l^s, W_m^t\right> dV
    $$

    In addition, recall that the integral of $\left<W_m^t, W_m^u\right>$ defines
    the elements of the consistent $m$-mass matrix $M_m$. Taken together, we can
    write the Galerkin projection equation more concisely as the linear system

    $$M_m \mu = b$$

    where $b$ is the load vector defined elementwise as

    $$
    b^t = \xi_r \eta_s T^{rst}
    $$

    which is computed by this function. In practice, this function first compute
    $b^t$ locally over the $k$-, $l$-, and $m$-faces in each top-level simplex
    before scatter_add() the results to the global, canonical $m$-simplices.

    In general, the wedge product computed this way satisfies graded commutativity,

    $$\xi \wedge \eta = (-1)^{kl} \eta \wedge \xi$$

    but not associativity $(\xi \wedge \eta) \wedge \mu = \xi \wedge (\eta \wedge \mu)$
    or the Leibniz rule $d(\xi \wedge \eta) = d\xi \wedge \eta + (-1)^k \xi \wedge d\eta$.

    Compared with the cup product and anti-symmetric cup product, the Galerkin wedge
    product takes into account metric information. As such, the `WhitneyWedgeL2Projector`
    needs to be re-initialized each time the underlying mesh geometry is modified.
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

        self.m_face_idx_flat: Integer[Tensor, " top_splx_by_m_face"]
        self.register_buffer("m_face_idx_flat", m_faces.idx.flatten())

        self.n_m_splx = mesh.splx[m].size(0)

        # Compute the triple tensor product. When k + l = 3, a special optimized
        # version of the method is applied that is more memory efficient.
        if m == 3:
            triple_prod = compute_3_form_triple_prod_tensor(k, l, mesh)
        else:
            triple_prod = compute_triple_prod_tensor(k, l, mesh)

        triple_prod_with_parity = einsum(
            triple_prod,
            m_faces.parity,
            "splx k l m, splx m -> splx k l m",
        )

        self.triple_prod_with_parity: Float[Tensor, "top_splx k_face l_face m_face"]
        self.register_buffer("triple_prod_with_parity", triple_prod_with_parity)

    def forward(
        self,
        k_cochain: Float[Tensor, " k_splx *ch_in"],
        l_cochain: Float[Tensor, " l_splx *ch_in"],
        pairing: Literal["scalar", "dot", "cross", "outer"] = "scalar",
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
        k_cochain_at_k_face = torch.einsum(
            "tf,tf...->tf...", self.k_face_parity, k_cochain[self.k_face_idx]
        )
        l_cochain_at_l_face = torch.einsum(
            "tf,tf...->tf...", self.l_face_parity, l_cochain[self.l_face_idx]
        )

        match pairing:
            case "scalar":
                m_cochain_at_m_face = einsum(
                    self.triple_prod_with_parity,
                    k_cochain_at_k_face,
                    l_cochain_at_l_face,
                    "splx k l m, splx k ..., splx l ... -> splx m ...",
                )

            case "dot":
                m_cochain_at_m_face = einsum(
                    self.triple_prod_with_parity,
                    k_cochain_at_k_face,
                    l_cochain_at_l_face,
                    "splx k l m, splx k ch, splx l ch -> splx m",
                ).unsqueeze(-1)

            case "cross":
                # ϵ_ijk is the 3D Levi-Civita symbol used to compute cross products.
                # For any 3D vectors u and v, (u x v)_k = u^i v^j ϵ_ijk
                epsilon = torch.tensor(
                    [
                        [[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]],
                        [[0.0, 0.0, -1.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
                        [[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                    ],
                    device=self.triple_prod_with_parity.device,
                    dtype=self.triple_prod_with_parity.dtype,
                )

                m_cochain_at_m_face = einsum(
                    self.triple_prod_with_parity,
                    k_cochain_at_k_face,
                    l_cochain_at_l_face,
                    epsilon,
                    "splx k l m, splx k e_i, splx l e_j, e_i e_j e_k -> splx m e_k",
                )

            case "outer":
                m_cochain_at_m_face = einsum(
                    self.triple_prod_with_parity,
                    k_cochain_at_k_face,
                    l_cochain_at_l_face,
                    "splx k l m, splx k ch1, splx l ch2 -> splx m ch1 ch2",
                )

            case _:
                raise ValueError(f"Unknown pairing method '{pairing}'.")

        ch_out_shape = m_cochain_at_m_face.shape[2:]
        load = torch.zeros(
            (self.n_m_splx,) + ch_out_shape,
            device=self.triple_prod_with_parity.device,
            dtype=self.triple_prod_with_parity.dtype,
        )

        # load[m_face_idx[i], ...] = m_cochain_at_m_face[i, ...]
        load.index_add_(
            dim=0,
            index=self.m_face_idx_flat,
            source=m_cochain_at_m_face.flatten(end_dim=1),
        )

        return load

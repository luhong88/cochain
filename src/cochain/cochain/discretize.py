__all__ = ["DeRhamMap"]

from dataclasses import dataclass

import torch
from einops import einsum
from jaxtyping import Float
from torch import Tensor

from ..complex import SimplicialMesh
from ..utils import quadrature


@dataclass
class DeRhamMap:
    """
    Discretize k-forms via the de Rham map.

    This class implements the de Rham map and discretizes k-forms by mapping them
    to discrete k-cochains via numerical integration. In particular, this class
    uses Gauss-Legendre, Dunavant, and Keast numerical quadrature rules for
    integrating over 1-, 2-, and 2-simplices, respectively. Notably, these rules
    are invariant to vertex permutation.

    To use this class, first call `sample_points()` to get a set of points on the
    mesh at which to evaluate the k-form. Then, call `discretize()` with the
    sampled k-form to perform the integration. It is possible to call `discretize()`
    directly without having called `sample_points()`.

    Attributes
    ----------
    k
        The degree of the k-form. Only k = 1, 2, and 3 are supported.
    quad_degree
        The degree of the numerical quadrature/integration rule. Currently, degrees
        between 0 and 5 (inclusive) are supported. In general, higher degree rules
        require more sampled points.
    mesh
        A simplicial mesh.
    allow_neg_weights
        Whether to allow for negative weights. Note that only some of the Keast
        rules employ negative weights; such rules require fewer points to achieve
        the same exactness, but may be less numerically stable due to potential
        cancellations with adjacent positively weighted points.

    Notes
    -----
    This class does not support 0-forms. For 0-forms, the de Rham map is "trivial"
    and equivalent to sampling the 0-form (scalar function) at the vertex positions.
    """

    k: int
    quad_degree: int
    mesh: SimplicialMesh
    allow_neg_weights: bool = True

    def __post_init__(self):
        if self.k > self.mesh.dim:
            raise ValueError(
                f"k-form degree ({self.k}) cannot be greater than mesh dimension ({self.mesh.dim})."
            )

        match self.k:
            case 1:
                self.quad = quadrature.GaussLegendre
            case 2:
                self.quad = quadrature.Dunavant
            case 3:
                self.quad = quadrature.Keast
            case _:
                raise ValueError(
                    f"Unsupported k-form degree: {self.k}. Only 1, 2, and 3 are supported."
                )

    def _get_quad_rule(self):
        dtype = self.mesh.dtype
        device = self.mesh.device

        bary_coords, weights = self.quad(dtype, device).get_rule(
            self.quad_degree, allow_neg_weights=self.allow_neg_weights
        )

        self.bary_coords = bary_coords
        self.weights = weights

    def sample_points(self) -> Float[Tensor, "k_splx pt coord=3"]:
        """
        Get the k-form sample points for the given quadrature.

        Returns
        -------
        [k_splx, pt, coord=3]
            The spatial points on each k-simplex over which to sample the k-form.
        """
        if not hasattr(self, "bary_coords"):
            self._get_quad_rule()

        # The barycentric coordinates in ref_barys provide the weights for the
        # linear combination of the simplex vertex coordinates to identify
        # the sample points in the simplex, and these weights are the same
        # for the reference and physical simplices.
        splx_vert_coords = self.mesh.vert_coords[self.mesh.splx[self.k]]
        sampled_points = einsum(
            splx_vert_coords,
            self.bary_coords,
            "k_splx vert coord, pt vert -> k_splx pt coord",
        )

        return sampled_points

    def discretize(
        self,
        k_forms: Float[Tensor, "k_splx pt *ch coord"],
    ) -> Float[Tensor, " k_splx *ch"]:
        r"""
        Discretize a k-form using the sampled values.

        Parameters
        ----------
        k_forms : [k_splx, pt, *ch, coord]
            The k-form sampled at the points specified by the quadrature. Note that
            this function sampled k-forms with arbitrary batch/channel dimensions
            (but the batch/channel dimensions must precede the final coordiate
            dimension).

        Returns
        -------
        [k_splx, *ch]
            The discretized k-cochain associated with the k-simplices.

        Notes
        -----
        For 2-forms, we assume that they are represented with the basis
        $\{dy \wedge dz, dz \wedge dx, dx \wedge dy\}$, or, equivalently (under
        the Hodge star isomorphism), as proxy 1-forms represented with the basis
        $\{dx, dy, dz\}$. For 3-forms, we assume that they are represented as
        scalars and the coord dimension of `k_forms` is trivial.
        """
        if not hasattr(self, "bary_coords"):
            self._get_quad_rule()

        # Consider the pushforward map ϕ: λ -> x from the barycentric coordinates
        # on the reference simplex to the Cartesian coordinates of the "physical"
        # simplices. If we match the first vertex of each simplex with the point of
        # origin in the ref simplex, then the map can be written as ϕ(λ) = J@λ + v_0;
        # in particular, J is the jacobian and can be computed as the matrix of
        # the edge (column) vectors v_i - v_0.
        splx_vert_coords = self.mesh.vert_coords[self.mesh.splx[self.k]]
        jacs: Float[Tensor, "k_splx edge coord=3"] = (
            splx_vert_coords[:, 1:, :] - splx_vert_coords[:, [0], :]
        )

        match self.k:
            case 1:
                # For 1-forms, the Jacobian for each edge is the edge vector v1 - v0,
                # and the pullback is the dot product between the 1-form and
                # the edge vector. Note that, for 1-forms, the edge dimension of
                # jacs is trivial, since each 1-simplex has only one edge.
                pullback = einsum(
                    jacs,
                    k_forms,
                    "k_splx edge coord, k_splx pt ... coord -> k_splx pt ...",
                )
                circulation = einsum(
                    pullback, self.weights, "k_splx pt ..., pt -> k_splx ..."
                )
                return circulation

            case 2:
                # The Jacobian for edge triangle consists of the edge column
                # vectors {v1 - v0, v2 - v0}, and the pullback is the dot product
                # between the proxy 1-form and the triangle normal vector (oriented
                # to satisfy the right-hand rule and scaled to the triangle area).
                area_normal: Float[Tensor, "k_splx coord=3"] = 0.5 * torch.cross(
                    jacs[:, 0, :], jacs[:, 1, :], dim=-1
                )
                pullback = einsum(
                    area_normal,
                    k_forms,
                    "k_splx coord, k_splx pt ... coord -> k_splx pt ...",
                )
                flux = einsum(pullback, self.weights, "k_splx pt ..., pt -> k_splx ...")
                return flux

            case 3:
                # For 3-forms, the pullback consists of the scalar product between
                # the 3-form (a scalar) and the determinant of the Jacobian (scaled
                # to the tet volume by the 1/6 factor).
                signed_vol: Float[Tensor, " k_splx"] = torch.linalg.det(jacs) / 6.0
                pullback = einsum(
                    signed_vol,
                    k_forms,
                    "k_splx, k_splx pt ... coord -> k_splx pt ...",
                )
                density = einsum(
                    pullback, self.weights, "k_splx pt ..., pt -> k_splx ..."
                )
                return density

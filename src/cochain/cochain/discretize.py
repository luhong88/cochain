from dataclasses import dataclass

import torch as t
from einops import einsum
from jaxtyping import Float

from ..complex import SimplicialComplex
from ..utils import quadrature


@dataclass
class DeRhamMap:
    """
    This class implements the de Rham map and discretizes k-forms by mapping them
    to discrete k-cochains via numerical integration. Note that, for 0-forms, the
    de Rham map is "trivial" and equivalent to sampling the 0-form (scalar function)
    at the vertex positions; therefore, only k = 1, 2, or 3 are supported.

    To use this class, first call `sample_points()` to get a set of points on the
    mesh at which to evaluate the k-form. Then, call `discretize()` with the
    sampled k-form to perform the integration. It is possible to call `discretize()`
    directly without having called `sample_points()`. Note that this function
    supports input sampled k-forms with arbitrary batch/channel dimensions (but
    note that the batch/channel dimensions precedes the final coordiate dimension).

    This class uses Gauss-Legendre, Dunavant, and Keast numerical quadrature
    rules for integrating over 1-, 2-, and 2-simplices, respectively. These rules
    have the following properties:

    * Invariance to vertex permutation.
    * Can integrate polynomial functions exactly (currently, up to degree 5 is
      supported); in general, higher degree rules require more sampled points. The
      degree is specified via the `quad_degree` argument.

    Note that some of the Keast rules employ negative weights; such rules require
    fewer points to achieve the same exactness, but may be less numerically
    stable due to potential cancellations with adjacent positively weighted points.
    The argument `allow_neg_weights` specifies whether such rules are allowed.
    """

    k: int
    quad_degree: int
    mesh: SimplicialComplex
    allow_neg_weights: bool = True

    def __post_init__(self):
        if self.k > self.mesh.dim:
            raise ValueError()

        match self.k:
            case 1:
                self.quad = quadrature.GaussLegendre
            case 2:
                self.quad = quadrature.Dunavant
            case 3:
                self.quad = quadrature.Keast
            case _:
                raise ValueError()

    def _get_quad_rule(self):
        dtype = self.mesh.vert_coords.dtype
        device = self.mesh.vert_coords.device

        bary_coords, weights = self.quad(dtype, device).get_rule(
            self.quad_degree, allow_neg_weights=self.allow_neg_weights
        )

        self.bary_coords = bary_coords
        self.weights = weights

    def sample_points(self) -> Float[t.Tensor, "k_simp pt coord=3"]:
        if not hasattr(self, "bary_coords"):
            self._get_quad_rule()

        # The barycentric coordinates in ref_barys provide the weights for the
        # linear combination of the simplex vertex coordinates to identify
        # the sample points in the simplex, and these weights are the same
        # for the reference and physical simplices.
        simp_vert_coords = self.mesh.vert_coords[self.mesh.simplices[self.k]]
        sampled_points = einsum(
            simp_vert_coords,
            self.bary_coords,
            "k_simp vert coord, pt vert -> k_simp pt coord",
        )

        return sampled_points

    def discretize(
        self,
        k_forms: Float[t.Tensor, "k_simp pt *ch coord"],
    ) -> Float[t.Tensor, " k_simp *ch"]:
        if not hasattr(self, "bary_coords"):
            self._get_quad_rule()

        # Consider the pushforward map ϕ: λ -> x from the barycentric coordinates
        # on the reference simplex to the Cartesian coordinates of the "physical"
        # simplices. If we match the first vertex of each simplex with the point of
        # origin in the ref simplex, then the map can be written as ϕ(λ) = J@λ + v_0;
        # in particular, J is the jacobian and can be computed as the matrix of
        # the edge (column) vectors v_i - v_0.
        simp_vert_coords = self.mesh.vert_coords[self.mesh.simplices[self.k]]
        jacs: Float[t.Tensor, "k_simp edge coord=3"] = (
            simp_vert_coords[:, 1:, :] - simp_vert_coords[:, [0], :]
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
                    "k_simp edge coord, k_simp pt ... coord -> k_simp pt ...",
                )
                circulation = einsum(
                    pullback, self.weights, "k_simp pt ..., pt -> k_simp ..."
                )
                return circulation

            case 2:
                # For 2-forms, we assume that they are represented with the basis
                # {dy⋀dz, dz⋀dx, dx⋀dy}, or, equivalently (under the Hodge star
                # isomorphism), as proxy 1-forms represented with the basis {dx,
                # dy, dz}. The Jacobian for edge triangle consists of the edge column
                # vectors {v1 - v0, v2 - v0}, and the pullback is the dot product
                # between the proxy 1-form and the triangle normal vector (oriented
                # to satisfy the right-hand rule and scaled to the triangle area).
                area_normal: Float[t.Tensor, "k_simp coord=3"] = 0.5 * t.cross(
                    jacs[:, 0, :], jacs[:, 1, :], dim=-1
                )
                pullback = einsum(
                    area_normal,
                    k_forms,
                    "k_simp coord, k_simp pt ... coord -> k_simp pt ...",
                )
                flux = einsum(pullback, self.weights, "k_simp pt ..., pt -> k_simp ...")
                return flux

            case 3:
                # For 3-forms, the pullback consists of the scalar product between
                # the 3-form (a scalar) and the determinant of the Jacobian (scaled
                # to the tet volume by the 1/6 factor). Note that, for 3-forms,
                # the coord dimension is trivial.
                signed_vol: Float[t.Tensor, " k_simp"] = t.linalg.det(jacs) / 6.0
                pullback = einsum(
                    signed_vol,
                    k_forms,
                    "k_simp, k_simp pt ... coord -> k_simp pt ...",
                )
                density = einsum(
                    pullback, self.weights, "k_simp pt ..., pt -> k_simp ..."
                )
                return density

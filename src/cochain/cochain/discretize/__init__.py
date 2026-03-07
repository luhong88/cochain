import torch as t
from jaxtyping import Float, Integer

from ...utils import quadrature
from ..tet.tet_geometry import get_tet_signed_vols


class DeRhamMap:
    def __init__(
        self,
        k: int,
        degree: int,
        allow_neg_weights: bool = True,
    ):
        """
        For 0-form, the de Rham map is equivalent to sampling the 0-form (scalar
        function) at the vertex positions; therefore, only k = 1, 2, or 3 are supported.
        """
        self.k = k
        self.degree = degree
        self.allow_neg_weights = allow_neg_weights

        self.ref_barys: Float[t.Tensor, "point vert"]
        self.weights: Float[t.Tensor, "1 point"]

        match k:
            case 1:
                self.quad = quadrature.GaussLegendre
            case 2:
                self.quad = quadrature.Dunavant
            case 3:
                self.quad = quadrature.Keast
            case _:
                raise ValueError()

    def _get_quad_rule(
        self,
        vert_coords: Float[t.Tensor, "vert 3"],
    ):
        dtype = vert_coords.dtype
        device = vert_coords.device

        ref_barys, weights = self.quad(dtype, device).get_rule(
            self.degree, allow_neg_weights=self.allow_neg_weights
        )

        self.ref_barys = ref_barys
        self.weights = weights.view(1, -1)

    def sample_points(
        self,
        vert_coords: Float[t.Tensor, "vert 3"],
        k_simps: Integer[t.LongTensor, "simp vert"],
    ) -> Float[t.Tensor, "simp point 3"]:
        if not hasattr(self, "ref_barys"):
            self._get_quad_rule(vert_coords)

        # The barycentric coordinates in ref_barys provide the weights for the
        # linear combination of the simplex vertex coordinates to identify
        # the sample points in the simplex, and these weights are the same
        # for the reference and physical simplices.

        # (simp, vert, coord), (point, vert) -> (simp, point, coord)
        simp_vert_coords = vert_coords[k_simps]
        sampled_points = t.einsum("svc,pv->spc", simp_vert_coords, self.ref_barys)

        return sampled_points

    def discretize(
        self,
        vert_coords: Float[t.Tensor, "vert 3"],
        k_simps: Integer[t.LongTensor, "simp vert"],
        k_forms: Float[t.Tensor, "simp point *covariant"],
    ):
        if not hasattr(self, "weights"):
            self._get_quad_rule(vert_coords)

        # For discretizing 1-forms and 2-forms, we need to compute the
        # Jacobian of the ref -> phys transformations (i.e., the pushforward
        # map); for discretizing 3-forms, we simply need the unsigned determinant
        # of the Jacobian (i.e., the tet volume).
        if self.k in [1, 2]:
            simp_vert_coords = vert_coords[k_simps]

            # Pick the first vertex of each simplex as the point of origin
            # in the ref simplex and compute the jacobian as the matrix
            # of the edge (column) vectors v_i - v_0.
            jacs: Float[t.Tensor, "simp edge 3"] = (
                simp_vert_coords[:, 1:, :] - simp_vert_coords[:, [0], :]
            )

        match self.k:
            case 1:
                # For 1-forms, the Jacobian for each edge is the edge vector v1 - v0,
                # and the pullback is the dot product between the 1-form and
                # the edge vector.
                pullback: Float[t.Tensor, "simp point"] = t.sum(jacs * k_forms, dim=-1)
                circulation = t.sum(pullback * self.weights, dim=-1)
                return circulation

            case 2:
                # For 2-forms, we assume that they are represented with the basis
                # {dy⋀dz, dz⋀dx, dx⋀dy}, or, equivalently (under the Hodge star
                # isomorphism), as proxy 1-forms represented with the basis {dx,
                # dy, dz}. The Jacobian for edge triangle consists of the edge column
                # vectors {v1 - v0, v2 - v0}, and the pullback is the dot product
                # between the proxy 1-form and the triangle normal vector (oriented
                # to satisfy the right-hand rule and scaled to the triangle area).
                area_normal: Float[t.Tensor, "simp 1 3"] = 0.5 * t.cross(
                    jacs[:, [0], :], jacs[:, [1], :], dim=-1
                )
                pullback: Float[t.Tensor, "simp point"] = t.sum(
                    area_normal * k_forms, dim=-1
                )
                flux = t.sum(pullback * self.weights, dim=-1)
                return flux

            case 3:
                # For 3-forms, the pullback consists of the scalar product between
                # the 3-form (a scalar) and the scalar triple product of the Jacobian
                # (or, equivalently, the signed tet volume).
                signed_vol: Float[t.Tensor, "simp 1"] = get_tet_signed_vols(
                    vert_coords, k_simps
                ).view(-1, 1)

                pullback = signed_vol * k_forms
                density = t.sum(pullback * self.weights, dim=-1)

                return density

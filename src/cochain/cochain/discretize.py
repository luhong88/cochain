import torch as t
from einops import einsum, rearrange, reduce, repeat
from jaxtyping import Float

from ..complex import SimplicialComplex
from ..utils import quadrature


class DeRhamMap:
    def __init__(
        self,
        k: int,
        quad_degree: int,
        mesh: SimplicialComplex,
        allow_neg_weights: bool = True,
    ):
        """
        For 0-form, the de Rham map is equivalent to sampling the 0-form (scalar
        function) at the vertex positions; therefore, only k = 1, 2, or 3 are supported.
        """
        self.mesh = mesh
        self.k = k
        self.degree = quad_degree
        self.allow_neg_weights = allow_neg_weights

        self.ref_barys: Float[t.Tensor, "pt vert"]
        self.weights: Float[t.Tensor, " pt"]

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
    ):
        dtype = self.mesh.vert_coords.dtype
        device = self.mesh.vert_coords.device

        ref_barys, weights = self.quad(dtype, device).get_rule(
            self.degree, allow_neg_weights=self.allow_neg_weights
        )

        self.ref_barys = ref_barys
        self.weights = weights

    def sample_points(
        self,
    ) -> Float[t.Tensor, "k_simp pt coord=3"]:
        if not hasattr(self, "ref_barys"):
            self._get_quad_rule()

        # The barycentric coordinates in ref_barys provide the weights for the
        # linear combination of the simplex vertex coordinates to identify
        # the sample points in the simplex, and these weights are the same
        # for the reference and physical simplices.
        simp_vert_coords = self.mesh.vert_coords[self.mesh.simplices[self.k]]
        sampled_points = einsum(
            "k_simp vert coord, pt vert -> k_simp pt coord",
            simp_vert_coords,
            self.ref_barys,
        )

        return sampled_points

    def discretize(
        self,
        k_forms: Float[t.Tensor, "k_simp pt *ch coord"],
    ) -> Float[t.Tensor, " k_simp *ch"]:
        if not hasattr(self, "weights"):
            self._get_quad_rule()

        # Compute the Jacobian of the ref -> phys transformations (i.e., the
        # pushforward map).
        simp_vert_coords = self.mesh.vert_coords[self.mesh.simplices[self.k]]

        # Pick the first vertex of each simplex as the point of origin in the ref
        # simplex and compute the jacobian as the matrix of the edge (column)
        # vectors v_i - v_0.
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
                    "k_simp edge coord, simp pt ... coord -> k_simp pt ...",
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

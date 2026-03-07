import torch as t
from jaxtyping import Float

from ...complex import SimplicialComplex
from ...utils import quadrature
from ..tet.tet_geometry import get_tet_signed_vols


class DeRhamMap:
    def __init__(self, dim: int):
        self.dim = dim

        match dim:
            case 0:
                self.quad = None
            case 1:
                self.quad = quadrature.GaussLegendre
            case 2:
                self.quad = quadrature.Dunavant
            case 3:
                self.quad = quadrature.Keast
            case _:
                raise ValueError()

    def sample_points(
        self, mesh: SimplicialComplex, degree: int, allow_neg_weights: bool = True
    ) -> Float[t.Tensor, "simp point 3"]:
        if self.dim == 0:
            return mesh.vert_coords

        else:
            ref_barys, weights = self.quad(
                mesh.vert_coords.dtype, mesh.vert_coords.device
            ).get_rule(degree, allow_neg_weights=allow_neg_weights)

            self.weights: Float[t.Tensor, "1 point"] = weights.view(1, -1)

            simp_vert_coords = mesh.vert_coords[mesh.simplices[self.dim]]

            # For discretizing 1-forms and 2-forms, we need to compute the
            # Jacobian of the ref -> phys transformations (i.e., the pushforward
            # map); for discretizing 3-forms, we simply need the unsigned determinant
            # of the Jacobian (i.e., the tet volume).
            match self.dim:
                case 1 | 2:
                    # Pick the first vertex of each simplex as the point of origin
                    # in the ref simplex and compute the jacobian as the matrix
                    # of the edge (column) vectors v_i - v_0.
                    self.jacs: Float[t.Tensor, "simp edge 3"] = (
                        simp_vert_coords[:, 1:, :] - simp_vert_coords[:, [0], :]
                    )
                case 3:
                    self.vol: Float[t.Tensor, "simp 1"] = (
                        get_tet_signed_vols(mesh.vert_coords, mesh.tets)
                        .abs()
                        .view(-1, 1)
                    )

            # The barycentric coordinates in ref_barys provide the weights for the
            # linear combination of the simplex vertex coordinates to identify
            # the sample points in the simplex, and these weights are the same
            # for the reference and physical simplices.

            # (simp, vert, coord), (point, vert) -> (simp, point, coord)
            sampled_points = t.einsum("svc,pv->spc", simp_vert_coords, ref_barys)

            return sampled_points

    # TODO: handle simplex orientation
    def discretize(self, forms: Float[t.Tensor, "simp point *covariant"]):
        match self.dim:
            case 0:
                # Discretizing a 0-form (scalar function) is equivalent to sampling
                # the scalar function.
                return forms

            case 1:
                # For 1-forms, the Jacobian for each edge is the edge vector v1 - v0,
                # and the pullback is the dot product between the 1-form and
                # the edge vector.
                pullback: Float[t.Tensor, "simp point"] = t.sum(
                    self.jacs * forms, dim=-1
                )
                circulation = pullback * self.weights
                return circulation

            case 2:
                # For 2-forms, we assume that they are represented with the basis
                # {dy⋀dz, dz⋀dx, dx⋀dy}, or, equivalently (under the Hodge star
                # isomorphism), as proxy 1-forms represented with the basis {dx,
                # dy, dz}. The Jacobian for edge triangle consists of the edge column
                # vectors {v1 - v0, v2 - v0}, and the pullback is the dot product
                # between the proxy 1-form and the triangle normal vector (oriented
                # to satisfy the right-hand rule).
                area_normal: Float[t.Tensor, "simp 1 3"] = t.cross(
                    self.jacs[:, [0], :], self.jacs[:, [1], :], dim=-1
                )
                pullback: Float[t.Tensor, "simp point"] = t.sum(
                    area_normal * forms, dim=-1
                )
                # 0.5 scales the numerical quadrature with the area of the ref triangle.
                flux = 0.5 * pullback * self.weights
                return flux

            case 3:
                # For 3-forms, the pullback consists of the scalar product between
                # the 3-form (a scalar) and the scalar triple product of the
                # Jacobian (the tet volume).
                pullback = self.vol * forms
                # 1/6 scales the numerical quadrature with the volume of the ref tet.
                density = pullback * self.weights / 6.0
                return density

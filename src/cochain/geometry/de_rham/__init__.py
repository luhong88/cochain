import torch as t
from jaxtyping import Float

from ...complex import SimplicialComplex
from ...utils import quadrature
from ...utils.perm_parity import compute_lex_rel_orient
from ..tet.tet_geometry import get_tet_signed_vols


class _DeRhamMap0Form:
    def __init__(self, mesh: SimplicialComplex):
        self.mesh = mesh

    def sample_points(self):
        return self.mesh.vert_coords

    def discretize(self, forms: Float[t.Tensor, "simp point"]):
        # Discretizing a 0-form (scalar function) is equivalent to sampling the
        # scalar function.
        return forms


class _DeRhamMapKForm:
    def __init__(self, mesh: SimplicialComplex, dim: int):
        self.mesh = mesh
        self.dim = dim

        dtype = self.mesh.vert_coords.dtype
        device = self.mesh.vert_coords.device

        match dim:
            case 1:
                self.quad = quadrature.GaussLegendre(dtype, device)
            case 2:
                self.quad = quadrature.Dunavant(dtype, device)
            case 3:
                self.quad = quadrature.Keast(dtype, device)
            case _:
                raise ValueError()

        # In the SimplicialComplex class, we assume that the vertex ordering
        # is lex sorted for all but the top level n-simplices, where the vertex
        # index ordering carries information on geometric orientation. Therefore,
        # an orientation parity sign correction is needed to account for the
        # orientation of such simplices relative to the mesh.
        self.parity = compute_lex_rel_orient(mesh.simplices[self.dim])

    def sample_points(
        self, degree: int, allow_neg_weights: bool = True
    ) -> Float[t.Tensor, "simp point 3"]:
        ref_barys, weights = self.quad.get_rule(
            degree, allow_neg_weights=allow_neg_weights
        )

        self.weights: Float[t.Tensor, "1 point"] = weights.view(1, -1)

        simp_vert_coords = self.mesh.vert_coords[self.mesh.simplices[self.dim]]

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
                self.signed_vol: Float[t.Tensor, "simp 1"] = get_tet_signed_vols(
                    self.mesh.vert_coords, self.mesh.tets
                ).view(-1, 1)

        # The barycentric coordinates in ref_barys provide the weights for the
        # linear combination of the simplex vertex coordinates to identify
        # the sample points in the simplex, and these weights are the same
        # for the reference and physical simplices.

        # (simp, vert, coord), (point, vert) -> (simp, point, coord)
        sampled_points = t.einsum("svc,pv->spc", simp_vert_coords, ref_barys)

        return sampled_points

    def discretize(self, forms: Float[t.Tensor, "simp point *covariant"]):
        match self.dim:
            case 1:
                # For 1-forms, the Jacobian for each edge is the edge vector v1 - v0,
                # and the pullback is the dot product between the 1-form and
                # the edge vector.
                pullback: Float[t.Tensor, "simp point"] = t.sum(
                    self.jacs * forms, dim=-1
                )
                circulation = self.parity * pullback * self.weights
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
                flux = 0.5 * self.parity * pullback * self.weights
                return flux

            case 3:
                # For 3-forms, the pullback consists of the scalar product between
                # the 3-form (a scalar) and the scalar triple product of the
                # Jacobian (the signed tet volume).
                pullback = self.signed_vol * forms
                # 1/6 scales the numerical quadrature with the volume of the ref tet.
                density = self.parity * pullback * self.weights / 6.0
                return density


class DeRhamMap:
    def __init__(self, mesh: SimplicialComplex, dim: int):
        if dim == 0:
            return _DeRhamMap0Form(mesh)
        else:
            return _DeRhamMapKForm(mesh, dim)

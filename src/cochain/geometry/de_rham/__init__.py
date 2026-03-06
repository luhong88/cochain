import torch as t
from jaxtyping import Float, Integer

from ...complex import SimplicialComplex
from ...utils import quadrature
from ..tet.tet_geometry import get_tet_signed_vols
from ..tri.tri_geometry import compute_tri_areas


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

            self.weights = weights

            simp_vert_coords = mesh.vert_coords[mesh.simplices[self.dim]]

            # Pick the first vertex of each simplex as the point of origin in
            # the ref simplex and compute the jacobian of the ref -> phys
            # transformations, which is also the pushforward map.
            self.jacs = simp_vert_coords[:, 1:, :] - simp_vert_coords[:, [0], :]

            # Compute the simplex volumes
            match self.dim:
                case 1:
                    self.vol = t.linalg.norm(
                        simp_vert_coords[:, 1] - simp_vert_coords[:, 0]
                    )
                case 2:
                    self.vol = compute_tri_areas(mesh.vert_coords, mesh.tris)
                case 3:
                    self.vol = get_tet_signed_vols(mesh.vert_coords, mesh.tets).abs()

            # The barycentric coordinates in ref_barys provide the weights for the
            # linear combination of the simplex vertex coordinates to identify
            # the sample points in the simplex, and these weights are the same
            # for the reference and physical simplices.

            # (simp, vert, coord), (point, vert) -> (simp, point, coord)
            sampled_points = t.einsum("svc,pv->spc", simp_vert_coords, ref_barys)

            return sampled_points

    def discretize(self, forms):
        if self.dim == 0:
            return forms

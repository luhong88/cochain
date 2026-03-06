import torch as t
from jaxtyping import Float, Integer

from ...complex import SimplicialComplex
from ...utils import quadrature


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

            # (simp, vert, coord), (point, vert) -> (simp, point, coord)
            sampled_points = t.einsum("svc,pv->spc", simp_vert_coords, ref_barys)

            return sampled_points

    def discretize(self, forms):
        if self.dim == 0:
            return forms

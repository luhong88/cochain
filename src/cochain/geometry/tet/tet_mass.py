import torch as t
from jaxtyping import Float, Integer

from ...complex import SimplicialComplex
from ...utils.constants import EPS
from .tet_geometry import tet_signed_vols


def mass_0(tet_mesh: SimplicialComplex) -> Float[t.Tensor, "vert"]:
    """
    Compute the diagonal elements of the vertex mass matrix, which is equivalent
    to the barycentric 0-star.

    The barycentric dual volume for each vertex is the sum of 1/4 of the volumes
    of all tetrahedra that share the vertex as a face.
    """
    n_verts = tet_mesh.n_verts

    tri_area = t.abs(tet_signed_vols(tet_mesh.vert_coords, tet_mesh.tets))

    diag = t.zeros(n_verts, device=tet_mesh.vert_coords.device)
    diag.scatter_add_(
        dim=0,
        index=tet_mesh.tets.flatten(),
        src=t.repeat_interleave(tri_area / 4.0, 4),
    )

    return diag

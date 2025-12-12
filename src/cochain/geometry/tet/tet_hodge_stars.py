import torch as t
from jaxtyping import Float, Integer

from ...complex import SimplicialComplex
from ..tri.tri_geometry import _tri_areas
from .tet_masses import _tet_tri_face_idx


def star_2(tet_mesh: SimplicialComplex) -> Float[t.Tensor, "tri"]:
    """
    Compute the barycentric Hodge 2-star operator.
    """
    i, j, k, l = 0, 1, 2, 3

    tet_barycenters: Float[t.Tensor, "tet 1 3"] = t.mean(
        tet_mesh.vert_coords[tet_mesh.tets], dim=-2, keepdim=True
    )
    tet_face_barycenters = Float[t.Tensor, "tet 4 3"] = t.mean(
        tet_mesh.vert_coords[:, [[j, k, l], [i, l, k], [i, j, l], [i, k, j]]], dim=-2
    )

    dual_edges = tet_barycenters - tet_face_barycenters
    dual_edge_lens: Float[t.Tensor, "tet 4"] = t.linalg.norm(dual_edges, dim=-1)

    all_canon_tris_idx: Integer[t.LongTensor, "tet 4"] = _tet_tri_face_idx(tet_mesh)

    diag = t.zeros(
        tet_mesh.n_tris,
        dtype=tet_mesh.vert_coords.dtype,
        device=tet_mesh.vert_coords.device,
    )
    diag.scatter_add_(
        dim=0,
        index=all_canon_tris_idx.flatten(),
        src=dual_edge_lens.flatten(),
    )

    tri_areas = _tri_areas(tet_mesh.vert_coords, tet_mesh.tris)
    diag.divide_(tri_areas)

    return diag

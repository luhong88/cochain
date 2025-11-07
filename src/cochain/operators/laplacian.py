import torch as t
from jaxtyping import Float, Integer

from ..data.simplicial_mesh import SimplicialMesh


def cotan_laplacian(mesh: SimplicialMesh) -> Float[t.Tensor, "vert vert"]:
    asym_laplacian = t.zeros(mesh.n_vert, mesh.n_vert)

    # for each face, compute the cotan of the angle at each vertex
    face_vert_coord: Float[t.Tensor, "face 3 3"] = mesh.vert[mesh.face]

    face_vert_vec1 = face_vert_coord[:, [1, 2, 0], :] - face_vert_coord
    face_vert_uvec1 = face_vert_vec1 / t.linalg.norm(
        face_vert_vec1, dim=-1, keepdim=True
    )

    face_vert_vec2 = face_vert_coord[:, [2, 0, 1], :] - face_vert_coord
    face_vert_uvec2 = face_vert_vec2 / t.linalg.norm(
        face_vert_vec2, dim=-1, keepdim=True
    )

    face_vert_ang_cos = t.sum(face_vert_uvec1 * face_vert_uvec2, dim=-1)
    face_vert_ang_sin = t.linalg.norm(
        t.linalg.cross(face_vert_uvec1, face_vert_uvec2, dim=-1), dim=-1
    )
    face_vert_ang_cotan: Float[t.Tensor, "face 3"] = face_vert_ang_cos / (
        1e-9 + face_vert_ang_sin
    )

    # for each face ijk, scatter the cotan at i to edge jk in the laplacian (L_jk)
    asym_laplacian[mesh.face[:, 1], mesh.face[:, 2]] += -0.5 * face_vert_ang_cotan[:, 0]
    asym_laplacian[mesh.face[:, 0], mesh.face[:, 2]] += -0.5 * face_vert_ang_cotan[:, 1]
    asym_laplacian[mesh.face[:, 0], mesh.face[:, 1]] += -0.5 * face_vert_ang_cotan[:, 2]

    # symmetrize so that the cotan at i is scattered to both jk and kj
    sym_laplacian = asym_laplacian + asym_laplacian.T

    # populate the diagonal elements of the laplacian
    laplacian = sym_laplacian - t.diagflat(t.sum(sym_laplacian, dim=-1))

    return laplacian

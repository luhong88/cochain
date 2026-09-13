from typing import Literal

import pytest
import torch
from jaxtyping import Float

from cochain.complex import SimplicialMesh, collate_fn
from cochain.metric import hodge_laplacians
from cochain.metric.tri import tri_hodge_stars
from cochain.sparse.decoupled_tensor import SparseDecoupledTensor


def _codifferential_1(
    tri_mesh: SimplicialMesh,
    dual_complex: Literal["circumcentric", "barycentric"] = "barycentric",
) -> Float[SparseDecoupledTensor, "vert edge"]:
    """Codifferential on discrete 1-forms for a tri mesh."""
    return hodge_laplacians.codifferential(
        cbd_km1=tri_mesh.cbd[0],
        mass_k=tri_hodge_stars.star_1(tri_mesh, dual_complex),
        inv_mass_km1=tri_hodge_stars.star_0(tri_mesh).inv,
    )


def _codifferential_2(
    tri_mesh: SimplicialMesh,
    dual_complex: Literal["circumcentric", "barycentric"] = "barycentric",
) -> Float[SparseDecoupledTensor, "edge tri"]:
    """Codifferential on discrete 2-forms for a tri mesh."""
    return hodge_laplacians.codifferential(
        cbd_km1=tri_mesh.cbd[1],
        mass_k=tri_hodge_stars.star_2(tri_mesh),
        inv_mass_km1=tri_hodge_stars.star_1(tri_mesh, dual_complex).inv,
    )


def _hodge_laplacian_0(
    tri_mesh: SimplicialMesh,
    dual_complex: Literal["circumcentric", "barycentric"] = "barycentric",
) -> Float[SparseDecoupledTensor, "vert vert"]:
    """Classical Hodge 0-Laplacian for a tri mesh."""
    return _codifferential_1(tri_mesh, dual_complex) @ tri_mesh.cbd[0]


def _hodge_laplacian_1_grad_div(
    tri_mesh: SimplicialMesh,
    dual_complex: Literal["circumcentric", "barycentric"] = "barycentric",
) -> Float[SparseDecoupledTensor, "edge edge"]:
    """Grad-div component of the classical 1-Laplacian for a tri mesh."""
    return tri_mesh.cbd[0] @ _codifferential_1(tri_mesh, dual_complex)


def _hodge_laplacian_1_curl_curl(
    tri_mesh: SimplicialMesh,
    dual_complex: Literal["circumcentric", "barycentric"] = "barycentric",
) -> Float[SparseDecoupledTensor, "edge edge"]:
    """Curl-curl component of the classical 1-Laplacian for a tri mesh."""
    return _codifferential_2(tri_mesh, dual_complex) @ tri_mesh.cbd[1]


def _hodge_laplacian_1(
    tri_mesh: SimplicialMesh,
    dual_complex: Literal["circumcentric", "barycentric"] = "barycentric",
) -> Float[SparseDecoupledTensor, "edge edge"]:
    """Classical Hodge 1-Laplacian for a tri mesh."""
    return SparseDecoupledTensor.assemble(
        _hodge_laplacian_1_grad_div(tri_mesh, dual_complex),
        _hodge_laplacian_1_curl_curl(tri_mesh, dual_complex),
    )


def _hodge_laplacian_2(
    tri_mesh: SimplicialMesh,
    dual_complex: Literal["circumcentric", "barycentric"] = "barycentric",
) -> Float[SparseDecoupledTensor, "tri tri"]:
    """Classical Hodge 2-Laplacian for a tri mesh."""
    return tri_mesh.cbd[1] @ _codifferential_2(tri_mesh, dual_complex)


laplacian_constructors = {
    0: _hodge_laplacian_0,
    1: _hodge_laplacian_1,
    2: _hodge_laplacian_2,
}


@pytest.mark.parametrize("k", [0, 1, 2])
def test_block_diag_batching(
    k: int, two_tris_mesh: SimplicialMesh, hollow_tet_mesh: SimplicialMesh, device
):
    # Compute matrix-vector product per mesh.
    mesh_1 = two_tris_mesh.to(device)
    mesh_2 = hollow_tet_mesh.to(device)

    cochain_1 = torch.randn(mesh_1.n_splx[k], dtype=mesh_1.dtype, device=mesh_1.device)
    cochain_2 = torch.randn(mesh_2.n_splx[k], dtype=mesh_2.dtype, device=mesh_2.device)

    laplacian_1 = laplacian_constructors[k](mesh_1)
    laplacian_2 = laplacian_constructors[k](mesh_2)

    output_1_true = laplacian_1 @ cochain_1
    output_2_true = laplacian_2 @ cochain_2

    # Compute matrix-vector product in a batch.
    mesh_batch = collate_fn([mesh_1, mesh_2])
    cochain_batch = torch.concat((cochain_1, cochain_2))

    laplacian_batch = laplacian_constructors[k](mesh_batch)

    output_batch = laplacian_batch @ cochain_batch

    output_1 = output_batch[mesh_batch.ptrs[k] == 0]
    output_2 = output_batch[mesh_batch.ptrs[k] == 1]

    torch.testing.assert_close(output_1, output_1_true)
    torch.testing.assert_close(output_2, output_2_true)

    # Check Laplacian unpacking.
    laplacians = laplacian_batch.unpack_by_ptrs(
        n_blocks=2, row_ptrs=mesh_batch.ptrs[k], col_ptrs=mesh_batch.ptrs[k]
    )

    torch.testing.assert_close(laplacians[0].to_dense(), laplacian_1.to_dense())
    torch.testing.assert_close(laplacians[1].to_dense(), laplacian_2.to_dense())

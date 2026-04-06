import pytest
import torch

from cochain.complex import SimplicialMesh, collate_fn
from cochain.geometry.tri import tri_laplacians


@pytest.mark.parametrize("k", [0, 1, 2])
def test_block_diag_batching(
    k: int, two_tris_mesh: SimplicialMesh, hollow_tet_mesh: SimplicialMesh, device
):
    # Compute matrix-vector product per mesh.
    mesh_1 = two_tris_mesh.to(device)
    mesh_2 = hollow_tet_mesh.to(device)

    cochain_1 = torch.randn(mesh_1.n_splx[k], dtype=mesh_1.dtype, device=mesh_1.device)
    cochain_2 = torch.randn(mesh_2.n_splx[k], dtype=mesh_2.dtype, device=mesh_2.device)

    laplacian_1 = getattr(tri_laplacians, f"laplacian_{k}")(mesh_1)
    laplacian_2 = getattr(tri_laplacians, f"laplacian_{k}")(mesh_2)

    output_1_true = laplacian_1 @ cochain_1
    output_2_true = laplacian_2 @ cochain_2

    # Compute matrix-vector product in a batch.
    mesh_batch = collate_fn([mesh_1, mesh_2])
    cochain_batch = torch.concat((cochain_1, cochain_2))

    laplacian_batch = getattr(tri_laplacians, f"laplacian_{k}")(mesh_batch)

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

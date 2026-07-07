# %%
import math
import os

import matplotlib.pyplot as plt
import numpy as np
import polyscope as ps
import pyvista as pv
import torch
from jaxtyping import Float, Integer
from nvmath.sparse.advanced import DirectSolverOptions
from torch import Tensor

from cochain.complex import SimplicialMesh
from cochain.metric.tri import tri_hodge_stars, tri_masses
from cochain.sparse.decoupled_tensor import SparseDecoupledTensor
from cochain.sparse.linalg.eigen import LOBPCGConfig, LOBPCGPrecondConfig, lobpcg
from cochain.sparse.linalg.solvers import (
    DirectSolverConfig,
    NVMathDirectSolver,
)
from cochain.topology import betti
from cochain.vis import PolyscopeViewer


# %%
def _compute_edge_matrix(
    vert_coords: Float[Tensor, "tri coord=3"],
    tris: Integer[Tensor, "tri vert=3"],
) -> Float[Tensor, "tri 2 2"]:
    tris_ref = vert_coords[tris]

    e01 = tris_ref[:, 1] - tris_ref[:, 0]
    e02 = tris_ref[:, 2] - tris_ref[:, 0]

    e01_norm = torch.linalg.norm(e01, dim=-1)
    e02_proj_x_coord = torch.sum(e01 * e02, dim=-1) / e01_norm
    e02_proj_y_coord = (
        torch.linalg.norm(torch.linalg.cross(e01, e02, dim=-1), dim=-1) / e01_norm
    )

    e01_proj_coord = torch.stack([e01_norm, torch.zeros_like(e01_norm)], dim=-1)
    e02_proj_coord = torch.stack([e02_proj_x_coord, e02_proj_y_coord], dim=-1)

    edge_mat = torch.stack([e01_proj_coord, e02_proj_coord], dim=-1)

    return edge_mat


def compute_deformation_gradient(
    edge_mat_inv_ref: Float[Tensor, "tri 2 2"],
    vert_coords_deformed: Float[Tensor, "tri coord=3"],
    tris: Integer[Tensor, "tri vert=3"],
) -> Float[Tensor, "tri 2 2"]:
    edge_mat_deformed = _compute_edge_matrix(vert_coords_deformed, tris)
    deform_grad = edge_mat_deformed @ edge_mat_inv_ref

    return deform_grad


# %%
def consistent_stiff_0(
    mesh: SimplicialMesh,
) -> Float[SparseDecoupledTensor, "vert vert"]:
    # S_0 = d_0^T \star_1 d_0
    m_1 = tri_masses.mass_1(mesh)
    d_0 = mesh.cbd[0]

    l_0 = d_0.T @ m_1 @ d_0

    return l_0


# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# %%
mt_lib_path = f"{os.environ['VIRTUAL_ENV']}/lib/python3.13/site-packages/nvidia/cu12/lib/libcudss_mtlayer_gomp.so.0"
nvmath_config = DirectSolverConfig(
    options=DirectSolverOptions(multithreading_lib=mt_lib_path)
)

lobpcg_config = LOBPCGConfig(largest=False, maxiter=200, tol=1e-6)
precond_config = LOBPCGPrecondConfig(method="cholesky", nvmath_config=nvmath_config)

# %%
ps.init()

# %%
ps.remove_all_structures()

# %%
toroid_pv = pv.ParametricSuperToroid(n1=0.75, n2=0.5, u_res=50, v_res=50).smooth_taubin(
    n_iter=50, pass_band=0.1
)
toroid_pv_tri = toroid_pv.clean().triangulate().smooth_taubin(n_iter=50, pass_band=0.1)

# %%
toroid = SimplicialMesh.from_tri_mesh(
    vert_coords=torch.from_numpy(np.asarray(toroid_pv_tri.points)).to(
        dtype=torch.float64
    ),
    tris=torch.from_numpy(np.asarray(toroid_pv_tri.regular_faces)).to(
        dtype=torch.int64
    ),
).to(device)

# %%
tris = toroid.vert_coords[toroid.tris]
tri_normals = torch.linalg.cross(
    tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0], dim=-1
)

# %%
viewer_original = PolyscopeViewer(name="original_mesh", mesh=toroid)
viewer_original.add_vector_field(k=2, name="area_norm", vec_field=tri_normals)

# %%
ps.show()

# %%
assert not toroid.bd_edge_mask.any()

# %%
b0, b1, b2 = betti.compute_betti_numbers(mesh=toroid, manifold=True)
assert b0 == 1
assert b1 == 2
assert b2 == 1

# %%
tris = toroid.vert_coords[toroid.tris]

init_areas = tri_hodge_stars.compute_tri_areas(toroid.vert_coords, toroid.tris)
init_area = init_areas.sum()
init_vol = torch.sum(tris[:, 0] * torch.linalg.cross(tris[:, 1], tris[:, 2], dim=-1))

# %%
ideal_tri_area = init_area / toroid.n_tris

equilat_vert_coords = torch.tensor(
    [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.5, math.sqrt(3.0) / 2.0, 0.0]],
    dtype=toroid.dtype,
    device=toroid.device,
) * math.sqrt(ideal_tri_area / (math.sqrt(3.0) / 4.0))

equilat_tri = torch.tensor([[0, 1, 2]], dtype=toroid.tris.dtype, device=toroid.device)

edge_mat_ref = _compute_edge_matrix(equilat_vert_coords, equilat_tri)
edge_mat_inv_ref = torch.linalg.inv(edge_mat_ref)

# %%
sobolev_strength = 1.0
tau = sobolev_strength * init_area

m0 = tri_hodge_stars.star_0(tri_mesh=toroid).to_sdt()

s0 = consistent_stiff_0(toroid)

op = SparseDecoupledTensor.assemble(tau * s0, m0)
solver = NVMathDirectSolver(
    a=op, b=toroid.vert_coords, sparse_system_type="spd", config=nvmath_config
)

# %%
n_eigs = b0 + 5

# %%
eig_vals, eig_vecs = lobpcg(
    a=s0,
    m=m0,
    n=2 * n_eigs,
    k=n_eigs,
    eps=0,
    lobpcg_config=lobpcg_config,
    precond_config=precond_config,
)

# %%
fundamental = eig_vals[b0]
target_eig_vals = fundamental * torch.arange(
    b0, n_eigs, dtype=eig_vals.dtype, device=eig_vals.device
)

# %%
toroid.requires_grad_()

# %%
n_steps = 1000

optimizer = torch.optim.SGD([toroid.vert_coords], lr=10.0)

reg_strength = 5e-3
willmore_strength = 1e-3

eps = torch.finfo(toroid.vert_coords.dtype).eps

update_frequency = 50

log_mse_traj = []
dirichlet_loss_traj = []
willmore_loss_traj = []

for t in range(n_steps):
    optimizer.zero_grad()

    m0_ddt = tri_hodge_stars.star_0(toroid)
    m0 = m0_ddt.to_sdt()
    s0 = consistent_stiff_0(toroid)

    eig_vals, eig_vecs = lobpcg(
        a=s0,
        m=m0,
        n=2 * n_eigs,
        k=n_eigs,
        eps=0,
        lobpcg_config=lobpcg_config,
        precond_config=precond_config,
    )

    mesh_area = tri_hodge_stars.compute_tri_areas(toroid.vert_coords, toroid.tris).sum()

    spectral_log_mse = torch.sum(
        (torch.log(target_eig_vals * init_area) - torch.log(eig_vals[b0:] * mesh_area))
        ** 2
    )

    log_mse_traj.append(spectral_log_mse.detach().cpu().numpy())

    deform_grad = compute_deformation_gradient(
        edge_mat_inv_ref, toroid.vert_coords, toroid.tris
    )
    deform_grad_sym = torch.transpose(deform_grad, -1, -2) @ deform_grad

    deform_grad_sym_inv = torch.linalg.inv(deform_grad_sym)

    dirichlet = torch.einsum("bii->b", deform_grad_sym)
    dirichlet_inv = torch.einsum("bii->b", deform_grad_sym_inv)
    dirichlet_sym = (
        reg_strength
        * 0.99 ** (t / 5.0)
        * torch.sum((dirichlet + dirichlet_inv) * ideal_tri_area)
    )
    dirichlet_loss_traj.append(dirichlet_sym.detach().cpu().numpy())

    willmore = (
        willmore_strength
        * 0.99 ** (t / 5.0)
        * torch.trace(toroid.vert_coords.T @ s0 @ m0_ddt.inv @ s0 @ toroid.vert_coords)
    )
    willmore_loss_traj.append(willmore.detach().cpu().numpy())

    energy = spectral_log_mse + dirichlet_sym + willmore
    energy.backward()

    with torch.no_grad():
        grad = toroid.vert_coords.grad.detach().clone()
        grad_smoothed = solver(grad, trans="N")
        toroid.vert_coords.grad = grad_smoothed.contiguous()

    optimizer.step()

    with torch.no_grad():
        toroid.vert_coords.sub_(toroid.vert_coords.mean(dim=0, keepdim=True))

        tris = toroid.vert_coords[toroid.tris]
        updated_vol = torch.sum(
            tris[:, 0] * torch.linalg.cross(tris[:, 1], tris[:, 2], dim=-1)
        )
        vol_ratio = (init_vol / updated_vol).pow(1.0 / 3.0)
        toroid.vert_coords.mul_(vol_ratio)

        if t > 0 and t % update_frequency == 0:
            # edge_mat_ref = _compute_edge_matrix(toroid.vert_coords, toroid.tris)
            # edge_mat_inv_ref = torch.linalg.inv(edge_mat_ref)
            # init_areas = tri_hodge_stars.compute_tri_areas(
            #     toroid.vert_coords, toroid.tris
            # )

            m0_ref = tri_hodge_stars.star_0(tri_mesh=toroid).to_sdt()
            s0_ref = consistent_stiff_0(toroid)

            op = SparseDecoupledTensor.assemble(tau * s0_ref, m0_ref)
            solver = NVMathDirectSolver(
                a=op,
                b=toroid.vert_coords,
                sparse_system_type="spd",
                config=nvmath_config,
            )

    if t % update_frequency == 0:
        print(
            f"Timestep: {t:04d}; "
            f"spectral_log_mse: {spectral_log_mse:.4f}; "
            f"dirichlet: {dirichlet_sym:.4f}; "
            f"willmore: {willmore:.4f}"
        )

# %%
viewer_updated = PolyscopeViewer(name="optimized_mesh", mesh=toroid)

# %%
for i in range(1, n_eigs):
    viewer_updated.add_k_cochain(
        k=0, name=f"eig_{i}", cochain=eig_vecs[:, i], datatype="symmetric"
    )

# %%
ps.show()

# %%
log_mse_traj = np.asarray(log_mse_traj)
dirichlet_loss_traj = np.asarray(dirichlet_loss_traj)
willmore_loss_traj = np.asarray(willmore_loss_traj)

t_traj = np.asarray(range(len(log_mse_traj))) + 1

# %%
fig, axes = plt.subplots(1, 3, figsize=(8, 4), layout="constrained")


axes[0].plot(t_traj, log_mse_traj, color="dodgerblue")
axes[0].set_xscale("log")
axes[0].set(xlabel="Time", ylabel="log_mse")


axes[1].set_xscale("log")
axes[1].plot(t_traj, willmore_loss_traj, color="dodgerblue")
axes[1].set(xlabel="Time", ylabel="willmore_loss")

axes[2].plot(t_traj, dirichlet_loss_traj, color="dodgerblue")
axes[2].set_xscale("log")
axes[2].set(xlabel="Time", ylabel="sym_dirichlet_loss")

plt.show()
plt.close()

# %%
with torch.no_grad():
    print(eig_vals[b0:] / eig_vals[b0])

# %%

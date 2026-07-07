# %%
import math
import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import polyscope as ps
import pytetwild
import pyvista as pv
import torch
from jaxtyping import Float, Integer
from nvmath.sparse.advanced import DirectSolverOptions
from torch import Tensor

from cochain.complex import SimplicialMesh
from cochain.datasets.synthetic_tet_meshes import load_regular_tet_mesh
from cochain.metric.tet import tet_hodge_stars, tet_laplacians, tet_masses
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
    vert_coords: Float[Tensor, "vert coord=3"],
    tets: Integer[Tensor, "tet local_vert=4"],
) -> Float[Tensor, "tet coord=3 local_edge=3"]:
    tet_verts = vert_coords[tets]
    edge_mat_T = tet_verts[:, 1:] - tet_verts[:, [0]]
    return edge_mat_T.transpose(-1, -2)


def compute_deformation_gradient(
    edge_mat_inv_ref: Float[Tensor, "tet coord=3 local_edge=3"],
    vert_coords_deformed: Float[Tensor, "vert coord=3"],
    tets: Integer[Tensor, "tet local_vert=4"],
) -> Float[Tensor, "tet coord=3 coord=3"]:
    edge_mat_deformed = _compute_edge_matrix(vert_coords_deformed, tets)
    deform_grad = edge_mat_deformed @ edge_mat_inv_ref

    return deform_grad


# %%
def consistent_tri_stiff_0(
    mesh: SimplicialMesh,
) -> Float[SparseDecoupledTensor, "vert vert"]:
    # S_0 = d_0^T \star_1 d_0
    m_1 = tri_masses.mass_1(mesh)
    d_0 = mesh.cbd[0]

    l_0 = d_0.T @ m_1 @ d_0

    return l_0


# %%
def project_to_superellipsoid(mesh, n1, n2):
    """
    Mathematically maps a uniform spherical mesh onto a superellipsoid surface.
    """
    # Extract coordinates from the uniform base mesh
    pts = mesh.points.copy()

    # Project to an exact unit sphere first to guarantee normalized inputs
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    pts /= np.where(norms == 0, 1, norms)

    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]

    # Calculate horizontal radius in the XY plane
    xy_radius = np.sqrt(x**2 + y**2)

    # Avoid division by zero at the exact poles
    with np.errstate(divide="ignore", invalid="ignore"):
        cos_omega = np.where(xy_radius == 0, 0, x / xy_radius)
        sin_omega = np.where(xy_radius == 0, 0, y / xy_radius)

    cos_eta = xy_radius
    sin_eta = z

    # Apply the superellipsoid signed power transformations
    def signed_power(val, power):
        return np.sign(val) * (np.abs(val) ** power)

    new_x = signed_power(cos_eta, n1) * signed_power(cos_omega, n2)
    new_y = signed_power(cos_eta, n1) * signed_power(sin_omega, n2)
    new_z = signed_power(sin_eta, n1)

    # Overwrite the mesh points with the newly deformed coordinates
    mesh.points = np.column_stack((new_x, new_y, new_z))
    return mesh


# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# %%
mt_lib_path = f"{os.environ['VIRTUAL_ENV']}/lib/python3.13/site-packages/nvidia/cu12/lib/libcudss_mtlayer_gomp.so.0"
nvmath_config = DirectSolverConfig(
    options=DirectSolverOptions(multithreading_lib=mt_lib_path)
)

lobpcg_config = LOBPCGConfig(largest=False, maxiter=20, tol=1e-6)
precond_config = LOBPCGPrecondConfig(method="cholesky", nvmath_config=nvmath_config)
# precond_config = LOBPCGPrecondConfig(method="ilu")

# %%
ps.init()

# %%
ps.remove_all_structures()

# %%
# # 1. Create a highly uniform, regular base mesh (Icosahedron)
# # Increasing 'subdivisions' gives a dense, uniform triangulation without pole crowding
# base_sphere = pv.Icosahedron(radius=1.0).subdivide(4)

# # 2. Map the uniform vertices into the superellipsoid geometry
# # ellipsoid_mesh = project_to_superellipsoid(base_sphere, n1=0.75, n2=0.5)
# ellipsoid_mesh = project_to_superellipsoid(base_sphere, n1=1.6, n2=1.6)

# # 3. Clean up any duplicate vertices if necessary
# ellipsoid_pv_tri = (
#     ellipsoid_mesh.clean().triangulate().smooth_taubin(n_iter=50, pass_band=0.1)
# )

ellipsoid_pv_tri = pv.examples.download_bunny_coarse().clean().triangulate()

# %%
v_tet, t_tet = pytetwild.tetrahedralize(
    ellipsoid_pv_tri.points,
    ellipsoid_pv_tri.faces.reshape(-1, 4)[:, 1:],
    edge_length_fac=0.7,
)

# Swap the second and third vertex indices to invert the winding order and avoid neg volume.
t_tet = t_tet[:, [0, 2, 1, 3]]

# %%
ellipsoid = SimplicialMesh.from_tet_mesh(
    vert_coords=15.0 * torch.from_numpy(v_tet).to(dtype=torch.float64),
    tets=torch.from_numpy(t_tet).to(dtype=torch.int64),
).to(device)

ellipsoid.vert_coords.sub_(ellipsoid.vert_coords.mean(dim=0, keepdim=True))

# %%
viewer_original = PolyscopeViewer(name="original_mesh", mesh=ellipsoid)

# %%
b0, b1, b2 = betti.compute_betti_numbers(mesh=ellipsoid)
assert b0 == 1
assert b1 == 0
assert b2 == 0

# %%
vert_idx_map = torch.cumsum(ellipsoid.bd_vert_mask, dim=0) - 1

bd_mesh = SimplicialMesh.from_tri_mesh(
    vert_coords=ellipsoid.vert_coords[ellipsoid.bd_vert_mask],
    tris=vert_idx_map[ellipsoid.tris[ellipsoid.bd_tri_mask]],
)

# %%
reg_tet = load_regular_tet_mesh().to(dtype=torch.float64, device=device)
reg_tet_vol = tet_masses.compute_tet_signed_vols(
    reg_tet.vert_coords, reg_tet.tets
).item()

# %%
init_vol = tet_masses.compute_tet_signed_vols(
    ellipsoid.vert_coords, ellipsoid.tets
).sum()

ideal_tet_vol = init_vol / ellipsoid.n_tets

reg_tet_vert_coords = reg_tet.vert_coords * math.cbrt(ideal_tet_vol / reg_tet_vol)

edge_mat_ref = _compute_edge_matrix(reg_tet_vert_coords, reg_tet.tets)
edge_mat_inv_ref = torch.linalg.inv(edge_mat_ref)

# %%
sobolev_strength = 1.0
tau = sobolev_strength * init_vol

m0 = tet_hodge_stars.star_0(tet_mesh=ellipsoid).to_sdt()
s0 = tet_laplacians.weak_laplacian_0(ellipsoid, method="consistent")

op = SparseDecoupledTensor.assemble(tau * s0, m0)
solver = NVMathDirectSolver(
    a=op, b=ellipsoid.vert_coords, sparse_system_type="spd", config=nvmath_config
)

# %%
m1_ddt = tet_hodge_stars.star_1(ellipsoid)
m1 = m1_ddt.to_sdt()
# hybrid definition: S_1 = d_1^T M_2 d_1 + M_1 d_0 M_0^{-1} d_0^T M_1
s1 = tet_laplacians.weak_laplacian_1(ellipsoid)

eig_vals, eig_vecs = lobpcg(
    a=s1, m=m1, k=3, eps=0, lobpcg_config=lobpcg_config, precond_config=precond_config
)

eig_vals

# %%
init_eig_vec = eig_vecs[:, 0]

# %%
right_ear_tip = torch.tensor([-1.10, 2.66, -2.01], dtype=torch.float64, device=device)
right_ear_base = torch.tensor([-1.52, 1.89, -0.20], dtype=torch.float64, device=device)
right_ear_len = torch.linalg.norm(right_ear_tip - right_ear_base)

left_ear_tip = torch.tensor([0.60, 2.52, -1.19], dtype=torch.float64, device=device)
left_ear_base = torch.tensor([0.68, 1.80, 0.05], dtype=torch.float64, device=device)
left_ear_len = torch.linalg.norm(left_ear_tip - left_ear_base)

# %%
tet_centroids = ellipsoid.vert_coords[ellipsoid.tets].mean(dim=-2)
tet_mask = (
    (
        torch.linalg.norm(tet_centroids - right_ear_tip.unsqueeze(0), dim=-1)
        < 0.8 * right_ear_len
    )
    & (
        torch.linalg.norm(tet_centroids - right_ear_base.unsqueeze(0), dim=-1)
        < right_ear_len
    )
    & ~(
        (
            torch.linalg.norm(tet_centroids - left_ear_tip.unsqueeze(0), dim=-1)
            < left_ear_len
        )
        & (
            torch.linalg.norm(tet_centroids - left_ear_base.unsqueeze(0), dim=-1)
            < left_ear_len
        )
    )
).to(dtype=torch.float64)

# %%
viewer_original.add_k_cochain(k=3, name="tet_mask", cochain=tet_mask)

# %%
edge_mask = ellipsoid.cbd[1].abs().T @ ellipsoid.cbd[2].abs().T @ tet_mask

# %%
edge_mask.shape

# %%
edge_mask.sum()

# %%
torch.abs(init_eig_vec * edge_mask).sum() / edge_mask.sum()

# %%
torch.abs(init_eig_vec * (1 - edge_mask)).sum() / torch.abs(1 - edge_mask).sum()

# %%
viewer_original.add_k_cochain(k=1, name="edge_mask", cochain=edge_mask)
viewer_original.add_k_cochain(
    k=1, name="init_eig", cochain=init_eig_vec, datatype="symmetric"
)

# %%
ps.show()

# %%
ellipsoid.requires_grad_()

# %%
n_steps = 200

optimizer = torch.optim.SGD([ellipsoid.vert_coords], lr=0.001)

reg_strength = 1e-3
willmore_strength = 1e-3

eps = torch.finfo(ellipsoid.vert_coords.dtype).eps

update_frequency = 20

nodal_loss_traj = []
dirichlet_loss_traj = []
willmore_loss_traj = []

bd_verts_mask = ellipsoid.bd_vert_mask
bd_tris_mask = ellipsoid.bd_tri_mask


for t in range(n_steps):
    optimizer.zero_grad()

    m1_ddt = tet_hodge_stars.star_1(ellipsoid)
    m1 = m1_ddt.to_sdt()
    # hybrid definition: S_1 = d_1^T M_2 d_1 + M_1 d_0 M_0^{-1} d_0^T M_1
    s1 = tet_laplacians.weak_laplacian_1(ellipsoid)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)

        eig_vals, eig_vecs = lobpcg(
            a=s1, m=m1, k=3, lobpcg_config=lobpcg_config, precond_config=precond_config
        )

    target_eig_val = eig_vals[0]
    target_eig_vec = eig_vecs[:, 0]

    # tet_centroids = ellipsoid.vert_coords[ellipsoid.tets].mean(dim=-2)
    # tet_mask = (torch.linalg.norm(tet_centroids, dim=-1) < 0.8).to(torch.float64)
    # edge_mask = ellipsoid.cbd[1].abs().T @ ellipsoid.cbd[2].abs().T @ tet_mask

    masked_eig_vecs = edge_mask * target_eig_vec
    dual_masked_vecs = m1_ddt @ masked_eig_vecs
    nodal_loss = torch.sum(masked_eig_vecs * dual_masked_vecs)

    nodal_loss_traj.append(nodal_loss.detach().cpu().numpy())

    deform_grad = compute_deformation_gradient(
        edge_mat_inv_ref, ellipsoid.vert_coords, ellipsoid.tets
    )
    deform_grad_sym = torch.transpose(deform_grad, -1, -2) @ deform_grad

    deform_grad_sym_inv = torch.linalg.inv(deform_grad_sym)

    dirichlet = torch.einsum("bii->b", deform_grad_sym)
    dirichlet_inv = torch.einsum("bii->b", deform_grad_sym_inv)
    dirichlet_sym = (
        reg_strength
        * 0.99 ** (t / 5.0)
        * torch.sum((dirichlet + dirichlet_inv) * torch.abs(ideal_tet_vol))
    )
    dirichlet_loss_traj.append(dirichlet_sym.detach().cpu().numpy())

    bd_mesh.vert_coords = ellipsoid.vert_coords[ellipsoid.bd_vert_mask]
    bd_m0_ddt = tri_hodge_stars.star_0(bd_mesh)
    bd_m0 = bd_m0_ddt.to_sdt()
    bd_s0 = consistent_tri_stiff_0(bd_mesh)

    willmore = (
        willmore_strength
        * 0.99 ** (t / 5.0)
        * torch.trace(
            bd_mesh.vert_coords.T @ bd_s0 @ bd_m0_ddt.inv @ bd_s0 @ bd_mesh.vert_coords
        )
    )
    willmore_loss_traj.append(willmore.detach().cpu().numpy())

    energy = nodal_loss + dirichlet_sym + willmore
    energy.backward()

    with torch.no_grad():
        grad = ellipsoid.vert_coords.grad.detach().clone()
        grad_smoothed = solver(grad, trans="N")
        ellipsoid.vert_coords.grad = grad_smoothed.contiguous()

    optimizer.step()

    with torch.no_grad():
        ellipsoid.vert_coords.sub_(ellipsoid.vert_coords.mean(dim=0, keepdim=True))

        updated_vol = tet_masses.compute_tet_signed_vols(
            ellipsoid.vert_coords, ellipsoid.tets
        ).sum()
        vol_ratio = (init_vol / updated_vol).pow(1.0 / 3.0)
        ellipsoid.vert_coords.mul_(vol_ratio)

        if t > 0 and t % update_frequency == 0:
            m0_ref = tet_hodge_stars.star_0(tet_mesh=ellipsoid).to_sdt()
            s0_ref = tet_laplacians.weak_laplacian_0(ellipsoid, method="consistent")

            op = SparseDecoupledTensor.assemble(tau * s0_ref, m0_ref)
            solver = NVMathDirectSolver(
                a=op,
                b=ellipsoid.vert_coords,
                sparse_system_type="spd",
                config=nvmath_config,
            )

    if t % update_frequency == 0:
        print(
            f"Timestep: {t:04d}; "
            f"nodal_loss: {nodal_loss:.4f}; "
            f"dirichlet: {dirichlet_sym:.4f}; "
            f"willmore: {willmore:.4f}"
        )

# %%
viewer_updated = PolyscopeViewer(name="optimized_mesh", mesh=ellipsoid)

# %%
viewer_updated.add_k_cochain(
    k=1, name="target_eig", cochain=target_eig_vec, datatype="symmetric"
)

# %%
ps.show()

# %%
target_eig_vecs_np = target_eig_vec.detach().cpu().numpy()
edge_mask_np = edge_mask.detach().cpu().numpy()

# %%
edge_mask_np.shape

# %%
edge_mask_np.sum()

# %%
np.abs(target_eig_vecs_np * edge_mask_np).mean()

# %%
np.abs(target_eig_vecs_np * (1 - edge_mask_np)).mean()

# %%
nodal_loss_traj = np.asarray(nodal_loss_traj)
dirichlet_loss_traj = np.asarray(dirichlet_loss_traj)
willmore_loss_traj = np.asarray(willmore_loss_traj)

t_traj = np.asarray(range(len(nodal_loss_traj))) + 1

# %%
fig, axes = plt.subplots(1, 3, figsize=(8, 4), layout="constrained")


axes[0].plot(t_traj, nodal_loss_traj, color="dodgerblue")
axes[0].set_xscale("log")
axes[0].set(xlabel="Time", ylabel="nodal_loss")


axes[1].set_xscale("log")
axes[1].plot(t_traj, willmore_loss_traj, color="dodgerblue")
axes[1].set(xlabel="Time", ylabel="willmore_loss")

axes[2].plot(t_traj, dirichlet_loss_traj, color="dodgerblue")
axes[2].set_xscale("log")
axes[2].set(xlabel="Time", ylabel="sym_dirichlet_loss")

plt.show()
plt.close()

# %%
eig_vals

import itertools
import math

import torch as t
from jaxtyping import Float

# TODO: this script is to be depreciated


def moments_2simp_0order(
    device: t.device, dtype: t.dtype = t.float
) -> Float[t.Tensor, ""]:
    return t.tensor(1.0, dtype=dtype, device=device)


def moments_2simp_1order(
    device: t.device, dtype: t.dtype = t.float
) -> Float[t.Tensor, "lambda=4"]:
    """
    For a tet of unit volume, let λ_x(p) be the barycentric coordinate function
    for p wrt a vertex x of the tet. This function computes all first moments of
    the form

    M_x = int[λ_x(p) dV]

    for all vertex x.
    """

    # There are only one unique using the magic formula: M_x = V/4

    moments = t.ones(4, dtype=dtype, device=device) / 4.0

    return moments


def moments_2simp_2order(
    device: t.device, dtype: t.dtype = t.float
) -> Float[t.Tensor, "lambda=4 lambda=4"]:
    """
    For a tet of unit volume, let λ_x(p) be the barycentric coordinate function
    for p wrt a vertex x of the tet. This function computes all second moments of
    the form

    M_xy = int[λ_x(p)*λ_y(p) dV]

    for all vertex pairs (x, y).
    """

    # There are only two unique values: xy, xx. Using the magic formula, these are:
    #
    # M_xy = V/20
    # M_xx = V/10

    # Let delta_xy be the Kronecker delta function; with this, we can condense the
    # results as:
    #
    # M_xy = (1 + delta_xy)/20

    delta_xy = t.eye(4, dtype=dtype, device=device)

    moments = (1.0 + delta_xy) / 20.0

    return moments


def moments_3simp_0order(
    device: t.device, dtype: t.dtype = t.float
) -> Float[t.Tensor, ""]:
    return t.tensor(1.0, dtype=dtype, device=device)


def moments_3simp_1order(
    device: t.device, dtype: t.dtype = t.float
) -> Float[t.Tensor, "lambda=4"]:
    """
    For a tet of unit volume, let λ_x(p) be the barycentric coordinate function
    for p wrt a vertex x of the tet. This function computes all first moments of
    the form

    M_x = int[λ_x(p) dV]

    for all vertex x.
    """

    # There are only one unique using the magic formula: M_x = V/4

    moments = t.ones(4, dtype=dtype, device=device) / 4.0

    return moments


def moments_3simp_2order(
    device: t.device, dtype: t.dtype = t.float
) -> Float[t.Tensor, "lambda=4 lambda=4"]:
    """
    For a tet of unit volume, let λ_x(p) be the barycentric coordinate function
    for p wrt a vertex x of the tet. This function computes all second moments of
    the form

    M_xy = int[λ_x(p)*λ_y(p) dV]

    for all vertex pairs (x, y).
    """

    # There are only two unique values: xy, xx. Using the magic formula, these are:
    #
    # M_xy = V/20
    # M_xx = V/10

    # Let delta_xy be the Kronecker delta function; with this, we can condense the
    # results as:
    #
    # M_xy = (1 + delta_xy)/20

    delta_xy = t.eye(4, dtype=dtype, device=device)

    moments = (1.0 + delta_xy) / 20.0

    return moments


def moments_3simp_3order(
    device: t.device, dtype: t.dtype = t.float
) -> Float[t.Tensor, "lambda=4 lambda=4 lambda=4"]:
    """
    For a tet of unit volume, let λ_x(p) be the barycentric coordinate function
    for p wrt a vertex x of the tet. This function computes all third moments of
    the form

    M_xyz = int[λ_x(p)*λ_y(p)*λ_z(p) dV]

    for all vertex triplet (x, y, z).
    """

    # There are only three unique values: xyz, xxz, xxx. Using the magic formula,
    # these are:
    #
    # M_xyz = V/120
    # M_xxz = V/60
    # M_xxx = V/20
    #
    # Let delta_xy be the Kronecker delta function; with this, we can condense the
    # results as:
    #
    # M_xyz = (1 + delta_xy + delta_xz + delta_yz + 2*delta_xyz)/120

    diag = t.eye(4, dtype=dtype, device=device)

    delta_xy = diag.view(4, 4, 1)
    delta_xz = diag.view(4, 1, 4)
    delta_yz = diag.view(1, 4, 4)
    delta_xyz = delta_xy * delta_yz

    moments = (1.0 + delta_xy + delta_xz + delta_yz + 2 * delta_xyz) / 120.0

    return moments


def moments_3simp_4order(
    device: t.device, dtype: t.dtype = t.float
) -> Float[t.Tensor, "lambda=4 lambda=4 lambda=4 lambda=4"]:
    """
    For a tet of unit volume, let λ_x(p) be the barycentric coordinate function
    for p wrt a vertex x of the tet. This function computes all fourth moments of
    the form

    M_wxyz = int[λ_w(p)*λ_x(p)*λ_y(p)*λ_z(p) dV]

    for all vertex tuples (w, x, y, z).
    """

    # There are only four unique values: wxyz, wwyz, wwwz, wwww. Using the magic
    # formula, these are:
    #
    # M_wxyz = V/120
    # M_wwyz = V/60
    # M_wwwz = V/20
    # M_wwww =
    #
    # Let delta_xy be the Kronecker delta function; with this, we can condense the
    # results as:
    #
    # M_xyz = (1 + delta_xy + delta_xz + delta_yz + 2*delta_xyz)/120

    diag = t.eye(4, dtype=dtype, device=device)

    delta_xy = diag.view(4, 4, 1)
    delta_xz = diag.view(4, 1, 4)
    delta_yz = diag.view(1, 4, 4)
    delta_xyz = delta_xy * delta_yz

    moments = (1.0 + delta_xy + delta_xz + delta_yz + 2 * delta_xyz) / 120.0

    return moments

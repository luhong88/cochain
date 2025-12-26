import torch as t
from jaxtyping import Integer


def whitney_router(
    simp_dim: int, form_deg: int, device: t.device, dtype: t.dtype = t.int64
):
    match simp_dim:
        case 2:
            match form_deg:
                case 0:
                    return router_2simp_0form(device, dtype)
                case 1:
                    return router_2simp_1form(device, dtype)
                case 2:
                    return router_2simp_2form(device, dtype)
                case _:
                    raise ValueError()
        case 3:
            match form_deg:
                case 0:
                    return router_3simp_0form(device, dtype)
                case 1:
                    return router_3simp_1form(device, dtype)
                case 2:
                    return router_3simp_2form(device, dtype)
                case 3:
                    return router_3simp_3form(device, dtype)
                case _:
                    raise ValueError()


def router_2simp_0form(
    device: t.device, dtype: t.dtype = t.int64
) -> Integer[t.Tensor, "vert=3 lambda=3"]:
    """
    The Whitney 0-form of a vert i is defined by w_i = λ_i
    """
    router = t.eye(3, dtype=dtype, device=device)

    return router


def router_2simp_1form(
    device: t.device, dtype: t.dtype = t.int64
) -> Integer[t.Tensor, "edge=3 lambda=3 d_lambda=3"]:
    """
    The Whitney 1-form of an edge ij is defined by

    w_ij = λ_i * dλ_j - λ_j * dλ_i

    This function contains the +1/-1 coefficients that specify how each 1-form
    is to be constructed from the λ's and the dλ's.
    """
    i, j, k = 0, 1, 2
    all_edges = t.tensor(
        [
            [0, i, j],
            [1, i, k],
            [2, j, k],
        ],
        dtype=dtype,
        device=device,
    )

    router = t.zeros(3, 3, 3, dtype=dtype, device=device)
    router[all_edges.T.unbind(0)] = 1.0
    router.sub_(router.transpose(-1, -2).clone())

    return router


def router_2simp_2form(
    device: t.device, dtype: t.dtype = t.int64
) -> Integer[t.Tensor, "tri=1 lambda=3 d_lambda=3 d_lambda=3"]:
    """
    The Whitney 2-form of an triangle ijk is defined by

    w_ijk = 2*(
        λ_i * dλ_j ∧ dλ_k +
        λ_j * dλ_k ∧ dλ_i +
        λ_k * dλ_i ∧ dλ_j +
    )

    This function contains the +1/-1 coefficients that specify how each 2-form
    is to be constructed from the λ's and the dλ's.
    """
    i, j, k = 0, 1, 2, 3
    all_tris_cyclic_perm = t.tensor(
        [
            # tri: ijk
            [3, i, j, k],
            [3, k, i, j],
            [3, j, k, i],
        ],
        dtype=dtype,
        device=device,
    )

    router = t.zeros(1, 3, 3, 3, dtype=dtype, device=device)
    router[all_tris_cyclic_perm.T.unbind(0)] = 2.0

    return router


def router_3simp_0form(
    device: t.device, dtype: t.dtype = t.int64
) -> Integer[t.Tensor, "vert=4 lambda=4"]:
    """
    The Whitney 0-form of a vert i is defined by w_i = λ_i
    """
    router = t.eye(4, dtype=dtype, device=device)

    return router


def router_3simp_1form(
    device: t.device, dtype: t.dtype = t.int64
) -> Integer[t.Tensor, "edge=6 lambda=4 d_lambda=4"]:
    """
    The Whitney 1-form of an edge ij is defined by

    w_ij = λ_i * dλ_j - λ_j * dλ_i

    This function contains the +1/-1 coefficients that specify how each 1-form
    is to be constructed from the λ's and the dλ's.
    """
    i, j, k, l = 0, 1, 2, 3
    all_edges = t.tensor(
        [
            [0, i, j],
            [1, i, k],
            [2, i, l],
            [3, j, k],
            [4, j, l],
            [5, k, l],
        ],
        dtype=dtype,
        device=device,
    )

    router = t.zeros(6, 4, 4, dtype=dtype, device=device)
    router[all_edges.T.unbind(0)] = 1.0
    router.sub_(router.transpose(-1, -2).clone())

    return router


def router_3simp_2form(
    device: t.device, dtype: t.dtype = t.int64
) -> Integer[t.Tensor, "tri=4 lambda=4 d_lambda=4 d_lambda=4"]:
    """
    The Whitney 2-form of an triangle ijk is defined by

    w_ijk = 2*(
        λ_i * dλ_j ∧ dλ_k +
        λ_j * dλ_k ∧ dλ_i +
        λ_k * dλ_i ∧ dλ_j +
    )

    This function contains the +1/-1 coefficients that specify how each 2-form
    is to be constructed from the λ's and the dλ's.
    """
    i, j, k, l = 0, 1, 2, 3
    all_tris_cyclic_perm = t.tensor(
        [
            # tri: jkl
            [0, j, k, l],
            [0, l, j, k],
            [0, k, l, j],
            # tri: ilk
            [1, i, l, k],
            [1, k, i, l],
            [1, l, k, i],
            # tri: ijl
            [2, i, j, l],
            [2, l, i, j],
            [2, j, l, i],
            # tri: ikj
            [3, i, k, j],
            [3, j, i, k],
            [3, k, j, i],
        ],
        dtype=dtype,
        device=device,
    )

    router = t.zeros(4, 4, 4, 4, dtype=dtype, device=device)
    router[all_tris_cyclic_perm.T.unbind(0)] = 2.0

    return router


def router_3simp_3form(
    device: t.device, dtype: t.dtype = t.int64
) -> Integer[t.Tensor, "tet=1 lambda=4 d_lambda=4 d_lambda=4 d_lambda=4"]:
    """
    The Whitney 3-form of a tet ijkl is defined by

    w_ijkl = 6*(
        λ_i * dλ_j ∧ dλ_k ∧ dλ_l -
        λ_j * dλ_i ∧ dλ_k ∧ dλ_l +
        λ_k * dλ_i ∧ dλ_j ∧ dλ_l -
        λ_l * dλ_i ∧ dλ_j ∧ dλ_k
    )

    This function contains the coefficients that specify how each 3-form
    is to be constructed from the λ's and the dλ's.
    """
    i, j, k, l = 0, 1, 2, 3

    pos_perm = t.tensor(
        [
            [0, i, j, k, l],
            [0, k, i, j, l],
        ],
        dtype=dtype,
        device=device,
    )

    neg_perm = t.tensor(
        [
            [0, j, i, k, l],
            [0, l, i, j, k],
        ],
        dtype=dtype,
        device=device,
    )

    router = t.zeros(1, 4, 4, 4, 4, dtype=dtype, device=device)
    router[pos_perm.T.unbind(0)] = 6.0
    router[neg_perm.T.unbind(0)] = -6.0

    return router

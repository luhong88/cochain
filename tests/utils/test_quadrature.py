import itertools
from math import factorial

import pytest
import torch as t

from cochain.utils import quadrature


@pytest.mark.parametrize(
    "dtype",
    [t.float32, t.float64],
)
def test_GaussLegendre_polynomial_basis(dtype, device):
    """
    A numerical quadrature on a unit edge of degree k should integrate a polynomial
    basis function λ_0^a * λ_1^b (a + b = k) exactly.
    """
    for k in [1, 2, 3, 4, 5]:
        quad = quadrature.GaussLegendre(dtype=dtype, device=device)
        bc, w = quad.get_rule(degree=k)

        for a, b in itertools.product(range(k + 1), repeat=2):
            if a + b == k:
                # Integrate the basis function over the ref edge using the
                # magic formula
                int_true = t.tensor(
                    (factorial(a) * factorial(b)) / factorial(a + b + 1),
                    dtype=dtype,
                    device=device,
                )

                int_test = t.sum(w * (bc[:, 0] ** a * bc[:, 1] ** b))

                t.testing.assert_close(int_test, int_true)


@pytest.mark.parametrize(
    "dtype",
    [t.float32, t.float64],
)
def test_Dunavant_polynomial_basis(dtype, device):
    """
    A numerical quadrature on a ref tri of degree k should integrate a polynomial
    basis function λ_0^a * λ_1^b * λ_2^c (a + b + c = k) exactly.
    """
    for k in [1, 2, 3, 4, 5]:
        quad = quadrature.Dunavant(dtype=dtype, device=device)
        bc, w = quad.get_rule(degree=k)

        for a, b, c in itertools.product(range(k + 1), repeat=3):
            if a + b + c == k:
                # Integrate the basis function over the ref triangle using the
                # magic formula
                int_true = t.tensor(
                    (factorial(a) * factorial(b) * factorial(c))
                    / factorial(a + b + c + 2),
                    dtype=dtype,
                    device=device,
                )

                # Multiply by 1/2 to account for ref tri area (recall that
                # the weights from the quadrature rules are barycentric).
                int_test = (
                    t.sum(w * (bc[:, 0] ** a * bc[:, 1] ** b * bc[:, 2] ** c)) / 2.0
                )

                t.testing.assert_close(int_test, int_true)


@pytest.mark.parametrize(
    "dtype",
    [t.float32, t.float64],
)
def test_Keast_polynomial_basis(dtype, device):
    """
    A numerical quadrature on a ref tet of degree k should integrate a polynomial
    basis function λ_0^a * λ_1^b * λ_2^c * λ_3^d (a + b + c + d = k) exactly.
    """
    for k, neg_weights in [
        (1, False),
        (2, False),
        (3, True),
        (3, False),
        (4, True),
        (4, False),
        (5, False),
    ]:
        quad = quadrature.Keast(dtype=dtype, device=device)
        bc, w = quad.get_rule(degree=k, allow_neg_weights=neg_weights)

        for a, b, c, d in itertools.product(range(k + 1), repeat=4):
            if a + b + c + d == k:
                # Integrate the basis function over the ref tet using the
                # magic formula
                int_true = t.tensor(
                    (factorial(a) * factorial(b) * factorial(c) * factorial(d))
                    / factorial(a + b + c + d + 3),
                    dtype=dtype,
                    device=device,
                )

                # Multiply by 1/6 to account for ref tet volume (recall that
                # the weights from the quadrature rules are barycentric).
                int_test = (
                    t.sum(
                        w
                        * (
                            bc[:, 0] ** a
                            * bc[:, 1] ** b
                            * bc[:, 2] ** c
                            * bc[:, 3] ** d
                        )
                    )
                    / 6.0
                )

                t.testing.assert_close(int_test, int_true)

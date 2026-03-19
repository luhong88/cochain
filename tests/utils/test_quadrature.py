import random

import pytest
import torch as t

from cochain.utils import quadrature


@pytest.mark.parametrize(
    "dtype",
    [t.float32, t.float64],
)
def test_gauss_legendre_polynomial_deg_1(dtype, device):
    """
    A numerical quadrature on [0, 1] of degree 1 should integrate a linear function
    f(x) = a*x + b exactly.
    """
    a = random.random()
    b = random.random()

    int_true = t.tensor(0.5 * a + b, dtype=dtype, device=device)

    quad = quadrature.GaussLegendre(dtype=dtype, device=device)
    bc, w = quad.get_rule(degree=1)

    int_test_1 = t.sum(w * (a * bc[:, 0] + b))
    int_test_2 = t.sum(w * (a * (1.0 - bc[:, 1]) + b))

    t.testing.assert_close(int_test_1, int_true)
    t.testing.assert_close(int_test_2, int_true)


@pytest.mark.parametrize(
    "dtype",
    [t.float32, t.float64],
)
def test_gauss_legendre_polynomial_deg_3(dtype, device):
    """
    A numerical quadrature on [0, 1] of degree 3 should integrate a cubic function
    f(x) = a*x^3 + b*x^2 + c*x + d exactly.
    """
    a = random.random()
    b = random.random()
    c = random.random()
    d = random.random()

    int_true = t.tensor(a / 4.0 + b / 3.0 + c / 2.0 + d, dtype=dtype, device=device)

    quad = quadrature.GaussLegendre(dtype=dtype, device=device)
    bc, w = quad.get_rule(degree=3)

    bc1 = bc[:, 0]
    int_test_1 = t.sum(w * (a * bc1**3 + b * bc1**2 + c * bc1 + d))

    bc2 = 1.0 - bc[:, 1]
    int_test_2 = t.sum(w * (a * bc2**3 + b * bc2**2 + c * bc2 + d))

    t.testing.assert_close(int_test_1, int_true)
    t.testing.assert_close(int_test_2, int_true)


@pytest.mark.parametrize(
    "dtype",
    [t.float32, t.float64],
)
def test_gauss_legendre_polynomial_deg_5(dtype, device):
    """
    A numerical quadrature on [0, 1] of degree 5 should integrate a cubic function
    f(x) = a*x^5 + b*x^4 + c*x^3 + d*x^2 + e*x + f exactly.
    """
    a = random.random()
    b = random.random()
    c = random.random()
    d = random.random()
    e = random.random()
    f = random.random()

    int_true = t.tensor(
        a / 6.0 + b / 5.0 + c / 4.0 + d / 3.0 + e / 2.0 + f, dtype=dtype, device=device
    )

    quad = quadrature.GaussLegendre(dtype=dtype, device=device)
    bc, w = quad.get_rule(degree=5)

    bc1 = bc[:, 0]
    int_test_1 = t.sum(
        w * (a * bc1**5 + b * bc1**4 + c * bc1**3 + d * bc1**2 + e * bc1 + f)
    )

    bc2 = 1.0 - bc[:, 1]
    int_test_2 = t.sum(
        w * (a * bc2**5 + b * bc2**4 + c * bc2**3 + d * bc2**2 + e * bc2 + f)
    )

    t.testing.assert_close(int_test_1, int_true)
    t.testing.assert_close(int_test_2, int_true)

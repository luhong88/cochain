from dataclasses import dataclass
from functools import cached_property
from math import sqrt

import torch as t
from jaxtyping import Float


@dataclass
class GaussLegendre:
    dtype: t.dtype = t.float64
    device: t.device = t.device("cpu")

    def get_rule(
        self, degree: int, *args, **kwargs
    ) -> tuple[Float[t.Tensor, "point 2"], Float[t.Tensor, " point"]]:
        # The n-point quadrature has a polynomial degree of exactness of 2n - 1.
        match degree:
            case 0:
                return self.one_point
            case 1:
                return self.one_point
            case 2:
                return self.two_points
            case 3:
                return self.two_points
            case 4:
                return self.three_points
            case 5:
                return self.three_points
            case _:
                raise ValueError()

    @cached_property
    def one_point(self) -> tuple[Float[t.Tensor, "1 2"], Float[t.Tensor, "1"]]:
        bary = t.tensor([[0.5, 0.5]], dtype=self.dtype, device=self.device)
        weight = t.tensor([1.0], dtype=self.dtype, device=self.device)
        return bary, weight

    @cached_property
    def two_points(self) -> tuple[Float[t.Tensor, "2 2"], Float[t.Tensor, "2"]]:
        bary = t.tensor(
            [
                [0.5 - sqrt(3.0) / 6.0, 0.5 + sqrt(3.0) / 6.0],
                [0.5 + sqrt(3.0) / 6.0, 0.5 - sqrt(3.0) / 6.0],
            ],
            dtype=self.dtype,
            device=self.device,
        )
        weight = t.tensor([0.5, 0.5], dtype=self.dtype, device=self.device)
        return bary, weight

    @cached_property
    def three_points(self) -> tuple[Float[t.Tensor, "3 2"], Float[t.Tensor, "3"]]:
        bary = t.tensor(
            [
                [0.5 - sqrt(15.0) / 10.0, 0.5 + sqrt(15.0) / 10.0],
                [0.5, 0.5],
                [0.5 + sqrt(15.0) / 10.0, 0.5 - sqrt(15.0) / 10.0],
            ],
            dtype=self.dtype,
            device=self.device,
        )
        weight = t.tensor(
            [5.0 / 18.0, 8.0 / 18.0, 5.0 / 18.0], dtype=self.dtype, device=self.device
        )
        return bary, weight


@dataclass
class Dunavant:
    """
    Look up table for the Dunavant quadrature rule on a reference triangle.

    In general, rule #k guarantees that the numerical integration of a polynomial
    of degree up to k is exact. Currently, a polynomial degree of exactness up to
    5 is supported. Note that rule #3 contains a negative weight.

    Note that the weights returned are the "barycentric"/normalized weights in that
    they sum up to 1, rather than the area of the reference triangle (1/2).

    Reference: David Dunavant, High degree efficient symmetrical Gaussian quadrature
    rules for the triangle, Int. J. Numer. Methods Eng., 1985.
    """

    dtype: t.dtype = t.float64
    device: t.device = t.device("cpu")

    def get_rule(
        self, degree: int, *args, **kwargs
    ) -> tuple[Float[t.Tensor, "point 3"], Float[t.Tensor, " point"]]:
        match degree:
            case 0:
                return self.rule_1
            case 1:
                return self.rule_1
            case 2:
                return self.rule_2
            case 3:
                return self.rule_3
            case 4:
                return self.rule_4
            case 5:
                return self.rule_5
            case _:
                raise ValueError()

    @cached_property
    def rule_1(self) -> tuple[Float[t.Tensor, "1 3"], Float[t.Tensor, "1"]]:
        bary = t.tensor([[1.0 / 3.0] * 3], dtype=self.dtype, device=self.device)
        weight = t.tensor([1.0], dtype=self.dtype, device=self.device)
        return bary, weight

    @cached_property
    def rule_2(self) -> tuple[Float[t.Tensor, "3 3"], Float[t.Tensor, "3"]]:
        a = 2.0 / 3.0
        b = 1.0 / 6.0
        bary = t.tensor(
            [
                [a, b, b],
                [b, a, b],
                [b, b, a],
            ],
            dtype=self.dtype,
            device=self.device,
        )

        weight = t.tensor([1.0 / 3.0] * 3, dtype=self.dtype, device=self.device)

        return bary, weight

    @cached_property
    def rule_3(self) -> tuple[Float[t.Tensor, "4 3"], Float[t.Tensor, "4"]]:
        a = 0.6
        b = 0.2
        bary = t.tensor(
            [
                [1.0 / 3.0] * 3,
                [a, b, b],
                [b, a, b],
                [b, b, a],
            ],
            dtype=self.dtype,
            device=self.device,
        )

        pt1_weight = -0.5625
        pt234_weight = (1.0 - pt1_weight) / 3.0
        weight = t.tensor(
            [pt1_weight, pt234_weight, pt234_weight, pt234_weight],
            dtype=self.dtype,
            device=self.device,
        )

        return bary, weight

    @cached_property
    def rule_4(self) -> tuple[Float[t.Tensor, "6 3"], Float[t.Tensor, "6"]]:
        a1 = 0.108103018168070
        b1 = 0.445948490915965
        a2 = 0.816847572980459
        b2 = 0.091576213509771
        bary = t.tensor(
            [
                [a1, b1, b1],
                [b1, a1, b1],
                [b1, b1, a1],
                [a2, b2, b2],
                [b2, a2, b2],
                [b2, b2, a2],
            ],
            dtype=self.dtype,
            device=self.device,
        )

        weight = t.tensor(
            [0.223381589678011] * 3 + [0.109951743655322] * 3,
            dtype=self.dtype,
            device=self.device,
        )

        return bary, weight

    @cached_property
    def rule_5(self) -> tuple[Float[t.Tensor, "7 3"], Float[t.Tensor, "7"]]:
        a1 = 0.059715871789770
        b1 = 0.470142064105115
        a2 = 0.797426985353087
        b2 = 0.101286507323456
        bary = t.tensor(
            [
                [1.0 / 3.0] * 3,
                [a1, b1, b1],
                [b1, a1, b1],
                [b1, b1, a1],
                [a2, b2, b2],
                [b2, a2, b2],
                [b2, b2, a2],
            ],
            dtype=self.dtype,
            device=self.device,
        )

        weight = t.tensor(
            [0.225] + [0.132394152788506] * 3 + [0.125939180544827] * 3,
            dtype=self.dtype,
            device=self.device,
        )

        return bary, weight


@dataclass
class Keast:
    """
    Look up table for the Keast quadrature rule on a reference tetrahedron.

    Currently, a polynomial degree of exactness up to 5 is supported; note that,
    for degree of exactness 3 and 4, two rules are provided that differ by whether
    negative weights are allowed.

    Note that the weights returned are the "barycentric"/normalized weights in that
    they sum up to 1, rather than the area of the reference tet (1/6).

    Reference: Patrick Keast, Moderate Degree Tetrahedral Quadrature Formulas,
    Comput. Methods Appl. Mech. Eng., 1986.
    """

    dtype: t.dtype = t.float64
    device: t.device = t.device("cpu")

    def get_rule(
        self, degree: int, allow_neg_weights: bool = True, *args, **kwargs
    ) -> tuple[Float[t.Tensor, "point 4"], Float[t.Tensor, " point"]]:
        match degree:
            case 0:
                return self.rule_1
            case 1:
                return self.rule_1
            case 2:
                return self.rule_2
            case 3:
                return self.rule_3 if allow_neg_weights else self.rule_4
            case 4:
                return self.rule_5 if allow_neg_weights else self.rule_6
            case 5:
                return self.rule_7
            case _:
                raise ValueError()

    def _suborder_1(self, a: float) -> Float[t.Tensor, "1 4"]:
        return t.tensor([[a, a, a, a]], dtype=self.dtype, device=self.device)

    def _suborder_4(self, a: float, b: float) -> Float[t.Tensor, "4 4"]:
        return t.tensor(
            [[a, b, b, b], [b, a, b, b], [b, b, a, b], [b, b, b, a]],
            dtype=self.dtype,
            device=self.device,
        )

    def _suborder_6(self, a: float, b: float) -> Float[t.Tensor, "6 4"]:
        return t.tensor(
            [
                [a, a, b, b],
                [a, b, a, b],
                [a, b, b, a],
                [b, a, b, a],
                [b, a, a, b],
                [b, b, a, a],
            ],
            dtype=self.dtype,
            device=self.device,
        )

    @cached_property
    def rule_1(self) -> tuple[Float[t.Tensor, "1 4"], Float[t.Tensor, "1"]]:
        bary = self._suborder_1(0.25)

        weight = t.tensor(
            [1.0],
            dtype=self.dtype,
            device=self.device,
        )

        return bary, weight

    @cached_property
    def rule_2(self) -> tuple[Float[t.Tensor, "4 4"], Float[t.Tensor, "4"]]:
        bary = self._suborder_4(0.585410196624968500, 0.138196601125010500)

        weight = 6.0 * t.tensor(
            [0.0416666666666666667] * 4,
            dtype=self.dtype,
            device=self.device,
        )

        return bary, weight

    @cached_property
    def rule_3(self) -> tuple[Float[t.Tensor, "5 4"], Float[t.Tensor, "5"]]:
        bary = t.vstack(
            (
                self._suborder_1(0.25),
                self._suborder_4(
                    0.5,
                    0.166666666666666667,
                ),
            )
        )

        weight = 6.0 * t.tensor(
            [-0.133333333333333333] + [0.075] * 4,
            dtype=self.dtype,
            device=self.device,
        )

        return bary, weight

    @cached_property
    def rule_4(self) -> tuple[Float[t.Tensor, "10 4"], Float[t.Tensor, "10"]]:
        bary = t.vstack(
            (
                self._suborder_4(0.568430584196844400, 0.143856471934385200),
                self._suborder_6(
                    0.5,
                    0.0,
                ),
            )
        )

        weight = 6.0 * t.tensor(
            [0.0362941783134009000] * 4 + [0.00358165890217718333] * 6,
            dtype=self.dtype,
            device=self.device,
        )

        return bary, weight

    @cached_property
    def rule_5(self) -> tuple[Float[t.Tensor, "11 4"], Float[t.Tensor, "11"]]:
        bary = t.vstack(
            (
                self._suborder_1(0.25),
                self._suborder_4(0.785714285714285714, 0.0714285714285714285),
                self._suborder_6(
                    0.399403576166799219,
                    0.100596423833200785,
                ),
            )
        )

        weight = 6.0 * t.tensor(
            [-0.0131555555555555556]
            + [0.00762222222222222222] * 4
            + [0.0248888888888888889] * 6,
            dtype=self.dtype,
            device=self.device,
        )

        return bary, weight

    @cached_property
    def rule_6(self) -> tuple[Float[t.Tensor, "14 4"], Float[t.Tensor, "14"]]:
        bary = t.vstack(
            (
                self._suborder_6(0.5, 0.0),
                self._suborder_4(0.698419704324386603, 0.100526765225204467),
                self._suborder_4(0.0568813795204234229, 0.314372873493192195),
            )
        )

        weight = 6.0 * t.tensor(
            [0.00317460317460317450] * 6
            + [0.0147649707904967828] * 4
            + [0.0221397911142651221] * 4,
            dtype=self.dtype,
            device=self.device,
        )

        return bary, weight

    @cached_property
    def rule_7(self) -> tuple[Float[t.Tensor, "15 4"], Float[t.Tensor, "15"]]:
        bary = t.vstack(
            (
                self._suborder_1(0.25),
                self._suborder_4(0.0, 1.0 / 3.0),
                self._suborder_4(0.727272727272727273, 0.0909090909090909091),
                self._suborder_6(0.0665501535736642813, 0.433449846426335728),
            )
        )

        weight = 6.0 * t.tensor(
            [0.0302836780970891856]
            + [0.00602678571428571597] * 4
            + [0.0116452490860289742] * 4
            + [0.0109491415613864534] * 6,
            dtype=self.dtype,
            device=self.device,
        )

        return bary, weight

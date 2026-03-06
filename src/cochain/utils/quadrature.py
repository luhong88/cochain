from dataclasses import dataclass

import torch as t
from jaxtyping import Float


@dataclass(frozen=True)
class Dunavant:
    """
    Look up table for the quadrature rule on a reference triangle.

    Reference: David Dunavant, High degree efficient symmetrical Gaussian quadrature
    rules for the triangle, Int. J. Numer. Methods Eng., (1985).
    """

    dtype: t.dtype
    device: t.device

    def get_rule(self, degree: int):
        match degree:
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

    @property
    def rule_1(self) -> tuple[Float[t.Tensor, "1 3"], Float[t.Tensor, "1"]]:
        bary = t.tensor([[1.0 / 3.0] * 3], dtype=self.dtype, device=self.device)
        weight = t.tensor([1.0], dtype=self.dtype, device=self.device)
        return bary, weight

    @property
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

    @property
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

    @property
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

    @property
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

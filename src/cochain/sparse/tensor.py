from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

import torch as t
from jaxtyping import Float, Integer


class SparseOperator:
    def __init__(
        self, values: Float[t.Tensor, " nnz *d"], sparse_topology: SparseTopology
    ):
        self.val = values
        self.sp_topo = sparse_topology

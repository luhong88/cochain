from __future__ import annotations

import contextlib
from typing import Iterator

import torch

try:
    import cupy as cp

    _HAS_CUPY = True

except ImportError:
    _HAS_CUPY = False


if _HAS_CUPY:

    class _TorchStreamAdapter:
        """Adapt older PyTorch streams to the modern CUDA Stream Protocol."""

        def __init__(self, stream: torch.cuda.Stream):
            self.stream = stream

        def __cuda_stream__(self) -> tuple[int, int]:
            # The protocol strictly expects a 2-tuple: (version_number, pointer).
            return (0, self.stream.cuda_stream)

    @contextlib.contextmanager
    def cupy_in_torch_stream() -> Iterator[None]:
        """Synchronize CuPy operations with the current PyTorch CUDA stream."""
        torch_stream = torch.cuda.current_stream()

        if hasattr(cp.cuda.Stream, "from_external"):
            # For CuPy >= 14.0.
            if hasattr(torch_stream, "__cuda_stream__"):
                ctx = cp.cuda.Stream.from_external(torch_stream)
            else:
                adapter = _TorchStreamAdapter(torch_stream)
                ctx = cp.cuda.Stream.from_external(adapter)
        else:
            # For CuPy < 14.0.
            ctx = cp.cuda.ExternalStream(
                torch_stream.cuda_stream, torch_stream.device_index
            )

        with ctx:
            yield

else:

    @contextlib.contextmanager
    def cupy_in_torch_stream() -> Iterator[None]:
        yield

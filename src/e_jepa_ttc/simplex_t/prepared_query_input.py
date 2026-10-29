"""One-entry CPU input reuse; never share producer outputs or observation IDs."""

from collections.abc import Callable

import torch
from torch import Tensor


class PreparedQueryInput:
    """Retain at most one FP32 query tensor under an explicit byte ceiling.

    Keys must describe every sensor/preprocessing dependency, not merely a query
    number. Callers must not mutate the returned input. This class alone neither
    changes queue ordering nor authorizes simultaneous producer residency.
    """

    def __init__(self, max_bytes: int = 64 * 1024**2) -> None:
        if type(max_bytes) is not int or not 0 < max_bytes <= 64 * 1024**2:
            raise ValueError("prepared input ceiling must be within 64 MiB")
        self.max_bytes = max_bytes
        self.key: tuple[bytes, ...] | None = None
        self.tensor: Tensor | None = None
        self.preparations = 0
        self.hits = 0

    def get(self, key: tuple[bytes, ...], prepare: Callable[[], Tensor]) -> Tensor:
        """Reuse exactly matching inputs; discard the old entry before preparing."""
        if not key or any(not isinstance(part, bytes) for part in key):
            raise ValueError("explicit byte-valued input dependency key required")
        if key == self.key and self.tensor is not None:
            self.hits += 1
            return self.tensor
        self.clear()
        tensor = prepare()
        if tensor.device.type != "cpu" or tensor.dtype != torch.float32:
            raise ValueError("prepared inputs must be CPU FP32")
        if tensor.numel() * tensor.element_size() > self.max_bytes:
            raise ValueError("prepared query exceeds byte ceiling")
        self.key, self.tensor = key, tensor
        self.preparations += 1
        return tensor

    def clear(self) -> None:
        """Release the retained reference; external consumers own their lifetimes."""
        self.key, self.tensor = None, None

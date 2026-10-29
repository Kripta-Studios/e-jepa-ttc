"""Exact upper-tail selection for the frozen event-voxel normalization.

The 95th percentile of occupied values only needs the largest five percent of
the complete row. A fixed-size descending top-k avoids sorting the remaining
95 percent and supports different occupancy counts without a device-to-host
transfer. This is an experimental execution change, not a different quantile.
"""

from __future__ import annotations

import torch
from torch import Tensor


def percentile95_tail_size(columns: int) -> int:
    """Return a conservative fixed top-k length containing both P95 statistics."""
    if isinstance(columns, bool) or not isinstance(columns, int) or columns < 1:
        raise ValueError("columns must be a positive integer")
    # For m occupied values, the lower ascending rank is floor(.95*(m-1)).
    # Its rank from the maximum is ceil(.05*(m-1)), at most ceil((N-1)/20).
    return min(columns, 1 + (columns - 1 + 19) // 20)


class ExactTopKNormalizer:
    """In-place float64 voxel normalization with bounded reusable top-k buffers.

    Input is a contiguous [windows, columns] matrix of finite voxel values.
    Occupancy remains ``value > 1e-6``. The rank expression, float64 interpolation,
    denominator clamp and zero masking follow IntegralEventHistory._normalize.
    Callers retain responsibility for output float32 conversion and the unchanged
    event-count scalar channels. No CPU reads, precision changes or CUDA graphs
    are introduced here. Instances and scratch buffers have single-stream owners.
    """

    def __init__(self, max_windows: int, columns: int, *, device: str | torch.device) -> None:
        if isinstance(max_windows, bool) or not isinstance(max_windows, int) or max_windows < 1:
            raise ValueError("max_windows must be a positive integer")
        self.max_windows = max_windows
        self.columns = columns
        self.tail_size = percentile95_tail_size(columns)
        self.device = torch.device(device)
        self.occupied = torch.empty((max_windows, columns), dtype=torch.bool, device=self.device)
        self.values = torch.empty(
            (max_windows, self.tail_size), dtype=torch.float64, device=self.device
        )
        self.indices = torch.empty(
            (max_windows, self.tail_size), dtype=torch.int64, device=self.device
        )
        # Resolve an unspecified CUDA index without an extra query during calls.
        self.device = self.values.device

    @property
    def allocated_bytes(self) -> int:
        """Retained scratch bytes; operation temporaries are accounted separately."""
        return sum(t.numel() * t.element_size() for t in (self.occupied, self.values, self.indices))

    def normalize_(self, voxel: Tensor) -> Tensor:
        """Normalize voxel in place and return its per-row unclamped P95 scale."""
        if (
            voxel.ndim != 2
            or voxel.shape[1] != self.columns
            or not 1 <= voxel.shape[0] <= self.max_windows
            or voxel.dtype != torch.float64
            or voxel.device != self.device
            or not voxel.is_contiguous()
        ):
            raise ValueError(
                "voxel must be a contiguous resident float64 [windows, columns] matrix"
            )
        rows = voxel.shape[0]
        occupied = self.occupied[:rows]
        values, indices = self.values[:rows], self.indices[:rows]
        torch.gt(voxel, 1e-6, out=occupied)
        nonzero = occupied.sum(1)
        torch.topk(voxel, self.tail_size, dim=1, largest=True, sorted=True, out=(values, indices))
        rank = (nonzero - 1).clamp_min(0).double() * 0.95
        lower, upper = rank.floor().long(), rank.ceil().long()
        offset = self.columns - nonzero
        lower_from_max = self.columns - 1 - (offset + lower).clamp_max(self.columns - 1)
        upper_from_max = self.columns - 1 - (offset + upper).clamp_max(self.columns - 1)
        a = values.gather(1, lower_from_max[:, None])[:, 0]
        b = values.gather(1, upper_from_max[:, None])[:, 0]
        fraction = rank - lower
        scale = torch.where(fraction >= 0.5, b - (b - a) * (1.0 - fraction), a + (b - a) * fraction)
        voxel.div_(scale.clamp_min(1e-6)[:, None])
        voxel.masked_fill_(~occupied, 0.0)
        return scale


__all__ = ["ExactTopKNormalizer", "percentile95_tail_size"]

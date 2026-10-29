"""Experimental compiled integer selector for bounded TRAIN40 temporal prefixes.

Only spatial rectangle and temporal-difference arithmetic is compiled. Voxel
interpolation, normalization, model weights and predictions keep their existing
implementations. This route remains opt-in until sensor and final TTC parity
have been demonstrated for its declared population.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import torch
from torch import Tensor

from e_jepa_ttc.data.eap import EAP_IMAGE_SIZE
from operational.train40_runtime.temporal_prefix import TemporalPrefixEventHistory


def prefix_quarter_moments(
    counts: Tensor,
    moments: Tensor,
    previous_count: Tensor,
    previous_moment: Tensor,
    xb: Tensor,
    yb: Tensor,
    slots: Tensor,
    left_relative: Tensor,
    crosses_epoch: Tensor,
) -> tuple[Tensor, Tensor]:
    """Return integer quarter counts/moments without explicit address matrices.

    ``slots`` contains the 25 current temporal boundaries, while ``left_relative``
    and ``crosses_epoch`` have shape [1,24,1,1,1]. ROI bounds exactly match the
    canonical float64 projection. Every loaded count is widened before arithmetic
    so compiler reassociation cannot introduce an int32 intermediate overflow.
    """
    polarity = torch.arange(2, device=counts.device, dtype=torch.int64)

    def rectangle(source: Tensor) -> Tensor:
        def corner(yi: int, xi: int) -> Tensor:
            return source[
                slots[None, :, None, None, None],
                polarity[None, None, :, None, None],
                yb[:, None, None, yi : yi + yb.shape[1] - 1, None],
                xb[:, None, None, None, xi : xi + xb.shape[1] - 1],
            ].to(torch.int64)

        return corner(1, 1) - corner(0, 1) - corner(1, 0) + corner(0, 0)

    def preceding_rectangle(source: Tensor) -> Tensor:
        def corner(yi: int, xi: int) -> Tensor:
            return source[
                polarity[None, :, None, None],
                yb[:, None, yi : yi + yb.shape[1] - 1, None],
                xb[:, None, None, xi : xi + xb.shape[1] - 1],
            ].to(torch.int64)

        return (corner(1, 1) - corner(0, 1) - corner(1, 0) + corner(0, 0))[:, None]

    count_boundaries = rectangle(counts)
    moment_boundaries = rectangle(moments)
    count_previous = preceding_rectangle(previous_count)
    moment_previous = preceding_rectangle(previous_moment)
    quarter_count = (
        count_boundaries[:, 1:] - count_boundaries[:, :-1] + crosses_epoch * count_previous
    )
    quarter_moment = (
        moment_boundaries[:, 1:]
        - moment_boundaries[:, :-1]
        + crosses_epoch * (moment_previous + 600_000 * count_boundaries[:, 1:])
    )
    return quarter_count, quarter_moment - left_relative * quarter_count


def _select_into(
    counts: Tensor,
    moments: Tensor,
    previous_count: Tensor,
    previous_moment: Tensor,
    xb: Tensor,
    yb: Tensor,
    slots: Tensor,
    left_relative: Tensor,
    crosses_epoch: Tensor,
    output_count: Tensor,
    output_moment: Tensor,
) -> tuple[Tensor, Tensor]:
    count, moment = prefix_quarter_moments(
        counts,
        moments,
        previous_count,
        previous_moment,
        xb,
        yb,
        slots,
        left_relative,
        crosses_epoch,
    )
    output_count.copy_(count)
    output_moment.copy_(moment)
    return output_count, output_moment


class CompiledTemporalPrefixEventHistory(TemporalPrefixEventHistory):
    """Opt-in selector fusion with unchanged event-history and query contracts.

    ``compile_backend=None`` executes the pure tensor selector for diagnostics.
    A compiled backend requires explicit synthetic ``prime_selectors(counts)``
    before history installation. ``prime_query_graphs`` performs that priming
    first when query graphs are enabled. Compilation never consumes real ROI or
    sensor values outside the measured query clock.

    Unused legacy address/gather buffers are released after base construction;
    both the initial buffer peak and retained bytes are disclosed. The memory
    budget still covers that initial allocation, and allocator-reserved memory
    is distinct from live retained tensors.
    """

    def __init__(
        self,
        *,
        device: str | torch.device = "cuda",
        width: int = EAP_IMAGE_SIZE[0],
        height: int = EAP_IMAGE_SIZE[1],
        roi_size: int = 128,
        max_packet_events: int = 2_000_000,
        max_memory_bytes: int = 4 * 1024**3,
        async_transfers: bool = False,
        max_objects: int = 20,
        roi_batch_size: int = 4,
        normalization: str = "sort",
        graphs: bool = False,
        compile_backend: str | None = "inductor",
    ) -> None:
        super().__init__(
            device=device,
            width=width,
            height=height,
            roi_size=roi_size,
            max_packet_events=max_packet_events,
            max_memory_bytes=max_memory_bytes,
            async_transfers=async_transfers,
            max_objects=max_objects,
            roi_batch_size=roi_batch_size,
            normalization=normalization,
            graphs=graphs,
        )
        self.compile_backend = compile_backend
        self._selector_shapes: set[int] = set()
        self._selector_priming = False
        self._selector_synthetic_calls = 0
        self.initial_buffer_bytes_before_release = self.allocated_bytes
        unused_names = (
            "_address",
            "_quarter_work",
            "_count_gather_0",
            "_count_gather_1",
            "_moment_gather_0",
            "_moment_gather_1",
        )
        unused_ids = {id(getattr(self, name)) for name in unused_names}
        self._buffers = [value for value in self._buffers if id(value) not in unused_ids]
        self._count_gather = []
        self._moment_gather = []
        for name in unused_names:
            delattr(self, name)
        self.allocated_bytes = sum(value.numel() * value.element_size() for value in self._buffers)
        self.released_gather_bytes = self.initial_buffer_bytes_before_release - self.allocated_bytes
        self._selector: Callable[..., tuple[Tensor, Tensor]] = _select_into
        if compile_backend is not None:
            options: dict[str, Any] | None = None
            if compile_backend == "inductor":
                options = {
                    "use_fast_math": False,
                    "force_same_precision": True,
                    "triton.cudagraphs": False,
                }
            self._selector = torch.compile(
                _select_into,
                backend=compile_backend,
                fullgraph=True,
                dynamic=False,
                options=options,
            )

    @property
    def stats(self) -> dict[str, int | bool | str | None]:
        """Report selector identity and actual retained versus startup storage."""
        result = super().stats
        result.update(
            backend="COMPILED_INTEGER_TEMPORAL_PREFIX",
            selector_backend=self.compile_backend or "eager_tensor_diagnostic",
            selector_compiled_shapes=",".join(map(str, sorted(self._selector_shapes))),
            selector_synthetic_priming_calls=self._selector_synthetic_calls,
            initial_buffer_bytes_before_release=self.initial_buffer_bytes_before_release,
            released_gather_bytes=self.released_gather_bytes,
            legacy_gather_storage_released=True,
            max_take_temporary_bytes=0,
            selector_scope="INTEGER_RECTANGLE_AND_TEMPORAL_DIFFERENCES_ONLY",
            selector_eager_materializes_temporaries=self.compile_backend is None,
        )
        return result

    @torch.inference_mode()
    def prime_selectors(self, counts: Sequence[int]) -> None:
        """Prime required workspace shapes on synthetic zero startup buffers."""
        if self.end_us is not None or self._ingested_packets or self._total_buckets:
            raise ValueError("selectors must be primed before any sensor history or reset")
        if (
            not counts
            or any(
                isinstance(count, bool)
                or not isinstance(count, int)
                or not 1 <= count <= self.max_objects
                for count in counts
            )
            or len(set(counts)) != len(counts)
        ):
            raise ValueError(f"startup counts must be unique integers in 1..{self.max_objects}")
        self._query_state.zero_()
        self._selector_priming = True
        try:
            for count in counts:
                self._execute_query_many_device(count)
        finally:
            self._selector_priming = False
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def prime_query_graphs(self, counts: Sequence[int]) -> None:
        """Compile selector shapes before capturing query replay graphs."""
        if not self.graphs or self.device.type != "cuda":
            raise ValueError("enable graphs on CUDA before explicit startup priming")
        self.prime_selectors(counts)
        super().prime_query_graphs(counts)

    @torch.inference_mode()
    def _query_chunk_device(self, xb: Tensor, yb: Tensor, count: int) -> Tensor:
        if (
            self.compile_backend is not None
            and count not in self._selector_shapes
            and not self._selector_priming
        ):
            raise ValueError("selector shape was not primed before sensor observations")
        self._selector(
            self._counts,
            self._moments,
            self._previous_count,
            self._previous_moment,
            xb,
            yb,
            self._slot_indices,
            self._left_relative,
            self._cross,
            self._quarter_count[:count],
            self._quarter_moment[:count],
        )
        if self._selector_priming:
            self._selector_shapes.add(count)
            self._selector_synthetic_calls += 1
        # Keep the original float64 expressions and normalization as independent
        # operators. Only the preceding integer selector is compiled.
        size = self.roi_size
        self._upper[:count].copy_(self._quarter_moment[:count]).div_(25_000.0)
        self._lower[:count].copy_(self._quarter_count[:count]).sub_(self._upper[:count])
        upper = self._upper[:count].reshape(count, 6, 4, 2, size, size).transpose(2, 3)
        lower = self._lower[:count].reshape(count, 6, 4, 2, size, size).transpose(2, 3)
        self._voxel[:count].zero_()
        self._voxel[:count, :, :, :4].copy_(lower)
        self._voxel[:count, :, :, 1:].add_(upper)
        self._normalize(count)
        torch.index_select(
            self._six[: count * 6], 0, self._triplets[: count * 12], out=self._output[: count * 12]
        )
        return self._output[: count * 12].view(count, 4, 3, 12, size, size)


__all__ = ["CompiledTemporalPrefixEventHistory", "prefix_quarter_moments"]

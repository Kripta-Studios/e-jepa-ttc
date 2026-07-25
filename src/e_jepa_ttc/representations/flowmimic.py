"""Physics-constrained FlowMimic-style event generation.

The generator renders log-intensity frames first and applies a contrast-threshold
event model afterwards. It deliberately never warps an already accumulated
event voxel grid.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class FlowMimicEventBatch:
    """Synthetic causal context/future event windows and their known dynamics."""

    context: torch.Tensor
    future: torch.Tensor
    inverse_ttc_at_context_end: torch.Tensor


def physical_approach_scale(
    ttc_at_reference_s: torch.Tensor,
    delta_from_reference_s: torch.Tensor,
) -> torch.Tensor:
    """Return pinhole apparent scale for constant-speed frontal approach.

    ``delta_from_reference_s=0`` is the context reference time. Positive deltas
    move towards collision and negative deltas describe earlier observations.
    """

    denominator = ttc_at_reference_s - delta_from_reference_s
    if bool(torch.any(ttc_at_reference_s <= 0.0)):
        raise ValueError("TTC must be positive.")
    if bool(torch.any(denominator <= 0.0)):
        raise ValueError("Requested render time reaches or crosses collision.")
    return ttc_at_reference_s / denominator


def _render_log_intensity(
    *,
    grid_x: torch.Tensor,
    grid_y: torch.Tensor,
    times_s: torch.Tensor,
    ttc_s: torch.Tensor,
    radius_x: torch.Tensor,
    radius_y: torch.Tensor,
    center_x: torch.Tensor,
    center_y: torch.Tensor,
    velocity_x: torch.Tensor,
    velocity_y: torch.Tensor,
    contrast: torch.Tensor,
    phase: torch.Tensor,
) -> torch.Tensor:
    scale = physical_approach_scale(ttc_s[:, None], times_s)
    cx = center_x[:, None] + velocity_x[:, None] * times_s
    cy = center_y[:, None] + velocity_y[:, None] * times_s
    rx = radius_x[:, None] * scale
    ry = radius_y[:, None] * scale

    normalized_radius = torch.sqrt(
        torch.square((grid_x - cx[:, :, None, None]) / rx[:, :, None, None])
        + torch.square((grid_y - cy[:, :, None, None]) / ry[:, :, None, None])
        + 1e-8
    )
    soft_object = torch.sigmoid((1.0 - normalized_radius) * 48.0)
    background = 0.08 * (
        torch.sin(5.0 * grid_x + phase[:, None, None, None])
        + torch.cos(4.0 * grid_y - phase[:, None, None, None])
    )
    return background + contrast[:, None, None, None] * soft_object


def _events_from_window(
    *,
    window_start_s: torch.Tensor,
    window_duration_s: float,
    bins: int,
    contrast_threshold: float,
    render_parameters: dict[str, torch.Tensor],
    grid_x: torch.Tensor,
    grid_y: torch.Tensor,
    render_substeps_per_bin: int = 4,
) -> torch.Tensor:
    if render_substeps_per_bin <= 0:
        raise ValueError("render_substeps_per_bin must be positive.")
    render_steps = bins * render_substeps_per_bin
    offsets = torch.linspace(
        0.0,
        window_duration_s,
        render_steps + 1,
        device=window_start_s.device,
        dtype=torch.float32,
    )
    times = window_start_s[:, None] + offsets[None, :]
    frames = _render_log_intensity(
        grid_x=grid_x,
        grid_y=grid_y,
        times_s=times,
        **render_parameters,
    )
    # Contrast is accumulated relative to the last emitted-event reference.
    # Resetting that reference at every render frame would silently discard
    # sub-threshold motion and undercount events for distant approaches.
    reference = frames[:, 0].clone()
    positive = torch.zeros(frames.shape[0], bins, *frames.shape[-2:], device=frames.device)
    negative = torch.zeros_like(positive)
    for step in range(1, render_steps + 1):
        log_delta = frames[:, step] - reference
        positive_step = torch.floor(torch.relu(log_delta) / contrast_threshold)
        negative_step = torch.floor(torch.relu(-log_delta) / contrast_threshold)
        reference.add_((positive_step - negative_step) * contrast_threshold)
        bin_index = (step - 1) // render_substeps_per_bin
        positive[:, bin_index].add_(positive_step)
        negative[:, bin_index].add_(negative_step)
    return torch.cat([positive, negative], dim=1)


def _normalize_occupied_per_window(events: torch.Tensor) -> torch.Tensor:
    normalized = torch.zeros_like(events)
    for batch_idx in range(events.shape[0]):
        occupied = events[batch_idx][events[batch_idx] > 0.0]
        if occupied.numel() == 0:
            continue
        scale = torch.quantile(occupied.float(), 0.95).clamp_min(1e-6)
        normalized[batch_idx] = events[batch_idx] / scale
    return normalized


def _append_auxiliary_channels(
    events: torch.Tensor,
    *,
    output_channels: int,
    metadata_channels: bool,
    window_duration_s: float,
) -> torch.Tensor:
    pieces = [events]
    if metadata_channels:
        event_count = events.sum(dim=(1, 2, 3))
        log_count = torch.log1p(event_count)
        log_rate = torch.log1p(event_count / max(window_duration_s, 1e-6))
        height, width = events.shape[-2:]
        pieces.append(
            torch.stack([log_count, log_rate], dim=1)[:, :, None, None].expand(
                -1, -1, height, width
            )
        )
    combined = torch.cat(pieces, dim=1)
    if combined.shape[1] > output_channels:
        raise ValueError(
            f"FlowMimic output needs {combined.shape[1]} channels, got {output_channels}."
        )
    if combined.shape[1] < output_channels:
        padding = torch.zeros(
            combined.shape[0],
            output_channels - combined.shape[1],
            combined.shape[2],
            combined.shape[3],
            device=combined.device,
            dtype=combined.dtype,
        )
        combined = torch.cat([combined, padding], dim=1)
    return combined


@torch.no_grad()
def generate_physical_event_approach_batch(
    *,
    batch_size: int,
    output_channels: int,
    height: int,
    width: int,
    bins: int,
    horizons_ms: tuple[int, ...],
    context_ms: float,
    device: torch.device,
    metadata_channels: bool = False,
    normalize_events: bool = False,
    minimum_ttc_s: float = 0.8,
    maximum_ttc_s: float = 6.0,
    contrast_threshold: float = 0.08,
) -> FlowMimicEventBatch:
    """Generate physically scaled event windows for FlowMimic regularization."""

    if batch_size <= 0 or height <= 8 or width <= 8 or bins <= 0:
        raise ValueError("Batch, resolution and bin counts must be positive and non-trivial.")
    if not horizons_ms or any(horizon <= 0 for horizon in horizons_ms):
        raise ValueError("FlowMimic requires positive future horizons.")
    if context_ms <= 0.0 or contrast_threshold <= 0.0:
        raise ValueError("Context duration and contrast threshold must be positive.")

    context_s = context_ms / 1000.0
    latest_future_end_s = max(horizons_ms) / 1000.0 + context_s
    safe_minimum = max(float(minimum_ttc_s), latest_future_end_s + 0.2)
    if maximum_ttc_s <= safe_minimum:
        raise ValueError("maximum_ttc_s must leave margin after the latest future window.")

    ttc_s = safe_minimum + (maximum_ttc_s - safe_minimum) * torch.rand(batch_size, device=device)
    radius_x = 0.10 + 0.10 * torch.rand(batch_size, device=device)
    radius_y = radius_x * (0.70 + 0.60 * torch.rand(batch_size, device=device))
    center_x = -0.15 + 0.30 * torch.rand(batch_size, device=device)
    center_y = -0.12 + 0.24 * torch.rand(batch_size, device=device)
    velocity_x = -0.12 + 0.24 * torch.rand(batch_size, device=device)
    velocity_y = -0.08 + 0.16 * torch.rand(batch_size, device=device)
    contrast_magnitude = 0.9 + 1.2 * torch.rand(batch_size, device=device)
    polarity_sign = torch.where(
        torch.rand(batch_size, device=device) < 0.5,
        -torch.ones(batch_size, device=device),
        torch.ones(batch_size, device=device),
    )
    contrast = contrast_magnitude * polarity_sign
    phase = torch.rand(batch_size, device=device) * (2.0 * torch.pi)

    x_axis = torch.linspace(-1.0, 1.0, width, device=device)
    y_axis = torch.linspace(-1.0, 1.0, height, device=device)
    grid_y, grid_x = torch.meshgrid(y_axis, x_axis, indexing="ij")
    grid_x = grid_x[None, None]
    grid_y = grid_y[None, None]
    render_parameters = {
        "ttc_s": ttc_s,
        "radius_x": radius_x,
        "radius_y": radius_y,
        "center_x": center_x,
        "center_y": center_y,
        "velocity_x": velocity_x,
        "velocity_y": velocity_y,
        "contrast": contrast,
        "phase": phase,
    }

    context_events = _events_from_window(
        window_start_s=torch.full((batch_size,), -context_s, device=device),
        window_duration_s=context_s,
        bins=bins,
        contrast_threshold=contrast_threshold,
        render_parameters=render_parameters,
        grid_x=grid_x,
        grid_y=grid_y,
    )
    future_events = torch.stack(
        [
            _events_from_window(
                window_start_s=torch.full((batch_size,), horizon_ms / 1000.0, device=device),
                window_duration_s=context_s,
                bins=bins,
                contrast_threshold=contrast_threshold,
                render_parameters=render_parameters,
                grid_x=grid_x,
                grid_y=grid_y,
            )
            for horizon_ms in horizons_ms
        ],
        dim=1,
    )
    if normalize_events:
        context_events = _normalize_occupied_per_window(context_events)
        future_shape = future_events.shape
        future_events = _normalize_occupied_per_window(future_events.flatten(0, 1)).view(
            future_shape
        )

    context = _append_auxiliary_channels(
        context_events,
        output_channels=output_channels,
        metadata_channels=metadata_channels,
        window_duration_s=context_s,
    )
    future_shape = future_events.shape
    future = _append_auxiliary_channels(
        future_events.flatten(0, 1),
        output_channels=output_channels,
        metadata_channels=metadata_channels,
        window_duration_s=context_s,
    ).view(batch_size, len(horizons_ms), output_channels, height, width)

    if not bool(torch.all(torch.isfinite(context))) or not bool(torch.all(torch.isfinite(future))):
        raise RuntimeError("FlowMimic event generation produced non-finite values.")
    # Individual empty windows are physically possible for distant, low-motion
    # objects. Preserve them instead of injecting non-physical background events;
    # only a wholly empty batch indicates unusable simulator parameters.
    if float(context[:, : bins * 2].sum().cpu()) <= 0.0:
        raise RuntimeError("FlowMimic generated a wholly empty context event batch.")
    return FlowMimicEventBatch(
        context=context,
        future=future,
        inverse_ttc_at_context_end=ttc_s.reciprocal(),
    )


__all__ = [
    "FlowMimicEventBatch",
    "generate_physical_event_approach_batch",
    "physical_approach_scale",
]

"""Temporal window indexing for TTC experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from e_jepa_ttc.data.evttc import read_manifest
from e_jepa_ttc.data.targets import interpolate_ttc_seconds, load_ttc_csv
from e_jepa_ttc.data.types import TemporalIndexEntry
from e_jepa_ttc.utils.io import write_structured


def build_temporal_index(
    *,
    manifest_path: str | Path,
    context_ms: int = 100,
    stride_ms: int = 20,
    horizons_ms: tuple[int, ...] = (25, 50, 100, 250, 500),
    clip_ttc_seconds: tuple[float, float] | None = (0.1, 12.0),
) -> list[TemporalIndexEntry]:
    """Build dense temporal windows using TTC timestamps as anchors."""

    sequences = read_manifest(manifest_path)
    entries: list[TemporalIndexEntry] = []
    last_anchor_by_sequence: dict[str, int] = {}
    stride_us = int(stride_ms * 1000)
    context_us = int(context_ms * 1000)
    horizons_us = tuple(int(horizon * 1000) for horizon in horizons_ms)

    for sequence in sequences:
        ttc_csv = sequence.resolve("ttc_csv")
        if ttc_csv is None:
            continue
        table = load_ttc_csv(ttc_csv)
        min_us = int(float(table["timestamp_s"][0]) * 1_000_000)
        max_us = int(float(table["timestamp_s"][-1]) * 1_000_000)

        for timestamp_s, ttc_s in zip(table["timestamp_s"], table["ttc_s"], strict=True):
            timestamp_us = int(float(timestamp_s) * 1_000_000)
            previous = last_anchor_by_sequence.get(sequence.sequence_id)
            if previous is not None and timestamp_us - previous < stride_us:
                continue
            context_start_us = timestamp_us - context_us
            if context_start_us < min_us:
                continue
            # A horizon is the gap between the end of the causal context and
            # the start of a disjoint future window of the same duration.
            # Previous versions paired against a cache entry ending at t+h,
            # which overlapped the context for h < context_ms.
            max_horizon_end = timestamp_us + max(horizons_us, default=0) + context_us
            if max_horizon_end > max_us:
                continue
            interpolated = interpolate_ttc_seconds(table, timestamp_us)
            if interpolated is None:
                continue
            if clip_ttc_seconds is not None:
                low, high = clip_ttc_seconds
                if not (low <= interpolated <= high):
                    continue

            horizons = {
                int(horizon_ms): (
                    timestamp_us + horizon_us,
                    timestamp_us + horizon_us + context_us,
                )
                for horizon_ms, horizon_us in zip(horizons_ms, horizons_us, strict=True)
            }
            entries.append(
                TemporalIndexEntry(
                    sequence_id=sequence.sequence_id,
                    timestamp_us=timestamp_us,
                    context_start_us=context_start_us,
                    context_end_us=timestamp_us,
                    horizons_us=horizons,
                    ttc_seconds=float(ttc_s),
                    metadata={
                        "scenario_family": sequence.scenario_family,
                        "speed_bucket": sequence.speed_bucket,
                        "target_type": sequence.target_type,
                    },
                )
            )
            last_anchor_by_sequence[sequence.sequence_id] = timestamp_us

    return entries


def write_index(path: str | Path, entries: list[TemporalIndexEntry]) -> None:
    """Write a temporal index to JSON/YAML."""

    payload: dict[str, Any] = {
        "version": 2,
        "future_window_semantics": "disjoint_window_start_after_context_plus_horizon",
        "window_count": len(entries),
        "windows": [entry.to_dict() for entry in entries],
    }
    write_structured(path, payload)

"""Validate and bind native observation bytes to one physical query."""

from pathlib import Path

import numpy as np

from operational.evttc_transfer_audit.common import file_sha
from operational.phase_adapt_cache import atomic_receipt
from operational.phase_campaign.data import read


def _clock(value: object) -> int:
    if type(value) is not int:
        raise ValueError("Clock must be an integer, excluding booleans")
    return value


def validated_observations(destination: Path, query_us: int) -> dict:
    """Reject malformed clocks, unbounded sensory spans and coerced feature payloads."""
    query_us = _clock(query_us)
    metadata = read(destination / "OBSERVATIONS.json")
    if set(metadata) != {"event", "rgb"}:
        raise ValueError("Unknown observation metadata keys")
    allowed = {
        "anchor_us",
        "available_us",
        "first_sensor_ts_us",
        "timestamps_us",
        "windows_us",
        "delta_t_s",
        "square_xyxy",
    }
    result = {}
    with np.load(destination / "OBSERVATIONS.npz", allow_pickle=False) as stored:
        expected = {f"{m}.{k}" for m in ("event", "rgb") for k in ("raw17", "expert3")}
        if set(stored.files) != expected:
            raise ValueError("Unknown observation array keys")
        for modality in ("event", "rgb"):
            items = metadata[modality]
            raw, experts = stored[f"{modality}.raw17"], stored[f"{modality}.expert3"]
            if len(items) > 8 or raw.shape != (len(items), 17) or experts.shape != (len(items), 3):
                raise ValueError("Observation array/metadata dimensions differ")
            if any(a.dtype != np.float32 or not np.isfinite(a).all() for a in (raw, experts)):
                raise ValueError("Observation arrays must be finite FP32 without coercion")
            previous = None
            result[modality] = []
            for i, item in enumerate(items):
                if not set(item) <= allowed:
                    raise ValueError("Unknown per-observation metadata")
                anchor, available, first = (
                    _clock(item[k]) for k in ("anchor_us", "available_us", "first_sensor_ts_us")
                )
                timestamps = [_clock(t) for t in item["timestamps_us"]]
                if (
                    len(timestamps) not in (2, 3)
                    or timestamps[-1] != anchor
                    or any(b <= a for a, b in zip(timestamps, timestamps[1:], strict=False))
                    or first > timestamps[0]
                    or not anchor <= available <= query_us
                    or not 0 <= query_us - first <= 650_000
                    or (previous is not None and anchor <= previous)
                    or (modality == "rgb" and first != timestamps[0])
                ):
                    raise ValueError("Invalid causal observation clocks or full information span")
                windows = item.get("windows_us")
                if modality == "event" and windows is None:
                    raise ValueError("Accepted event observations require recorded windows")
                delta = item.get("delta_t_s")
                if delta is not None:
                    value = np.asarray(delta)
                    expected_delta = np.diff(timestamps) / 1_000_000
                    if (
                        value.shape != expected_delta.shape
                        or not np.isfinite(value).all()
                        or not (value > 0).all()
                        or not np.allclose(value, expected_delta, rtol=1e-6, atol=1e-8)
                    ):
                        raise ValueError("Producer deltas differ from physical timestamps")
                roi = item.get("square_xyxy")
                if roi is not None:
                    value = np.asarray(roi)
                    if (
                        value.shape != (4,)
                        or not np.isfinite(value).all()
                        or not (value[2:] > value[:2]).all()
                    ):
                        raise ValueError("ROI must contain finite coordinates and positive extents")
                if windows is not None:
                    if len(windows) != len(timestamps):
                        raise ValueError("Window/timestamp counts differ")
                    end_previous = first
                    for window, timestamp in zip(windows, timestamps, strict=True):
                        if len(window) != 2:
                            raise ValueError("Window needs two integer endpoints")
                        start, end = [_clock(t) for t in window]
                        if not end_previous <= start < end or end != timestamp:
                            raise ValueError(
                                "Windows are unordered or differ from observation clocks"
                            )
                        end_previous = end
                    if windows[0][0] != first:
                        raise ValueError("Window starts differ from first sensor clock")
                previous = anchor
                result[modality].append(
                    {**item, "raw17": raw[i].copy(), "expert3": experts[i].copy()}
                )
    return result


def seal_observations(destination: Path, binding: dict) -> dict:
    """Create an immutable identity and byte-digest receipt after validation."""
    validated_observations(destination, binding["query_us"])
    required = {
        "token",
        "group",
        "query_us",
        "query_sha256",
        "calibration_sha256",
        "model_sha256",
        "source_sha256",
    }
    if set(binding) != required or any(
        not isinstance(binding[k], str) or not binding[k] for k in required - {"query_us"}
    ):
        raise ValueError("Complete physical query identity and dependency binding required")
    receipt = {
        "binding": binding,
        "artifacts": {
            name: file_sha(destination / name) for name in ("OBSERVATIONS.json", "OBSERVATIONS.npz")
        },
    }
    atomic_receipt(destination / "OBSERVATION_SEAL.json", receipt)
    return receipt


def load_bound_observations(destination: Path, binding: dict) -> dict:
    """Reject exchanged query identities or modified NPZ/JSON before replay."""
    receipt = read(destination / "OBSERVATION_SEAL.json")
    if set(receipt) != {"binding", "artifacts"} or receipt["binding"] != binding:
        raise ValueError("Observation query binding differs")
    expected = {"OBSERVATIONS.json", "OBSERVATIONS.npz"}
    if set(receipt["artifacts"]) != expected or any(
        file_sha(destination / name) != digest for name, digest in receipt["artifacts"].items()
    ):
        raise ValueError("Observation bytes differ from sealed identity")
    return validated_observations(destination, binding["query_us"])

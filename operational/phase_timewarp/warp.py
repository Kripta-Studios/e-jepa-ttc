"""Pure numeric cadence transforms of cached producer representations, not raw re-encoding."""

from copy import deepcopy
from numbers import Integral, Real

import numpy as np

SPEEDS = (0.5, 1.0, 2.0)


def _integer(value: object) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError("Clock requires an integer, excluding booleans")
    value = int(value)
    if not -(2**63) <= value < 2**63:
        raise ValueError("Clock exceeds signed int64")
    return value


def warp_clock(t: int, query_us: int, speed: float) -> int:
    """Fix q and use integer-exact nearest-even rounding of (t-q)/a, without float clocks."""
    t, query_us = _integer(t), _integer(query_us)
    if isinstance(speed, (bool, np.bool_)) or not isinstance(speed, Real) or speed not in SPEEDS:
        raise ValueError("Speed must be one of the preregistered0.5,1,2")
    if t > query_us:
        raise ValueError("Noncausal clock exceeds query")
    offset = t - query_us
    if speed == 0.5:
        offset *= 2
    elif speed == 2.0:
        floor, remainder = divmod(offset, 2)
        offset = floor + (remainder if floor % 2 else 0)
    return _integer(query_us + offset)


def _windows(values: object, count: int) -> np.ndarray:
    windows = np.asarray(values)
    if windows.shape != (count, 2) or windows.dtype.kind not in "iu":
        raise ValueError("Windows require integer[T,2] endpoints")
    normalized = np.array([[_integer(v) for v in row] for row in windows], np.int64)
    if np.any(normalized[:, 1] <= normalized[:, 0]):
        raise ValueError("Windows require positive integer duration")
    if count > 1 and np.any(normalized[1:, 0] < normalized[:-1, 1]):
        raise ValueError("Windows must be chronological and nonoverlapping")
    return normalized


def _validate_metadata(obs: dict, query_us: int) -> None:
    required = {"timestamps_us", "first_sensor_ts_us", "anchor_us", "available_us"}
    if not isinstance(obs, dict) or not required.issubset(obs):
        raise ValueError("Missing observation clocks")
    timestamps = [_integer(t) for t in obs["timestamps_us"]]
    first, anchor, available = [
        _integer(obs[k]) for k in ("first_sensor_ts_us", "anchor_us", "available_us")
    ]
    if len(timestamps) not in (2, 3) or any(
        b <= a for a, b in zip(timestamps, timestamps[1:], strict=False)
    ):
        raise ValueError("Observation needs2or3 strictly increasing timestamps")
    if not first <= timestamps[0] or not timestamps[-1] == anchor <= available <= query_us:
        raise ValueError("Invalid causal observation endpoints")
    if "windows_us" in obs:
        windows = _windows(obs["windows_us"], len(timestamps))
        if windows[0, 0] != first or not np.array_equal(windows[:, 1], timestamps):
            raise ValueError("Windows differ from observation endpoints")
    elif first != timestamps[0]:
        raise ValueError("Windowless RGB first sensor must equal first timestamp")
    if "delta_t_s" in obs:
        delta = np.asarray(obs["delta_t_s"])
        expected = np.diff(np.array(timestamps, np.int64).astype(object)).astype(np.float64) / 1e6
        if (
            delta.shape != expected.shape
            or delta.dtype.kind not in "fiu"
            or not np.isfinite(delta).all()
            or not np.allclose(delta, expected, atol=1e-8, rtol=1e-6)
        ):
            raise ValueError("Producer deltas differ from timestamp endpoints")


def warp_metadata(obs: dict, query_us: int, speed: float) -> dict:
    """Copy metadata, transform every declared clock and recompute dt from rounded endpoints."""
    query_us = _integer(query_us)
    _validate_metadata(obs, query_us)
    result = deepcopy(obs)
    for key in ("first_sensor_ts_us", "anchor_us", "available_us"):
        result[key] = warp_clock(obs[key], query_us, speed)
    result["timestamps_us"] = [warp_clock(t, query_us, speed) for t in obs["timestamps_us"]]
    if "windows_us" in obs:
        result["windows_us"] = [
            [warp_clock(t, query_us, speed) for t in window] for window in obs["windows_us"]
        ]
    result["delta_t_s"] = [
        (b - a) / 1e6
        for a, b in zip(result["timestamps_us"], result["timestamps_us"][1:], strict=False)
    ]
    try:
        _validate_metadata(result, query_us)
    except ValueError as error:
        raise ValueError(
            "Warped clock quantization collapsed or invalidated observation"
        ) from error
    return result


def warp_pixels(
    pixels: np.ndarray, modality: str, old_windows: object, new_windows: object
) -> np.ndarray:
    """Copy RGB exactly; change only event channel11 by the actual integer-duration ratio."""
    if modality not in ("rgb", "event"):
        raise ValueError("Modality must be rgb or event")
    channels = 3 if modality == "rgb" else 12
    if (
        not isinstance(pixels, np.ndarray)
        or pixels.dtype != np.float32
        or pixels.ndim != 4
        or pixels.shape[1:] != (channels, 128, 128)
        or pixels.shape[0] not in (1, 2, 3)
        or not np.isfinite(pixels).all()
    ):
        raise ValueError("Pixels require finite FP32[T,C,128,128] with declared modality channels")
    result = pixels.copy()
    if modality == "rgb":
        return result
    old, new = _windows(old_windows, len(pixels)), _windows(new_windows, len(pixels))
    if np.any(pixels[:, 10:] < 0):
        raise ValueError("Event count and rate must be nonnegative log1p values")
    # Both scalar channels are global per-window values in the frozen encoder.
    if not np.all(pixels[:, 10:] == pixels[:, 10:, :1, :1]):
        raise ValueError("Event count and rate channels must be spatially constant")
    for index in range(len(pixels)):
        old_duration = int(old[index, 1]) - int(old[index, 0])
        new_duration = int(new[index, 1]) - int(new[index, 0])
        if old_duration != new_duration:
            with np.errstate(over="raise", invalid="raise"):
                try:
                    value = np.log1p(
                        np.expm1(float(pixels[index, 11, 0, 0])) * old_duration / new_duration
                    )
                except FloatingPointError as error:
                    raise ValueError("Event rate transform overflow") from error
            if not np.isfinite(value) or value > np.finfo(np.float32).max:
                raise ValueError("Event rate transform is not finite FP32")
            result[index, 11] = value
    return result


def filter_retained(
    observations: list[dict],
    query_us: int,
    *,
    budget_us: int = 650000,
    max_observations: int = 8,
) -> tuple[list[dict], list[dict]]:
    """Retain chronological identities within budget; report every discard without filling gaps."""
    query_us, budget_us, max_observations = map(_integer, (query_us, budget_us, max_observations))
    if budget_us != 650000 or max_observations != 8:
        raise ValueError("Pilot fixes650ms history and maximum8 observations")
    retained, discarded = [], []
    previous = None
    for index, obs in enumerate(observations):
        _validate_metadata(obs, query_us)
        anchor = _integer(obs["anchor_us"])
        if previous is not None and anchor <= previous:
            raise ValueError("Observation anchors must be strictly chronological")
        previous = anchor
        if query_us - _integer(obs["first_sensor_ts_us"]) > budget_us:
            discarded.append(
                {"index": index, "reason": "history_budget", "observation": deepcopy(obs)}
            )
        else:
            retained.append((index, deepcopy(obs)))
    for index, obs in retained[:-max_observations]:
        discarded.append({"index": index, "reason": "observation_limit", "observation": obs})
    return [obs for _, obs in retained[-max_observations:]], sorted(
        discarded, key=lambda x: x["index"]
    )

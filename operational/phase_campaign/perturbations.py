"""Deterministic raw sensor interventions before representation construction."""

from __future__ import annotations

import numpy as np


def perturb_events(
    raw: dict[str, np.ndarray], variant: str, start: int, end: int, width: int, height: int
) -> dict[str, np.ndarray]:
    """Perturb one complete half-open window; keep source buffers immutable.

    Tuple hashes make dropout/jitter independent of read chunking. Identical
    timestamp/pixel/polarity tuples receive the same perturbation. Background
    injection uses a fixed seed and is local to the declared complete window.
    """
    result = {k: v.copy() for k, v in raw.items()}
    code = (
        raw["t"].astype(np.uint64) * np.uint64(6364136223846793005)
        ^ raw["x"].astype(np.uint64) * np.uint64(1442695040888963407)
        ^ raw["y"].astype(np.uint64) * np.uint64(2862933555777941757)
        ^ raw["p"].astype(np.uint64)
        ^ np.uint64(7)
    )
    code ^= code >> np.uint64(33)
    uniform = (code % np.uint64(1000003)).astype(np.float64) / 1000003
    keep = np.ones(len(code), dtype=bool)
    if variant.startswith("drop_") and variant.split("_")[1].isdigit():
        keep = uniform >= int(variant.split("_")[1]) / 100
    elif variant == "drop_positive":
        keep = raw["p"] <= 0
    elif variant == "drop_negative":
        keep = raw["p"] > 0
    elif variant.startswith("jitter_"):
        radius = int(variant.split("_")[1])
        result["t"] = result["t"] + np.rint((2 * uniform - 1) * radius).astype(np.int64)
        keep = (result["t"] >= start) & (result["t"] < end)
    elif variant == "background_5":
        rng = np.random.default_rng(7)
        size = int(0.05 * len(code))
        extra = {
            "x": rng.integers(width, size=size),
            "y": rng.integers(height, size=size),
            "t": rng.integers(start, end, size=size),
            "p": rng.integers(2, size=size),
        }
        result = {k: np.concatenate((v, extra[k].astype(v.dtype))) for k, v in result.items()}
        keep = np.ones(len(result["t"]), dtype=bool)
    elif variant != "clean":
        raise ValueError(f"Unknown sensor perturbation: {variant}")
    result = {k: v[keep] for k, v in result.items()}
    order = np.argsort(result["t"], kind="stable")
    return {k: v[order] for k, v in result.items()}

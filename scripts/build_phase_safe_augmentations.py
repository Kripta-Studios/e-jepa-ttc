"""Build causal geometry-channel perturbations for the frozen eligible population.

Native predictions and real targets are copied unchanged. This is neither raw
sensor augmentation nor a cold-start rerun of native predictors. No benchmark
sensor/target files are opened and no models are trained or selected.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from operational.phase_robust.data import histories
from operational.phase_startup.geometry import startup_geometry

VARIANTS = ("last2", "last4", "jitter2px", "dropout50")
HISTORY_US = 650_000
SENSOR_FIELDS = ("sequence_id", "query_id", "anchor_us", "rgb_timestamps_us", "rgb_boxes")


@dataclass(frozen=True)
class GeometryQuery:
    """A group-bound geometry query without native predictions or labels."""

    token: str
    group: str
    anchor_us: int


def eligible_sensor_rows(
    metadata: list[dict[str, Any]], allowed_groups: set[str]
) -> list[dict[str, Any]]:
    """Inspect only group ID for excluded rows; copy allowed sensor fields only."""
    return [
        {key: row[key] for key in SENSOR_FIELDS}
        for row in metadata
        if row["sequence_id"] in allowed_groups
    ]


def perturb_history(
    boxes: np.ndarray,
    timestamps_us: np.ndarray,
    query_us: int,
    token: str,
    variant: str,
    *,
    seed: int = 7,
) -> tuple[np.ndarray, np.ndarray]:
    """Cut to causal history before deterministic per-query bbox perturbation."""
    boxes = np.asarray(boxes, dtype=np.float64)
    clocks = np.asarray(timestamps_us, dtype=np.int64)
    if clocks.ndim != 1 or boxes.shape != (len(clocks), 4):
        raise ValueError("Expected matching [N,4] boxes and [N] clocks")
    if np.any(np.diff(clocks) <= 0):
        raise ValueError("History clocks must be strictly increasing")
    if variant not in (*VARIANTS, "clean"):
        raise ValueError("Unknown geometry augmentation")
    admitted = (clocks <= query_us) & (clocks >= query_us - HISTORY_US)
    boxes, clocks = boxes[admitted].copy(), clocks[admitted].copy()
    if variant in ("last2", "last4"):
        count = 2 if variant == "last2" else 4
        return boxes[-count:], clocks[-count:]
    digest = hashlib.sha256(f"{seed}\0{token}\0{variant}".encode()).digest()
    rng = np.random.Generator(np.random.PCG64(int.from_bytes(digest[:16], "big")))
    if variant == "jitter2px":
        boxes += rng.uniform(-2.0, 2.0, size=boxes.shape)
    elif variant == "dropout50":
        keep = rng.random(len(clocks)) >= 0.5
        boxes, clocks = boxes[keep], clocks[keep]
    return boxes, clocks


def build_geometry_cache(
    queries: list[GeometryQuery],
    tracks: dict[str, tuple[np.ndarray, np.ndarray]],
    *,
    variants: tuple[str, ...] = VARIANTS,
    seed: int = 7,
) -> dict[str, np.ndarray]:
    """Recompute each geometry estimate and its missing mask from observations."""
    shape = (len(variants), len(queries))
    geometry = np.full(shape, np.nan, np.float32)
    quality = np.zeros((*shape, 20), np.float32)
    valid = np.zeros(shape, bool)
    reasons = np.full(shape, "", dtype="U64")
    observation_count = np.zeros(shape, np.int64)
    admitted_count = np.zeros(shape, np.int64)
    latest_observation_us = np.full(shape, -1, np.int64)
    for k, variant in enumerate(variants):
        for i, query in enumerate(queries):
            boxes, clocks = perturb_history(
                *tracks[query.group], query.anchor_us, query.token, variant, seed=seed
            )
            result = startup_geometry(boxes, clocks, query.anchor_us, history_us=HISTORY_US)
            if result["valid"]:
                geometry[k, i] = np.clip(result["ttc"], -60, 60)
            quality[k, i] = result["quality"]
            valid[k, i] = result["valid"]
            reasons[k, i] = result["reason"]
            observation_count[k, i] = len(clocks)
            admitted_count[k, i] = result["count"]
            if len(clocks):
                latest_observation_us[k, i] = clocks[-1]
    if not np.array_equal(valid, np.isfinite(geometry)) or not np.isfinite(quality).all():
        raise ValueError("Invalid geometry availability or quality")
    return {
        "geometry": geometry,
        "quality": quality,
        "valid": valid,
        "reason": reasons,
        "observation_count": observation_count,
        "admitted_count": admitted_count,
        "latest_observation_us": latest_observation_us,
    }


def file_sha(path: Path) -> str:
    """Hash the exact source or artifact bytes."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def array_sha(value: np.ndarray) -> str:
    """Bind array shape, dtype and exact contents, including missing values."""
    digest = hashlib.sha256(str((value.shape, value.dtype.str)).encode())
    digest.update(value.tobytes())
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--population", type=Path, default=Path("output/phase_submission/training/POPULATION.npz")
    )
    parser.add_argument(
        "--training-config", type=Path, default=Path("output/phase_submission/training/CONFIG.json")
    )
    parser.add_argument(
        "--metadata", type=Path, default=Path("output/phase_adapt_final/data_v3/POPULATION.json")
    )
    parser.add_argument("--campaign", type=Path, default=Path("output/phase_campaign"))
    parser.add_argument(
        "--output", type=Path, default=Path("output/phase_safe_fusion/augmentations")
    )
    args = parser.parse_args()
    with np.load(args.population, allow_pickle=False) as archive:
        original = {key: archive[key].copy() for key in archive.files}
    config = json.loads(args.training_config.read_text())
    excluded = set(config["excluded_road"]) | {"Slider-750", "Slider-1000"}
    allowed = set(original["group"].tolist())
    if (
        len(original["token"]) != 2608
        or len(set(original["token"])) != 2608
        or len(allowed) != 25
        or allowed & excluded
        or allowed != set(config["fit_groups"])
    ):
        raise ValueError(
            "Expected exactly frozen eligible 2608 rows/25 groups with benchmark excluded"
        )
    metadata = json.loads(args.metadata.read_text())["rows"]
    sensor_rows = eligible_sensor_rows(metadata, allowed - {"Slider-500"})
    del metadata
    by_token = {"evttc:" + row["query_id"]: row for row in sensor_rows}
    if len(by_token) != len(sensor_rows):
        raise ValueError("Duplicate eligible metadata identity")
    tracks = histories(sensor_rows)
    slider_box_path = args.campaign / "Slider500/inputs/Slider-500/bbox.csv"
    slider_clock_path = args.campaign / "Slider500/FREEZE.json"
    raw_boxes = np.loadtxt(slider_box_path, delimiter=",")
    clocks = np.array(
        [row["anchor_us"] for row in json.loads(slider_clock_path.read_text())["queries"]], np.int64
    )
    if len(raw_boxes) != len(clocks) or np.max(abs(raw_boxes[:, 1] * 1e6 - clocks)) > 2:
        raise ValueError("Slider500 original box and sensor clocks disagree")
    tracks["Slider-500"] = raw_boxes[:, 2:6], clocks
    queries = []
    for token, group in zip(original["token"], original["group"], strict=True):
        if str(token).startswith("evttc:"):
            row = by_token[str(token)]
            if row["sequence_id"] != group:
                raise ValueError("Metadata and frozen population groups disagree")
            anchor = int(row["anchor_us"])
        else:
            slider_group, index = str(token).split(":")
            if slider_group != group or group != "Slider-500":
                raise ValueError("Noneligible slider token")
            anchor = int(clocks[int(index)])
        queries.append(GeometryQuery(str(token), str(group), anchor))
    clean = build_geometry_cache(queries, tracks, variants=("clean",))
    # Parity is a gate before generating perturbations, with no target-dependent tolerance.
    np.testing.assert_allclose(clean["geometry"][0], original["geometry"], rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(clean["quality"][0], original["quality"], rtol=1e-6, atol=1e-6)
    args.output.mkdir(parents=True, exist_ok=True)
    parity_path = args.output / "CLEAN_PARITY.json"
    clean_parity = {
        "passed": True,
        "rtol": 1e-6,
        "atol": 1e-6,
        "geometry_exact": np.array_equal(
            clean["geometry"][0], original["geometry"], equal_nan=True
        ),
        "quality_exact": np.array_equal(clean["quality"][0], original["quality"], equal_nan=True),
        "before_augmentation": True,
        "rows": len(queries),
        "population_sha256": file_sha(args.population),
        "metadata_sha256": file_sha(args.metadata),
        "script_sha256": file_sha(Path(__file__)),
    }
    parity_path.write_text(json.dumps(clean_parity, indent=2) + "\n", encoding="utf-8")
    result = build_geometry_cache(queries, tracks, seed=7)
    result.update(
        {key: original[key].copy() for key in ("native", "target", "group", "family", "token")}
    )
    result.update(
        variant_names=np.array(VARIANTS),
        anchor_us=np.array([q.anchor_us for q in queries], np.int64),
        clean_geometry=clean["geometry"][0],
        clean_quality=clean["quality"][0],
    )
    args.output.mkdir(parents=True, exist_ok=True)
    output = args.output / "AUGMENTATIONS.npz"
    pending = args.output / ".AUGMENTATIONS.building.npz"
    np.savez_compressed(pending, allow_pickle=False, **result)
    pending.replace(output)
    unchanged = {
        key: array_sha(result[key]) == array_sha(original[key])
        for key in ("native", "target", "group", "family", "token")
    }
    if not all(unchanged.values()):
        raise ValueError("Frozen native/target/identity fields changed")
    source_paths = [
        Path(__file__),
        Path("operational/phase_startup/geometry.py"),
        Path("operational/phase_robust/data.py"),
    ]
    receipt = {
        "schema": "phase_safe_geometry_augmentations_v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "scope": (
            "Geometry observation channel only; not raw-sensor or native cold-start augmentation"
        ),
        "native_scope": (
            "Cached native predictions retain original unperturbed bbox/sensor histories"
        ),
        "targets": "Real frozen targets copied unchanged; no synthetic labels or label adjustment",
        "rows": 2608,
        "road_rows": 2531,
        "slider500_rows": 77,
        "groups": sorted(allowed),
        "excluded_groups": sorted(excluded),
        "eligible_sensor_metadata_rows": len(sensor_rows),
        "history_policy": (
            "All sensor-only observations from eligible groups, cutoff at query and 650 ms history"
        ),
        "variants": list(VARIANTS),
        "seed": 7,
        "rng": "PCG64 seeded by first16bytes SHA256(UTF8(seed\\0token\\0variant)), big-endian",
        "jitter": "Independent continuous uniform[-2,+2]pixels per box coordinate; no label access",
        "dropout": "Independent Bernoulli50% per causal box; no forced endpoint or imputation",
        "clean_parity": clean_parity,
        "clean_parity_receipt_sha256": file_sha(parity_path),
        "unchanged_array_checks": unchanged,
        "unchanged_array_sha256": {key: array_sha(original[key]) for key in unchanged},
        "variant_support": {
            variant: {
                "valid_geometry": int(result["valid"][k].sum()),
                "reasons": dict(Counter(result["reason"][k].tolist())),
                "mean_observation_count": float(result["observation_count"][k].mean()),
            }
            for k, variant in enumerate(VARIANTS)
        },
        "source_sha256": {str(p): file_sha(p) for p in source_paths},
        "input_sha256": {
            str(p): file_sha(p)
            for p in (
                args.population,
                args.training_config,
                args.metadata,
                slider_box_path,
                slider_clock_path,
            )
        },
        "output_sha256": {output.name: file_sha(output)},
        "benchmark_sensor_or_target_files_opened": False,
        "excluded_metadata_access": "Only sequence_id inspected before dropping excluded groups",
        "geometry_helpers_receive_targets": False,
        "training_performed": False,
        "model_selection_performed": False,
    }
    (args.output / "RECEIPT.json").write_text(
        json.dumps(receipt, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "rows": 2608,
                "shape": list(result["geometry"].shape),
                "clean_parity": receipt["clean_parity"],
                "support": receipt["variant_support"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

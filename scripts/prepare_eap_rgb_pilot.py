"""Prepare and inspect a train-only RGB pilot without approving physical geometry."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from e_jepa_ttc.detection.eap_rgb_data import (
    EAPRGBDataset,
    safe_dataset_path,
    select_pilot_tokens,
    sha256_file,
)

CLASSES = (
    "car",
    "bicycle",
    "truck",
    "pedestrian",
    "motorcycle",
    "bus",
    "tricycle",
    "other_vehicle",
)


def inspect_event_slice(root: Path, row: dict[str, Any], duration_us: int = 5000) -> dict[str, Any]:
    """Read a bounded real event interval ending at RGB exposure end.

    The millisecond index only brackets the window: an explicit timestamp mask
    removes events outside it. This proves sampled data access, not sensor arrival
    latency or monotonicity of the entire stream.
    """
    import h5py
    import hdf5plugin  # noqa: F401 -- registers the dataset's compression filters

    end_us = int(row["rgb_exposure_end_timestamp_us"])
    start_us = end_us - duration_us
    path = safe_dataset_path(root, str(row["events_path"]))
    with h5py.File(path, "r") as handle:
        index = handle["ms_to_idx"]
        first_ms, after_ms = start_us // 1000, end_us // 1000 + 1
        if first_ms < 0 or after_ms >= len(index):
            raise ValueError("Event window is outside the millisecond index")
        first, after = int(index[first_ms]), int(index[after_ms])
        if not 0 <= first <= after <= len(handle["events/t"]):
            raise ValueError("Invalid event index offsets")
        if after - first > 2_000_000:
            raise ValueError("Bounded event audit would exceed two million events")
        arrays = {key: handle[f"events/{key}"][first:after] for key in ("x", "y", "t", "p")}
    timestamps = arrays["t"]
    if len(timestamps) and np.any(np.diff(timestamps) < 0):
        raise ValueError("Event timestamps decrease in the sampled slice")
    valid = (timestamps >= start_us) & (timestamps <= end_us)
    if not valid.any():
        raise ValueError("No events in the sampled interval")
    trimmed = {key: value[valid] for key, value in arrays.items()}
    digest = hashlib.sha256()
    for key in ("x", "y", "t", "p"):
        digest.update(np.ascontiguousarray(trimmed[key]).tobytes())
    return {
        "sequence_id": str(row["sequence_id"]),
        "events_path": str(row["events_path"]),
        "window_us_closed": [start_us, end_us],
        "decision_reference": "RGB exposure end; arrival latency not measured",
        "index_slice": [first, after],
        "bracket_event_count": len(timestamps),
        "retained_event_count": int(valid.sum()),
        "removed_before_start": int((timestamps < start_us).sum()),
        "removed_after_end": int((timestamps > end_us).sum()),
        "retained_timestamp_min_us": int(trimmed["t"].min()),
        "retained_timestamp_max_us": int(trimmed["t"].max()),
        "retained_polarities": np.unique(trimmed["p"]).tolist(),
        "retained_x_range": [int(trimmed["x"].min()), int(trimmed["x"].max())],
        "retained_y_range": [int(trimmed["y"].min()), int(trimmed["y"].max())],
        "retained_array_bytes_sha256_xy_tp_order": digest.hexdigest(),
        "timestamp_mask_applied": True,
        "future_events_retained": int((trimmed["t"] > end_us).sum()),
    }


def prepare(
    root: Path,
    split_path: Path,
    output: Path,
    receipt_path: Path,
    seed: int,
    frames_per_sequence: int,
    check_events: bool,
) -> dict[str, Any]:
    """Write a declared protocol, then verify every selected RGB frame."""
    started = time.perf_counter()
    source = root / "data/train.parquet"
    split = json.loads(split_path.read_text(encoding="utf-8"))
    manifest = select_pilot_tokens(
        pd.read_parquet(source),
        split["assignments"],
        seed=seed,
        frames_per_sequence=frames_per_sequence,
    )
    if sha256_file(source) != split["media_sha256"]:
        raise ValueError("Media manifest differs from the candidate split source")
    manifest.update(
        {
            "schema": "eap_rgb_pilot_v1",
            "created_at_utc": datetime.now(UTC).isoformat(),
            "dataset_revision": split["dataset_revision"],
            "source_split": "train",
            "source_media_sha256": sha256_file(source),
            "candidate_split_sha256": sha256_file(split_path),
            "sequence_selection": "First two sha256(pilot:seed:role:sequence_id) ranks per role",
            "frame_selection": (
                "Evenly spaced after exposure-start/token sort, no label/image selection"
            ),
            "candidate_grouping_limit": split["grouping"],
            "protocol": {
                "overfit16_scope": "Functional memorization test, not evidence of generalization",
                "selection_use": (
                    "No training on selection; pilot model/hyperparameter selection permitted"
                ),
                "calibration_use": "Untouched",
                "baseline_inputs": "Full RGB image plus calibration; no GT ROI or object ID input",
                "overfit16_predeclared_checks": [
                    "All losses, predictions and gradients finite before any sanitization",
                    "Learned parameters change after optimization",
                    "Training loss lower than initial evaluation on the same fixed 16 frames",
                    "Measure object recovery on the same frames with a declared development metric",
                    "Save predictions, checkpoint, seed, config, source hashes and runtime",
                ],
                "note": "Training runner must predeclare numerical budget/criteria before its run",
                "does_not_approve_geometry_or_official_submission": True,
            },
        }
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    tokens = manifest["train"]["sample_tokens"] + manifest["selection"]["sample_tokens"]
    train_tokens = set(manifest["train"]["sample_tokens"])
    overfit_tokens = set(manifest["overfit16"]["sample_tokens"])
    counts = {role: Counter() for role in ("train", "selection", "overfit16")}
    frames = []
    errors = []
    label_sources = {}
    events = []
    with EAPRGBDataset(root, tokens) as dataset:
        for index, token in enumerate(tokens):
            try:
                frame = dataset[index]
                role = "train" if token in train_tokens else "selection"
                labels = Counter(str(row["category"]) for row in frame.annotations)
                if set(labels) - set(CLASSES):
                    raise ValueError(f"Unknown classes: {set(labels) - set(CLASSES)}")
                counts[role].update(labels)
                if token in overfit_tokens:
                    counts["overfit16"].update(labels)
                frames.append(
                    {
                        "sample_token": token,
                        "role": role,
                        "overfit16": token in overfit_tokens,
                        "shape_hwc": list(frame.image.shape),
                        "member_sha256": frame.member_sha256,
                        "annotation_count": len(frame.annotations),
                        "class_counts": dict(labels),
                        "exposure_start_us": frame.exposure_start_us,
                        "exposure_end_us": frame.exposure_end_us,
                    }
                )
                label_sources[str(frame.media["labels_path"])] = None
            except (OSError, ValueError, KeyError) as error:
                errors.append({"sample_token": token, "error": f"{type(error).__name__}: {error}"})
            if (index + 1) % 64 == 0:
                print(json.dumps({"rgb_checked": index + 1, "total": len(tokens)}), flush=True)
        if check_events:
            for sequence in manifest["train"]["sequences"]:
                token = next(
                    t
                    for t in manifest["overfit16"]["sample_tokens"]
                    if t.startswith(sequence + ":")
                )
                try:
                    events.append(inspect_event_slice(root, dataset.rows[token]))
                except (OSError, ValueError, KeyError, ImportError) as error:
                    errors.append(
                        {"sequence_id": sequence, "event_error": f"{type(error).__name__}: {error}"}
                    )
    receipt = {
        "schema": "eap_rgb_pilot_inspection_v1",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "status": "PASS_SELECTED_MEDIA_ONLY" if not errors else "FAIL_SELECTED_MEDIA",
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "source_sha256": {
            "scripts/prepare_eap_rgb_pilot.py": sha256_file(Path(__file__)),
            "src/e_jepa_ttc/detection/eap_rgb_data.py": sha256_file(
                Path("src/e_jepa_ttc/detection/eap_rgb_data.py")
            ),
        },
        "pilot_manifest_path": str(output),
        "pilot_manifest_sha256": sha256_file(output),
        "source_media_sha256": manifest["source_media_sha256"],
        "candidate_split_sha256": manifest["candidate_split_sha256"],
        "label_source_sha256": {p: sha256_file(safe_dataset_path(root, p)) for p in label_sources},
        "selected_sequences": {
            role: manifest[role]["sequences"] for role in ("train", "selection")
        },
        "selected_frames": len(tokens),
        "readable_frames": len(frames),
        "class_counts": {role: {name: counts[role][name] for name in CLASSES} for role in counts},
        "frames": frames,
        "event_slices": events,
        "errors": errors,
        "elapsed_seconds": time.perf_counter() - started,
        "limitations": [
            "Verifies selected members, not full TAR/HDF5 hashes or all dataset contents",
            "Only public train rows used; no test media or test labels read",
            "Sequence-disjoint candidate; cross-sequence acquisition groups not verified",
            "Geometry conversion and RGB/event pixel rectification not approved by this receipt",
            "Event slice checks do not measure physical sensor arrival or full-stream monotonicity",
            "Not evidence of detector training, generalization or official benchmark performance",
        ],
    }
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    return receipt


def main() -> int:
    """Prepare the pilot and return nonzero if any selected member is invalid."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eap-root", type=Path, required=True)
    parser.add_argument(
        "--split", type=Path, default=Path("data/splits/eap_detection_candidate_v1.json")
    )
    parser.add_argument("--output", type=Path, default=Path("output/eap_rgb_pilot/manifest.json"))
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--frames-per-sequence", type=int, default=128)
    parser.add_argument("--check-events", action="store_true")
    args = parser.parse_args()
    receipt = prepare(
        args.eap_root,
        args.split,
        args.output,
        args.receipt,
        args.seed,
        args.frames_per_sequence,
        args.check_events,
    )
    print(
        json.dumps(
            {key: receipt[key] for key in ("status", "readable_frames", "class_counts", "errors")}
        )
    )
    return 0 if not receipt["errors"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

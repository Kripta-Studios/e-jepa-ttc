"""Describe preserved TRAIN40 predictions by fixed strata; no model execution."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def run(campaign: Path, output: Path) -> None:
    """Verify original predictions and retain every query in each partition."""
    output.mkdir(parents=True, exist_ok=True)
    hashes: dict[str, str] = {}

    def pin(path: Path, expected: str | None = None) -> None:
        with path.open("rb") as handle:
            value = hashlib.file_digest(handle, "sha256").hexdigest()
        if expected is not None and value != expected:
            raise ValueError(f"Source hash mismatch: {path}")
        hashes[str(path)] = value

    def read(path: Path) -> dict:
        return json.loads(path.read_text(encoding="utf-8"))

    index = campaign / "TRAIN40_INDEX.npz"
    pin(index, read(campaign / "DATA_AUDIT.json")["index_sha256"])
    with np.load(index, allow_pickle=False) as stored:
        truth = stored["ttc_s"]
        sequences = stored["sequences"]
        boxes = stored["boxes_xyxy"][:, -1]
    n = len(truth)
    if n != 88744 or len(np.unique(sequences)) != 40:
        raise ValueError("Unexpected fixed TRAIN40 population")
    predictions = {f"H8_seed{s}": np.empty(n, np.float32) for s in (7, 13, 23)}
    binding_path = campaign / "train_predictions/BINDING.json"
    pin(binding_path)
    for start in range(0, n, 128):
        stop = min(start + 128, n)
        path = campaign / "train_predictions" / f"batch_{start:06d}.npz"
        receipt = read(path.with_suffix(".json"))
        pin(path, receipt["sha256"])
        if receipt["binding_sha256"] != hashes[str(binding_path)]:
            raise ValueError("Prediction lineage changed")
        with np.load(path, allow_pickle=False) as stored:
            if not np.array_equal(stored["ordinals"], np.arange(start, stop)):
                raise ValueError("Prediction order changed")
            for seed in (7, 13, 23):
                predictions[f"H8_seed{seed}"][start:stop] = stored[f"ttc_seed{seed}"]
    manifest = read(campaign / "PUBLIC_GARL_PREDICTION_MANIFEST.json")
    if manifest["status"] != "COMPLETE_VERIFIED" or manifest["row_count"] != n:
        raise ValueError("Incomplete Garl predictions")
    entry = manifest["files"][0]
    path = campaign / entry["path"]
    pin(path, entry["sha256"])
    with np.load(path, allow_pickle=False) as stored:
        if not np.array_equal(stored["ordinals"], np.arange(n)):
            raise ValueError("Garl prediction order changed")
        predictions["public_Garl_event_lhr"] = stored["ttc"]
    height = boxes[:, 3].astype(float) - boxes[:, 1].astype(float)
    if not np.isfinite(truth).all() or not np.all(np.isfinite(height) & (height > 0)):
        raise ValueError("Nonfinite targets or invalid boxes")
    partitions = {
        "signed_target": np.where(truth < 0, "negative", "positive"),
        "absolute_ttc_seconds": np.asarray(["<0.5", "0.5-1", "1-2", "2-4", "4-8", ">=8"])[
            np.digitize(np.abs(truth), [0.5, 1, 2, 4, 8])
        ],
        "last_bbox_height_sensor_px": np.asarray(["<32", "32-64", "64-128", ">=128"])[
            np.digitize(height, [32, 64, 128])
        ],
    }
    rows = []
    for model, prediction in predictions.items():
        if not np.isfinite(prediction).all():
            raise ValueError("Original complete finite coverage changed")
        error = prediction.astype(float) - truth
        for partition, labels in partitions.items():
            selected = 0
            total_absolute = 0.0
            for label in np.unique(labels):
                mask = labels == label
                values = error[mask]
                count = int(mask.sum())
                selected += count
                total_absolute += float(np.abs(values).sum())
                rows.append(dict(
                    model=model, partition=partition, stratum=str(label), population=count,
                    sequences=int(len(np.unique(sequences[mask]))),
                    mae_seconds=float(np.abs(values).mean()),
                    rmse_seconds=float(np.sqrt(np.square(values).mean())),
                    median_absolute_error_seconds=float(np.median(np.abs(values))),
                ))
            if selected != n or not np.isclose(total_absolute, np.abs(error).sum(), rtol=1e-12):
                raise ValueError("Partition does not reconcile to original population")
    with (output / "STRATA.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    pin(Path(__file__))
    result = dict(
        status="COMPLETE", role="TRAIN_FIT_DESCRIPTION_NOT_INDEPENDENT_EVALUATION",
        population=n, sequences=40, optimizer_updates=0, new_predictions=0,
        fixed_partitions=list(partitions), partition_reconciliation="PASSED",
        clipping_or_model_selection=False, inputs_sha256=hashes, rows=rows,
        limitation="Native own TTC bounded to +/-60s; Garl native output unbounded.",
    )
    (output / "RESULT.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({k: v for k, v in result.items() if k not in ("rows", "inputs_sha256")}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.campaign, args.output)

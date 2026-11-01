"""Score sealed predictions on supported local targets without changing the candidate."""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path

import numpy as np
import torch

from operational.evttc_transfer_audit.common import file_sha
from operational.phase_adapt_cache import atomic_receipt
from operational.phase_campaign.data import read
from operational.phase_submission.protocol import interpolate_negative_predictions
from operational.phase_submission.runtime import load_submission_model
from operational.phase_submission.verify import verify_results


def metric(target: np.ndarray, prediction: np.ndarray) -> dict:
    """Report full population/target/prediction coverage, never silently drop failures."""
    supported = np.isfinite(target) & (target > 0)
    mask = supported & np.isfinite(prediction)
    if not mask.any():
        return {"population": len(target), "targets": int(supported.sum()), "scored": 0}
    error = abs(prediction[mask] - target[mask])
    return {
        "population": len(target),
        "targets": int(supported.sum()),
        "scored": int(mask.sum()),
        "mae_seconds": float(error.mean()),
        "rmse_seconds": float(np.sqrt(np.mean(error**2))),
        "relative_error_pct": float(np.mean(error / target[mask]) * 100),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("replay", "bundle", "historical_run", "road_targets", "slider_contract", "output"):
        parser.add_argument("--" + key.replace("_", "-"), type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    complete = read(args.replay / "COMPLETE.json")  # Must exist BEFORE opening targets.
    freeze = read(args.replay / "RUN_FREEZE.json")
    for name, key in (
        ("RUN_FREEZE.json", "freeze_sha256"),
        ("all_rows.jsonl", "ledger_sha256"),
        ("results.zip", "results_zip_sha256"),
    ):
        if file_sha(args.replay / name) != complete[key]:
            raise ValueError("Changed sealed predictions")
    if file_sha(args.bundle / "MODEL.json") != freeze["model_sha256"]:
        raise ValueError("Changed model identity")
    records = [
        json.loads(line) for line in (args.replay / "all_rows.jsonl").read_text().splitlines()
    ]
    expected = {
        seq: [(row["index"], row["timestamp_text"]) for row in rows]
        for seq, rows in freeze["queries"].items()
    }
    verification = verify_results(args.replay / "results.zip", records, expected)
    if verification["rows"] != 434:
        raise ValueError("Expected434rows")
    model = load_submission_model(args.bundle)
    exported = torch.jit.load(io.BytesIO((args.bundle / "router/router.ts").read_bytes()))
    native = np.array(
        [[v if v is not None else np.nan for v in r["native_ttc_seconds"]] for r in records],
        np.float32,
    )
    geometry = np.array(
        [r["geometry"]["ttc"] if r["geometry"]["valid"] else np.nan for r in records], np.float32
    )
    quality = np.array([r["geometry"]["quality"] for r in records], np.float32)
    prediction = np.array([r["prediction_ttc_seconds"] for r in records])
    tensors = [torch.from_numpy(a) for a in (native, geometry, quality)]
    with torch.inference_mode():
        reloaded = model(*tensors).numpy()
        export = exported(*tensors).numpy()
    # Batch GEMM may differ from measured per-query GEMV in the last FP32 bits.
    np.testing.assert_allclose(reloaded, prediction, rtol=2e-6, atol=2e-6)
    np.testing.assert_array_equal(reloaded, export)
    verification["router_replay_max_abs_difference"] = float(abs(reloaded - prediction).max())
    verification["export_batch_exact_rows"] = len(prediction)
    # Ground-truth access starts only after verifying the immutable prediction receipt.
    road = {r["query_id"]: r for r in read(args.road_targets)["rows"]}
    historical = {
        r["identity"]["sequence_id"] + ":" + str(r["identity"]["row_index"]): r
        for r in map(json.loads, (args.historical_run / "all_rows.jsonl").read_text().splitlines())
    }
    gt = {
        f"Slider-{speed}": {
            int(r[0]): r
            for r in np.loadtxt(args.slider_contract / f"{speed}_gt.csv", delimiter=",")
        }
        for speed in (750, 1000)
    }
    targets = []
    originals = []
    missing = []
    sequence = []
    for row in records:
        identity = row["identity"]
        seq = identity["sequence_id"]
        key = seq + ":" + str(identity["row_index"])
        old = historical[key]
        if old["identity"] != identity:
            raise ValueError("Historical/current query identity differs")
        originals.append(old["prediction_ttc_seconds"])
        sequence.append(seq)
        if seq.startswith("Slider"):
            target = gt[seq][identity["index"]]
            if abs(target[1] - float(identity["timestamp_text"])) > 2e-6:
                raise ValueError("Slider target timestamp differs")
            targets.append(float(target[4]))
        else:
            target_row = road[key]
            if abs(target_row["query_us"] - identity["anchor_us"]) > 1:
                raise ValueError("Road target timestamp differs")
            valid = target_row["target_status"] == "valid"
            targets.append(float(target_row["target_ttc"]) if valid else np.nan)
            if not valid:
                missing.append({"query_id": key, "reason": target_row["reason"]})
    target = np.array(targets)
    old = np.array(originals)
    seq_array = np.array(sequence)
    processed_old = old.copy()
    processed_new = prediction.copy()
    adjustments = {}
    for seq in expected:
        selected = seq_array == seq
        times = np.array(
            [
                float(r["identity"]["timestamp_text"])
                for r in records
                if r["identity"]["sequence_id"] == seq
            ]
        )
        processed_old[selected], old_changes = interpolate_negative_predictions(
            times, old[selected]
        )
        processed_new[selected], new_changes = interpolate_negative_predictions(
            times, prediction[selected]
        )
        adjustments[seq] = {
            "original_negative_row_indices": old_changes,
            "candidate_negative_row_indices": new_changes,
        }
    columns = {
        "original_raw": old,
        "original_email_processed": processed_old,
        "adapted_raw": prediction,
        "adapted_email_processed": processed_new,
        "causal_bbox_geometry": geometry,
    }
    populations = {
        "all": np.ones(len(target), bool),
        "road": ~np.char.startswith(seq_array, "Slider"),
        "slider": np.char.startswith(seq_array, "Slider"),
        **{seq: seq_array == seq for seq in expected},
    }
    report = {
        "scope": "local supported targets only; not official results; historical exposure remains",
        "prediction_seal_before_scoring": complete["results_zip_sha256"],
        "model_sha256": freeze["model_sha256"],
        "verification": verification,
        "missing_targets": missing,
        "negative_processing": adjustments,
        "metrics": {
            name: {label: metric(target[mask], value[mask]) for label, value in columns.items()}
            for name, mask in populations.items()
        },
        "target_sources": {
            str(p): file_sha(p)
            for p in [
                args.road_targets,
                args.slider_contract / "750_gt.csv",
                args.slider_contract / "1000_gt.csv",
            ]
        },
        "official_comparison_limit": (
            "11road targets unavailable locally; Slider750 original PDF not reproduced"
        ),
    }
    atomic_receipt(args.output / "LOCAL_EVALUATION.json", report)
    print(json.dumps(report["metrics"]["all"], indent=2))


if __name__ == "__main__":
    main()

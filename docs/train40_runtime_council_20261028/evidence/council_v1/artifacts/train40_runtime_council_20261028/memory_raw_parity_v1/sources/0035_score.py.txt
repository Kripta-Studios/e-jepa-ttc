"""Score sealed EvTTC predictions separately from all model inputs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from operational.efficient_context.common import atomic_json, digest

MODELS = ("H8_seed7", "H8_seed13", "H8_seed23", "public_Garl_event_lhr")


def interpolate_label(table: np.ndarray, timestamp_s: float) -> float | None:
    """Interpolate only between adjacent finite labels without extrapolation."""
    x, y = table[:, 1], table[:, 4]
    if not np.isfinite(x).all() or (np.diff(x) <= 0).any():
        raise ValueError("TTC timestamps must be finite and strictly increasing")
    if not x[0] <= timestamp_s <= x[-1]:
        return None
    pos = int(np.searchsorted(x, timestamp_s))
    if pos < len(x) and x[pos] == timestamp_s:
        return float(y[pos]) if np.isfinite(y[pos]) else None
    if not np.isfinite(y[pos - 1 : pos + 1]).all():
        return None
    return float(np.interp(timestamp_s, x[pos - 1 : pos + 1], y[pos - 1 : pos + 1]))


def metrics(prediction: np.ndarray, truth: np.ndarray) -> dict:
    """Report finite coverage alongside error; never clip native predictions."""
    labeled = np.isfinite(truth)
    finite = labeled & np.isfinite(prediction)
    error = prediction[finite] - truth[finite]
    return dict(
        queries=len(truth),
        labeled_queries=int(labeled.sum()),
        finite_predictions_on_labels=int(finite.sum()),
        coverage=float(finite.sum() / labeled.sum()) if labeled.any() else None,
        mae_seconds=float(np.abs(error).mean()) if error.size else None,
        rmse_seconds=float(np.sqrt(np.square(error).mean())) if error.size else None,
        median_absolute_error_seconds=float(np.median(np.abs(error))) if error.size else None,
        signed_bias_seconds=float(error.mean()) if error.size else None,
    )


def run(output: Path, inventory: Path, data_root: Path) -> None:
    """Read labels only after validating the complete prediction seal."""
    import yaml

    manifest_path = output / "QUERY_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    seal = json.loads((output / "PREDICTIONS_SEALED.json").read_text(encoding="utf-8"))
    if seal["status"] != "COMPLETE":
        raise ValueError("Only complete sealed predictions may be scored")
    if seal["binding_sha256"] != digest(output / "INFERENCE_FREEZE.json"):
        raise ValueError("Frozen inference binding differs from prediction seal")
    if seal["manifest_sha256"] != digest(manifest_path):
        raise ValueError("Prediction population changed after sealing")
    rows = manifest["rows"]
    if seal["queries"] != len(rows) or len(seal["fragments"]) != len(rows):
        raise ValueError("All query results, including failures, must be sealed before scoring")
    verified_predictions = []
    for i, row in enumerate(rows):
        name = f"predictions/query_{i:05d}.json"
        if digest(output / name) != seal["fragments"][name]:
            raise ValueError(f"Prediction changed after sealing: {name}")
        pred = json.loads((output / name).read_text(encoding="utf-8"))
        if pred["query_id"] != row["query_id"] or pred["binding_sha256"] != seal["binding_sha256"]:
            raise ValueError("Prediction identity or inference binding mismatch")
        verified_predictions.append(pred)
    sequence_meta = {
        s["sequence_id"]: s
        for s in yaml.safe_load(inventory.read_text(encoding="utf-8"))["sequences"]
    }
    tables, labels_sha = {}, {}
    for sequence in sorted({r["sequence_id"] for r in rows}):
        s = sequence_meta[sequence]
        # data_root is the repository containing the original relative local_path.
        path = data_root / s["local_path"] / s["ttc_csv"]
        table = np.genfromtxt(path, dtype=np.float64, ndmin=2)
        if table.shape[1] != 5 or not len(table):
            raise ValueError(f"Expected official five-column TTC table: {path}")
        tables[sequence] = table
        labels_sha[sequence] = digest(path)
    truth = np.full(len(rows), np.nan)
    predictions = {name: np.full(len(rows), np.nan) for name in MODELS}
    scored_rows = []
    for i, row in enumerate(rows):
        pred = verified_predictions[i]
        value = interpolate_label(tables[row["sequence_id"]], row["anchor_us"] / 1e6)
        truth[i] = value if value is not None else np.nan
        scored = dict(
            query_id=row["query_id"],
            sequence_id=row["sequence_id"],
            anchor_us=row["anchor_us"],
            truth_ttc_seconds=value,
            prediction_status=pred["status"],
        )
        for model in MODELS:
            v = pred.get("ttc", {}).get(model)
            predictions[model][i] = float(v) if v is not None else np.nan
            scored[model] = v
        scored_rows.append(scored)
    sequences = np.asarray([r["sequence_id"] for r in rows])
    summaries = {m: metrics(p, truth) for m, p in predictions.items()}
    per_sequence = []
    for sequence in np.unique(sequences):
        mask = sequences == sequence
        for model, p in predictions.items():
            per_sequence.append(
                dict(sequence_id=str(sequence), model=model, **metrics(p[mask], truth[mask]))
            )
    comparison = {}
    garl = predictions["public_Garl_event_lhr"]
    for model in MODELS[:-1]:
        p = predictions[model]
        differences = []
        common = np.isfinite(p) & np.isfinite(garl) & np.isfinite(truth)
        for sequence in np.unique(sequences):
            mask = common & (sequences == sequence)
            if mask.any():
                differences.append(
                    float((np.abs(p[mask] - truth[mask]) - np.abs(garl[mask] - truth[mask])).mean())
                )
        if differences:
            d = np.asarray(differences)
            rng = np.random.default_rng(20261008)
            boot = d[rng.integers(0, len(d), size=(2000, len(d)))].mean(1)
            comparison[model] = dict(
                metric="macro_sequence_MAE_difference_ours_minus_Garl",
                common_queries=int(common.sum()),
                sequences=len(d),
                difference_seconds=float(d.mean()),
                sequence_bootstrap_95pct_seconds=np.quantile(boot, [0.025, 0.975]).tolist(),
            )
    for filename, records in (
        ("SCORED_PREDICTIONS.csv", scored_rows),
        ("PER_SEQUENCE.csv", per_sequence),
    ):
        with (output / filename).open("w", newline="", encoding="utf-8") as h:
            writer = csv.DictWriter(h, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
    result: dict = dict(
        status="COMPLETE",
        role="FROZEN_EVTTC_TRANSFER_DEVELOPMENT_COMPARISON",
        manifest_sha256=digest(manifest_path),
        prediction_seal_sha256=digest(output / "PREDICTIONS_SEALED.json"),
        label_sha256=labels_sha,
        metrics=summaries,
        paired_sequence_comparison=comparison,
        optimizer_updates=0,
        checkpoint_selection_on_EvTTC=False,
        limitations=[
            "Historical EvTTC development exposure; not a wholly blind model-selection test.",
            "Garl checkpoint training/selection ancestry not independently certified.",
            "Oracle RGB ROI with documented rotation-only projection; parallax ignored.",
            "Conservative past annotation availability; no TTC/distance in inference.",
            "Native H8 TTC support +/-60 seconds; native Garl unbounded.",
            "Related depth/relative-velocity TTC definitions; "
            "annotation construction differs across datasets.",
            "Fixed temporal sample per sequence, not all frames or official benchmark score.",
        ],
    )
    atomic_json(output / "RESULT.json", result)
    lines = [
        "# Frozen EvTTC transfer comparison",
        "",
        "No training or calibration. All three H8 heads retained.",
        "",
        "| Model | Labeled | Finite coverage | MAE s | RMSE s | Median AE s |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for model, values in summaries.items():
        lines.append(
            f"| {model} | {values['labeled_queries']} | {values['coverage']} | "
            f"{values['mae_seconds']} | {values['rmse_seconds']} | "
            f"{values['median_absolute_error_seconds']} |"
        )
    lines += ["", "## Limits", ""] + [f"- {x}" for x in result["limitations"]]
    (output / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    args = parser.parse_args()
    run(args.output, args.inventory, args.data_root)

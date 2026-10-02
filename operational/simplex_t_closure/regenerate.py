"""Regenerate delivered scores and compact-head outputs from extracted bytes only."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import psutil


def sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.bundle_root.resolve(strict=True)
    policy = json.loads((root / "operational/resource_policy.json").read_text(encoding="utf-8"))

    def resource_check() -> None:
        if (
            psutil.virtual_memory().available < policy["host_available_ram_floor_bytes"]
            or psutil.Process().memory_info().rss > policy["process_tree_rss_ceiling_bytes"]
        ):
            raise InterruptedError("PAUSED_RESOURCE: extracted regeneration; inputs preserved")

    resource_check()
    manifest = json.loads((root / "CONTENT_MANIFEST.json").read_text(encoding="utf-8"))
    for name, pin in manifest["members"].items():
        path = (root / name).resolve(strict=True)
        if (
            not path.is_relative_to(root)
            or path.stat().st_size != pin["bytes"]
            or sha(path) != pin["sha256"]
        ):
            raise ValueError(f"extracted content hash/size mismatch: {name}")
    sys.path.insert(0, str(root / "provenance/work/src"))
    import torch

    from e_jepa_ttc.evaluation.stage63_65 import benchmark_phase, strict_macro_mass
    from e_jepa_ttc.simplex_t.compact_weights import load_compact_endpoint
    from e_jepa_ttc.simplex_t.expert_phase import expert_benchmark_phase
    from e_jepa_ttc.simplex_t.phase import phase_to_ttc

    torch.set_num_threads(4)
    resource_check()
    export = json.loads((root / "supplement/HEAD_EXPORT.json").read_text(encoding="utf-8"))
    if str(torch.__version__) != export["torch_version"]:
        raise ValueError("use the recorded torch version for numerical head reproduction")
    pieces, heads = {}, []
    for key, pin in export["heads"].items():
        resource_check()
        stage, name, _, seed = key.split("/")
        path = root / "publications" / stage / pin["prediction_path"]
        if sha(path) != pin["prediction_sha256"]:
            raise ValueError("head prediction pin differs")
        frame = pd.read_parquet(path)
        phase_fn = expert_benchmark_phase if name.startswith("SELECTOR") else benchmark_phase
        phase = phase_fn(frame.prediction_ttc_s.to_numpy(np.float64))
        losses = 10000 * np.abs(phase - benchmark_phase(frame.target_ttc.to_numpy(np.float64)))
        if not np.array_equal(losses, frame.loss.to_numpy()):
            raise ValueError("query losses do not regenerate exactly")
        pieces.setdefault(name + "@" + seed.removeprefix("seed"), []).append(frame)
        model = load_compact_endpoint(
            root / "postprocessing" / pin["compact_path"],
            manifest_sha256=pin["compact_manifest_sha256"],
        )
        with np.load(root / "supplement" / pin["inputs"]["path"], allow_pickle=False) as archive:
            inputs = {field: archive[field] for field in ("features", "times", "valid", "experts")}
            if not np.array_equal(archive["sample_token"], frame.sample_token.to_numpy(str)):
                raise ValueError("compact query order differs")
        point, choices = [], []
        raw_fields = {name: [] for name in ("raw_location", "raw_residual", "q10", "q90")}
        relative_costs = []
        with torch.inference_mode():
            for start in range(0, len(frame), 128):
                output = model(
                    *[
                        torch.from_numpy(inputs[field][start : start + 128])
                        for field in ("features", "times", "valid", "experts")
                    ]
                )
                point.append(output["point_phase"].numpy())
                choices.append(output["expert_index"].numpy())
                for name, parts in raw_fields.items():
                    parts.append(output[name].numpy())
                relative_costs.append(output["relative_cost"].numpy())
        # Fixed before export/inference: raw FP32 heads use abs tolerance1e-7;
        # emitted TTC retains its stricter historical tolerance1e-10.
        field_errors = {}
        for name, parts in raw_fields.items():
            actual = np.concatenate(parts).astype(np.float64)
            error = float(np.max(np.abs(actual - frame[name].to_numpy())))
            if not np.isfinite(error) or error > 1e-7:
                raise ValueError(f"raw head field differs: {key}: {name}: {error}")
            field_errors[name] = error
        actual_costs = np.concatenate(relative_costs).astype(np.float64)
        expected_costs = frame[[f"predicted_relative_cost{i}" for i in range(3)]].to_numpy()
        cost_error = float(np.max(np.abs(actual_costs - expected_costs)))
        if not np.isfinite(cost_error) or cost_error > 1e-7:
            raise ValueError(f"relative cost head differs: {key}: {cost_error}")
        field_errors["relative_cost"] = cost_error
        choices = np.concatenate(choices)
        if not np.array_equal(choices, frame.diagnostic_selector.to_numpy()):
            raise ValueError("diagnostic expert decision changed")
        point = np.concatenate(point).astype(np.float64)
        if model.cfg.output_mode == "selector":
            # The unchanged original experts' signed TTC bytes are included.
            predicted = frame[["expert0_ttc", "expert1_ttc", "expert2_ttc"]].to_numpy()[
                np.arange(len(frame)), choices
            ]
        else:
            predicted = phase_to_ttc(torch.from_numpy(point)).numpy()
        expected_ttc = frame.prediction_ttc_s.to_numpy()
        if np.isnan(predicted).any() or np.isnan(expected_ttc).any():
            raise ValueError("head TTC must not contain NaN")
        if not np.array_equal(
            np.isposinf(predicted), np.isposinf(expected_ttc)
        ) or not np.array_equal(np.isneginf(predicted), np.isneginf(expected_ttc)):
            raise ValueError("selector infinite original expert TTC differs")
        finite = np.isfinite(predicted)
        error = (
            float(np.max(np.abs(predicted[finite] - expected_ttc[finite]))) if finite.any() else 0.0
        )
        if error > 1e-10:
            raise ValueError(f"compact output regeneration differs: {key}: {error}")
        heads.append(
            {
                "fit": key,
                "queries": len(frame),
                "max_abs_TTC_error": error,
                "raw_field_errors": field_errors,
                "raw_field_tolerance": 1e-7,
                "TTC_tolerance": 1e-10,
            }
        )
        print(json.dumps({"heads_regenerated": len(heads), "fit": key}), flush=True)
        del inputs, model, frame
    metrics = pd.read_csv(root / "supplement/METRICS.csv")
    scores = []
    for name, parts in pieces.items():
        frame = pd.concat(parts).sort_values("sample_token").reset_index(drop=True)
        mass = strict_macro_mass(frame.target_ttc.to_numpy(), frame.sequence_id.to_numpy())
        score = float(mass @ frame.loss.to_numpy())
        expected = float(metrics[(metrics.arm_seed == name) & (metrics.scope == "all")].MiD.iloc[0])
        if abs(score - expected) > 1e-10:
            raise ValueError("aggregate differs from included physical CSV")
        scores.append({"arm_seed": name, "MiD": score, "CSV_error": score - expected})
    for name in ("CURRENT_MEDIAN", "EWMA_0P3S_H8", "RISK17", "SIMPLEX17"):
        frame = pd.read_parquet(root / f"supplement/{name}_OLD.parquet")
        loss = 10000 * np.abs(
            benchmark_phase(frame.prediction_ttc_s.to_numpy())
            - benchmark_phase(frame.target_ttc.to_numpy())
        )
        if not np.array_equal(loss, frame.loss.to_numpy()):
            raise ValueError("fixed baseline losses differ")
        mass = strict_macro_mass(frame.target_ttc.to_numpy(), frame.sequence_id.to_numpy())
        score = float(mass @ loss)
        series_name = name + ("@7" if name in {"RISK17", "SIMPLEX17"} else "@fixed")
        expected = float(
            metrics[(metrics.arm_seed == series_name) & (metrics.scope == "all")].MiD.iloc[0]
        )
        if abs(score - expected) > 1e-10:
            raise ValueError("fixed baseline aggregate differs")
        scores.append({"arm_seed": series_name, "MiD": score, "CSV_error": score - expected})
    curves = pd.read_parquet(root / "supplement/TRAINING_CURVES_ALL_180000_UPDATES.parquet")
    if (
        len(curves) != 180000
        or curves.fit.nunique() != 72
        or not np.all(curves.groupby("fit").size().to_numpy() == 2500)
    ):
        raise ValueError("72 endpoint curve inventory differs")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                "status": "EXTRACTED_BUNDLE_REGENERATION_VERIFIED",
                "members_hashed": len(manifest["members"]),
                "heads": heads,
                "scores": scores,
                "raw_autonomous": False,
                "expert_forwards": 0,
                "optimizer_updates": 0,
                "scope": (
                    "72 compact heads from included normalized contexts "
                    "and current expert outputs; "
                    "metrics from saved predictions"
                ),
            },
            indent=2,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

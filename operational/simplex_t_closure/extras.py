"""Read-only arithmetic supplement and compact-input export, with zero updates."""

from __future__ import annotations

import argparse
import gc
import json
import sys
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from runtime import atomic_json, digest, publish_json, resumable_hierarchical_losses

from e_jepa_ttc.evaluation.stage63_65 import benchmark_phase, strict_macro_mass
from e_jepa_ttc.simplex_t.training import state_digest

CAMPAIGN = ROOT / "artifacts/simplex_t/scientific_campaign"
OUT = ROOT / "artifacts/simplex_t/closure_20261002/supplement"
ANALYSIS = CAMPAIGN / "T6/CHECKPOINTED_WORK/analysis"
IDENTITY = ["sample_token", "sequence_id", "track_id", "outer_fold", "target_ttc"]


def record(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def check() -> None:
    import psutil

    policy = record(Path(__file__).with_name("resource_policy.json"))
    if (
        psutil.virtual_memory().available < policy["host_available_ram_floor_bytes"]
        or psutil.Process().memory_info().rss > policy["process_tree_rss_ceiling_bytes"]
        or psutil.disk_usage(str(ROOT)).free - 536_870_912
        < policy["written_volume_emergency_floor_bytes"]
    ):
        raise InterruptedError("PAUSED_RESOURCE: closure supplement; outputs preserved")


def frames() -> dict[str, pd.DataFrame]:
    result = {}
    for stage in ("T2", "T3", "T4", "T5"):
        path = CAMPAIGN / "publication" / f"{stage}_PREDICTIONS.json"
        pieces = defaultdict(list)
        for key, row in record(path)["fits"].items():
            payload = path.parent / row["path"]
            if digest(payload) != row["sha256"]:
                raise ValueError(f"publication changed: {payload}")
            frame = pd.read_parquet(payload)
            pieces[key.split("/")[1] + "@" + key.split("seed")[1]].append(frame)
        for name, parts in pieces.items():
            result[name] = pd.concat(parts).sort_values("sample_token").reset_index(drop=True)
    base = result["TPR-D1-H8-C160@7"]
    for name, frame in result.items():
        if not frame[IDENTITY].equals(base[IDENTITY]):
            raise ValueError(f"cohort differs: {name}")
    return result


def arithmetic() -> None:
    check()
    launch = record(CAMPAIGN / "launches/T6.json")
    freeze = record(Path(launch["freeze"]))
    data = frames()
    base = data["TPR-D1-H8-C160@7"]
    baseline_pins = {}
    pieces = defaultdict(list)
    for fold in range(3):
        folder = ROOT / f"artifacts/simplex_t/T1/fixed_baselines_outer{fold}_fp64_emission"
        receipt = record(folder / "BASELINES.json")
        pin = receipt["roles"]["outer_dev"]
        path = folder / pin["path"]
        if (
            digest(path) != pin["sha256"]
            or receipt["outer_fold"] != fold
            or receipt["optimizer_updates"] != 0
            or state_digest(
                {
                    "cache": pin["source_sha256"],
                    "pool": "D0",
                    "history": 8,
                    "control": "NONE",
                    "zero_latent": False,
                }
            )
            != freeze["source_identities"][f"T2/TPR-D0-H8-C160/fold{fold}/seed7"]["outer_dev"]
        ):
            raise ValueError("fixed baseline source binding differs")
        compiled = ROOT / f"artifacts/simplex_t/T1/compiled_context/outer{fold}/COMPILED.json"
        if digest(compiled) != receipt["compiled_sha256"]:
            raise ValueError("baseline compiled source changed")
        metadata = pd.read_csv(
            Path(launch["roots"]["historical"]) / "tables" / f"outer{fold}_outer_dev.csv"
        )
        baseline_pins[str(folder.relative_to(ROOT))] = {
            "receipt_sha256": digest(folder / "BASELINES.json"),
            "payload_sha256": digest(path),
            "source_sha256": pin["source_sha256"],
            "registered_arm_source_sha256": freeze["source_identities"][
                f"T2/TPR-D0-H8-C160/fold{fold}/seed7"
            ]["outer_dev"],
            "binding": "original cache identity wrapped by frozen ArmBinding D0/H8/NONE",
        }
        with np.load(path, allow_pickle=False) as archive:
            if not np.array_equal(archive["sample_token"], metadata.sample_token.to_numpy()):
                raise ValueError("baseline independent query order differs")
            if not np.array_equal(archive["sequence_id"], metadata.sequence_id.to_numpy()):
                raise ValueError("baseline independent sequence order differs")
            aligned = base.set_index("sample_token").loc[metadata.sample_token]
            histories = aligned[[f"history_index{i}" for i in range(8)]].to_numpy()
            if not np.array_equal(histories, archive["history"]):
                raise ValueError("fixed baseline histories differ from source-bound publication")
            for name in ("CURRENT_MEDIAN", "EWMA_0P3S_H8"):
                frame = metadata[IDENTITY[:-2] + ["target_ttc"]].copy()
                frame["outer_fold"] = fold
                frame["prediction_ttc_s"] = archive[f"{name}_prediction_ttc_s"]
                frame["prediction_phase"] = benchmark_phase(frame.prediction_ttc_s.to_numpy())
                frame["target_phase"] = benchmark_phase(frame.target_ttc.to_numpy())
                frame["loss"] = 10000 * np.abs(frame.prediction_phase - frame.target_phase)
                frame["wrong_sign"] = np.sign(frame.prediction_ttc_s) != np.sign(frame.target_ttc)
                pieces[name].append(frame)
    for name, parts in pieces.items():
        frame = pd.concat(parts).sort_values("sample_token").reset_index(drop=True)
        if not frame[IDENTITY].equals(base[IDENTITY]):
            raise ValueError("fixed baseline cohort differs")
        data[name + "@fixed"] = frame
        frame.to_parquet(OUT / f"{name}_OLD.parquet", index=False)
    for name, folder in (
        ("RISK17", "risk17_frozen_replay"),
        ("SIMPLEX17", "simplex17_frozen_replay"),
    ):
        directory = ROOT / "artifacts/simplex_t/T1" / folder
        receipt = record(directory / "REPLAY.json")
        path = directory / (name + "_OLD.parquet")
        if digest(path) != receipt["payload_sha256"]:
            raise ValueError("historical comparator replay changed")
        frame = pd.read_parquet(path).sort_values("sample_token").reset_index(drop=True)
        if not frame[IDENTITY].equals(base[IDENTITY]):
            raise ValueError("historical comparator OLD cohort differs")
        frame["loss"] = 10000 * np.abs(
            benchmark_phase(frame.prediction_ttc_s.to_numpy())
            - benchmark_phase(frame.target_ttc.to_numpy())
        )
        frame["wrong_sign"] = np.sign(frame.prediction_ttc_s) != np.sign(frame.target_ttc)
        data[name + "@7"] = frame
        frame.to_parquet(OUT / (name + "_OLD.parquet"), index=False)
    review = record(ROOT / "artifacts/simplex_t/review_20261002/evidence.json")
    score_oracle = {row["arm_seed"]: row["mid"] for row in review["scores"]}
    evidence_rows = {row["arm_seed"]: row for row in review["scores"]}
    csv_rows = pd.read_csv(
        ROOT / "artifacts/simplex_t/review_20261002/scores.csv", float_precision="round_trip"
    ).set_index("arm_seed")
    metrics, residuals, reconciliation = [], [], []
    for name, frame in data.items():
        check()
        mass = strict_macro_mass(frame.target_ttc.to_numpy(), frame.sequence_id.to_numpy())
        score = float(mass @ frame.loss.to_numpy())
        if name in score_oracle and abs(score - score_oracle[name]) > 1e-10:
            raise ValueError(f"evidence.json numerical mismatch: {name}")
        if name in score_oracle:
            wrong_sign = float(mass @ frame.wrong_sign.to_numpy(float))
            values = {
                "mid": score,
                "weighted_wrong_sign": wrong_sign,
            }
            for metric, actual in values.items():
                csv_value = float(csv_rows.loc[name, metric])
                json_value = float(evidence_rows[name][metric])
                if max(abs(actual - csv_value), abs(actual - json_value)) > 1e-10:
                    raise ValueError(f"physical CSV/JSON mismatch: {name}: {metric}")
                reconciliation.append(
                    {
                        "arm_seed": name,
                        "metric": metric,
                        "actual": actual,
                        "physical_CSV": csv_value,
                        "evidence_JSON": json_value,
                        "CSV_difference": actual - csv_value,
                        "JSON_difference": actual - json_value,
                        "absolute_tolerance": 1e-10,
                    }
                )
        scopes = [("all", "ALL", np.ones(len(frame), bool))]
        scopes += [("fold", str(f), frame.outer_fold.to_numpy() == f) for f in range(3)]
        scopes += [
            ("sequence", str(s), frame.sequence_id.to_numpy() == s)
            for s in sorted(frame.sequence_id.unique())
        ]
        for scope, value, mask in scopes:
            w = mass[mask] / mass[mask].sum()
            metrics.append(
                {
                    "arm_seed": name,
                    "scope": scope,
                    "value": value,
                    "queries": int(mask.sum()),
                    "MiD": float(w @ frame.loss.to_numpy()[mask]),
                    "wrong_sign": float(w @ frame.wrong_sign.to_numpy(float)[mask]),
                }
            )
        if name.startswith(("LATENT", "TPR-D1-H8")):
            for scope, value, scope_mask in scopes:
                for stratum in ("all", "positive_crucial_0_to_3s"):
                    mask = scope_mask.copy()
                    if stratum != "all":
                        mask &= (frame.target_ttc.to_numpy() > 0) & (
                            frame.target_ttc.to_numpy() <= 3
                        )
                    truth, pred = frame.target_ttc.to_numpy(), frame.prediction_ttc_s.to_numpy()
                    positive_pair = (truth > 0) & (pred > 0)
                    labels = {
                        "positive_TTC_overestimate": positive_pair & (pred > truth),
                        "positive_TTC_underestimate": positive_pair & (pred < truth),
                        "wrong_sign": frame.wrong_sign.to_numpy(),
                        "harmful_phase_escape": frame.escape_harm.to_numpy() > 0,
                    }
                    row = {
                        "arm_seed": name,
                        "scope": scope,
                        "value": value,
                        "stratum": stratum,
                        "queries": int(mask.sum()),
                        "global_mass": float(mass[mask].sum()),
                    }
                    for label, selection in labels.items():
                        row[label + "_count"] = int((mask & selection).sum())
                        row[label + "_conditional_mass"] = (
                            float(mass[mask & selection].sum() / mass[mask].sum())
                            if mask.any()
                            else None
                        )
                    row["escape_harm_MiD_global"] = float(
                        mass[mask] @ frame.escape_harm.to_numpy()[mask]
                    )
                    residuals.append(row)
    pd.DataFrame(metrics).to_csv(OUT / "METRICS.csv", index=False, float_format="%.17g")
    pd.DataFrame(residuals).to_csv(
        OUT / "RESIDUAL_DECOMPOSITION.csv", index=False, float_format="%.17g"
    )
    pd.DataFrame(reconciliation).to_csv(
        OUT / "NUMERICAL_RECONCILIATION.csv", index=False, float_format="%.17g"
    )
    selected = {
        name: data[name]
        for name in (
            "TPR-D1-H8-C160@7",
            "TPR-D1-H1-C160@7",
            "CURRENT_MEDIAN@fixed",
            "EWMA_0P3S_H8@fixed",
        )
    }
    from adapters import UNCERTAINTY_DIRECTORY_GUARD, UNCERTAINTY_EXISTS_GUARD, resumable_function

    from e_jepa_ttc.simplex_t import uncertainty_analysis

    def hierarchical(
        frame: pd.DataFrame, losses: np.ndarray, output: Path, callback: Callable[[], None]
    ) -> tuple[np.ndarray, dict]:
        return resumable_hierarchical_losses(
            frame,
            losses,
            output,
            callback,
            draws_path=ANALYSIS / "analyses/T2/paired_uncertainty/HIERARCHICAL_DRAWS.jsonl",
            binding={
                "freeze": launch["freeze_sha256"],
                "fixed_baselines": baseline_pins,
                "recipe": digest(Path(uncertainty_analysis.__file__)),
            },
        )

    paired = resumable_function(
        uncertainty_analysis.paired_uncertainty,
        {
            UNCERTAINTY_EXISTS_GUARD: UNCERTAINTY_DIRECTORY_GUARD,
            "output.mkdir(parents=True)": "output.mkdir(parents=True, exist_ok=True)",
        },
        {"hierarchical_losses": hierarchical, "write_new_json": publish_json},
    )
    result = paired(
        selected,
        reference="EWMA_0P3S_H8@fixed",
        output=OUT / "ewma_uncertainty",
        resource_check=check,
    )
    atomic_json(
        OUT / "ARITHMETIC_VERIFICATION.json",
        {
            "status": "ALL_24_ARM_SEEDS_RECONCILED_TO_CSV_AND_EVIDENCE_PLUS_FIXED_EWMA_ALL_FOLDS",
            "queries": 8192,
            "sequences": 9,
            "folds": 3,
            "optimizer_updates": 0,
            "fixed_baseline_inputs": baseline_pins,
            "ewma_comparisons": result["comparisons"],
            "review_evidence_sha256": digest(
                ROOT / "artifacts/simplex_t/review_20261002/evidence.json"
            ),
            "posthoc_residuals": True,
            "over_under_definition": (
                "TTC direction only for positive target AND positive prediction; "
                "wrong signs separate"
            ),
        },
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["arithmetic", "sources"], required=True)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.mode == "arithmetic":
        arithmetic()
    else:
        from source_export import export_sources

        export_sources()
    gc.collect()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

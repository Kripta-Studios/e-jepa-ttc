"""Low-memory saved-prediction analysis; does not certify canonical T6 admission."""

from __future__ import annotations

import gc
import inspect
import json
import sys
from collections.abc import Callable
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
import psutil
from adapters import UNCERTAINTY_DIRECTORY_GUARD, UNCERTAINTY_EXISTS_GUARD, resumable_function
from runtime import atomic_json, digest, publish_json, resumable_hierarchical_losses

from e_jepa_ttc.evaluation.stage63_65 import benchmark_phase, strict_macro_mass
from e_jepa_ttc.simplex_t import uncertainty_analysis
from e_jepa_ttc.simplex_t.diagnostic_summary import summarize_diagnostics
from e_jepa_ttc.simplex_t.replication_summary import three_seed_losses
from e_jepa_ttc.simplex_t.temporal_diagnostics import sampled_temporal_diagnostics

CAMPAIGN = ROOT / "artifacts/simplex_t/scientific_campaign"
OUT = ROOT / "artifacts/simplex_t/closure_20261002/saved_analysis"
IDENTITY = ["sample_token", "sequence_id", "track_id", "outer_fold", "target_ttc"]


def check() -> None:
    p = psutil.Process()
    snapshot = {
        "host_available_bytes": psutil.virtual_memory().available,
        "process_tree_rss_bytes": p.memory_info().rss,
        "written_volume_free_bytes": psutil.disk_usage(str(ROOT)).free,
    }
    if (
        snapshot["host_available_bytes"] < 2 * 1024**3
        or snapshot["process_tree_rss_bytes"] > 4 * 1024**3
        or snapshot["written_volume_free_bytes"] - 67_108_864 < 10_000_000_000
    ):
        atomic_json(OUT / "RESOURCE_PAUSE.json", snapshot)
        raise InterruptedError("PAUSED_RESOURCE: saved analysis")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    check()
    if "torch" in sys.modules:
        raise RuntimeError("saved arithmetic worker must not import PyTorch")
    launch = json.loads((CAMPAIGN / "launches/T6.json").read_text(encoding="utf-8"))
    review = json.loads(
        (ROOT / "artifacts/simplex_t/review_20261002/evidence.json").read_text(encoding="utf-8")
    )
    evidence_scores = {row["arm_seed"]: row["mid"] for row in review["scores"]}
    freeze = json.loads(Path(launch["freeze"]).read_text(encoding="utf-8"))
    if digest(Path(launch["freeze"])) != launch["freeze_sha256"]:
        raise ValueError("scientific freeze changed")
    pins = {}

    def pin(path: Path, expected: str) -> None:
        if digest(path) != expected:
            raise ValueError(f"saved input changed: {path}")
        pins[str(path.relative_to(ROOT))] = expected

    # Independently frozen OLD targets; no raw streams or protected groups.
    tables = []
    for fold in range(3):
        path = Path(launch["roots"]["historical"]) / "tables" / f"outer{fold}_outer_dev.csv"
        expected = next(
            p["sha256"]
            for p in freeze["files"]
            if p["root"] == "historical"
            and p["relative_path"] == f"tables/outer{fold}_outer_dev.csv"
        )
        if digest(path) != expected:
            raise ValueError(f"independent OLD targets changed: {path}")
        table = pd.read_csv(path)
        table["outer_fold"] = fold
        tables.append(table[IDENTITY])
    cohort = pd.concat(tables).sort_values("sample_token").reset_index(drop=True)
    if len(cohort) != 8192 or cohort.sample_token.duplicated().any():
        raise ValueError("OLD cohort incomplete")
    publications = {}
    for stage, binding in launch["publications"].items():
        path = Path(binding["publication"])
        pin(path, binding["publication_sha256"])
        endpoints = Path(binding["endpoints"])
        pin(endpoints, binding["endpoints_sha256"])
        record = json.loads(endpoints.read_text(encoding="utf-8"))
        for endpoint in record["fits"]:
            checkpoint = Path(binding["checkpoint_root"]) / endpoint["checkpoint"]
            pin(checkpoint, endpoint["checkpoint_sha256"])
            if (
                endpoint["train_source_sha256"]
                != freeze["source_identities"][endpoint["key"]]["inner_oof"]
            ):
                raise ValueError("TRAIN source seal changed")
        publications[stage] = json.loads(path.read_text(encoding="utf-8"))

    def load_arm(stage: str, name: str, seed: int) -> pd.DataFrame:
        parts = []
        for key, row in publications[stage]["fits"].items():
            if f"/{name}/" not in key or not key.endswith(f"/seed{seed}"):
                continue
            path = CAMPAIGN / "publication" / row["path"]
            pin(path, row["sha256"])
            frame = pd.read_parquet(path)
            if not frame.source_sha256.eq(freeze["source_identities"][key]["outer_dev"]).all():
                raise ValueError("OLD prediction source seal changed")
            parts.append(frame)
        frame = pd.concat(parts).sort_values("sample_token").reset_index(drop=True)
        if len(parts) != 3 or not frame[IDENTITY].equals(cohort):
            raise ValueError("saved prediction query/target binding changed")
        if name.startswith("SELECTOR"):
            phase = -np.log1p(-0.1 / frame.prediction_ttc_s.to_numpy(np.float64))
            phase[np.isinf(frame.prediction_ttc_s)] = 0
        else:
            phase = benchmark_phase(frame.prediction_ttc_s.to_numpy(np.float64))
        loss = 10000 * np.abs(phase - benchmark_phase(frame.target_ttc.to_numpy(np.float64)))
        if not np.array_equal(loss, frame.loss.to_numpy()) or not np.array_equal(
            phase, frame.prediction_phase.to_numpy()
        ):
            raise ValueError(f"emitted-TTC arithmetic differs: {name}@{seed}")
        return frame

    draws = (
        CAMPAIGN
        / "T6/CHECKPOINTED_WORK/analysis/analyses/T2/paired_uncertainty/HIERARCHICAL_DRAWS.jsonl"
    )
    binding = {
        "scientific_freeze_sha256": launch["freeze_sha256"],
        "publications": {
            stage: {k: row[k] for k in ("publication_sha256", "endpoints_sha256")}
            for stage, row in launch["publications"].items()
        },
        "recipe_sha256": digest(Path(inspect.getfile(uncertainty_analysis))),
        "historical_recipe_sha256": digest(ROOT / "src/e_jepa_ttc/evaluation/risk_geometry_v10.py"),
    }

    def hierarchical(
        frame: pd.DataFrame, losses: np.ndarray, output: Path, callback: Callable[[], None]
    ) -> tuple[np.ndarray, dict]:
        return resumable_hierarchical_losses(
            frame, losses, output, callback, draws_path=draws, binding=binding
        )

    paired = resumable_function(
        uncertainty_analysis.paired_uncertainty,
        {
            UNCERTAINTY_EXISTS_GUARD: UNCERTAINTY_DIRECTORY_GUARD,
            "output.mkdir(parents=True)": "output.mkdir(parents=True, exist_ok=True)",
        },
        {"hierarchical_losses": hierarchical, "write_new_json": publish_json},
    )
    scores = []
    loss_columns = {}
    for stage in ("T2", "T3", "T4", "T5"):
        names = sorted(
            {(key.split("/")[1], int(key.split("/seed")[1])) for key in publications[stage]["fits"]}
        )
        for name, seed in names:
            check()
            frame = load_arm(stage, name, seed)
            mass = strict_macro_mass(frame.target_ttc.to_numpy(), frame.sequence_id.to_numpy())
            score = float(mass @ frame.loss.to_numpy())
            expected = (
                pd.read_csv(ROOT / "artifacts/simplex_t/review_20261002/scores.csv")
                .set_index("arm_seed")
                .loc[f"{name}@{seed}", "mid"]
            )
            if abs(score - expected) > 1e-10:
                raise ValueError(f"review CSV mismatch: {name}@{seed}")
            if abs(score - evidence_scores[f"{name}@{seed}"]) > 1e-10:
                raise ValueError(f"review evidence.json mismatch: {name}@{seed}")
            scores.append(
                {
                    "arm_seed": f"{name}@{seed}",
                    "mid": score,
                    "review_csv_mid": float(expected),
                    "delta": score - expected,
                }
            )
            loss_columns[f"{name}@{seed}"] = frame.loss.to_numpy().copy()
            del frame
            gc.collect()
    pd.DataFrame(scores).to_csv(OUT / "SCORE_RECONCILIATION.csv", index=False, float_format="%.17g")
    for stage, groups in (
        (
            "T4",
            [
                (
                    7,
                    ["LATENT-D1-H1-C160", "LATENT-D1-H8-C160", "LATENT_ZERO-D1-H8-C160"],
                    "latent_seed7",
                )
            ],
        ),
        (
            "T5",
            [(seed, ["TPR-D1-H1-C160", "TPR-D1-H8-C160"], f"TPR_seed{seed}") for seed in (13, 23)],
        ),
    ):
        for seed, names, group in groups:
            check()
            frames = {name: load_arm(stage, name, seed) for name in names}
            target = CAMPAIGN / "T6/CHECKPOINTED_WORK/analysis/analyses" / stage / group
            paired(frames, reference=names[0], output=target, resource_check=check)
            pd.concat(
                [
                    summarize_diagnostics(frame, history_length=int(name.split("-")[2][1:]))
                    for name, frame in frames.items()
                ],
                ignore_index=True,
            ).to_parquet(target / "DIAGNOSTICS.parquet", index=False)
            temporal = [sampled_temporal_diagnostics(frame) for frame in frames.values()]
            for pos, name in enumerate(
                ("TEMPORAL_QUERY_DIAGNOSTICS.parquet", "TEMPORAL_STRATA.parquet")
            ):
                pd.concat([r[pos] for r in temporal], ignore_index=True).to_parquet(
                    target / name, index=False
                )
            print(
                json.dumps(
                    {
                        "status": "SAVED_ANALYSIS_GROUP_DURABLE_NOT_CANONICAL_ADMISSION",
                        "stage": stage,
                        "group": group,
                        "draws": 8192,
                    }
                ),
                flush=True,
            )
            del frames, temporal
            gc.collect()
    selected = {
        (name, seed): load_arm("T2" if seed == 7 else "T5", name, seed)
        for name in ("TPR-D1-H1-C160", "TPR-D1-H8-C160")
        for seed in (7, 13, 23)
    }
    averaged, summary = three_seed_losses(selected, family="TPR", pool="D1")
    del selected
    gc.collect()
    destination = CAMPAIGN / "T6/CHECKPOINTED_WORK/analysis/three_seed/TPR"
    destination.mkdir(parents=True, exist_ok=True)
    pd.concat(
        [frame.assign(loss_series=name) for name, frame in averaged.items()], ignore_index=True
    ).to_parquet(destination / "MEAN_LOSSES_NOT_TTC_PREDICTIONS.parquet", index=False)
    paired(
        averaged,
        reference="TPR-D1-H1-C160",
        output=destination / "uncertainty",
        resource_check=check,
    )
    atomic_json(
        OUT / "SUMMARY.json",
        {
            "status": "SAVED_PREDICTIONS_NUMERICALLY_RECONCILED_CANONICAL_T6_PENDING",
            "optimizer_updates_executed": 0,
            "three_seed_summary": summary,
            "review_evidence_sha256": digest(
                ROOT / "artifacts/simplex_t/review_20261002/evidence.json"
            ),
            "input_hashes": pins,
            "canonical_admission_complete": False,
            "torch_imported": "torch" in sys.modules,
            "bootstrap_groups": 4,
            "valid_draws_per_group": 8192,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

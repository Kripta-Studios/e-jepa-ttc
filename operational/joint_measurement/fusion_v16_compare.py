"""Supplementary same-cohort contrast with V16's already frozen PHASE_ONLY family."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from operational.range_transfer.analyze import gate, metrics
from operational.range_transfer.common import freeze, read, sha, write

from .bootstrap import compare
from .common import ROOT, owner
from .data import role_rows
from .fusion_replication_evaluate import loss_bootstrap, mean_metrics
from .fusion_replication_train import DIRECTORY as REPLICAS
from .fusion_train import DIRECTORY as FUSION
from .fusion_train import freeze_endpoints
from .train import committed_sources
from .verify_predictions import verify

V16 = ROOT.parent / "e-jepa-ttc-v16-multimodal-validation/artifacts/multimodal_research_20261012"
OUT = FUSION / "v16_compare"


def admit() -> dict:
    """Preserve V16 checkpoints/prediction bytes; do not reuse its looser historical gates."""
    freeze_endpoints()
    sources = {"operational/joint_measurement/fusion_v16_compare.py": sha(Path(__file__))}
    committed_sources(sources)
    originals = {}
    for seed in (7, 13, 23):
        study = "PHASE_ONLY" if seed == 7 else "PHASE_ONLY_REPLICATION"
        folder = V16 / "studies" / study / f"S{seed}"
        seal = read(folder / "V_PREDICTIONS_SEALED.json")
        path = folder / "V_PREDICTIONS_UNSCORED.csv"
        if sha(path) != seal["sha256"] or seal["targets_read_for_scoring"]:
            raise ValueError("V16 sealed predictions changed")
        for arm, receipt in seal["endpoints"].items():
            if (
                receipt["status"] != "COMPLETE"
                or not receipt["scientific_endpoint"]
                or receipt["completed_updates"] != 2500
                or receipt["accumulation_index"] != 0
                or sha(Path(receipt["checkpoint_path"])) != receipt["checkpoint_sha256"]
            ):
                raise ValueError(f"V16 frozen {arm} endpoint changed")
        originals[str(seed)] = {
            "path": str(path),
            "predictions_sha256": seal["sha256"],
            "seal_sha256": sha(folder / "V_PREDICTIONS_SEALED.json"),
            "endpoints": seal["endpoints"],
        }
    manifest = read(owner() / "V_MANIFEST.json")
    if sha(Path(manifest["rows_path"])) != manifest["rows_sha256"]:
        raise ValueError("Canonical native V targets changed")
    registration = {
        "sources": sources,
        "V16": originals,
        "manifest_sha256": sha(owner() / "V_MANIFEST.json"),
        "V18_replication_registration_sha256": sha(REPLICAS / "REGISTRATION.json"),
        "purpose": (
            "Supplementary descriptive contrast after V18 recipe/replica "
            "selection; never selection feedback"
        ),
        "V16_scope": ("Previously trained frozen phase heads; no new training or official score"),
        "target_policy": (
            "Attach unchanged native V targets to sealed query tokens; do "
            "not import rescored V16 label roundoff"
        ),
        "gates": (
            "Original V18 system gate shown descriptively; historical V16 "
            "2% decision is not retroactively altered"
        ),
        "official_comparison": False,
    }
    freeze(OUT / "REGISTRATION.json", registration)
    return registration


def run(*, include_replicas: bool = False) -> dict:
    """Recalculate frozen historical outputs under the identical current scorer and query set."""
    record = admit()
    rows = role_rows("V")
    frames, scores, proofs = {}, {}, {}
    for seed in (7, 13, 23):
        entry = record["V16"][str(seed)]
        # Native pandas parser supports round_trip; installed stubs omit this literal.
        source = pd.read_csv(entry["path"], float_precision="round_trip")  # pyright: ignore[reportCallIssue, reportArgumentType]
        if not source.sample_token.is_unique or not np.array_equal(
            source.sample_token, rows.sample_token
        ):
            raise ValueError("V16 prediction population/order differs from V18")
        frame = rows[["sample_token", "group_id", "target_ttc"]].copy()
        frame["prediction_ttc"] = source.R_ANCHOR_EVENT.to_numpy(dtype=np.float64)
        name = f"V16-PHASE-ONLY-S{seed}"
        folder = OUT / "predictions/V" / name
        freeze(
            folder / "BINDING.json",
            {
                "registration_sha256": sha(OUT / "REGISTRATION.json"),
                "original_predictions_sha256": entry["predictions_sha256"],
            },
        )
        path = folder / "predictions.parquet"
        if path.exists():
            pd.testing.assert_frame_equal(pd.read_parquet(path), frame)
        else:
            frame.to_parquet(path, index=False)
        result = metrics(
            frame.target_ttc.to_numpy(), frame.prediction_ttc.to_numpy(), frame.group_id.to_numpy()
        )
        freeze(folder / "METRICS.json", result)
        freeze(
            folder / "COMPLETE.json",
            {"status": "COMPLETE", "rows": len(frame), "sha256": sha(path)},
        )
        proofs[str(seed)] = verify(folder)
        frames[seed], scores[seed] = frame, result
    candidate_path = FUSION / "predictions/V/FUSION-TRUE-S7/predictions.parquet"
    if sha(candidate_path) != read(candidate_path.parent / "COMPLETE.json")["sha256"]:
        raise ValueError("Frozen V18 seed7 output changed")
    candidate = pd.read_parquet(candidate_path)
    c_score = read(candidate_path.parent / "METRICS.json")
    comparison = compare(
        OUT / "predictions/V/V16-PHASE-ONLY-S7/predictions.parquet",
        candidate_path,
        OUT / "bootstrap/seed7",
    )
    result = {
        "status": "COMPLETE_DESCRIPTIVE_SEED7",
        "reference_metrics": scores,
        "candidate_seed7_metrics": c_score,
        "gate_seed7": gate(scores[7], c_score),
        "bootstrap_seed7": comparison,
        "verification": proofs,
        "selection_feedback_allowed": False,
        "system_promotion": False,
    }
    if include_replicas:
        result["status"] = "COMPLETE_DESCRIPTIVE_THREE_SEEDS"
        candidates, candidate_scores = [candidate], [c_score]
        for seed in (13, 23):
            folder = REPLICAS / "predictions/V" / f"FUSION-TRUE-S{seed}"
            if sha(folder / "predictions.parquet") != read(folder / "COMPLETE.json")["sha256"]:
                raise ValueError("V18 replica prediction changed")
            candidates.append(pd.read_parquet(folder / "predictions.parquet"))
            candidate_scores.append(read(folder / "METRICS.json"))
        result["reference_mean"] = mean_metrics(list(scores.values()))
        result["candidate_mean"] = mean_metrics(candidate_scores)
        result["gate_seed_mean"] = gate(result["reference_mean"], result["candidate_mean"])
        result["group_bootstrap_seed_mean_loss"] = loss_bootstrap(list(frames.values()), candidates)
    write(OUT / ("THREE_SEEDS.json" if include_replicas else "SEED7.json"), result)
    return {
        "status": result["status"],
        "V16_seed7_aggregate": scores[7]["overall_MiD"],
        "V18_seed7_aggregate": c_score["overall_MiD"],
        "gate_seed7": result["gate_seed7"],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--include-replicas", action="store_true")
    args = parser.parse_args()
    print(run(include_replicas=args.include_replicas))

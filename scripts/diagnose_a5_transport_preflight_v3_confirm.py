#!/usr/bin/env python
"""A5-PREFLIGHT-V3 confirmation on train rows disjoint from V2 discovery.

V2 discovered two facts that must remain separate:

1. real temporal DINO-relation pairs have much lower *absolute post-match error*
   than same-sequence shuffled and spatial-roll nulls; and
2. the A4 soft expected flow at r=1, tau=0.02 had the lowest train-only physical
   EPE of the preregistered V2 r x tau grid, while hard r=4 flow was harmful.

V2 nevertheless rejected all radii because it compared *relative error
reductions*.  The nulls start from much larger fixed errors, so that statistic is
not a fair best-of-K control.  V3 does not rewrite or re-score V2.  It freezes
exactly one candidate (r=1, tau=0.02) and confirms it on the complementary train
indices that V2 never inspected.  Validation/test remain unopened and no
optimizer step is taken.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
import diagnose_a5_transport_preflight_v2 as v2

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_CONFIG = v2.DEFAULT_SOURCE_CONFIG
DEFAULT_PROTOCOL = ROOT / "configs/experiment/e_jepa_garl_event_causal_scale_a5_preflight_v3_confirm.yaml"
DEFAULT_V2_ARTIFACT = ROOT / "artifacts/metrics/a5_transport_preflight_v2/a5_transport_preflight_v2.json"
DEFAULT_A4 = v2.DEFAULT_A4


def _mean(values: list[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return float(arr.mean()) if arr.size else float("nan")


def _sequence_specificity(
    rows: list[dict[str, Any]],
    *,
    kind: str,
    teacher_shuffled_min: float,
    teacher_spatial_min: float,
) -> tuple[dict[str, Any], float]:
    selected = [r for r in rows if r.get("kind") == kind]
    by_seq: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in selected:
        by_seq[str(row["sequence_id"])].append(row)
    result: dict[str, Any] = {}
    passes = 0
    for sequence, seq_rows in sorted(by_seq.items()):
        if kind == "teacher":
            real = _mean([float(r["real_best_error"]) for r in seq_rows])
            shuffled = _mean([float(r["shuffled_best_error"]) for r in seq_rows])
            spatial = _mean([float(r["spatial_null_best_error"]) for r in seq_rows])
            vs_shuffled = (shuffled - real) / shuffled if shuffled > 0 else float("nan")
            vs_spatial = (spatial - real) / spatial if spatial > 0 else float("nan")
            passed = bool(vs_shuffled >= teacher_shuffled_min and vs_spatial >= teacher_spatial_min)
            result[sequence] = {
                "real_best_error": real,
                "shuffled_best_error": shuffled,
                "spatial_null_best_error": spatial,
                "real_vs_shuffled_improvement": vs_shuffled,
                "real_vs_spatial_null_improvement": vs_spatial,
                "passed": passed,
            }
        else:
            real = _mean([float(r["real_top1_cosine"]) for r in seq_rows])
            shuffled = _mean([float(r["shuffled_top1_cosine"]) for r in seq_rows])
            spatial = _mean([float(r["spatial_null_top1_cosine"]) for r in seq_rows])
            passed = bool(real > shuffled and real > spatial)
            result[sequence] = {
                "real_top1_cosine": real,
                "shuffled_top1_cosine": shuffled,
                "spatial_null_top1_cosine": spatial,
                "passed": passed,
            }
        passes += int(passed)
    fraction = float(passes / len(result)) if result else 0.0
    return result, fraction


def _load_v2_discovery(path: Path, candidate_radius: int, candidate_tau: float) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if payload.get("artifact_type") != "a5_transport_preflight_train_only_v2":
        raise ValueError("V3 requires the immutable A5-PREFLIGHT-V2 artifact")
    scope = payload.get("scope", {})
    if scope.get("public_train_only") is not True or scope.get("validation_or_test_opened") is not False:
        raise ValueError("V2 artifact violated train-only contract")
    if int(scope.get("optimizer_steps", -1)) != 0:
        raise ValueError("V2 artifact contains optimizer steps")

    temp_name = payload.get("files", {}).get("temperature_csv")
    if not temp_name:
        raise ValueError("V2 artifact does not reference its temperature CSV")
    temp_path = path.parent / str(temp_name)
    frame = pd.read_csv(temp_path)
    a4 = frame[frame["model"] == "A4"].copy()
    if a4.empty:
        raise ValueError("V2 temperature CSV lacks A4 rows")
    best = a4.sort_values(
        ["soft_physical_epe", "radius", "temperature"], ascending=[True, True, False]
    ).iloc[0]
    if int(best["radius"]) != candidate_radius or not np.isclose(float(best["temperature"]), candidate_tau):
        raise ValueError(
            "Frozen V3 candidate is not the minimum-A4-soft-EPE point in the supplied V2 artifact: "
            f"observed r={int(best['radius'])}, tau={float(best['temperature'])}"
        )
    return {
        "payload": payload,
        "file_sha256": v2._sha256(path),
        "temperature_csv": temp_path,
        "temperature_csv_sha256": v2._sha256(temp_path),
        "discovery_best_soft_physical_epe": float(best["soft_physical_epe"]),
        "discovery_best_soft_epe_improvement_over_zero": float(best["soft_physical_epe_improvement_over_zero"]),
    }


def run(
    *,
    source_config_path: Path,
    protocol_path: Path,
    v2_artifact_path: Path,
    a4_checkpoint_path: Path,
    output_dir: Path,
    device_name: str,
    batch_size: int,
) -> dict[str, Any]:
    source = v2._read_yaml(source_config_path)
    protocol_root = v2._read_yaml(protocol_path)
    protocol = protocol_root.get("protocol")
    if not isinstance(protocol, dict):
        raise ValueError("V3 protocol YAML lacks protocol mapping")
    if protocol.get("public_train_only") is not True or protocol.get("validation_or_test_opened") is not False:
        raise ValueError("V3 refuses non-train-only protocol")
    if int(protocol.get("optimizer_steps", -1)) != 0:
        raise ValueError("V3 must have zero optimizer steps")

    candidate = protocol["candidate"]
    radius = int(candidate["radius"])
    tau = float(candidate["temperature"])
    if (radius, tau) != (1, 0.02):
        raise ValueError("V3 confirmation candidate is frozen to r=1, tau=0.02")
    discovery = _load_v2_discovery(v2_artifact_path, radius, tau)

    data = source.get("data")
    if not isinstance(data, dict):
        raise ValueError("source config lacks data mapping")
    if any(data.get(key) is not False for key in ("official_test_opened", "codabench_opened", "evttc_test_opened")):
        raise ValueError("V3 refuses any config authorizing private/test access")

    v2.seed_everything(int(protocol["seed"]), deterministic=True)
    device = v2.resolve_device(device_name)
    cache_manifest = (ROOT / str(data["cache_manifest"])).resolve(strict=True)
    teacher_cfg = data.get("dinov3_relational_teacher")
    if not isinstance(teacher_cfg, dict):
        raise ValueError("source config lacks DINO teacher")
    teacher_manifest = (ROOT / str(teacher_cfg["manifest"])).resolve(strict=True)
    base = v2.GarlTTCObjectEventV4Dataset(str(cache_manifest), splits=("train",))
    dataset = v2.DINOv3RelationalTeacherDataset(
        base,
        manifest_path=teacher_manifest,
        expected_artifact_sha256=str(teacher_cfg["artifact_sha256"]),
        expected_manifest_sha256=str(teacher_cfg["manifest_sha256"]),
    )

    discovery_count = int(protocol["discovery_v2_sample_count"])
    v2_count = int(discovery["payload"]["scope"]["samples"])
    if v2_count != discovery_count:
        raise ValueError(f"V2 sample count drifted: expected {discovery_count}, observed {v2_count}")
    v2_indices = set(v2._selected_indices(len(dataset), discovery_count))
    confirm_indices = [idx for idx in range(len(dataset)) if idx not in v2_indices]
    if not confirm_indices:
        raise RuntimeError("no train rows remain for disjoint V3 confirmation")
    if set(confirm_indices) & v2_indices:
        raise RuntimeError("V3 confirmation is not disjoint from V2 discovery")

    loader = DataLoader(
        Subset(dataset, confirm_indices), batch_size=batch_size, shuffle=False,
        num_workers=0, collate_fn=v2.collate_object_event_v4,
    )
    bank = v2._collect_metadata_and_teacher(loader)
    partner, different_track_fraction = v2._same_sequence_null_partners(
        bank["sequence_ids"], bank["track_ids"], confirm_indices
    )
    null_fraction = float((partner >= 0).mean())
    if null_fraction < 0.95:
        raise RuntimeError(f"same-sequence null coverage too low: {null_fraction:.3f}")

    teacher_summary, rows = v2._teacher_audit(
        bank, partner, device=device, radii=(radius,), batch_size=batch_size
    )
    a4_model, a4_checkpoint = v2._load_checkpoint_model(a4_checkpoint_path, device)
    random_model = v2._random_model_like(a4_checkpoint, device, seed=7001)

    students: dict[str, Any] = {}
    temperature_rows: list[dict[str, Any]] = []
    all_rows = list(rows)
    for name, model in (("A4", a4_model), ("RANDOM", random_model)):
        features = v2._collect_student_features(model, loader, device)
        summary, temps, student_rows = v2._student_audit(
            name, features, bank, partner, device=device,
            radii=(radius,), temperatures=(tau,), batch_size=batch_size,
        )
        students[name] = summary
        temperature_rows.extend(temps)
        all_rows.extend(student_rows)
        del features
        if device.type == "cuda":
            torch.cuda.empty_cache()

    thresholds = protocol["confirmation"]
    teacher_row = teacher_summary[str(radius)]
    a4_row = students["A4"][str(radius)]
    random_row = students["RANDOM"][str(radius)]
    a4_temp = next(r for r in temperature_rows if r["model"] == "A4")
    random_temp = next(r for r in temperature_rows if r["model"] == "RANDOM")

    teacher_seq, teacher_seq_fraction = _sequence_specificity(
        all_rows, kind="teacher",
        teacher_shuffled_min=float(thresholds["teacher_real_vs_shuffled_best_error_improvement_min"]),
        teacher_spatial_min=float(thresholds["teacher_real_vs_spatial_null_best_error_improvement_min"]),
    )
    student_rows_only = [r for r in all_rows if r.get("kind") == "student" and r.get("model") == "A4"]
    student_seq, student_seq_fraction = _sequence_specificity(
        student_rows_only, kind="student", teacher_shuffled_min=0.0, teacher_spatial_min=0.0
    )

    soft_improvement = float(a4_temp["soft_physical_epe_improvement_over_zero"])
    random_soft_improvement = float(random_temp["soft_physical_epe_improvement_over_zero"])
    checks = {
        "teacher_real_vs_shuffled_absolute_post_match": float(teacher_row["real_vs_shuffled_best_error_improvement"]) >= float(thresholds["teacher_real_vs_shuffled_best_error_improvement_min"]),
        "teacher_real_vs_spatial_null_absolute_post_match": float(teacher_row["real_vs_spatial_null_best_error_improvement"]) >= float(thresholds["teacher_real_vs_spatial_null_best_error_improvement_min"]),
        "teacher_sequence_specificity": teacher_seq_fraction >= float(thresholds["teacher_sequence_specificity_fraction_min"]),
        "student_soft_physics": soft_improvement >= float(thresholds["student_soft_epe_improvement_over_zero_min"]),
        "student_soft_physics_over_random": (soft_improvement - random_soft_improvement) >= float(thresholds["student_soft_epe_advantage_over_random_min"]),
        "student_temporal_specificity": float(a4_row["real_minus_shuffled_top1_cosine"]) >= float(thresholds["student_real_vs_shuffled_top1_cosine_min"]),
        "student_spatial_null_specificity": float(a4_row["real_minus_spatial_null_top1_cosine"]) >= float(thresholds["student_real_vs_spatial_null_top1_cosine_min"]),
        "student_confidence_margin": float(a4_row["real_margin"]) >= float(thresholds["student_confidence_margin_min"]),
        "student_sequence_temporal_specificity": student_seq_fraction >= float(thresholds["student_sequence_temporal_specificity_fraction_min"]),
        "local_physical_coverage": float(teacher_row["bbox_physical_coverage"]) >= float(thresholds["bbox_physical_coverage_min"]),
        "boundary_not_degenerate": float(a4_row["hard_boundary_fraction"]) <= float(thresholds["hard_boundary_fraction_max"]),
        "vertical_scale_signal": float(a4_row["divergence_y_vs_log_height_pearson"]) >= float(thresholds["divergence_y_vs_log_height_pearson_min"]),
    }
    authorized = all(checks.values())
    failed = [name for name, passed in checks.items() if not passed]
    decision = {
        "a5_corr_authorized": authorized,
        "selected_radius": radius if authorized else None,
        "selected_temperature": tau if authorized else None,
        "candidate_radius": radius,
        "candidate_temperature": tau,
        "checks": checks,
        "failed_checks": failed,
        "teacher_sequence_specificity_fraction": teacher_seq_fraction,
        "student_sequence_temporal_specificity_fraction": student_seq_fraction,
        "a4_soft_physical_epe_improvement_over_zero": soft_improvement,
        "random_soft_physical_epe_improvement_over_zero": random_soft_improvement,
        "a4_soft_epe_advantage_over_random": soft_improvement - random_soft_improvement,
        "next_action": "AUTHORIZE_A5_CORR_R1_T002" if authorized else "STOP_CONFIRMATION_FAILED",
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    teacher_csv = output_dir / "a5_preflight_v3_teacher.csv"
    student_csv = output_dir / "a5_preflight_v3_students.csv"
    temp_csv = output_dir / "a5_preflight_v3_temperature.csv"
    sequence_csv = output_dir / "a5_preflight_v3_sequences.csv"
    pd.DataFrame([{"radius": radius, **teacher_row}]).to_csv(teacher_csv, index=False, lineterminator="\n")
    pd.DataFrame([
        {"model": name, "radius": radius, **students[name][str(radius)]}
        for name in ("A4", "RANDOM")
    ]).to_csv(student_csv, index=False, lineterminator="\n")
    pd.DataFrame(temperature_rows).to_csv(temp_csv, index=False, lineterminator="\n")
    seq_rows: list[dict[str, Any]] = []
    for seq, values in teacher_seq.items():
        seq_rows.append({"kind": "teacher", "sequence_id": seq, **values})
    for seq, values in student_seq.items():
        seq_rows.append({"kind": "student_A4", "sequence_id": seq, **values})
    pd.DataFrame(seq_rows).to_csv(sequence_csv, index=False, lineterminator="\n")

    payload: dict[str, Any] = {
        "artifact_type": "a5_transport_preflight_train_only_v3_confirmation",
        "created_at": datetime.now(UTC).isoformat(),
        "scope": {
            "public_train_only": True,
            "validation_or_test_opened": False,
            "optimizer_steps": 0,
            "dataset_rows": len(dataset),
            "v2_discovery_rows": len(v2_indices),
            "v3_confirmation_rows": len(confirm_indices),
            "v2_v3_index_overlap": 0,
            "sequence_count": len(set(bank["sequence_ids"])),
            "same_sequence_null_fraction": null_fraction,
            "null_different_track_fraction": different_track_fraction,
        },
        "source": {
            "source_config": str(source_config_path.relative_to(ROOT)),
            "source_config_sha256": v2._sha256(source_config_path),
            "protocol": str(protocol_path.relative_to(ROOT)),
            "protocol_sha256": v2._sha256(protocol_path),
            "v2_artifact": str(v2_artifact_path.relative_to(ROOT)),
            "v2_artifact_sha256": discovery["payload"].get("artifact_sha256"),
            "v2_file_sha256": discovery["file_sha256"],
            "v2_temperature_csv_sha256": discovery["temperature_csv_sha256"],
            "a4_checkpoint": str(a4_checkpoint_path.relative_to(ROOT)),
            "a4_checkpoint_sha256": v2._sha256(a4_checkpoint_path),
            "teacher_manifest": str(teacher_manifest.relative_to(ROOT)),
            "teacher_manifest_sha256": v2._sha256(teacher_manifest),
        },
        "discovery_contract": {
            "candidate_selection_rule": "minimum A4 soft physical EPE over the preregistered V2 r x tau grid",
            "candidate_radius": radius,
            "candidate_temperature": tau,
            "discovery_best_soft_physical_epe": discovery["discovery_best_soft_physical_epe"],
            "discovery_best_soft_epe_improvement_over_zero": discovery["discovery_best_soft_epe_improvement_over_zero"],
            "no_candidate_reselection_in_v3": True,
        },
        "teacher": teacher_row,
        "students": {name: students[name][str(radius)] for name in ("A4", "RANDOM")},
        "temperature": {r["model"]: r for r in temperature_rows},
        "sequence_confirmation": {"teacher": teacher_seq, "student_A4": student_seq},
        "decision": decision,
        "interpretation_contract": protocol.get("interpretation", {}),
        "files": {
            "teacher_csv": teacher_csv.name,
            "teacher_csv_sha256": v2._sha256(teacher_csv),
            "student_csv": student_csv.name,
            "student_csv_sha256": v2._sha256(student_csv),
            "temperature_csv": temp_csv.name,
            "temperature_csv_sha256": v2._sha256(temp_csv),
            "sequence_csv": sequence_csv.name,
            "sequence_csv_sha256": v2._sha256(sequence_csv),
        },
    }
    v2.sign_artifact(payload)
    v2._atomic_json(output_dir / "a5_transport_preflight_v3_confirm.json", payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-config", type=Path, default=DEFAULT_SOURCE_CONFIG)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--v2-artifact", type=Path, default=DEFAULT_V2_ARTIFACT)
    parser.add_argument("--a4-checkpoint", type=Path, default=DEFAULT_A4)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    payload = run(
        source_config_path=args.source_config.resolve(),
        protocol_path=args.protocol.resolve(),
        v2_artifact_path=args.v2_artifact.resolve(),
        a4_checkpoint_path=args.a4_checkpoint.resolve(),
        output_dir=args.output_dir.resolve(),
        device_name=args.device,
        batch_size=args.batch_size,
    )
    print(json.dumps(payload["decision"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

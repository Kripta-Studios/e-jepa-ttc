#!/usr/bin/env python
"""Freeze the V8 train-only temporal-recovery protocol and seed-7 configs.

This script intentionally reads only signed V5/V7 train-only contracts, train-only
OOF predictions, and local checkpoints.  It never opens a validation or test split.
The generated JSON is canonical and contains no wall-clock timestamp so a repeated
freeze is byte-for-byte stable while its signed inputs are unchanged.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any

import yaml

if str(Path(__file__).resolve().parents[1] / "src") not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from e_jepa_ttc.data.scientific_recovery_v8 import (
    EXP6_ALPHAS,
    EXP6_CONFIG_SOURCE_PATH,
    EXP6_CONFIG_SOURCE_SHA256,
    EXP6_INTERNAL_DT_MS,
    EXP6_OFFICIAL_COMMIT_SHA,
    EXP6_OUTPUT_INTERVAL_MS,
    EXP6_OUTPUT_TIME_BINS,
    EXP6_PROCESSOR_SOURCE_PATH,
    EXP6_PROCESSOR_SOURCE_SHA256,
    EXP6_RASTER_CONTRACT,
)
from e_jepa_ttc.models.causal_scale_ttc import CausalScaleTTC, CausalScaleTTCConfig

ROOT = Path(__file__).resolve().parents[1]
V5_PROTOCOL = Path("configs/protocol/scientific_recovery_v5_train_only_grouped_dev.json")
V7_PROTOCOL = Path("configs/protocol/scientific_recovery_v7_balanced_oof.json")
V7_BASELINES = Path("artifacts/scientific_recovery_v7/baselines/manifest.json")
A5_OOF = Path("artifacts/scientific_recovery_v7/baselines/a5_oof_predictions.csv")
GARL_OOF = Path("artifacts/scientific_recovery_v7/baselines/garl_oof_predictions.csv")
A5_MODEL = Path("configs/model/e_jepa_causal_scale_event_v9_transport_r1_t002_causal.yaml")
A5_TRAINING_CONFIG = Path(
    "configs/experiment/e_jepa_garl_event_causal_scale_eap_screen_a5_corr_v1.yaml"
)
OUTPUT_DIR = Path("configs/experiment/scientific_recovery_v8_fold_chain")
C1_PLAN_DIR = Path("configs/protocol/scientific_recovery_v8_c1_analysis_plans")
C1_PLAN_PATHS = {
    "autopsy_h3": C1_PLAN_DIR / "autopsy_h3.json",
    "exp6_regime": C1_PLAN_DIR / "exp6_regime.json",
    "router_regime": C1_PLAN_DIR / "router_regime.json",
}
MODEL_PATHS = {
    "timevol20_3": Path("configs/model/e_jepa_causal_scale_event_v8_timevol20_3.yaml"),
    "exp6_3": Path("configs/model/e_jepa_causal_scale_event_v8_exp6_3.yaml"),
    "pair20_2": Path("configs/model/e_jepa_causal_scale_event_v8_pair20_2.yaml"),
    "gated_exp6_3": Path("configs/model/e_jepa_causal_scale_event_v8_gated_exp6_3.yaml"),
}
MID_BUCKETS = (
    ("crucial", Decimal("0"), Decimal("3"), Decimal("0.5")),
    ("small", Decimal("3"), Decimal("6"), Decimal("0.3")),
    ("large", Decimal("6"), Decimal("10"), Decimal("0.1")),
    ("negative", Decimal("-10"), Decimal("0"), Decimal("0.1")),
)
ROUTER_FEATURES = [
    "shared_event_count_log1p",
    "shared_event_rate_log1p",
    "a5_flow",
    "a5_margin",
    "a5_log_variance",
    "c2f_flow",
    "c2f_margin",
    "c2f_log_variance",
]
ROUTER_PROHIBITED_FEATURES = [
    "target_ttc",
    "prediction_ttc",
    "bbox",
    "ttc_bucket",
    "outer_fold",
    "sequence_id",
    "track_id",
    "category",
    "future_features",
]
AGGREGATE_SCHEMA = [
    "schema_version",
    "status",
    "git_commit",
    "protocol_sha256",
    "config_sha256",
    "seed",
    "folds",
    "row_count",
    "row_identity_sha256",
    "target_sha256",
    "prediction_sha256",
    "checkpoint_sha256",
    "metrics",
    "per_sequence",
    "per_bucket",
    "bootstrap",
    "integrity_checks",
    "gate_decision",
    "coverage",
    "artifact_sha256",
]
OFFICIAL_BUCKET_IDS = ["crucial", "small", "large", "negative"]
H3_FACTORIAL_COMBINATIONS = [
    "analytic_only",
    "analytic_residual",
    "analytic_transport",
    "analytic_residual_transport",
    "analytic_residual_transport_history",
]
H3_FACTORIAL_COMBINATION_DEFINITIONS = {
    "analytic_only": {"analytic": True, "residual": False, "transport": False, "history": False},
    "analytic_residual": {"analytic": True, "residual": True, "transport": False, "history": False},
    "analytic_transport": {
        "analytic": True,
        "residual": False,
        "transport": True,
        "history": False,
    },
    "analytic_residual_transport": {
        "analytic": True,
        "residual": True,
        "transport": True,
        "history": False,
    },
    "analytic_residual_transport_history": {
        "analytic": True,
        "residual": True,
        "transport": True,
        "history": True,
    },
}
PRIMARY_METRIC_KEYS = [
    "mid_macro_sequence",
    "delta_mid_vs_a5",
    "finite_fraction",
    "failure_rate",
    "coverage_drop_max_pp",
]
PRIMARY_BOOTSTRAP_KEYS = ["probability_delta_lt_zero", "ci95_low", "ci95_high", "resamples"]
PER_GROUP_METRIC_KEYS = ["mid_macro_sequence", "delta_mid_vs_a5", "row_count"]
FACTORIAL_METRIC_KEYS = [
    "mid_macro_sequence",
    "delta_mid_vs_a5",
    "delta_residual_vs_analytic",
    "delta_transport_vs_without_transport",
    "delta_history_vs_without_history",
]
H3_DECISION_INPUTS = [
    "complementarity_present",
    "causal_regime_predictability_passed",
    "stable_across_outer_folds",
    "stable_across_sequences",
    "innocuous_change_invariance_passed",
    "analytic_or_residual_physics_supported",
    "sequence_concentration_detected",
    "residual_unrelated_to_dynamics",
]


@dataclass(frozen=True)
class Sample:
    """One immutable V8 row sourced from paired V7 OOF predictions."""

    token_id: str
    sequence_id: str
    track_id: str
    target: Decimal
    fold: int
    weight: Decimal


def canonical_json(value: object) -> bytes:
    """Return the repository's deterministic JSON representation."""

    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode(
        "utf-8"
    )


def artifact_hash(value: dict[str, Any]) -> str:
    """Hash a signed JSON artifact while excluding its self-signature."""

    unsigned = dict(value)
    unsigned.pop("artifact_sha256", None)
    return hashlib.sha256(canonical_json(unsigned)).hexdigest()


def sign(value: dict[str, Any]) -> dict[str, Any]:
    """Attach the standard artifact self-signature in place."""

    value["artifact_sha256"] = artifact_hash(value)
    return value


def sha256(path: Path) -> str:
    """Compute a streaming SHA-256 for a file."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def repo_path(path: Path) -> str:
    """Convert an existing path to a portable repository-relative POSIX path."""

    resolved = path.resolve()
    try:
        return resolved.relative_to(ROOT.resolve()).as_posix()
    except ValueError as error:
        raise ValueError(f"source must be inside repository: {path}") from error


def read_signed(path: Path) -> dict[str, Any]:
    """Load and verify a signed JSON artifact, failing closed on tampering."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("artifact_sha256") != artifact_hash(payload):
        raise ValueError(f"invalid signed artifact hash: {repo_path(path)}")
    return payload


def reject_sealed_paths(value: object, *, label: str = "source") -> None:
    """Reject an attempted reference to a public-validation, test, or CodaBench split."""

    if isinstance(value, dict):
        for key, nested in value.items():
            lowered = str(key).lower()
            if lowered in {
                "public_validation_used_for_selection",
                "private_test_opened",
                "evttc_test_opened",
                "codabench_opened",
            }:
                if nested is not False:
                    raise ValueError(f"sealed split flag is not false in {label}: {key}")
            reject_sealed_paths(nested, label=label)
    elif isinstance(value, list):
        for nested in value:
            reject_sealed_paths(nested, label=label)
    elif isinstance(value, str):
        lowered = value.lower().replace("\\", "/")
        if any(
            marker in lowered
            for marker in ("public_validation", "private_test", "evttc_test", "codabench")
        ):
            raise ValueError(f"sealed split path rejected in {label}: {value}")


def canonical_records(records: list[dict[str, str]]) -> str:
    """Hash sorted newline-delimited canonical JSON records (with a final newline)."""

    from e_jepa_ttc.data.canonical_token_identity import hash_canonical_json_records

    return hash_canonical_json_records(records)


def bucket_for(target: Decimal) -> tuple[str, Decimal]:
    """Return the signed MiD bucket and its official coefficient for a target."""

    for name, lower, upper, weight in MID_BUCKETS:
        if lower < target <= upper:
            return name, weight
    raise ValueError(f"target outside signed MiD domain: {target}")


def read_oof(path: Path) -> dict[str, dict[str, str]]:
    """Read a strict OOF CSV keyed by its exact sample token."""

    required = {"sample_token", "sequence_id", "track_id", "target_ttc_s", "fold"}
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"missing required OOF columns in {repo_path(path)}")
    keyed: dict[str, dict[str, str]] = {}
    for row in rows:
        token = row["sample_token"]
        if not token or token in keyed:
            raise ValueError(f"duplicate or empty sample token in {repo_path(path)}: {token!r}")
        keyed[token] = row
    return keyed


def derive_samples(a5_path: Path, garl_path: Path) -> list[Sample]:
    """Align A5 and Garl exactly and derive deterministic sequence-MiD row weights."""

    a5 = read_oof(a5_path)
    garl = read_oof(garl_path)
    if set(a5) != set(garl):
        missing_a5 = sorted(set(garl) - set(a5))[:3]
        missing_garl = sorted(set(a5) - set(garl))[:3]
        raise ValueError(f"A5/Garl row mismatch; only_a5={missing_garl}, only_garl={missing_a5}")
    unweighted: list[tuple[str, str, str, Decimal, int]] = []
    for token in sorted(a5):
        left, right = a5[token], garl[token]
        for field in ("sequence_id", "track_id", "fold"):
            if left[field] != right[field]:
                raise ValueError(f"A5/Garl {field} mismatch for token {token}")
        target = Decimal(left["target_ttc_s"])
        # Garl passed targets through a float64 CSV writer while A5 retained
        # float32 text.  Compare at the source target precision, not by text;
        # a materially changed label fails closed below.
        if abs(target - Decimal(right["target_ttc_s"])) > Decimal("0.000001"):
            raise ValueError(f"A5/Garl target_ttc_s mismatch for token {token}")
        if not target.is_finite():
            raise ValueError(f"non-finite target for token {token}")
        unweighted.append((token, left["sequence_id"], left["track_id"], target, int(left["fold"])))
    if len(unweighted) != 8192:
        raise ValueError(f"expected 8192 unique rows, got {len(unweighted)}")
    sequences = {row[1] for row in unweighted}
    folds = {row[4] for row in unweighted}
    if len(sequences) != 9 or folds != {0, 1, 2}:
        raise ValueError(f"expected 9 sequences and folds 0/1/2, got {len(sequences)} and {folds}")
    counts = Counter((sequence, bucket_for(target)[0]) for _, sequence, _, target, _ in unweighted)
    samples = []
    for token, sequence, track, target, fold in unweighted:
        _, coefficient = bucket_for(target)
        # Exact row coefficient of the macro-by-sequence signed MiD primary metric.
        weight = coefficient / Decimal(9) / Decimal(counts[(sequence, bucket_for(target)[0])])
        samples.append(Sample(token, sequence, track, target, fold, weight))
    return samples


def hash_contract(samples: list[Sample]) -> dict[str, str]:
    """Create the canonical row-contract hashes used by all V8 evidence."""

    target_identity_sha256 = canonical_records(
        [{"target_ttc_s": str(item.target), "token_id": item.token_id} for item in samples]
    )
    return {
        "canonical_hash_algorithm": (
            "sha256(canonical JSON records sorted by token_id, newline-delimited UTF-8)"
        ),
        "ordered_token_ids_sha256": canonical_records(
            [{"token_id": item.token_id} for item in samples]
        ),
        "row_identity_sha256": canonical_records(
            [
                {
                    "sequence_id": item.sequence_id,
                    "token_id": item.token_id,
                    "track_id": item.track_id,
                }
                for item in samples
            ]
        ),
        "target_sha256": target_identity_sha256,
        "target_identity_sha256": target_identity_sha256,
        "mid_sample_weight_sha256": canonical_records(
            [{"sample_weight": str(item.weight), "token_id": item.token_id} for item in samples]
        ),
        "fold_assignment_sha256": canonical_records(
            [
                {
                    "outer_fold": str(item.fold),
                    "sequence_id": item.sequence_id,
                    "token_id": item.token_id,
                }
                for item in samples
            ]
        ),
    }


def frozen_sample_contract(samples: list[Sample], v7: dict[str, Any]) -> dict[str, Any]:
    """Build the exact row, weight and fold contract shared by V8 artifacts."""

    by_fold = Counter(str(sample.fold) for sample in samples)
    by_sequence = Counter(sample.sequence_id for sample in samples)
    by_bucket = Counter(bucket_for(sample.target)[0] for sample in samples)
    by_sequence_bucket: dict[str, dict[str, int]] = {}
    for sequence in sorted(by_sequence):
        by_sequence_bucket[sequence] = {
            bucket: sum(
                1
                for sample in samples
                if sample.sequence_id == sequence and bucket_for(sample.target)[0] == bucket
            )
            for bucket in OFFICIAL_BUCKET_IDS
        }
    return {
        "rows": 8192,
        "sequences": 9,
        "folds": 3,
        "token_order": "lexicographic token_id",
        "weight_definition": (
            "official MiD bucket coefficient / 9 sequences / rows in that sequence-bucket"
        ),
        **hash_contract(samples),
        "row_count_contract": {
            "total": len(samples),
            "by_outer_fold": {fold: by_fold[fold] for fold in sorted(by_fold)},
            "by_sequence": {sequence: by_sequence[sequence] for sequence in sorted(by_sequence)},
            "by_bucket": {bucket: by_bucket[bucket] for bucket in OFFICIAL_BUCKET_IDS},
            "by_sequence_bucket": by_sequence_bucket,
        },
        "fold_definitions": v7["sample_contract"]["fold_definitions"],
    }


def current_branch() -> str:
    """Return the current Git branch without modifying repository state."""

    result = subprocess.run(
        ["git", "branch", "--show-current"], cwd=ROOT, capture_output=True, check=True, text=True
    )
    branch = result.stdout.strip()
    if not branch:
        raise ValueError("detached HEAD is not an admissible V8 freeze branch")
    return branch


def write_json(path: Path, value: dict[str, Any]) -> None:
    """Write canonical pretty JSON without timestamps or NaN values."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def write_yaml(path: Path, value: dict[str, Any]) -> None:
    """Write deterministic YAML, rejecting accidental NaN serialization."""

    path.parent.mkdir(parents=True, exist_ok=True)
    rendered = yaml.safe_dump(value, allow_unicode=True, default_flow_style=False, sort_keys=False)
    if ".nan" in rendered.lower() or ".inf" in rendered.lower():
        raise ValueError(f"non-finite YAML is forbidden: {repo_path(path)}")
    path.write_text(rendered, encoding="utf-8", newline="\n")


def source_entry(path: Path, *, artifact: dict[str, Any] | None = None) -> dict[str, str]:
    """Describe a file source using only a repository-relative path."""

    entry = {"path": repo_path(path), "sha256": sha256(path)}
    if artifact is not None:
        entry["artifact_sha256"] = str(artifact["artifact_sha256"])
    return entry


def verify_manifest_file(source: dict[str, Any], manifest_path: Path, *, key: str) -> Path:
    """Verify a manifest's declared checksum and return its repository-local file."""

    declared = source.get(key)
    if not isinstance(declared, dict) or not isinstance(declared.get("path"), str):
        raise ValueError(f"missing signed manifest source {key}")
    candidate = ROOT / Path(str(declared["path"]).replace("\\", "/"))
    # V7 stores absolute paths.  The trusted name is fixed by this script, not the path string.
    if not candidate.is_file():
        candidate = manifest_path.parent / Path(str(declared["path"])).name
    if not candidate.is_file() or sha256(candidate) != declared.get("sha256"):
        raise ValueError(f"checkpoint/prediction hash mismatch for signed source {key}")
    return candidate


def checkpoints(v7_baselines: dict[str, Any]) -> dict[str, list[dict[str, str]]]:
    """Hash A5/C2F/Garl parent checkpoints and verify available signed summaries."""

    result: dict[str, list[dict[str, str]]] = {"a5": [], "c2f": [], "garl": []}
    for fold in range(3):
        a5_entry = v7_baselines["sources"][f"a5_fold{fold}"]
        a5_path = (
            ROOT
            / "artifacts"
            / "runs"
            / f"scientific_recovery_v6_a5_causal_grouped_fold{fold}_seed7"
            / "model_best.pt"
        )
        actual = sha256(a5_path)
        if actual != a5_entry["checkpoint_sha256"]:
            raise ValueError(f"checkpoint hash mismatch for A5 fold {fold}")
        result["a5"].append({"fold": str(fold), "path": repo_path(a5_path), "sha256": actual})

        c2f_run = ROOT / "artifacts" / "runs" / f"scientific_recovery_v7_c2f_fold{fold}_seed7"
        c2f_summary = read_signed(c2f_run / "summary.json")
        c2f_path = c2f_run / "model_best.pt"
        if c2f_summary.get("checkpoint", {}).get("path") != "model_best.pt":
            raise ValueError(f"C2F summary does not bind model_best.pt for fold {fold}")
        result["c2f"].append(
            {
                "fold": str(fold),
                "path": repo_path(c2f_path),
                "sha256": sha256(c2f_path),
                "summary_artifact_sha256": str(c2f_summary["artifact_sha256"]),
            }
        )

        garl_run = (
            ROOT / "artifacts" / "runs" / f"scientific_recovery_v5_garl_fold_chain_fold{fold}_seed7"
        )
        garl_summary = read_signed(garl_run / "summary.json")
        garl_entry = garl_summary.get("artifacts", {}).get("model_best", {})
        garl_path = garl_run / "model_best.pt"
        if garl_entry.get("sha256") != sha256(garl_path):
            raise ValueError(f"checkpoint hash mismatch for Garl fold {fold}")
        result["garl"].append(
            {"fold": str(fold), "path": repo_path(garl_path), "sha256": sha256(garl_path)}
        )
    return result


def model_recipe(channels: int) -> dict[str, Any]:
    """Return the exact A5 model recipe with only its input-channel contract changed."""

    recipe = yaml.safe_load((ROOT / A5_MODEL).read_text(encoding="utf-8"))
    recipe["in_channels"] = channels
    return recipe


def a5_dino_teacher_contract() -> tuple[dict[str, str], str]:
    """Freeze the exact train-only A5 relational teacher identity for V8 arms."""

    source = yaml.safe_load((ROOT / A5_TRAINING_CONFIG).read_text(encoding="utf-8"))
    if not isinstance(source, dict):
        raise ValueError("A5 training source must be a mapping")
    data = source.get("data")
    training = source.get("training")
    if not isinstance(data, dict) or not isinstance(training, dict):
        raise ValueError("A5 training source lacks data/training")
    teacher = data.get("dinov3_relational_teacher")
    if not isinstance(teacher, dict):
        raise ValueError("A5 source has no DINO relational teacher contract")
    path = ROOT / str(teacher.get("manifest"))
    if not path.is_file() or sha256(path) != teacher.get("manifest_sha256"):
        raise ValueError("A5 DINO manifest file hash differs from frozen source")
    artifact = str(teacher.get("artifact_sha256"))
    if training.get("representation_teacher_cache_artifact_sha256") != artifact:
        raise ValueError("A5 DINO training/data artifact identities differ")
    return {
        "manifest": repo_path(path),
        "manifest_sha256": str(teacher["manifest_sha256"]),
        "artifact_sha256": artifact,
        "source_config": repo_path(ROOT / A5_TRAINING_CONFIG),
        "source_config_sha256": sha256(ROOT / A5_TRAINING_CONFIG),
    }, artifact


def timevol20_capacity_contract() -> dict[str, Any]:
    """Measure canonical A5/B1 models at the frozen V8 input shape.

    ``FlopCounterMode`` reports multiply-add work as two FLOPs.  The paired
    MAC estimate divides that count by two and records the exact input used.
    The count covers an eval-mode full forward and exists to disclose the
    mandatory input-stem capacity change, not to claim an accelerator benchmark.
    """

    from torch import full, no_grad, zeros
    from torch.utils.flop_counter import FlopCounterMode

    def measure(channels: int) -> dict[str, int]:
        raw = model_recipe(channels)
        raw.pop("model", None)
        raw["risk_thresholds_s"] = tuple(raw["risk_thresholds_s"])
        model = CausalScaleTTC(CausalScaleTTCConfig(**raw)).eval()
        stem = model.encoder.features[0]
        if not hasattr(stem, "weight"):
            raise RuntimeError("canonical CausalScaleTTC input stem has no convolution weight")
        values = zeros((1, 3, channels, 128, 128), dtype=next(model.parameters()).dtype)
        delta_t = full((1, 2), 0.02, dtype=values.dtype)
        with FlopCounterMode(display=False) as counter, no_grad():
            model(values, delta_t)
        total_flops = int(counter.get_total_flops())
        if total_flops <= 0:
            raise RuntimeError("torch flop counter returned no work for canonical model")
        return {
            "input_shape": [1, 3, channels, 128, 128],
            "input_stem_params": int(stem.weight.numel()),
            "total_params": int(sum(parameter.numel() for parameter in model.parameters())),
            "full_forward_flops": total_flops,
            "full_forward_macs": total_flops // 2,
        }

    a5 = measure(12)
    b1 = measure(20)
    return {
        "a5": a5,
        "timevol20_3": b1,
        "delta_input_stem_params": b1["input_stem_params"] - a5["input_stem_params"],
        "delta_total_params": b1["total_params"] - a5["total_params"],
        "delta_total_params_fraction": (
            (b1["total_params"] - a5["total_params"]) / a5["total_params"]
        ),
        "macs_estimate": {
            "method": "torch.utils.flop_counter.FlopCounterMode eval full forward; MACs=FLOPs/2",
            "delta_t_seconds": [0.02, 0.02],
            "model_mode": "eval",
            "scope": "CausalScaleTTC downstream plus mandatory input stem",
        },
        "capacity_control": {
            "default_enabled": False,
            "trigger": "only if B1 marginally passes the seed-7 screen gate",
            "purpose": "separate a near-threshold frontend result from input-stem capacity",
        },
    }


def write_models() -> dict[str, dict[str, str]]:
    """Freeze all V8 temporal frontend model recipes."""

    recipes = {
        "timevol20_3": model_recipe(20),
        "exp6_3": model_recipe(6),
        "pair20_2": model_recipe(20),
        "gated_exp6_3": model_recipe(6),
    }
    for name, recipe in recipes.items():
        write_yaml(ROOT / MODEL_PATHS[name], recipe)
    return {name: source_entry(ROOT / path) for name, path in MODEL_PATHS.items()}


def fold_groups(v5: dict[str, Any]) -> dict[int, dict[str, list[str]]]:
    """Read the signed outer groups and verify they form an exact 3-fold partition."""

    groups: dict[int, dict[str, list[str]]] = {}
    all_dev: list[str] = []
    for item in v5["folds"]:
        fold = int(item["fold"])
        dev = sorted(str(value) for value in item["dev_sequence_ids"])
        train = sorted(str(value) for value in item["train_sequence_ids"])
        if set(dev) & set(train):
            raise ValueError(f"outer fold {fold} overlaps its train/dev sequences")
        groups[fold] = {"dev": dev, "train": train}
        all_dev.extend(dev)
    if sorted(all_dev) != sorted(v5["sequence_ids"]):
        raise ValueError("signed V5 outer folds are not a sequence partition")
    return groups


def router_inner_folds(
    outer_fold: int, groups: dict[int, dict[str, list[str]]]
) -> list[dict[str, Any]]:
    """Pair the two remaining outer groups lexicographically for inner cross-fitting."""

    remaining = sorted(set(groups) - {outer_fold})
    left, right = (groups[index]["dev"] for index in remaining)
    if len(left) != 3 or len(right) != 3:
        raise ValueError(f"outer fold {outer_fold} does not leave two three-sequence groups")
    outer_train = set(groups[outer_fold]["train"])
    inner: list[dict[str, Any]] = []
    for index, (first, second) in enumerate(zip(left, right, strict=True)):
        dev = [first, second]
        train = sorted(outer_train - set(dev))
        if set(train) & set(dev) or len(train) != 4:
            raise ValueError(f"invalid nested fold {outer_fold}/{index}")
        inner.append({"inner_fold": index, "train_sequence_ids": train, "dev_sequence_ids": dev})
    return inner


def base_run_config(
    *,
    arm: str,
    fold: int,
    groups: dict[int, dict[str, list[str]]],
    model_config: str,
    frontend: dict[str, Any],
) -> dict[str, Any]:
    """Create one B seed-7 config preserving the A5 supervision contract."""

    dev = groups[fold]["dev"]
    train = groups[fold]["train"]
    teacher, teacher_artifact = a5_dino_teacher_contract()
    return {
        "experiment": {
            "name": f"scientific_recovery_v8_{arm}_fold{fold}_seed7",
            "protocol_version": "scientific_recovery_v8_temporal_v1",
            "evidence_scope": "public_train_only_grouped_development",
            "seed": 7,
            "arm": arm,
            "single_scientific_difference": frontend["single_scientific_difference"],
        },
        "model_config": model_config,
        "data": {
            "cache_manifest": frontend["cache_manifest"],
            "cache_manifest_expected": True,
            "cache_materialization_required_before_run": True,
            "train_sequence_ids": train,
            "dev_sequence_ids": dev,
            "opened_splits": ["train"],
            "expected_source_train_rows": 8192,
            "outer_fold": fold,
            "roi_size": [128, 128],
            "steps": frontend["steps"],
            "channels_per_endpoint": frontend["channels"],
            "temporal_frontend": frontend,
            "dinov3_relational_teacher": teacher,
        },
        "training": {
            "seed": 7,
            "epochs": 18,
            "minimum_epochs": 8,
            "early_stopping_patience": 5,
            "foreground_warmup_epochs": 3,
            "batch_size": 32,
            "gradient_accumulation_steps": 1,
            "learning_rate": 0.0003,
            "minimum_learning_rate": 0.00003,
            "weight_decay": 0.0001,
            "grad_clip_norm": 1.0,
            "num_workers": 0,
            "precision": "bf16",
            "maximum_runtime_hours": 6.0,
            "mask_t0_as_proxy": True,
            "foreground_supervision": "bbox_geometry",
            "representation_supervision": "dinov3_local_relational",
            "representation_teacher_cache_artifact_sha256": teacher_artifact,
            "representation_distillation_weight": 8.0,
            "freeze_encoder": False,
        },
        "loss": {
            "log_ratio_nll_weight": 1.0,
            "log_ratio_huber_weight": 0.0,
            "log_ratio_tail_weight": 2.0,
            "log_ratio_tail_fraction": 0.1,
            "foreground_bce_weight": 0.0,
            "foreground_dice_weight": 0.0,
            "foreground_extent_weight": 1.25,
            "foreground_width_weight": 1.25,
            "foreground_center_weight": 2.5,
            "foreground_pair_ratio_weight": 0.0,
            "risk_weight": 0.1,
            "auxiliary_inverse_ttc_weight": 0.05,
            "residual_regularization_weight": 0.1,
            "temporal_consistency_weight": 0.0,
            "smooth_l1_beta": 0.02,
            "supervise_pair_ratio_before_temporal_blend": True,
        },
        "decision_contract": {
            "checkpoint_selection": "dev_sequence_macro_MiD_then_failure_rate",
            "oracle_bbox_roi_preprocessing_shared_with_baseline": True,
            "bbox_coordinates_not_direct_model_features": True,
            "rows_folds_targets_weights_must_match_protocol": True,
            "public_validation_used_for_selection": False,
            "private_test_opened": False,
            "evttc_test_opened": False,
            "codabench_opened": False,
        },
    }


def generated_configs(v5: dict[str, Any]) -> dict[str, dict[str, str]]:
    """Write the nine enabled V8 seed-7 run configs and return their file hashes."""

    groups = fold_groups(v5)
    configs: dict[str, dict[str, str]] = {}
    for fold in range(3):
        inner = router_inner_folds(fold, groups)
        router = {
            "experiment": {
                "name": f"scientific_recovery_v8_router_fold{fold}_seed7",
                "seed": 7,
                "arm": "R",
            },
            "enabled": True,
            "outer_fold": fold,
            "outer_train_sequence_ids": groups[fold]["train"],
            "outer_dev_sequence_ids": groups[fold]["dev"],
            "inner_folds": inner,
            "experts": ["A5", "C2F"],
            "router": {
                "feature_names": ROUTER_FEATURES,
                "prohibited_features": ROUTER_PROHIBITED_FEATURES,
                "pipeline": {
                    "type": "sklearn.pipeline.Pipeline",
                    "scale": "StandardScaler",
                    "classifier": "LogisticRegression",
                    "penalty": "l2",
                    "C": 1.0,
                    "class_weight": None,
                    "solver": "liblinear",
                    "max_iter": 1000,
                    "random_state": 7,
                },
                "threshold": 0.5,
                "routing": "hard",
                "fit_rows": "inner_oof_only",
                "fit_sample_weight_only": True,
                "label": "lower_raw_official_eta_mid_loss",
                "base_sample_weight": (
                    "official MiD bucket coefficient / sequence-bucket rows / 9 sequences"
                ),
                "effective_sample_weight": (
                    "sequence_macro_mid_row_weight_times_absolute_expert_loss_delta"
                ),
                "signature": {
                    "ordered_effective_weight_sha256": "required_in_router_artifact",
                    "summary": ["sum", "minimum", "maximum", "positive_class_mass"],
                },
            },
            "closed_evaluation": {
                "opened_splits": ["train"],
                "public_validation_used_for_selection": False,
                "private_test_opened": False,
                "evttc_test_opened": False,
                "codabench_opened": False,
            },
        }
        name = f"router_fold{fold}_seed7.yaml"
        write_yaml(ROOT / OUTPUT_DIR / name, router)
        configs[name.removesuffix(".yaml")] = source_entry(ROOT / OUTPUT_DIR / name)
    frontends = {
        "timevol20_3": {
            "steps": 3,
            "channels": 20,
            "cache_manifest": "artifacts/scientific_recovery_v8/cache/timevol20_3/manifest.json",
            "single_scientific_difference": (
                "B1 changes the temporal frontend and mandatory 12-to-20 channel input stem; "
                "the downstream topology and training contract remain A5."
            ),
            "builder": "official_timevolume_roi_np",
            "support_ms": 100,
            "extra_tensor_channels": False,
        },
        "exp6_3": {
            "steps": 3,
            "channels": 6,
            "cache_manifest": "artifacts/scientific_recovery_v8/cache/exp6_3/manifest.json",
            "single_scientific_difference": (
                "B2 frontend/cache/in_channels only: signed causal EXP6 state 6x3."
            ),
            "builder": "CausalExponentialStateRepresentation",
            "alpha": list(EXP6_ALPHAS),
            "internal_dt_ms": EXP6_INTERNAL_DT_MS,
            "output_interval_ms": EXP6_OUTPUT_INTERVAL_MS,
            "time_bins": EXP6_OUTPUT_TIME_BINS,
            "source_parity": {
                "official_commit_sha": EXP6_OFFICIAL_COMMIT_SHA,
                "processor_source": {
                    "path": EXP6_PROCESSOR_SOURCE_PATH,
                    "sha256": EXP6_PROCESSOR_SOURCE_SHA256,
                },
                "config_source": {
                    "path": EXP6_CONFIG_SOURCE_PATH,
                    "sha256": EXP6_CONFIG_SOURCE_SHA256,
                },
                **EXP6_RASTER_CONTRACT,
            },
            "warmup_required": True,
        },
    }
    for arm, frontend in frontends.items():
        for fold in range(3):
            name = f"{arm}_fold{fold}_seed7.yaml"
            config = base_run_config(
                arm=arm,
                fold=fold,
                groups=groups,
                model_config=MODEL_PATHS[arm].as_posix(),
                frontend=frontend,
            )
            write_yaml(ROOT / OUTPUT_DIR / name, config)
            configs[name.removesuffix(".yaml")] = source_entry(ROOT / OUTPUT_DIR / name)
    return configs



def conditional_configs(v5: dict[str, Any]) -> dict[str, list[dict[str, str]]]:
    """Freeze B3/C1 fold configs before any V8 result can open their gates."""

    groups = fold_groups(v5)
    frontends = {
        "pair20_2": {
            "steps": 2,
            "channels": 20,
            "cache_manifest": "artifacts/scientific_recovery_v8/cache/timevol20_3/manifest.json",
            "single_scientific_difference": (
                "B3 uses the preregistered TIMEVOL20 frontend but only t1,t2; "
                "the cache and downstream A5 contract remain unchanged."
            ),
            "builder": "official_timevolume_roi_np",
            "support_ms": 100,
            "extra_tensor_channels": False,
        },
        "gated_exp6_3": {
            "steps": 3,
            "channels": 6,
            "cache_manifest": "artifacts/scientific_recovery_v8/cache/exp6_3/manifest.json",
            "single_scientific_difference": (
                "C1 gates the six preregistered causal EXP6 states with the frozen 4x4 "
                "patch router; the A5 downstream contract is otherwise unchanged."
            ),
            "builder": "CausalExponentialStateRepresentation+AdaptiveTemporalChannelGate",
            "alpha": list(EXP6_ALPHAS),
            "internal_dt_ms": EXP6_INTERNAL_DT_MS,
            "output_interval_ms": EXP6_OUTPUT_INTERVAL_MS,
            "time_bins": EXP6_OUTPUT_TIME_BINS,
            "warmup_required": True,
        },
    }
    frozen: dict[str, list[dict[str, str]]] = {}
    for arm, frontend in frontends.items():
        entries: list[dict[str, str]] = []
        for fold in range(3):
            name = f"{arm}_fold{fold}_seed7.yaml"
            config = base_run_config(
                arm=arm,
                fold=fold,
                groups=groups,
                model_config=MODEL_PATHS[arm].as_posix(),
                frontend=frontend,
            )
            config["enabled"] = False
            if arm == "gated_exp6_3":
                config["model_overrides"] = {
                    "temporal_channel_gate_enabled": True,
                    "temporal_channel_gate_patch_grid": 4,
                    "temporal_channel_gate_hidden_dim": 16,
                }
            config["decision_contract"]["conditional_arm"] = True
            config["decision_contract"]["opening_gate_required"] = (
                "B1_TIMEVOL20_3 seed-7 screen gate" if arm == "pair20_2"
                else "signed V8 C1 opening artifact"
            )
            write_yaml(ROOT / OUTPUT_DIR / name, config)
            entries.append(source_entry(ROOT / OUTPUT_DIR / name))
        frozen[arm] = entries
    return frozen

def write_c1_analysis_plans(
    samples: list[Sample], v7: dict[str, Any], configs: dict[str, dict[str, str]]
) -> dict[str, dict[str, str]]:
    """Freeze one pre-result C1 analysis plan for each allowed opening path."""

    sample_contract = frozen_sample_contract(samples, v7)

    def config_hashes(arm: str) -> dict[str, str]:
        return {str(fold): configs[f"{arm}_fold{fold}_seed7"]["sha256"] for fold in range(3)}

    source_contracts = {
        "autopsy_h3": {
            "stage": "autopsy",
            "arm": "autopsy",
            "candidate_id": "A_AUTOPSY",
            "seed": 7,
            "artifact_type": "scientific_recovery_v8_autopsy_seed7_aggregate_v1",
            "required_outputs": {
                "factorial_replay": "scientific_recovery_v8_autopsy_factorial_replay_v1",
                "diagnostic": "scientific_recovery_v8_autopsy_diagnostic_v1",
            },
            "factorial_replay_schema": {
                "combinations": H3_FACTORIAL_COMBINATIONS,
                "combination_definitions": H3_FACTORIAL_COMBINATION_DEFINITIONS,
                "required_cell_fields": [
                    "row_count",
                    "row_identity_sha256",
                    "target_sha256",
                    "prediction_sha256",
                    "metrics",
                    "settings",
                    "per_sequence",
                    "per_bucket",
                    "coverage",
                    "integrity_checks",
                ],
                "row_count": 8192,
                "metric_keys": FACTORIAL_METRIC_KEYS,
            },
            "diagnostic_schema": {
                "dimensions": {
                    "by_ttc_bucket": OFFICIAL_BUCKET_IDS,
                    "by_event_density": ["low", "medium", "high"],
                    "by_movement": ["low", "medium", "high"],
                    "by_sign": ["negative", "positive"],
                },
                "required_fields": [
                    "by_ttc_bucket",
                    "by_sequence",
                    "by_event_density",
                    "by_movement",
                    "by_sign",
                    "decision_rule_output",
                    "final_decision",
                    "integrity_checks",
                    "output_hashes",
                    "decision_inputs",
                ],
                "dimension_record_schema": {
                    "numeric": ["effect_size"],
                    "booleans": ["evidence_present", "stable"],
                },
                "decision_rule": {
                    "inputs": H3_DECISION_INPUTS,
                    "h3_all_true": H3_DECISION_INPUTS[:5],
                    "h1_all_true": ["analytic_or_residual_physics_supported"],
                    "h1_all_false": [
                        "sequence_concentration_detected",
                        "residual_unrelated_to_dynamics",
                    ],
                    "otherwise": "H2",
                },
            },
        },
        "exp6_regime": {
            "stage": "temporal",
            "arm": "exp6_3",
            "candidate_id": "B2_EXP6_3",
            "seed": 7,
            "artifact_type": "scientific_recovery_v8_exp6_3_seed7_aggregate_v1",
            "primary_ttc_gate_required": True,
            "aggregate_schema": AGGREGATE_SCHEMA,
            "config_sha256_by_fold": config_hashes("exp6_3"),
            "row_count": 8192,
            "required_bucket_ids": OFFICIAL_BUCKET_IDS,
            "primary_metric_keys": PRIMARY_METRIC_KEYS,
            "bootstrap_keys": PRIMARY_BOOTSTRAP_KEYS,
            "per_group_metric_keys": PER_GROUP_METRIC_KEYS,
        },
        "router_regime": {
            "stage": "router",
            "arm": "router",
            "candidate_id": "R",
            "seed": 7,
            "artifact_type": "scientific_recovery_v8_router_seed7_aggregate_v1",
            "primary_ttc_gate_required": True,
            "aggregate_schema": AGGREGATE_SCHEMA,
            "config_sha256_by_fold": config_hashes("router"),
            "row_count": 8192,
            "required_bucket_ids": OFFICIAL_BUCKET_IDS,
            "primary_metric_keys": PRIMARY_METRIC_KEYS,
            "bootstrap_keys": PRIMARY_BOOTSTRAP_KEYS,
            "per_group_metric_keys": PER_GROUP_METRIC_KEYS,
        },
    }
    sources: dict[str, dict[str, str]] = {}
    for plan_id, source_contract in source_contracts.items():
        plan = sign(
            {
                "artifact_type": "scientific_recovery_v8_preregistered_analysis_plan_v1",
                "schema_version": "scientific_recovery_v8_temporal_v1",
                "status": "frozen_before_v8_training",
                "plan_id": plan_id,
                "opening_route": plan_id,
                "source_aggregate_contract": source_contract,
                "sample_contract": sample_contract,
                "closed_evaluation": {
                    "allowed_splits": ["train_fold", "outer_dev_fold"],
                    "public_validation_used_for_selection": False,
                    "private_test_opened": False,
                    "evttc_test_opened": False,
                    "codabench_opened": False,
                },
            }
        )
        path = ROOT / C1_PLAN_PATHS[plan_id]
        write_json(path, plan)
        sources[plan_id] = source_entry(path, artifact=plan)
    return sources


def protocol_payload(
    *,
    samples: list[Sample],
    v5: dict[str, Any],
    v7: dict[str, Any],
    baselines: dict[str, Any],
    checkpoint_sources: dict[str, list[dict[str, str]]],
    c1_analysis_plans: dict[str, dict[str, str]],
) -> dict[str, Any]:
    """Build the immutable signed V8 protocol from verified train-only evidence."""

    sample_contract = frozen_sample_contract(samples, v7)
    capacity = timevol20_capacity_contract()
    return sign(
        {
            "artifact_type": "scientific_recovery_v8_temporal_protocol_v1",
            "schema_version": "scientific_recovery_v8_temporal_v1",
            "status": "frozen_before_v8_training",
            "git_base_commit": "f9331b29596c4107430af5a8c78935bd127ccf94",
            "git_branch": current_branch(),
            "created_at_utc": "FROZEN_NO_WALL_CLOCK",
            "closed_evaluation": {
                "allowed_splits": ["train_fold", "outer_dev_fold"],
                "public_validation_used_for_selection": False,
                "private_test_opened": False,
                "evttc_test_opened": False,
                "codabench_opened": False,
            },
            "v8_supersession": {
                "provisional_post_v7_commit": "f9331b29596c4107430af5a8c78935bd127ccf94",
                "historical_v7_a4_retention_gate": "immutable",
                "v8_ttc_candidate_gate": "supersedes provisional V8 A4 retention requirement",
                "a4_in_v8": "mechanistic diagnostic only",
            },
            "sources": {
                "v5_grouped_protocol": source_entry(ROOT / V5_PROTOCOL, artifact=v5),
                "v7_balanced_protocol": source_entry(ROOT / V7_PROTOCOL, artifact=v7),
                "v7_baseline_manifest": source_entry(ROOT / V7_BASELINES, artifact=baselines),
                "a5_oof_predictions": source_entry(ROOT / A5_OOF),
                "garl_oof_predictions": source_entry(ROOT / GARL_OOF),
                "a5_model_recipe": source_entry(ROOT / A5_MODEL),
                "garl_official_preprocessing": source_entry(
                    ROOT / Path("src/e_jepa_ttc/data/garl_official_preprocessing.py")
                ),
                "garl_release_commit": "256661242b8a7f5e56aa3c1c02348b30f6e89de6",
            },
            "sample_contract": sample_contract,
            "c1_analysis_plans": c1_analysis_plans,
            "parent_checkpoints": checkpoint_sources,
            "training_contract": {
                "screen_seed": 7,
                "multiseed_replication_seeds": [13, 23],
                "optimization_stability_only": True,
                "reselection_or_rescue_forbidden": True,
                "epochs": 18,
                "batch_size": 32,
                "learning_rate": 0.0003,
                "optimizer": "AdamW",
                "multiseed_replication_one_winner_only": True,
            },
            "external_confirmation": {
                "name": "single_user_authorized_sealed_public_validation",
                "requires_user_authorization": True,
                "selection_or_tuning_forbidden": True,
            },
            "evaluation_contract": {
                "primary_metric": "MiD macro-sequence at full coverage",
                "report_metrics": [
                    "MiD_sample_weighted",
                    "MAE",
                    "relative_error",
                    "finite_fraction",
                    "failure_rate",
                    "coverage",
                    "per_sequence",
                    "per_track",
                    "per_ttc_bucket",
                ],
                "bootstrap": {
                    "method": "paired_hierarchical_sequence_then_track",
                    "resamples": 5000,
                    "seed": 20260814,
                    "unit": ["sequence_id", "track_id"],
                },
                "oof_schema": [
                    "token_id",
                    "sequence_id",
                    "track_id",
                    "outer_fold",
                    "seed",
                    "target_ttc",
                    "sample_weight",
                    "prediction_ttc",
                    "prediction_log_variance",
                    "finite",
                    "failure_reason",
                    "event_count",
                    "event_rate",
                    "support_ms",
                    "model_name",
                    "config_sha256",
                    "checkpoint_sha256",
                ],
                "aggregate_schema": AGGREGATE_SCHEMA,
            },
            "router_contract": {
                "fit_rows": "inner_oof_only",
                "label": "lower_raw_official_eta_mid_loss",
                "base_sample_weight": (
                    "official MiD bucket coefficient / sequence-bucket rows / 9 sequences"
                ),
                "effective_sample_weight": (
                    "base_sample_weight * abs(raw_official_eta_mid_loss_C2F - "
                    "raw_official_eta_mid_loss_A5)"
                ),
                "classifier": {
                    "penalty": "l2",
                    "C": 1.0,
                    "class_weight": None,
                    "solver": "liblinear",
                    "max_iter": 1000,
                    "threshold": 0.5,
                },
                "fit_parameter": "router__sample_weight",
                "signature": {
                    "ordered_effective_weight_sha256": "required",
                    "summary": ["sum", "minimum", "maximum", "positive_class_mass"],
                },
            },
            "gates": {
                "ttc_candidate_gate": {
                    "delta_MiD_max": -3.0,
                    "bootstrap_probability_delta_below_zero_min": 0.9,
                    "finite_fraction_required": 1.0,
                    "failure_rate_required": 0.0,
                    "coverage_drop_max_pp": 1.0,
                    "exact_contract_identity_required": True,
                },
                "mechanistic_interpretability_gate": {
                    "a4_retention": "diagnostic",
                    "physical_ratio_correlation": "diagnostic",
                    "slope": "diagnostic",
                    "reversal": "diagnostic",
                },
                "multiseed_replication": {
                    "per_seed_delta_negative_required": True,
                    "mean_delta_MiD_max": -3.0,
                    "bootstrap_ci95_below_zero_required": True,
                    "same_config_except_seed": True,
                },
                "b3_open": "B1 passes ttc_candidate_gate in signed seed-7 aggregate",
                "c1_open": (
                    "frozen route plan plus signed seed-7 H3 autopsy OR signed seed-7 EXP6 "
                    "stable regime heterogeneity OR signed seed-7 router primary-gate pass with "
                    "frozen-plan stable temporal/density "
                    "causal-feature dependence across every outer fold and sequence plus "
                    "causal invariance"
                ),
            },
            "c1_evidence_contract": {
                "opening_artifact_type": "scientific_recovery_v8_c1_opening_decision_v1",
                "analysis_plan_artifact_type": (
                    "scientific_recovery_v8_preregistered_analysis_plan_v1"
                ),
                "frozen_plan_paths": {
                    route: path.as_posix() for route, path in C1_PLAN_PATHS.items()
                },
                "source_aggregate": (
                    "route-specific signed seed-7 aggregate, typed and bound by the frozen plan"
                ),
                "source_aggregate_requirements": {
                    "status": "completed",
                    "exact_sample_contract": True,
                    "exact_outer_fold_sequence_coverage": True,
                    "finite_recursive_content": True,
                    "primary_ttc_gate_fields": [
                        "delta_mid_vs_a5",
                        "bootstrap_probability_delta_lt_zero",
                        "finite_fraction",
                        "failure_rate",
                        "coverage_drop_max_pp",
                    ],
                    "autopsy_h3_outputs": ["factorial_replay", "diagnostic"],
                },
                "regime_evidence_artifact_type": "scientific_recovery_v8_regime_evidence_v1",
                "causal_invariance_artifact_type": ("scientific_recovery_v8_causal_invariance_v1"),
                "artifact_root": "artifacts/scientific_recovery_v8/",
                "required_binding": [
                    "protocol_artifact_sha256",
                    "protocol_file_sha256",
                    "exact sample_contract including fold_definitions",
                    "closed_evaluation",
                ],
                "coverage": "exact frozen outer fold dev sequence IDs",
                "self_reference_or_result_aggregate": "rejected",
            },
            "timevol20_3_capacity": capacity,
            "exp6_source_parity": {
                "official_commit_sha": EXP6_OFFICIAL_COMMIT_SHA,
                "processor_source": {
                    "path": EXP6_PROCESSOR_SOURCE_PATH,
                    "sha256": EXP6_PROCESSOR_SOURCE_SHA256,
                },
                "config_source": {
                    "path": EXP6_CONFIG_SOURCE_PATH,
                    "sha256": EXP6_CONFIG_SOURCE_SHA256,
                },
                "alphas": list(EXP6_ALPHAS),
                "internal_dt_ms": EXP6_INTERNAL_DT_MS,
                "output_interval_ms": EXP6_OUTPUT_INTERVAL_MS,
                "time_bins": EXP6_OUTPUT_TIME_BINS,
                **EXP6_RASTER_CONTRACT,
            },
            "jepa_d4_contract": {
                "cross_track_derangement": True,
                "fixed_points_forbidden": True,
                "fail_closed_when_cross_track_pairing_is_impossible": True,
                "track_ids_scope": "pairing_only; never model input or target",
                "equal_compute": [
                    "initialization",
                    "optimizer",
                    "updates",
                    "batches",
                    "masking",
                    "EMA",
                    "predictor",
                    "augmentations",
                ],
                "multiseed_if_jepa_positive": ["D0", "D1", "best(D2,D3)", "D4"],
            },
            "phase_d_transfer_contract": {
                "transfer_prefix": "CausalScaleTTC.encoder.*",
                "transfer_validation": ["strict ordered keys", "strict shapes", "state SHA-256"],
                "views": [
                    {"name": "dense", "weight": 1.0 / 3.0},
                    {"name": "global", "weight": 1.0 / 3.0},
                    {"name": "foreground", "weight": 1.0 / 3.0},
                ],
                "label_free": True,
                "gradient_coverage": "finite nonzero gradient for every transferred parameter",
                "collapse_monitoring": "per view",
                "d3_allowlist": ["encoder.final_residual_block.*", "foreground.*"],
                "router_winner_rule": (
                    "R remains the primary TTC winner, but JEPA attribution uses the frozen A5 "
                    "constituent encoder because a meta-router has no single transferable encoder; "
                    "the router itself is not credited as JEPA."
                ),
            },
        }
    )


def freeze(protocol_path: Path) -> dict[str, Any]:
    """Verify signed inputs, regenerate every owned V8 artifact, and return its manifest."""

    v5, v7, baselines = (
        read_signed(ROOT / path) for path in (V5_PROTOCOL, V7_PROTOCOL, V7_BASELINES)
    )
    reject_sealed_paths(
        {
            "v5": v5.get("checks", {}),
            "v7": v7["closed_evaluation"],
            "baselines": baselines["contracts"],
        }
    )
    if (
        sha256(ROOT / A5_OOF) != baselines["outputs"]["a5"]["sha256"]
        or sha256(ROOT / GARL_OOF) != baselines["outputs"]["garl"]["sha256"]
    ):
        raise ValueError("signed V7 baseline prediction checksum mismatch")
    samples = derive_samples(ROOT / A5_OOF, ROOT / GARL_OOF)
    groups = fold_groups(v5)
    expected_fold_by_sequence = {
        sequence: fold for fold, values in groups.items() for sequence in values["dev"]
    }
    if any(sample.fold != expected_fold_by_sequence[sample.sequence_id] for sample in samples):
        raise ValueError("OOF fold assignment disagrees with signed V5 sequence folds")
    checkpoint_sources = checkpoints(baselines)
    model_files = write_models()
    configs = generated_configs(v5)
    conditional = conditional_configs(v5)
    c1_analysis_plans = write_c1_analysis_plans(samples, v7, configs)
    protocol = protocol_payload(
        samples=samples,
        v5=v5,
        v7=v7,
        baselines=baselines,
        checkpoint_sources=checkpoint_sources,
        c1_analysis_plans=c1_analysis_plans,
    )
    write_json(protocol_path, protocol)
    templates = {
        "pair20_2": {
            "enabled": False,
            "fold_configs": conditional["pair20_2"],
            "opening_gate": protocol["gates"]["b3_open"],
            "model_config": MODEL_PATHS["pair20_2"].as_posix(),
            "steps": 2,
            "frontend_contract": "TIMEVOL20; V8-only steps=2; EVENT_V4_STEPS remains 3.",
            "runner_refusal": "require signed V8 B1 gate decision artifact",
        },
        "gated_exp6_3": {
            "enabled": False,
            "fold_configs": conditional["gated_exp6_3"],
            "opening_gate": protocol["gates"]["c1_open"],
            "model_config": MODEL_PATHS["gated_exp6_3"].as_posix(),
            "steps": 3,
            "frontend_contract": "causal EXP6 patch frontend gate; not an architecture sweep.",
            "runner_refusal": "require signed V8 C1 gate decision artifact",
        },
    }
    manifest = sign(
        {
            "artifact_type": "scientific_recovery_v8_frozen_config_manifest_v1",
            "schema_version": "scientific_recovery_v8_temporal_v1",
            "status": "frozen_before_v8_training",
            "protocol": source_entry(protocol_path, artifact=protocol),
            "closed_evaluation": protocol["closed_evaluation"],
            "enabled_seed7_configs": configs,
            "model_configs": model_files,
            "c1_analysis_plans": c1_analysis_plans,
            "conditional_templates": templates,
            "integrity": dict(protocol["sample_contract"]),
        }
    )
    write_json(ROOT / OUTPUT_DIR / "frozen_manifest.json", manifest)
    return manifest


def verify_frozen_outputs(protocol_path: Path) -> dict[str, Any]:
    """Verify existing signed V8 outputs without reading datasets or checkpoints."""

    protocol = read_signed(protocol_path)
    manifest_path = ROOT / OUTPUT_DIR / "frozen_manifest.json"
    manifest = read_signed(manifest_path)
    declared = manifest.get("protocol")
    if not isinstance(declared, dict):
        raise ValueError("frozen manifest has no signed protocol source")
    if declared.get("sha256") != sha256(protocol_path):
        raise ValueError("frozen manifest protocol file hash differs from protocol")
    if declared.get("artifact_sha256") != protocol.get("artifact_sha256"):
        raise ValueError("frozen manifest protocol signature differs from protocol")
    if manifest.get("integrity") != protocol.get("sample_contract"):
        raise ValueError("frozen manifest sample integrity differs from protocol")
    sources = [
        *manifest.get("enabled_seed7_configs", {}).values(),
        *manifest.get("model_configs", {}).values(),
    ]
    templates = manifest.get("conditional_templates", {})
    if isinstance(templates, dict):
        for template in templates.values():
            if isinstance(template, dict):
                fold_configs = template.get("fold_configs", [])
                if isinstance(fold_configs, list):
                    sources.extend(fold_configs)
    for source in sources:
        if not isinstance(source, dict) or not isinstance(source.get("path"), str):
            raise ValueError("frozen manifest contains an invalid source entry")
        path = ROOT / source["path"]
        if not path.is_file() or source.get("sha256") != sha256(path):
            raise ValueError(f"frozen manifest source checksum mismatch: {source.get('path')}")
    plans = manifest.get("c1_analysis_plans")
    if not isinstance(plans, dict) or plans != protocol.get("c1_analysis_plans"):
        raise ValueError("frozen C1 analysis plans differ from protocol")
    for route, source in plans.items():
        if not isinstance(route, str) or not isinstance(source, dict):
            raise ValueError("frozen C1 analysis plan entry is invalid")
        raw_path = source.get("path")
        if not isinstance(raw_path, str) or Path(raw_path).is_absolute():
            raise ValueError("frozen C1 analysis plan path must be relative")
        path = ROOT / raw_path
        if not path.is_file() or source.get("sha256") != sha256(path):
            raise ValueError(f"frozen C1 analysis plan checksum mismatch: {route}")
        plan = read_signed(path)
        if (
            plan.get("artifact_type") != "scientific_recovery_v8_preregistered_analysis_plan_v1"
            or source.get("artifact_sha256") != plan.get("artifact_sha256")
            or plan.get("plan_id") != route
            or plan.get("sample_contract") != protocol.get("sample_contract")
        ):
            raise ValueError(f"frozen C1 analysis plan contract mismatch: {route}")
    return manifest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the public freeze command arguments."""

    parser = argparse.ArgumentParser(
        description="Freeze signed V8 temporal-recovery configs from V5/V7 evidence."
    )
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path("configs/protocol/scientific_recovery_v8_temporal.json"),
        help="Repository-relative protocol output path.",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify existing signed outputs without regenerating them.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the deterministic V8 freeze and return a shell-safe exit code."""

    args = parse_args(argv)
    protocol_path = (
        (ROOT / args.protocol).resolve()
        if not args.protocol.is_absolute()
        else args.protocol.resolve()
    )
    try:
        repo_path(protocol_path)
        manifest = verify_frozen_outputs(protocol_path) if args.verify else freeze(protocol_path)
    except (FileNotFoundError, OSError, ValueError, subprocess.CalledProcessError) as error:
        print(f"V8 freeze failed closed: {error}", file=sys.stderr)
        return 2
    output = repo_path(ROOT / OUTPUT_DIR / "frozen_manifest.json")
    action = "verified" if args.verify else "frozen"
    print(f"{action} {len(manifest['enabled_seed7_configs'])} enabled seed-7 configs: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

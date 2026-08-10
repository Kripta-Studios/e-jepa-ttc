#!/usr/bin/env python
"""Select A4 DINOv3 relational weight using sequence-grouped train-only CV.

The selector never instantiates the public validation split.  The nine public
train sequences are partitioned into three preregistered folds; each sequence
is held out exactly once.  Every lambda is evaluated on the same folds/seeds.
The selected lambda minimizes nine-sequence macro MiD, then sample-weighted
failure rate, then the smaller lambda as a deterministic final tie-break.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence, Sized
from dataclasses import asdict, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
import yaml
from torch.utils.data import Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from e_jepa_ttc.artifacts.hashing import sign_artifact, verify_artifact_hash  # noqa: E402
from e_jepa_ttc.data.dinov3_relational_teacher_cache import (  # noqa: E402
    DINOv3RelationalTeacherDataset,
)
from e_jepa_ttc.data.object_event_v4 import GarlTTCObjectEventV4Dataset  # noqa: E402
from e_jepa_ttc.losses.causal_scale_ttc import CausalScaleTTCLossConfig  # noqa: E402
from e_jepa_ttc.models.causal_scale_ttc import (  # noqa: E402
    CausalScaleTTC,
    CausalScaleTTCConfig,
)
from e_jepa_ttc.reproducibility import resolve_device  # noqa: E402
from e_jepa_ttc.training.causal_scale_eap import (  # noqa: E402
    CausalScaleEAPTrainingConfig,
    train_real_causal_scale,
)

DEFAULT_CONFIG = (
    ROOT
    / "configs/experiment/e_jepa_garl_event_causal_scale_a4_lambda_cv_train8192_v1.yaml"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"YAML must contain a mapping: {path}")
    return value


def _resolve(value: object) -> Path:
    if not isinstance(value, str):
        raise ValueError("path references must be strings")
    path = (ROOT / value).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def _git_identity(parent_commit: str) -> dict[str, Any]:
    current = _git("rev-parse", "HEAD")
    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", parent_commit, current],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if ancestor.returncode != 0:
        raise RuntimeError(
            f"current HEAD {current} does not descend from required parent {parent_commit}"
        )
    status = _git("-c", "core.quotepath=false", "status", "--porcelain=v1", "--untracked-files=all")
    dirty_lines = [line for line in status.splitlines() if line.strip()]
    if dirty_lines:
        preview = "\n".join(dirty_lines[:20])
        raise RuntimeError(
            "lambda selection requires a completely clean Git worktree; "
            f"found:\n{preview}"
        )
    return {
        "git_commit": current,
        "git_dirty": False,
        "required_parent_commit": parent_commit,
    }


def _model_config(path: Path) -> CausalScaleTTCConfig:
    raw = _read_yaml(path)
    if raw.pop("model", None) != "e_jepa_causal_scale_event_v8":
        raise ValueError("lambda CV requires the causal-scale event v8 model")
    thresholds = raw.get("risk_thresholds_s")
    if not isinstance(thresholds, list):
        raise ValueError("risk_thresholds_s must be a list")
    raw["risk_thresholds_s"] = tuple(float(value) for value in thresholds)
    return CausalScaleTTCConfig(**raw)


class IndexedDataset(Dataset[dict[str, Any]]):
    """Index-preserving subset that retains cache-local shard sampling groups."""

    def __init__(self, dataset: Dataset[dict[str, Any]], indices: Sequence[int]) -> None:
        if not isinstance(dataset, Sized):
            raise TypeError("indexed dataset requires a sized base dataset")
        selected = tuple(int(index) for index in indices)
        if not selected:
            raise ValueError("indexed dataset cannot be empty")
        if len(set(selected)) != len(selected):
            raise ValueError("indexed dataset indices must be unique")
        if min(selected) < 0 or max(selected) >= len(dataset):
            raise IndexError("indexed dataset index outside base dataset")
        self.dataset = dataset
        self.indices = selected

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.dataset[self.indices[index]]

    def shard_index_groups(self) -> tuple[tuple[int, ...], ...]:
        provider = getattr(self.dataset, "shard_index_groups", None)
        if not callable(provider):
            raise TypeError("base dataset does not expose shard_index_groups")
        reverse = {base_index: subset_index for subset_index, base_index in enumerate(self.indices)}
        groups: list[tuple[int, ...]] = []
        base_groups = cast(tuple[tuple[int, ...], ...], provider())
        for base_group in base_groups:
            group = tuple(reverse[index] for index in base_group if index in reverse)
            if group:
                groups.append(group)
        flattened = sorted(index for group in groups for index in group)
        if flattened != list(range(len(self))):
            raise ValueError("subset shard groups do not partition the subset")
        return tuple(groups)


def _validate_fold_protocol(
    all_sequences: Sequence[str],
    folds: Sequence[Mapping[str, Any]],
) -> None:
    expected = set(all_sequences)
    if len(expected) != len(all_sequences):
        raise ValueError("train_sequence_ids must be unique")
    seen: list[str] = []
    names: set[str] = set()
    for fold in folds:
        name = str(fold.get("name", ""))
        if not name or name in names:
            raise ValueError("fold names must be non-empty and unique")
        names.add(name)
        heldout = [str(value) for value in fold.get("heldout_sequences", [])]
        if len(heldout) != 3 or len(set(heldout)) != 3:
            raise ValueError(f"{name} must hold out exactly three unique sequences")
        if not set(heldout) <= expected:
            raise ValueError(f"{name} contains a sequence outside train_sequence_ids")
        seed = int(fold.get("seed", -1))
        if seed < 0:
            raise ValueError(f"{name} requires a non-negative seed")
        seen.extend(heldout)
    if len(folds) != 3:
        raise ValueError("lambda CV requires exactly three folds")
    if len(seen) != len(expected) or set(seen) != expected:
        raise ValueError("the folds must hold out every train sequence exactly once")
    if len(set(seen)) != len(seen):
        raise ValueError("a train sequence is held out more than once")


def _aggregate_candidate(
    lambda_value: float,
    fold_results: Sequence[Mapping[str, Any]],
    expected_sequences: Sequence[str],
) -> dict[str, Any]:
    per_sequence_mid: dict[str, float] = {}
    weighted_failure_numerator = 0.0
    sample_count = 0
    for fold in fold_results:
        sequence_metrics = fold.get("per_sequence")
        if not isinstance(sequence_metrics, Mapping):
            raise ValueError("fold result lacks per_sequence metrics")
        for sequence, metrics in sequence_metrics.items():
            if sequence in per_sequence_mid:
                raise ValueError(f"sequence {sequence} appears in multiple held-out folds")
            if not isinstance(metrics, Mapping):
                raise ValueError("per-sequence metric entry must be a mapping")
            mid = float(metrics.get("paper_MiD_overall", float("nan")))
            if not math.isfinite(mid):
                raise FloatingPointError(f"non-finite MiD for held-out sequence {sequence}")
            per_sequence_mid[str(sequence)] = mid
        count = int(fold.get("num_samples", 0))
        failure = float(fold.get("failure_rate_pct", float("nan")))
        if count <= 0 or not math.isfinite(failure):
            raise ValueError("fold result has invalid sample count/failure rate")
        weighted_failure_numerator += failure * count
        sample_count += count
    if set(per_sequence_mid) != set(expected_sequences):
        missing = sorted(set(expected_sequences) - set(per_sequence_mid))
        extra = sorted(set(per_sequence_mid) - set(expected_sequences))
        raise ValueError(f"held-out sequence coverage mismatch: missing={missing}, extra={extra}")
    nine_sequence_macro_mid = float(np.mean(list(per_sequence_mid.values())))
    weighted_failure = weighted_failure_numerator / sample_count
    return {
        "lambda": float(lambda_value),
        "nine_sequence_macro_MiD": nine_sequence_macro_mid,
        "sample_weighted_failure_rate_pct": float(weighted_failure),
        "heldout_sample_count": sample_count,
        "per_sequence_paper_MiD_overall": dict(sorted(per_sequence_mid.items())),
        "selection_key": [nine_sequence_macro_mid, float(weighted_failure), float(lambda_value)],
    }


def _select_best_candidate(candidates: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    if not candidates:
        raise ValueError("no lambda candidates were evaluated")
    return min(
        candidates,
        key=lambda item: (
            float(item["nine_sequence_macro_MiD"]),
            float(item["sample_weighted_failure_rate_pct"]),
            float(item["lambda"]),
        ),
    )


def _write_predictions(path: Path, validation: Mapping[str, Any]) -> None:
    tokens = list(validation["sample_tokens"])
    sequences = list(validation["sequence_ids"])
    targets = list(validation["target_ttc_s"])
    predictions = list(validation["prediction_ttc_s"])
    if not (len(tokens) == len(sequences) == len(targets) == len(predictions)):
        raise ValueError("held-out prediction fields have inconsistent lengths")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["sample_token", "sequence_id", "target_ttc_s", "prediction_ttc_s"])
        writer.writerows(zip(tokens, sequences, targets, predictions, strict=True))


def _finite_json(value: object) -> object:
    if isinstance(value, np.generic):
        return _finite_json(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(key): _finite_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_finite_json(item) for item in value]
    return value


def run(config_path: Path, output_dir: Path, *, device_name: str) -> dict[str, Any]:
    raw = _read_yaml(config_path)
    experiment = raw.get("experiment")
    provenance = raw.get("provenance")
    data = raw.get("data")
    cv = raw.get("cv")
    decision = raw.get("decision_contract")
    if not all(isinstance(value, dict) for value in (experiment, provenance, data, cv, decision)):
        raise ValueError("experiment/provenance/data/cv/decision_contract mappings are required")
    experiment = cast(dict[str, Any], experiment)
    provenance = cast(dict[str, Any], provenance)
    data = cast(dict[str, Any], data)
    cv = cast(dict[str, Any], cv)
    decision = cast(dict[str, Any], decision)

    if data.get("opened_splits") != ["train"]:
        raise ValueError("lambda selection may open only the public train split")
    for key in ("official_test_opened", "codabench_opened", "evttc_test_opened"):
        if data.get(key) is not False:
            raise ValueError(f"{key} must remain false")
    forbidden_data_keys = {
        "validation_cache_manifest",
        "validation_cache_manifest_sha256",
        "validation_cache_artifact_sha256",
        "validation_sequence_ids",
    }
    present_forbidden = sorted(forbidden_data_keys & set(data))
    if present_forbidden:
        raise ValueError(
            "lambda CV config contains forbidden validation references: "
            f"{present_forbidden}"
        )

    parent_commit = str(experiment.get("parent_code_commit", ""))
    if not parent_commit:
        raise ValueError("experiment.parent_code_commit is required")
    code_identity = _git_identity(parent_commit)

    source_a4_config = _resolve(provenance["source_a4_config"])
    if _sha256(source_a4_config) != str(provenance["source_a4_config_sha256"]):
        raise ValueError("source A4 config hash differs from preregistered provenance")

    calibration_path = _resolve(provenance["original_calibration_artifact"])
    if _sha256(calibration_path) != str(provenance["original_calibration_file_sha256"]):
        raise ValueError("original A4 calibration file hash differs from preregistration")
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    if not verify_artifact_hash(calibration):
        raise ValueError("original A4 calibration artifact signature is invalid")
    if calibration.get("artifact_sha256") != provenance["original_calibration_artifact_sha256"]:
        raise ValueError("original A4 calibration artifact identity differs")
    if calibration.get("scope", {}).get("validation_or_test_opened") is not False:
        raise ValueError("original calibration did not preserve train-only scope")
    lambda_raw = float(calibration["lambda_raw"])
    selected_original = float(calibration["selected_weight"])
    clamp_range = calibration.get("clamp_range")
    if not isinstance(clamp_range, list) or len(clamp_range) != 2:
        raise ValueError("original calibration clamp range is malformed")
    if not math.isclose(
        lambda_raw,
        float(provenance["original_calibration_lambda_raw"]),
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError("lambda_raw differs from frozen provenance")
    if not math.isclose(selected_original, 4.0, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("original A4 selected weight must remain 4.0")
    if not math.isclose(float(clamp_range[1]), 4.0, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("original A4 upper clamp must remain 4.0")
    if lambda_raw <= float(clamp_range[1]):
        raise ValueError("this follow-up requires the original raw lambda to exceed the clamp")

    manifest_path = _resolve(data["cache_manifest"])
    if _sha256(manifest_path) != str(data["cache_manifest_sha256"]):
        raise ValueError("expanded train cache manifest hash differs")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("artifact_sha256") != data.get("cache_artifact_sha256"):
        raise ValueError("expanded train cache artifact identity differs")
    expected_train_rows = int(data.get("expected_train_rows", -1))
    if expected_train_rows != 8192:
        raise ValueError("A4 lambda CV v1 is frozen to exactly 8192 train rows")
    split_counts = manifest.get("split_counts")
    if not isinstance(split_counts, dict) or int(split_counts.get("train", -1)) != 8192:
        raise ValueError("expanded cache does not contain exactly 8192 train rows")

    train_sequences = [str(value) for value in data.get("train_sequence_ids", [])]
    if len(train_sequences) != 9 or len(set(train_sequences)) != 9:
        raise ValueError("lambda CV requires exactly nine unique train sequences")
    folds_raw = cv.get("folds")
    if not isinstance(folds_raw, list) or not all(isinstance(fold, dict) for fold in folds_raw):
        raise ValueError("cv.folds must be a list of mappings")
    folds = cast(list[dict[str, Any]], folds_raw)
    _validate_fold_protocol(train_sequences, folds)

    lambdas = [float(value) for value in cv.get("candidate_lambdas", [])]
    if (
        len(lambdas) != 5
        or len(set(lambdas)) != 5
        or not all(math.isfinite(value) and value > 0 for value in lambdas)
    ):
        raise ValueError("candidate_lambdas must contain five unique positive finite values")
    if not any(math.isclose(value, 4.0, rel_tol=0.0, abs_tol=1e-12) for value in lambdas):
        raise ValueError("lambda=4.0 control is mandatory")
    if not any(math.isclose(value, lambda_raw, rel_tol=0.0, abs_tol=1e-12) for value in lambdas):
        raise ValueError("the original unclipped lambda_raw must be a candidate")
    if max(lambdas) <= 4.0:
        raise ValueError("candidate grid must extend above the original clamp")

    teacher_cfg = data.get("dinov3_relational_teacher")
    if not isinstance(teacher_cfg, dict):
        raise ValueError("data.dinov3_relational_teacher is required")
    teacher_manifest_path = _resolve(teacher_cfg["manifest"])
    if _sha256(teacher_manifest_path) != str(teacher_cfg["manifest_sha256"]):
        raise ValueError("8192 DINO teacher manifest hash differs")

    base_dataset = GarlTTCObjectEventV4Dataset(str(manifest_path), splits=("train",))
    if len(base_dataset) != 8192:
        raise ValueError(f"event train dataset length is {len(base_dataset)}, expected 8192")

    print("[lambda-cv] indexing all 8192 train rows by sequence...")
    sequence_indices: dict[str, list[int]] = {sequence: [] for sequence in train_sequences}
    sample_tokens: set[str] = set()
    for index in range(len(base_dataset)):
        record = base_dataset[index]
        sequence = str(record["sequence_id"])
        token = str(record["sample_token"])
        if sequence not in sequence_indices:
            raise ValueError(f"train cache contains unexpected sequence {sequence}")
        if token in sample_tokens:
            raise ValueError(f"duplicate train sample_token: {token}")
        sample_tokens.add(token)
        sequence_indices[sequence].append(index)
    if any(not indices for indices in sequence_indices.values()):
        raise ValueError("at least one train sequence has zero rows")
    if sum(map(len, sequence_indices.values())) != 8192:
        raise ValueError("sequence partition does not cover all 8192 train rows")

    teacher_dataset = DINOv3RelationalTeacherDataset(
        base_dataset,
        manifest_path=teacher_manifest_path,
        expected_artifact_sha256=str(teacher_cfg["artifact_sha256"]),
        expected_manifest_sha256=str(teacher_cfg["manifest_sha256"]),
    )
    if len(teacher_dataset) != 8192:
        raise ValueError("DINO teacher wrapper length differs from 8192")

    model_path = _resolve(raw["model_config"])
    if _sha256(model_path) != str(raw.get("model_config_sha256", "")):
        raise ValueError("model config hash differs from frozen A4 architecture")
    model_config = _model_config(model_path)
    parameter_count = sum(
        parameter.numel() for parameter in CausalScaleTTC(model_config).parameters()
    )
    if parameter_count != int(decision.get("expected_parameter_count", -1)):
        raise ValueError("model parameter count differs from frozen A4 architecture")

    training_raw = raw.get("training")
    loss_raw = raw.get("loss")
    if not isinstance(training_raw, dict) or not isinstance(loss_raw, dict):
        raise ValueError("training and loss mappings are required")
    base_training = CausalScaleEAPTrainingConfig(**training_raw)
    loss_config = CausalScaleTTCLossConfig(**loss_raw)
    if base_training.representation_supervision != "dinov3_local_relational":
        raise ValueError("lambda CV requires DINO relational supervision")
    if base_training.representation_teacher_cache_artifact_sha256 != teacher_cfg["artifact_sha256"]:
        raise ValueError("training and DINO teacher artifact identities differ")

    if output_dir.exists():
        raise FileExistsError(f"output directory already exists: {output_dir}")
    output_dir.mkdir(parents=True)
    runs_dir = output_dir / "runs"
    runs_dir.mkdir()
    device = resolve_device(device_name)
    config_sha256 = _sha256(config_path)
    all_candidate_results: list[dict[str, Any]] = []
    started = time.perf_counter()

    for candidate_index, lambda_value in enumerate(lambdas, start=1):
        print(
            f"\n[lambda-cv] candidate {candidate_index}/{len(lambdas)}: "
            f"lambda={lambda_value:.12g}"
        )
        fold_summaries: list[dict[str, Any]] = []
        for fold in folds:
            fold_name = str(fold["name"])
            fold_seed = int(fold["seed"])
            heldout_sequences = [str(value) for value in fold["heldout_sequences"]]
            heldout_set = set(heldout_sequences)
            train_fold_sequences = [
                sequence for sequence in train_sequences if sequence not in heldout_set
            ]
            train_indices = sorted(
                index
                for sequence in train_fold_sequences
                for index in sequence_indices[sequence]
            )
            heldout_indices = sorted(
                index
                for sequence in heldout_sequences
                for index in sequence_indices[sequence]
            )
            if set(train_indices) & set(heldout_indices):
                raise RuntimeError("fold train/heldout indices overlap")
            if len(train_indices) + len(heldout_indices) != 8192:
                raise RuntimeError("fold train/heldout indices do not partition 8192 rows")

            train_subset = IndexedDataset(teacher_dataset, train_indices)
            heldout_subset = IndexedDataset(base_dataset, heldout_indices)
            training_config = replace(
                base_training,
                seed=fold_seed,
                representation_distillation_weight=lambda_value,
            )
            print(
                f"  [{fold_name}] seed={fold_seed} train={len(train_subset)} "
                f"heldout={len(heldout_subset)} sequences={heldout_sequences}"
            )
            fold_started = time.perf_counter()
            result = train_real_causal_scale(
                model_config,
                training_config,
                loss_config,
                train_subset,
                heldout_subset,
                device,
            )
            elapsed = time.perf_counter() - fold_started
            validation = result.best_validation
            observed_heldout = set(str(value) for value in validation["sequence_ids"])
            if observed_heldout != heldout_set:
                raise RuntimeError(
                    f"{fold_name} held-out sequences differ: "
                    f"{sorted(observed_heldout)} != {sorted(heldout_set)}"
                )
            per_sequence = validation["sequence_macro"].get("per_sequence")
            if not isinstance(per_sequence, dict) or set(per_sequence) != heldout_set:
                raise RuntimeError(
                    f"{fold_name} per-sequence metrics do not match held-out sequences"
                )

            lambda_tag = str(lambda_value).replace(".", "p")
            run_dir = runs_dir / f"lambda_{lambda_tag}" / fold_name
            run_dir.mkdir(parents=True, exist_ok=False)
            predictions_path = run_dir / "heldout_predictions.csv"
            _write_predictions(predictions_path, validation)
            compact_history = [
                {
                    "epoch": int(record["epoch"]),
                    "learning_rate": float(record["learning_rate"]),
                    "foreground_warmup": bool(record["foreground_warmup"]),
                    "selection": record["selection"],
                    "train": record["train"],
                    "validation": record["validation"],
                }
                for record in result.history
            ]
            fold_summary = {
                "fold": fold_name,
                "seed": fold_seed,
                "lambda": lambda_value,
                "train_sequences": train_fold_sequences,
                "heldout_sequences": heldout_sequences,
                "train_rows": len(train_subset),
                "heldout_rows": len(heldout_subset),
                "best_epoch": result.best_epoch,
                "best_selection": result.best_selection,
                "num_samples": int(validation["num_samples"]),
                "failure_rate_pct": float(validation["signed"]["failure_rate_pct"]),
                "per_sequence": per_sequence,
                "log_ratio_pearson": float(validation["log_ratio_pearson"]),
                "elapsed_seconds": elapsed,
                "predictions_file": predictions_path.relative_to(output_dir).as_posix(),
                "predictions_sha256": _sha256(predictions_path),
                "history": compact_history,
            }
            fold_summary = cast(dict[str, Any], _finite_json(fold_summary))
            fold_summary_path = run_dir / "summary.json"
            fold_summary_path.write_text(
                json.dumps(fold_summary, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            fold_summary["summary_file"] = fold_summary_path.relative_to(output_dir).as_posix()
            fold_summary["summary_sha256"] = _sha256(fold_summary_path)
            fold_summaries.append(fold_summary)
            print(
                f"  [{fold_name}] best_epoch={result.best_epoch} "
                f"MiD={result.best_selection['sequence_macro_MiD']:.6f} "
                f"failure={result.best_selection['failure_rate_pct']:.6f}%"
            )
            del result, train_subset, heldout_subset
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()

        aggregate = _aggregate_candidate(lambda_value, fold_summaries, train_sequences)
        aggregate["folds"] = fold_summaries
        all_candidate_results.append(aggregate)
        print(
            f"[lambda-cv] lambda={lambda_value:.12g} aggregate "
            f"MiD={aggregate['nine_sequence_macro_MiD']:.6f} "
            f"failure={aggregate['sample_weighted_failure_rate_pct']:.6f}%"
        )

    selected = _select_best_candidate(all_candidate_results)
    selected_lambda = float(selected["lambda"])
    lambda_min = min(lambdas)
    lambda_max = max(lambdas)
    boundary_hit = (
        math.isclose(selected_lambda, lambda_min, rel_tol=0.0, abs_tol=1e-12)
        or math.isclose(selected_lambda, lambda_max, rel_tol=0.0, abs_tol=1e-12)
    )
    payload: dict[str, Any] = {
        "artifact_type": "a4_dinov3_relational_lambda_train_only_group_cv_v1",
        "created_at": datetime.now(UTC).isoformat(),
        "status": "boundary_hit" if boundary_hit else "passed",
        "experiment": experiment,
        "scope": {
            "public_train_only": True,
            "opened_splits": ["train"],
            "public_validation_samples_opened": 0,
            "official_test_opened": False,
            "codabench_opened": False,
            "evttc_test_opened": False,
            "optimizer_steps_are_train_fold_only": True,
        },
        "code_identity": code_identity,
        "config": {
            "path": config_path.relative_to(ROOT).as_posix(),
            "sha256": config_sha256,
        },
        "data": {
            "train_cache_manifest": manifest_path.relative_to(ROOT).as_posix(),
            "train_cache_manifest_sha256": _sha256(manifest_path),
            "train_cache_artifact_sha256": str(manifest["artifact_sha256"]),
            "train_rows": len(base_dataset),
            "train_sequence_ids": train_sequences,
            "rows_per_sequence": {
                key: len(value) for key, value in sorted(sequence_indices.items())
            },
            "teacher_manifest": teacher_manifest_path.relative_to(ROOT).as_posix(),
            "teacher_manifest_sha256": _sha256(teacher_manifest_path),
            "teacher_artifact_sha256": str(teacher_cfg["artifact_sha256"]),
        },
        "original_a4_calibration": {
            "file": calibration_path.relative_to(ROOT).as_posix(),
            "file_sha256": _sha256(calibration_path),
            "artifact_sha256": str(calibration["artifact_sha256"]),
            "lambda_raw": lambda_raw,
            "selected_weight": selected_original,
            "clamp_range": clamp_range,
            "raw_lambda_exceeded_upper_clamp": True,
        },
        "protocol": {
            "candidate_lambdas": lambdas,
            "folds": folds,
            "training_config_template": asdict(base_training),
            "loss_config": asdict(loss_config),
            "parameter_count": parameter_count,
            "selection_order": cv.get("selection_order"),
            "every_train_sequence_held_out_exactly_once": True,
            "same_fold_seed_for_every_lambda": True,
            "DINO_teacher_is_never_attached_to_heldout_fold": True,
        },
        "candidates": all_candidate_results,
        "selected_lambda_candidate": selected_lambda,
        "lambda_grid_boundary_hit": boundary_hit,
        "promotion_ready": not boundary_hit,
        "selected_candidate": selected,
        "decision_contract": decision,
        "elapsed_seconds": time.perf_counter() - started,
        "claim_boundary": {
            "lambda_selection_is_adaptive_followup_after_A4": True,
            "original_A4_lambda_4_result_is_immutable": True,
            "selected_lambda_is_for_future_followup_only": not boundary_hit,
            "boundary_candidate_requires_grid_extension_before_promotion": boundary_hit,
            "public_validation_was_not_used_for_lambda_selection": True,
            "private_test_was_not_opened": True,
        },
    }
    payload = cast(dict[str, Any], _finite_json(payload))
    sign_artifact(payload)
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if boundary_hit:
        (output_dir / "boundary_lambda_candidate.txt").write_text(
            f"{selected_lambda:.17g}\n", encoding="utf-8"
        )
    else:
        (output_dir / "selected_lambda.txt").write_text(
            f"{selected_lambda:.17g}\n", encoding="utf-8"
        )
    print("\n============================================================")
    print("A4 DINO TRAIN-ONLY LAMBDA CV COMPLETE")
    print(f"selected_lambda_candidate = {selected_lambda:.17g}")
    print(f"lambda_grid_boundary_hit = {boundary_hit}")
    print(f"promotion_ready = {not boundary_hit}")
    print(f"nine_sequence_macro_MiD = {float(selected['nine_sequence_macro_MiD']):.6f}")
    print(
        "sample_weighted_failure_rate_pct = "
        f"{float(selected['sample_weighted_failure_rate_pct']):.6f}"
    )
    print(f"artifact_sha256 = {payload['artifact_sha256']}")
    print(f"summary = {summary_path}")
    print("public validation opened = 0")
    print("============================================================")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    device = args.device
    if device == "auto":
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    try:
        run(args.config.resolve(), args.output_dir.resolve(), device_name=device)
    except Exception as error:
        parser.exit(2, f"A4 lambda train-only CV failed: {type(error).__name__}: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

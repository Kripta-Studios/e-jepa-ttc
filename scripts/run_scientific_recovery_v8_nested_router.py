#!/usr/bin/env python
# ruff: noqa: E501, E701, E702
"""Execute one prospective, nested A5/C2F router outer fold.

Real mode accepts only signed expert prediction artifacts.  Those artifacts are
produced by the V8 expert-training contract: three disjoint inner-dev exports for
the six outer-train sequences, then one outer-dev export after final expert fits.
This script never substitutes historical OOF files, and it never evaluates a sealed
split.  ``--fixture-smoke`` is deliberately separate and marks every output as a
fixture so it cannot be aggregated as scientific evidence.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from e_jepa_ttc.artifacts.hashing import sign_artifact, verify_artifact_hash  # noqa: E402
from e_jepa_ttc.evaluation.nested_router import (  # noqa: E402
    INNER_OOF_COLUMNS,
    InnerFold,
    NestedRouterIntegrityError,
    fit_router_from_inner_oof,
    validate_inner_folds,
)
from e_jepa_ttc.evaluation.scientific_recovery_v8 import (  # noqa: E402
    OOF_V8_REQUIRED_COLUMNS,
)
from e_jepa_ttc.evaluation.scientific_recovery_v8_runner import (  # noqa: E402
    V8IntegrityError,
    verify_frozen_inputs,
)
from e_jepa_ttc.models.causal_expert_router import ROUTER_FEATURES  # noqa: E402
from e_jepa_ttc.scientific_provenance import (  # noqa: E402
    ScientificProvenanceError,
    assert_router_expert_reusable,
    refuse_scientific_bypass_env,
    require_clean_scientific_worktree,
)


class RouterStageError(ValueError):
    """Raised when a router run does not meet the frozen prospective contract."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _repo_relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError as error:
        raise RouterStageError(f"router artifact path escapes repository: {path}") from error


def _signed_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RouterStageError(f"cannot read signed {label}: {path}") from error
    if not isinstance(payload, dict) or not verify_artifact_hash(payload):
        raise RouterStageError(f"{label} is not a valid signed artifact: {path}")
    return payload


def _reference_path(reference: object, *, artifact: Path, label: str) -> Path:
    if not isinstance(reference, Mapping):
        raise RouterStageError(f"{label} artifact lacks a CSV reference")
    value = reference.get("path")
    expected = reference.get("sha256")
    if not isinstance(value, str) or not isinstance(expected, str) or len(expected) != 64:
        raise RouterStageError(f"{label} CSV reference is malformed")
    candidate = (ROOT / value).resolve() if not Path(value).is_absolute() else Path(value).resolve()
    try:
        candidate.relative_to(ROOT.resolve())
    except ValueError as error:
        raise RouterStageError(f"{label} CSV reference escapes repository") from error
    if not candidate.is_file() or _sha256(candidate) != expected:
        raise RouterStageError(f"{label} CSV digest does not match its signed artifact")
    return candidate


def _load_expert_artifact(
    path: Path,
    *,
    expert: str,
    role: str,
    protocol_sha256: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load one exact, signed expert export and bind its checkpoint/protocol."""

    payload = _signed_json(path, label=f"{expert} {role} expert")
    if payload.get("artifact_type") != "scientific_recovery_v8_router_expert_prediction_v1":
        raise RouterStageError("signed expert artifact has an incompatible artifact_type")
    if payload.get("status") != "completed" or payload.get("fixture") is True:
        raise RouterStageError(
            "real router runs require completed, non-fixture signed expert artifacts"
        )
    if payload.get("expert") != expert or payload.get("role") != role:
        raise RouterStageError(
            "signed expert artifact expert/role does not match the requested input"
        )
    if payload.get("protocol_sha256") != protocol_sha256:
        raise RouterStageError("signed expert artifact protocol hash differs from frozen V8")
    bound = dict(payload)
    summary_path = path.parent / "train" / "summary.json"
    if summary_path.is_file():
        summary = _signed_json(summary_path, label=f"{expert} {role} train summary")
        summary_dirty = summary.get("git_dirty")
        if bound.get("git_dirty") is None:
            bound["git_dirty"] = summary_dirty
            bound.setdefault("git_commit", summary.get("git_commit"))
        elif bound.get("git_dirty") != summary_dirty:
            raise RouterStageError(
                f"{expert} {role} expert git_dirty disagrees with train/summary.json"
            )
    elif bound.get("git_dirty") is None:
        raise RouterStageError(
            f"{expert} {role} expert omitted git_dirty and train/summary.json is missing"
        )
    try:
        assert_router_expert_reusable(bound)
    except ScientificProvenanceError as error:
        raise RouterStageError(str(error)) from error
    checkpoint = payload.get("checkpoint")
    if not isinstance(checkpoint, Mapping) or not isinstance(checkpoint.get("sha256"), str):
        raise RouterStageError("signed expert artifact lacks a bound checkpoint SHA-256")
    csv_path = _reference_path(payload.get("oof_csv"), artifact=path, label=f"{expert} {role}")
    frame = pd.read_csv(csv_path)
    return frame, payload


def _inner_folds(config: Mapping[str, Any]) -> tuple[InnerFold, ...]:
    raw = config.get("inner_folds")
    if not isinstance(raw, list):
        raise RouterStageError("router config lacks frozen inner_folds")
    folds: list[InnerFold] = []
    for item in raw:
        if not isinstance(item, Mapping):
            raise RouterStageError("router config has malformed inner fold")
        folds.append(
            InnerFold(
                index=int(item["inner_fold"]),
                train_sequences=tuple(str(value) for value in item["train_sequence_ids"]),
                dev_sequences=tuple(str(value) for value in item["dev_sequence_ids"]),
            )
        )
    outer_train = tuple(str(value) for value in config["outer_train_sequence_ids"])
    outer_dev = tuple(str(value) for value in config["outer_dev_sequence_ids"])
    validate_inner_folds(folds, outer_train_sequences=outer_train, outer_dev_sequences=outer_dev)
    return tuple(folds)


def _required_expert_columns(*, inner: bool) -> tuple[str, ...]:
    base = (*OOF_V8_REQUIRED_COLUMNS, *ROUTER_FEATURES)
    return (*base, "inner_fold") if inner else base


def _validate_expert_frame(frame: pd.DataFrame, *, label: str, inner: bool) -> pd.DataFrame:
    required = _required_expert_columns(inner=inner)
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise RouterStageError(f"{label} export lacks required columns: {missing}")
    checked = frame.loc[:, list(required)].copy()
    if checked["token_id"].duplicated().any():
        raise RouterStageError(f"{label} export has duplicate token_id values")
    if not checked["finite"].map(lambda value: isinstance(value, (bool, np.bool_))).all():
        raise RouterStageError(f"{label} export finite must be boolean")
    if not checked["finite"].all():
        raise RouterStageError(f"{label} export has non-finite expert predictions")
    numeric = (
        "target_ttc",
        "sample_weight",
        "prediction_ttc",
        "prediction_log_variance",
        *ROUTER_FEATURES,
    )
    for name in numeric:
        checked[name] = pd.to_numeric(checked[name], errors="raise")
        if not np.isfinite(checked[name].to_numpy(dtype=np.float64)).all():
            raise RouterStageError(f"{label} export contains non-finite {name}")
    return checked.sort_values("token_id", kind="stable").reset_index(drop=True)


def _combine_experts(
    a5: pd.DataFrame,
    c2f: pd.DataFrame,
    *,
    inner: bool,
) -> pd.DataFrame:
    """Align expert OOF exports and retain only the frozen label-free features."""

    a5_checked = _validate_expert_frame(a5, label="A5", inner=inner)
    c2f_checked = _validate_expert_frame(c2f, label="C2F", inner=inner)
    identities = ("token_id", "sequence_id", "track_id")
    merged = a5_checked.merge(
        c2f_checked,
        on=list(identities),
        suffixes=("_a5", "_c2f"),
        how="inner",
        validate="one_to_one",
    )
    if len(merged) != len(a5_checked) or len(merged) != len(c2f_checked):
        raise RouterStageError("A5/C2F expert exports do not have exact row identity alignment")
    for name in (
        "target_ttc",
        "sample_weight",
        "outer_fold",
        "seed",
        "event_count",
        "event_rate",
        "support_ms",
    ):
        if not np.array_equal(merged[f"{name}_a5"].to_numpy(), merged[f"{name}_c2f"].to_numpy()):
            raise RouterStageError(f"A5/C2F export mismatch for protected field {name}")
    if inner and not np.array_equal(
        merged["inner_fold_a5"].to_numpy(), merged["inner_fold_c2f"].to_numpy()
    ):
        raise RouterStageError("A5/C2F inner OOF fold assignments differ")
    for name in ("shared_event_count_log1p", "shared_event_rate_log1p"):
        if not np.allclose(
            merged[f"{name}_a5"].to_numpy(dtype=np.float64),
            merged[f"{name}_c2f"].to_numpy(dtype=np.float64),
            rtol=0.0,
            atol=0.0,
        ):
            raise RouterStageError(f"A5/C2F export mismatch for shared causal feature {name}")
    result = pd.DataFrame(
        {
            "sample_token": merged["token_id"].astype(str),
            "sequence_id": merged["sequence_id"].astype(str),
            "track_id": merged["track_id"].astype(str),
            "target_ttc_s": merged["target_ttc_a5"].astype(float),
            "a5_prediction_ttc_s": merged["prediction_ttc_a5"].astype(float),
            "c2f_prediction_ttc_s": merged["prediction_ttc_c2f"].astype(float),
            "sample_weight": merged["sample_weight_a5"].astype(float),
            "event_count": merged["event_count_a5"].astype(float),
            "event_rate": merged["event_rate_a5"].astype(float),
            "support_ms": merged["support_ms_a5"].astype(float),
            "shared_event_count_log1p": merged["shared_event_count_log1p_a5"].astype(float),
            "shared_event_rate_log1p": merged["shared_event_rate_log1p_a5"].astype(float),
            "a5_flow": merged["a5_flow_a5"].astype(float),
            "a5_margin": merged["a5_margin_a5"].astype(float),
            "a5_log_variance": merged["a5_log_variance_a5"].astype(float),
            "c2f_flow": merged["c2f_flow_c2f"].astype(float),
            "c2f_margin": merged["c2f_margin_c2f"].astype(float),
            "c2f_log_variance": merged["c2f_log_variance_c2f"].astype(float),
        }
    )
    if inner:
        result.insert(3, "inner_fold", merged["inner_fold_a5"].astype(int))
        return result.loc[:, list(INNER_OOF_COLUMNS)]
    result.insert(3, "outer_fold", merged["outer_fold_a5"].astype(int))
    result.insert(4, "seed", merged["seed_a5"].astype(int))
    return result


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise RouterStageError("unable to resolve current git commit") from error


def run_fold(
    *,
    config_path: Path,
    protocol_path: Path,
    manifest_path: Path,
    output_dir: Path,
    a5_inner: pd.DataFrame,
    c2f_inner: pd.DataFrame,
    a5_outer: pd.DataFrame,
    c2f_outer: pd.DataFrame,
    expert_artifacts: Mapping[str, Mapping[str, Any]] | None,
    fixture: bool,
) -> dict[str, Any]:
    """Fit on strict inner OOF and route one untouched outer-dev population."""

    if not fixture:
        refuse_scientific_bypass_env()
        require_clean_scientific_worktree()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, Mapping):
        raise RouterStageError("router config must be a mapping")
    folds = _inner_folds(config)
    outer_dev = tuple(str(value) for value in config["outer_dev_sequence_ids"])
    seed = int(config["experiment"]["seed"])
    outer_fold = int(config["outer_fold"])
    frozen = None
    if not fixture:
        frozen = verify_frozen_inputs(protocol_path, manifest_path)
        screen_seed = int(frozen.protocol["training_contract"]["screen_seed"])
        config_seed = int(config["experiment"]["seed"])
        replication = config["experiment"].get("multiseed_replication")
        allowed_replication = (
            config_seed in {13, 23}
            and isinstance(replication, Mapping)
            and int(replication.get("source_seed", -1)) == screen_seed
            and replication.get("no_tuning") is True
            and replication.get("no_reselection") is True
        )
        if config_seed != screen_seed and not allowed_replication:
            raise RouterStageError("router config seed is neither frozen screen seed nor derived replication")
    inner = _combine_experts(a5_inner, c2f_inner, inner=True)
    fitted = fit_router_from_inner_oof(
        inner, inner_folds=folds, outer_dev_sequences=outer_dev, seed=seed
    )
    outer = _combine_experts(a5_outer, c2f_outer, inner=False)
    if set(outer["sequence_id"]) != set(outer_dev):
        raise RouterStageError(
            "outer-dev expert exports do not contain exactly the frozen dev sequences"
        )
    if set(outer["outer_fold"].astype(int)) != {outer_fold}:
        raise RouterStageError("outer-dev expert export fold differs from router config")
    predictions, choose_c2f, probability = fitted.router.route(
        outer.loc[:, ROUTER_FEATURES],
        a5_prediction_ttc_s=outer["a5_prediction_ttc_s"].to_numpy(),
        c2f_prediction_ttc_s=outer["c2f_prediction_ttc_s"].to_numpy(),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    output = pd.DataFrame(
        {
            "token_id": outer["sample_token"],
            "sequence_id": outer["sequence_id"],
            "track_id": outer["track_id"],
            "outer_fold": outer["outer_fold"],
            "seed": outer["seed"],
            "target_ttc": outer["target_ttc_s"],
            "sample_weight": outer["sample_weight"],
            "prediction_ttc": predictions,
            "prediction_log_variance": np.where(
                choose_c2f, outer["c2f_log_variance"], outer["a5_log_variance"]
            ),
            "finite": True,
            "failure_reason": "",
            "event_count": outer["event_count"],
            "event_rate": outer["event_rate"],
            "support_ms": outer["support_ms"],
            "model_name": "scientific_recovery_v8_router_a5_c2f",
            "config_sha256": _sha256(config_path),
            "checkpoint_sha256": "pending_router_file_sha256",
            "a5_prediction_ttc": outer["a5_prediction_ttc_s"],
            "c2f_prediction_ttc": outer["c2f_prediction_ttc_s"],
            "choose_c2f": choose_c2f,
            "c2f_probability": probability,
        }
    )
    csv_path = output_dir / "outer_dev_predictions.csv"
    output.to_csv(csv_path, index=False, lineterminator="\n")
    router_path = output_dir / "router_signature.json"
    router_path.write_text(
        json.dumps(fitted.router.signature.payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    router_file_sha256 = _sha256(router_path)
    output["checkpoint_sha256"] = router_file_sha256
    # Rewrite after binding each row to the exact router file used for inference.
    output.to_csv(csv_path, index=False, lineterminator="\n")
    artifact: dict[str, Any] = {
        "artifact_type": "scientific_recovery_v8_router_fold_v1",
        "status": "fixture_completed" if fixture else "completed",
        "fixture": fixture,
        "stage": "router",
        "arm": "router",
        "run_name": str(config["experiment"]["name"]),
        "outer_fold": outer_fold,
        "seed": seed,
        "git_commit": _git_commit(),
        "protocol_sha256": "fixture" if fixture else str(frozen.protocol["artifact_sha256"]),
        "config_sha256": _sha256(config_path),
        "inner_oof_rows": int(len(inner)),
        "inner_oof_tokens_sha256": hashlib.sha256(
            "\n".join(sorted(inner["sample_token"].astype(str))).encode("utf-8")
        ).hexdigest(),
        "router_signature": {
            "path": str(router_path) if fixture else _repo_relative(router_path),
            "sha256": _sha256(router_path),
            "artifact_sha256": fitted.router.signature.artifact_sha256,
        },
        "outer_dev_oof": {
            "path": str(csv_path) if fixture else _repo_relative(csv_path),
            "sha256": _sha256(csv_path),
        },
        "router_choice": {
            "threshold": 0.5,
            "hard_routing": True,
            "c2f_fraction": float(np.mean(choose_c2f)),
        },
        "integrity_checks": {
            "inner_oof_only": True,
            "outer_dev_excluded_from_fit": True,
            "inner_folds_disjoint": True,
            "feature_schema_exact": True,
            "scaler_inner_oof_only": True,
            "class_weight_none": True,
            "threshold_fixed_at_0_5": True,
            "sealed_evaluation_closed": True,
        },
    }
    if expert_artifacts is not None:
        artifact["expert_artifacts"] = {
            key: {"artifact_sha256": value["artifact_sha256"]}
            for key, value in expert_artifacts.items()
        }
    artifact = sign_artifact(artifact)
    name = "router_fold_fixture.json" if fixture else f"router_fold{outer_fold}.json"
    fold_artifact_path = output_dir / name
    fold_artifact_path.write_text(
        json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if not fixture:
        canonical_csv = output_dir / "dev_predictions.csv"
        if canonical_csv != csv_path:
            canonical_csv.write_bytes(csv_path.read_bytes())
        summary = sign_artifact(
            {
                "artifact_type": "scientific_recovery_v8_fold_result_v1",
                "status": "completed",
                "fixture": False,
                "stage": "router",
                "arm": "router",
                "run_name": str(config["experiment"]["name"]),
                "outer_fold": outer_fold,
                "fold": outer_fold,
                "seed": seed,
                "base_git_commit": str(frozen.protocol.get("git_base_commit", "")),
                "implementation_git_commit": _git_commit(),
                "git_commit": _git_commit(),
                "protocol_sha256": str(frozen.protocol["artifact_sha256"]),
                "frozen_manifest_sha256": str(frozen.manifest["artifact_sha256"]),
                "config_sha256": _sha256(config_path),
                "checkpoint": {"path": "router_signature.json", "sha256": router_file_sha256},
                "dev_predictions": {"path": "dev_predictions.csv", "sha256": _sha256(canonical_csv)},
                "fold_artifact": {
                    "path": name,
                    "sha256": _sha256(fold_artifact_path),
                    "artifact_sha256": artifact["artifact_sha256"],
                },
                "closed_evaluation": {
                    "public_validation_used_for_selection": False,
                    "private_test_opened": False,
                    "evttc_test_opened": False,
                    "codabench_opened": False,
                },
            }
        )
        (output_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    return artifact


def _fixture_expert(*, expert: str, inner: bool) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    sequences = (("s0", "s3"), ("s1", "s4"), ("s2", "s5")) if inner else (("s6", "s7", "s8"),)
    row = 0
    for fold, values in enumerate(sequences):
        for sequence in values:
            for local in range(2):
                c2f_wins = row % 2 == 0
                prediction = 1.8 if (expert == "C2F") == c2f_wins else 2.8
                record = {
                    "token_id": f"fixture-{expert}-{int(inner)}-{row}-{local}",
                    "sequence_id": sequence,
                    "track_id": f"track-{sequence}",
                    "outer_fold": 0,
                    "seed": 7,
                    "target_ttc": 2.0,
                    "sample_weight": 1.0 / 36.0,
                    "prediction_ttc": prediction,
                    "prediction_log_variance": 0.1,
                    "finite": True,
                    "failure_reason": "",
                    "event_count": 10.0,
                    "event_rate": 5.0,
                    "support_ms": 20.0,
                    "model_name": expert,
                    "config_sha256": "a" * 64,
                    "checkpoint_sha256": "b" * 64,
                    "shared_event_count_log1p": np.log1p(10.0),
                    "shared_event_rate_log1p": np.log1p(5.0),
                    "a5_flow": 0.2,
                    "a5_margin": 0.1,
                    "a5_log_variance": 0.1,
                    "c2f_flow": 0.3,
                    "c2f_margin": 0.2,
                    "c2f_log_variance": 0.1,
                }
                if inner:
                    record["inner_fold"] = fold
                records.append(record)
                row += 1
    frame = pd.DataFrame(records)
    # Identical identities are mandatory across the two expert exports.
    frame["token_id"] = frame["token_id"].str.replace(f"fixture-{expert}", "fixture", regex=False)
    return frame


def _fixture_config(output_dir: Path) -> Path:
    config = {
        "experiment": {"name": "scientific_recovery_v8_router_fixture", "seed": 7},
        "outer_fold": 0,
        "outer_train_sequence_ids": ["s0", "s1", "s2", "s3", "s4", "s5"],
        "outer_dev_sequence_ids": ["s6", "s7", "s8"],
        "inner_folds": [
            {
                "inner_fold": 0,
                "train_sequence_ids": ["s1", "s2", "s4", "s5"],
                "dev_sequence_ids": ["s0", "s3"],
            },
            {
                "inner_fold": 1,
                "train_sequence_ids": ["s0", "s2", "s3", "s5"],
                "dev_sequence_ids": ["s1", "s4"],
            },
            {
                "inner_fold": 2,
                "train_sequence_ids": ["s0", "s1", "s3", "s4"],
                "dev_sequence_ids": ["s2", "s5"],
            },
        ],
    }
    path = output_dir / "fixture_router.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return path


def _router_stage_plan(
    *, frozen: object, results_root: Path, device: str, max_parallel: int,
    config_paths: Sequence[Path] | None = None, seed: int = 7,
) -> dict[str, Any]:
    """Return the exact 24 expert jobs required before the three router fits.

    The plan is derived only from the frozen router configurations.  It intentionally
    contains neither a historical OOF path nor a sealed evaluation split.  The
    dedicated trainer command is a repository contract: execution fails closed while
    the production expert trainer is unavailable rather than using fixture training.
    """

    manifest = frozen.manifest
    if config_paths is None:
        entries = manifest.get("enabled_seed7_configs")
        if not isinstance(entries, Mapping):
            raise RouterStageError("frozen manifest lacks enabled seed-7 configurations")
        config_paths = sorted(
            ROOT / str(entry["path"])
            for name, entry in entries.items()
            if str(name).startswith("router_fold") and str(name).endswith("_seed7")
        )
    else:
        config_paths = sorted(Path(path).resolve() for path in config_paths)
    if len(config_paths) != 3:
        raise RouterStageError(f"router stage requires exactly three configs for seed {seed}")
    expert_jobs: list[dict[str, Any]] = []
    router_commands: list[list[str]] = []
    for config_path in config_paths:
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        if not isinstance(config, Mapping):
            raise RouterStageError(f"router config is malformed: {config_path}")
        fold = int(config["outer_fold"])
        run_root = results_root / "router" / f"outer_fold{fold}_seed{seed}"
        inner_folds = config.get("inner_folds")
        if not isinstance(inner_folds, list) or len(inner_folds) != 3:
            raise RouterStageError(f"router config lacks three inner folds: {config_path}")
        for expert in ("A5", "C2F"):
            for inner in inner_folds:
                if not isinstance(inner, Mapping):
                    raise RouterStageError("router config inner fold is malformed")
                inner_fold = int(inner["inner_fold"])
                output = run_root / expert.lower() / f"inner{inner_fold}"
                expert_jobs.append(
                    {
                        "job_id": f"router_{expert.lower()}_outer{fold}_inner{inner_fold}_seed{seed}",
                        "expert": expert,
                        "role": "inner_oof",
                        "outer_fold": fold,
                        "inner_fold": inner_fold,
                        "train_sequence_ids": list(inner["train_sequence_ids"]),
                        "dev_sequence_ids": list(inner["dev_sequence_ids"]),
                        "config": _repo_relative(config_path),
                        "output_dir": _repo_relative(output),
                        "command": [
                            "uv",
                            "run",
                            "--no-sync",
                            "python",
                            "scripts/train_scientific_recovery_v8_router_expert.py",
                            "--config",
                            _repo_relative(config_path),
                            "--expert",
                            expert,
                            "--role",
                            "inner_oof",
                            "--inner-fold",
                            str(inner_fold),
                            "--output-dir",
                            _repo_relative(output),
                            "--device",
                            device,
                            "--protocol-sha256",
                            str(frozen.protocol["artifact_sha256"]),
                        ],
                    }
                )
            output = run_root / expert.lower() / "outer_dev"
            expert_jobs.append(
                {
                    "job_id": f"router_{expert.lower()}_outer{fold}_final_seed{seed}",
                    "expert": expert,
                    "role": "outer_dev",
                    "outer_fold": fold,
                    "inner_fold": None,
                    "train_sequence_ids": list(config["outer_train_sequence_ids"]),
                    "dev_sequence_ids": list(config["outer_dev_sequence_ids"]),
                    "config": _repo_relative(config_path),
                    "output_dir": _repo_relative(output),
                    "command": [
                        "uv",
                        "run",
                        "--no-sync",
                        "python",
                        "scripts/train_scientific_recovery_v8_router_expert.py",
                        "--config",
                        _repo_relative(config_path),
                        "--expert",
                        expert,
                        "--role",
                        "outer_dev",
                        "--output-dir",
                        _repo_relative(output),
                        "--device",
                        device,
                        "--protocol-sha256",
                        str(frozen.protocol["artifact_sha256"]),
                    ],
                }
            )
        router_commands.append(
            [
                "uv",
                "run",
                "--no-sync",
                "python",
                "scripts/run_scientific_recovery_v8_nested_router.py",
                "--config",
                _repo_relative(config_path),
                "--a5-inner-artifact",
                _repo_relative(run_root / "a5" / "inner_oof_artifact.json"),
                "--c2f-inner-artifact",
                _repo_relative(run_root / "c2f" / "inner_oof_artifact.json"),
                "--a5-outer-artifact",
                _repo_relative(run_root / "a5" / "outer_dev" / "expert_artifact.json"),
                "--c2f-outer-artifact",
                _repo_relative(run_root / "c2f" / "outer_dev" / "expert_artifact.json"),
                "--output-dir",
                _repo_relative(results_root / "runs" / f"router_fold{fold}_seed{seed}"),
            ]
        )
    return {
        "status": "planned",
        "stage": "router",
        "seed": seed,
        "protocol_sha256": frozen.protocol["artifact_sha256"],
        "device": device,
        "max_parallel": max_parallel,
        "results_root": _repo_relative(results_root),
        "expert_jobs": expert_jobs,
        "router_commands": router_commands,
        "aggregate_command": (
            [
                "uv", "run", "--no-sync", "python",
                "scripts/aggregate_scientific_recovery_v8_router.py",
                *[
                    item
                    for fold in range(3)
                    for item in (
                        "--fold-artifact",
                        _repo_relative(
                            results_root / "runs" / f"router_fold{fold}_seed{seed}" / f"router_fold{fold}.json"
                        ),
                    )
                ],
                "--output-dir", _repo_relative(results_root / "router" / f"aggregate_seed{seed}"),
            ]
            if seed == 7
            else [sys.executable, "-c", f"print('router seed {seed} fold artifacts complete; multiseed aggregator owns the comparison')"]
        ),
        "sealed_evaluation": "closed",
        "production_trainer_required": "scripts/train_scientific_recovery_v8_router_expert.py",
    }


def _execute_stage_plan(plan: Mapping[str, Any], *, max_parallel: int) -> None:
    """Execute a precomputed plan only when its production trainer is present.

    The repository currently keeps the temporal trainer explicitly smoke-only.  This
    guard makes an attempted production launch fail before any substitute command or
    fabricated expert artifact is created.
    """

    refuse_scientific_bypass_env()
    require_clean_scientific_worktree()
    trainer = ROOT / str(plan["production_trainer_required"])
    if not trainer.is_file():
        raise RouterStageError(
            "production V8 router expert trainer is unavailable; the exact plan was emitted "
            "but execution is blocked rather than replacing it with a fixture"
        )
    jobs = plan.get("expert_jobs")
    if not isinstance(jobs, list) or len(jobs) != 24:
        raise RouterStageError("nested router stage plan must contain exactly 24 expert jobs")

    def run_logged(command: list[str], *, label: str, output_dir: Path) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        log_dir = output_dir / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        stdout_path = log_dir / "stdout.log"
        stderr_path = log_dir / "stderr.log"
        command_path = log_dir / "command.json"
        command_path.write_text(
            json.dumps({"label": label, "command": command}, indent=2) + "\n", encoding="utf-8"
        )
        with stdout_path.open("a", encoding="utf-8", buffering=1) as stdout, stderr_path.open(
            "a", encoding="utf-8", buffering=1
        ) as stderr:
            stdout.write(f"\n=== START {label} ===\n")
            stdout.write("COMMAND: " + " ".join(command) + "\n")
            result = subprocess.run(command, cwd=ROOT, stdout=stdout, stderr=stderr, check=False)
            stdout.write(f"=== END {label} exit={result.returncode} ===\n")
        if result.returncode != 0:
            raise RouterStageError(
                f"router job failed: {label}; inspect {stderr_path.relative_to(ROOT)}"
            )

    def execute(job: Mapping[str, Any]) -> None:
        command = job.get("command")
        output_dir = job.get("output_dir")
        if not isinstance(command, list) or not isinstance(output_dir, str):
            raise RouterStageError(f"router expert job is malformed: {job.get('job_id')}")
        run_logged(
            [str(value) for value in command],
            label=str(job.get("job_id")),
            output_dir=ROOT / output_dir,
        )

    # Inner OOF experts are completed first.  Outer expert fits then run as a
    # separate bounded wave, preventing a future fixed-epoch contract from
    # racing ahead of its inner evidence.
    inner_jobs = [job for job in jobs if job.get("role") == "inner_oof"]
    outer_jobs = [job for job in jobs if job.get("role") == "outer_dev"]
    for wave in (inner_jobs, outer_jobs):
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_parallel) as executor:
            futures = [executor.submit(execute, job) for job in wave]
            for future in futures:
                future.result()
    protocol_sha = str(plan["protocol_sha256"])
    results_root = ROOT / str(plan["results_root"])
    seed = int(plan.get("seed", 7))
    for fold in range(3):
        run_root = results_root / "router" / f"outer_fold{fold}_seed{seed}"
        for expert in ("a5", "c2f"):
            artifacts = [
                _signed_json(
                    run_root / expert / f"inner{inner}" / "expert_artifact.json",
                    label="inner expert",
                )
                for inner in range(3)
            ]
            if any(
                item.get("expert") != expert.upper().replace("C2F", "C2F") for item in artifacts
            ):
                raise RouterStageError("inner expert artifact identity mismatch")
            frames = [
                pd.read_csv(
                    _reference_path(item["oof_csv"], artifact=run_root, label="inner expert")
                )
                for item in artifacts
            ]
            merged = pd.concat(frames, ignore_index=True).sort_values("token_id", kind="stable")
            csv_path = run_root / expert / "inner_oof.csv"
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            merged.to_csv(csv_path, index=False, lineterminator="\n")
            payload = sign_artifact(
                {
                    "artifact_type": "scientific_recovery_v8_router_expert_prediction_v1",
                    "status": "completed",
                    "expert": expert.upper().replace("C2F", "C2F"),
                    "role": "inner_oof",
                    "protocol_sha256": protocol_sha,
                    "checkpoint": {
                        "sha256": hashlib.sha256(
                            "".join(item["checkpoint"]["sha256"] for item in artifacts).encode()
                        ).hexdigest()
                    },
                    "oof_csv": {"path": _repo_relative(csv_path), "sha256": _sha256(csv_path)},
                    "source_inner_artifacts": [item["artifact_sha256"] for item in artifacts],
                }
            )
            (run_root / expert / "inner_oof_artifact.json").write_text(
                json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
    router_commands = plan.get("router_commands")
    if not isinstance(router_commands, list):
        raise RouterStageError("router plan lacks fold commands")
    for command in router_commands:
        if subprocess.run(command, cwd=ROOT, check=False).returncode != 0:
            raise RouterStageError("router fold execution failed")
    aggregate = plan.get("aggregate_command")
    if (
        not isinstance(aggregate, list)
        or subprocess.run(aggregate, cwd=ROOT, check=False).returncode != 0
    ):
        raise RouterStageError("router aggregate execution failed")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument(
        "--device",
        default="cuda",
        help="Recorded for stage-runner compatibility; the router itself runs on CPU sklearn.",
    )
    parser.add_argument(
        "--protocol",
        type=Path,
        default=ROOT / "configs/protocol/scientific_recovery_v8_temporal.json",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "configs/experiment/scientific_recovery_v8_fold_chain/frozen_manifest.json",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "artifacts/scientific_recovery_v8/router"
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=ROOT / "artifacts/scientific_recovery_v8",
        help="Root for automatic nested expert/router stage outputs.",
    )
    parser.add_argument("--max-parallel", type=int, default=1)
    parser.add_argument("--seed", type=int, choices=(7, 13, 23), default=7)
    parser.add_argument("--config-dir", type=Path)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Execute the generated stage plan when the production expert trainer is available.",
    )
    parser.add_argument("--a5-inner-artifact", type=Path)
    parser.add_argument("--c2f-inner-artifact", type=Path)
    parser.add_argument("--a5-outer-artifact", type=Path)
    parser.add_argument("--c2f-outer-artifact", type=Path)
    parser.add_argument("--fixture-smoke", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    try:
        if args.max_parallel < 1:
            raise RouterStageError("--max-parallel must be at least one")
        if args.fixture_smoke:
            if args.dry_run:
                print(
                    json.dumps(
                        {"status": "fixture_dry_run", "sealed_evaluation": "closed"}, sort_keys=True
                    )
                )
                return 0
            config = _fixture_config(args.output_dir)
            result = run_fold(
                config_path=config,
                protocol_path=args.protocol,
                manifest_path=args.manifest,
                output_dir=args.output_dir,
                a5_inner=_fixture_expert(expert="A5", inner=True),
                c2f_inner=_fixture_expert(expert="C2F", inner=True),
                a5_outer=_fixture_expert(expert="A5", inner=False),
                c2f_outer=_fixture_expert(expert="C2F", inner=False),
                expert_artifacts=None,
                fixture=True,
            )
            print(
                json.dumps(
                    {"status": "fixture_completed", "artifact": result["artifact_sha256"]},
                    sort_keys=True,
                )
            )
            return 0
        no_manual_artifacts = all(
            value is None
            for value in (
                args.a5_inner_artifact,
                args.c2f_inner_artifact,
                args.a5_outer_artifact,
                args.c2f_outer_artifact,
            )
        )
        if args.config is None and no_manual_artifacts:
            frozen = verify_frozen_inputs(args.protocol, args.manifest)
            external_configs = None
            if args.config_dir is not None:
                external_configs = sorted(args.config_dir.glob(f"router_fold*_seed{args.seed}.yaml"))
            plan = _router_stage_plan(
                frozen=frozen,
                results_root=args.results_root.resolve(),
                device=args.device,
                max_parallel=args.max_parallel,
                config_paths=external_configs,
                seed=args.seed,
            )
            if args.execute:
                _execute_stage_plan(plan, max_parallel=args.max_parallel)
            print(json.dumps(plan, indent=2, sort_keys=True))
            return 0
        if args.dry_run:
            frozen = verify_frozen_inputs(args.protocol, args.manifest)
            print(
                json.dumps(
                    {
                        "status": "dry_run",
                        "protocol_sha256": frozen.protocol["artifact_sha256"],
                        "sealed_evaluation": "closed",
                    },
                    sort_keys=True,
                )
            )
            return 0
        paths = (
            args.a5_inner_artifact,
            args.c2f_inner_artifact,
            args.a5_outer_artifact,
            args.c2f_outer_artifact,
        )
        if args.config is None or any(path is None for path in paths):
            raise RouterStageError(
                "real router execution requires config and four signed expert prediction artifacts"
            )
        frozen = verify_frozen_inputs(args.protocol, args.manifest)
        a5_inner, a5_inner_payload = _load_expert_artifact(
            args.a5_inner_artifact,
            expert="A5",
            role="inner_oof",
            protocol_sha256=str(frozen.protocol["artifact_sha256"]),
        )
        c2f_inner, c2f_inner_payload = _load_expert_artifact(
            args.c2f_inner_artifact,
            expert="C2F",
            role="inner_oof",
            protocol_sha256=str(frozen.protocol["artifact_sha256"]),
        )
        a5_outer, a5_outer_payload = _load_expert_artifact(
            args.a5_outer_artifact,
            expert="A5",
            role="outer_dev",
            protocol_sha256=str(frozen.protocol["artifact_sha256"]),
        )
        c2f_outer, c2f_outer_payload = _load_expert_artifact(
            args.c2f_outer_artifact,
            expert="C2F",
            role="outer_dev",
            protocol_sha256=str(frozen.protocol["artifact_sha256"]),
        )
        result = run_fold(
            config_path=args.config,
            protocol_path=args.protocol,
            manifest_path=args.manifest,
            output_dir=args.output_dir,
            a5_inner=a5_inner,
            c2f_inner=c2f_inner,
            a5_outer=a5_outer,
            c2f_outer=c2f_outer,
            expert_artifacts={
                "a5_inner": a5_inner_payload,
                "c2f_inner": c2f_inner_payload,
                "a5_outer": a5_outer_payload,
                "c2f_outer": c2f_outer_payload,
            },
            fixture=False,
        )
        print(
            json.dumps(
                {"status": "completed", "artifact": result["artifact_sha256"]}, sort_keys=True
            )
        )
        return 0
    except (
        OSError,
        ValueError,
        KeyError,
        NestedRouterIntegrityError,
        V8IntegrityError,
        RouterStageError,
    ) as error:
        parser.exit(2, f"V8 router stage failed closed: {type(error).__name__}: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())

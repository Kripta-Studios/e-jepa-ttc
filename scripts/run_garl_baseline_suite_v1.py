"""Validate and plan the official local Garl-TTC baseline suite.

This runner is deliberately preparation-only.  It validates the frozen protocol,
the preceding readiness gates, the public full cache and the official release,
then writes a reproducible matrix of commands.  It never starts training and it
never writes metrics or checkpoints.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

EXPECTED_SEEDS = (7, 13, 23)
EXPECTED_VARIANTS = ("event_only", "visual_only", "rgbe_late_fusion")
EXPECTED_PROTOCOL_ID = "garlttc_official_v1"
EXPECTED_CACHE_ARTIFACT = "garlttc_official_lhr_object_cache_v4"
EXPECTED_CACHE_SCHEMA = "garlttc_cache_v4"
EXPECTED_CACHE_INPUT_SCHEMA = "garlttc_input_v4"
EXPECTED_AUDIT_TYPES = {
    "release_audit": "garl_official_release_audit_v1",
    "preprocessing_parity": "garl_preprocessing_parity_v1",
    "model_parity": "garl_model_parity_v1",
    "cache_audit": "garlttc_lhr_cache_audit_v2",
}
ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


class BaselineSuiteError(ValueError):
    """Raised for an invalid preparation contract."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_metadata(repo_root: Path) -> dict[str, str | None]:
    metadata: dict[str, str | None] = {"commit": None, "dirty_diff_sha256": None}
    try:
        metadata["commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo_root, text=True
        ).strip()
        diff = subprocess.check_output(["git", "diff", "--binary"], cwd=repo_root)
        metadata["dirty_diff_sha256"] = hashlib.sha256(diff).hexdigest()
    except (OSError, subprocess.CalledProcessError):
        pass
    return metadata


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise BaselineSuiteError(f"Expected a YAML object: {path}")
    return value


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise BaselineSuiteError(f"Expected a JSON object: {path}")
    return value


def _expand_env(value: str) -> str:
    missing = sorted({name for name in ENV_PATTERN.findall(value) if name not in os.environ})
    if missing:
        raise BaselineSuiteError("Unresolved environment variables: " + ", ".join(missing))
    return os.path.expandvars(value)


def _resolve_path(value: object, repo_root: Path, *, label: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise BaselineSuiteError(f"Missing path for {label}")
    expanded = _expand_env(value)
    path = Path(expanded)
    if not path.is_absolute():
        path = repo_root / path
    return path.resolve()


def _status_passed(value: object) -> bool:
    return isinstance(value, str) and value.lower() == "pass"


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise BaselineSuiteError(f"{label} must be a mapping")
    return value


def _nested_strings(value: object) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, Mapping):
        values: list[str] = []
        for key, child in value.items():
            values.extend(_nested_strings(key))
            values.extend(_nested_strings(child))
        return values
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        values = []
        for child in value:
            values.extend(_nested_strings(child))
        return values
    return []


def validate_suite_config(config: Mapping[str, Any]) -> list[str]:
    """Return all preparation-contract errors without touching external data."""

    errors: list[str] = []
    suite = config.get("suite")
    if not isinstance(suite, Mapping):
        return ["suite must be a mapping"]

    expected_suite = {
        "artifact_type": "garl_baseline_suite_v1",
        "phase": 3,
        "track": "official_garl_local",
        "execution_mode": "plan_only",
    }
    for key, expected in expected_suite.items():
        if suite.get(key) != expected:
            errors.append(f"suite.{key} must be {expected!r}")

    seeds = suite.get("seeds")
    if not isinstance(seeds, list) or tuple(seeds) != EXPECTED_SEEDS:
        errors.append(f"suite.seeds must be exactly {list(EXPECTED_SEEDS)}")
    variants = suite.get("variants")
    if not isinstance(variants, list) or tuple(variants) != EXPECTED_VARIANTS:
        errors.append(f"suite.variants must be exactly {list(EXPECTED_VARIANTS)}")
    if suite.get("test_labels_available") is not False:
        errors.append("suite.test_labels_available must be false")
    if suite.get("test_used_for_selection") is not False:
        errors.append("suite.test_used_for_selection must be false")
    if suite.get("selection_metric") != "validation_sequence_macro_paper_MiD_overall_signed_v1":
        errors.append("suite.selection_metric must be validation-only signed MiD")

    execution = config.get("execution")
    if not isinstance(execution, Mapping) or execution.get("execute_training") is not False:
        errors.append("execution.execute_training must be false")
    elif (
        not isinstance(execution.get("checkpoint_snapshot_epochs"), list)
        or not execution.get("checkpoint_snapshot_epochs")
        or not all(
            isinstance(value, int) and value > 0
            for value in execution["checkpoint_snapshot_epochs"]
        )
    ):
        errors.append("execution.checkpoint_snapshot_epochs must be positive integers")

    protocol = config.get("protocol")
    if not isinstance(protocol, Mapping) or protocol.get("id") != EXPECTED_PROTOCOL_ID:
        errors.append(f"protocol.id must be {EXPECTED_PROTOCOL_ID!r}")

    training = config.get("training")
    if not isinstance(training, Mapping):
        errors.append("training must be a mapping")
    else:
        if training.get("selection_source") != "validation_eap_only":
            errors.append("training.selection_source must be validation_eap_only")
        if training.get("test_selection") is not False:
            errors.append("training.test_selection must be false")

    release = config.get("release")
    if not isinstance(release, Mapping):
        errors.append("release must be a mapping")
    else:
        if release.get("official_commit") != "256661242b8a7f5e56aa3c1c02348b30f6e89de6":
            errors.append("release.official_commit is not the audited release commit")
        checkpoints = release.get("checkpoints")
        if not isinstance(checkpoints, Mapping):
            errors.append("release.checkpoints must be a mapping")
        else:
            required_checkpoint_names = {
                "event_only": "paper_event_only_lhr.pth",
                "visual_only": "paper_visual_only_lhr.pth",
                "rgbe_late_fusion": "paper_ours_full.pth",
            }
            for variant, filename in required_checkpoint_names.items():
                item = checkpoints.get(variant)
                if not isinstance(item, Mapping) or item.get("path") != f"checkpoints/{filename}":
                    errors.append(f"release.checkpoints.{variant} has an invalid path")
                if not isinstance(item, Mapping) or not item.get("sha256"):
                    errors.append(f"release.checkpoints.{variant}.sha256 is required")
        variant_configs = release.get("variant_configs")
        if not isinstance(variant_configs, Mapping):
            errors.append("release.variant_configs must be a mapping")
        else:
            for variant in ("event_only", "visual_only", "rgbe_late_fusion"):
                value = variant_configs.get(variant)
                if not isinstance(value, str) or not value:
                    errors.append(f"release.variant_configs.{variant} is required")

    variant_specs = config.get("variant_specs")
    if not isinstance(variant_specs, Mapping):
        errors.append("variant_specs must be a mapping")
    else:
        expected_modes = {
            "event_only": "event_only",
            "visual_only": "image_only",
            "rgbe_late_fusion": "image_event",
        }
        for variant, mode in expected_modes.items():
            spec = variant_specs.get(variant)
            if not isinstance(spec, Mapping) or spec.get("official_dataset_mode") != mode:
                errors.append(f"variant_specs.{variant}.official_dataset_mode is invalid")
            if not isinstance(spec, Mapping) or spec.get("requires_branch_pretraining") is not True:
                errors.append(f"variant_specs.{variant} must require branch pretraining")

    selection_contract = {
        "suite": suite,
        "execution": execution,
        "training": training,
    }
    forbidden_tokens = ("evttc", "coda", "benchmark10")
    for text in _nested_strings(selection_contract):
        lowered = text.lower()
        if any(token in lowered for token in forbidden_tokens):
            errors.append("configuration contains a forbidden downstream/test-selection token")
            break
    return errors


def _resolve_paths(
    config: Mapping[str, Any],
    repo_root: Path,
    overrides: Mapping[str, Path] | None,
) -> dict[str, Path]:
    paths = _mapping(config.get("paths"), "paths")
    gate_paths = _mapping(config.get("gates"), "gates")
    overrides = overrides or {}

    resolved: dict[str, Path] = {}
    for key in ("eap_root", "garlttc_root", "release_root", "cache_manifest", "cache_audit"):
        value = overrides.get(key, paths.get(key))
        resolved[key] = (
            value.resolve()
            if isinstance(value, Path)
            else _resolve_path(value, repo_root, label=f"paths.{key}")
        )
    resolved["protocol"] = _resolve_path(
        _mapping(config.get("protocol"), "protocol").get("path"),
        repo_root,
        label="protocol.path",
    )
    for key in ("readiness", "release_audit", "preprocessing_parity", "model_parity"):
        resolved[key] = _resolve_path(gate_paths.get(key), repo_root, label=f"gates.{key}")
    return resolved


def _artifact_check(
    path: Path,
    *,
    label: str,
    expected_type: str,
) -> tuple[dict[str, Any] | None, str | None]:
    if not path.is_file():
        return None, f"{label} is missing: {path}"
    try:
        artifact = _read_json(path)
    except (OSError, json.JSONDecodeError, BaselineSuiteError) as exc:
        return None, f"{label} cannot be parsed: {exc}"
    if artifact.get("artifact_type") != expected_type:
        return None, f"{label} has artifact_type {artifact.get('artifact_type')!r}"
    if not _status_passed(artifact.get("status")):
        return None, f"{label} is not PASS: {artifact.get('status')!r}"
    if artifact.get("errors"):
        return None, f"{label} contains errors"
    return artifact, None


def _validate_cache(
    manifest_path: Path,
    audit_path: Path,
    config: Mapping[str, Any],
    protocol_hash: str,
) -> tuple[dict[str, Any] | None, list[str]]:
    errors: list[str] = []
    if not manifest_path.is_file():
        return None, [f"full cache manifest is missing: {manifest_path}"]
    try:
        manifest = _read_json(manifest_path)
    except (OSError, json.JSONDecodeError, BaselineSuiteError) as exc:
        return None, [f"full cache manifest cannot be parsed: {exc}"]
    cache = _mapping(config.get("cache"), "cache")
    expected_counts = {
        "train": cache.get("expected_train_count"),
        "validation": cache.get("expected_validation_count"),
    }
    checks = {
        "artifact_type": manifest.get("artifact_type") == EXPECTED_CACHE_ARTIFACT,
        "schema_version": manifest.get("schema_version") == EXPECTED_CACHE_SCHEMA,
        "input_schema": _mapping(manifest.get("input_schema", {}), "manifest.input_schema").get(
            "version"
        )
        == EXPECTED_CACHE_INPUT_SCHEMA,
        "event_roi_shape": _mapping(manifest.get("input_schema", {}), "manifest.input_schema").get(
            "event_roi_shape"
        )
        == [2, 20, 128, 128],
        "split_counts": manifest.get("split_counts") == expected_counts,
        "discard_count": manifest.get("discard_count") == 0,
        "discard_fraction": manifest.get("discard_fraction") == 0.0,
        "protocol_sha256": manifest.get("protocol_sha256") == protocol_hash,
        "jepa_pair_valid_fraction": manifest.get("jepa_pair_valid_fraction") == 1.0,
        "no_label_fallback": manifest.get("no_label_fallback") is True,
        "official_labels": manifest.get("uses_official_garl_ttc_labels") is True,
        "no_reconstructed_labels": manifest.get("uses_reconstructed_public_eap_ttc") is False,
    }
    for name, valid in checks.items():
        if not valid:
            errors.append(f"full cache check failed: {name}")
    allowed_compression = cache.get("allowed_compression", ["gzip", "none"])
    compression = manifest.get("shard_compression", "none")
    if compression not in allowed_compression:
        errors.append(f"unsupported cache compression: {compression!r}")
    shards = manifest.get("shards")
    if not isinstance(shards, list) or not shards:
        errors.append("full cache declares no shards")
    else:
        for shard in shards:
            if not isinstance(shard, Mapping):
                errors.append("full cache contains a malformed shard entry")
                continue
            relative = shard.get("path")
            if not isinstance(relative, str) or not (manifest_path.parent / relative).is_file():
                errors.append(f"missing cache shard: {relative!r}")
            if not isinstance(shard.get("sha256"), str) or not shard.get("sha256"):
                errors.append(f"cache shard has no declared hash: {relative!r}")
            if isinstance(relative, str):
                shard_path = manifest_path.parent / relative
                meta_path = shard_path.with_suffix("").with_suffix(".meta.json")
                if not meta_path.is_file():
                    errors.append(f"missing cache shard metadata: {meta_path}")
    audit, audit_error = _artifact_check(
        audit_path,
        label="cache audit",
        expected_type=EXPECTED_AUDIT_TYPES["cache_audit"],
    )
    if audit_error:
        errors.append(audit_error)
    if audit is not None:
        model_inputs = set(audit.get("model_input_fields", []))
        forbidden = {
            "ttc_s",
            "target_ttc",
            "box3d_h",
            "box3d_Fcam",
            "category_index",
            "context_depth_history_m",
        }
        if model_inputs & forbidden:
            errors.append("cache audit exposes privileged supervision as model input")
    return manifest, errors


def validate_gates(
    config: Mapping[str, Any],
    resolved: Mapping[str, Path],
) -> tuple[dict[str, Any], list[str]]:
    """Validate all gates required before Fase 3 training."""

    checks: dict[str, Any] = {}
    errors: list[str] = []
    for key in ("eap_root", "garlttc_root", "release_root"):
        checks[f"path:{key}"] = resolved[key].is_dir()
        if not checks[f"path:{key}"]:
            errors.append(f"required root is missing: {resolved[key]}")

    readiness_path = resolved["readiness"]
    if readiness_path.is_file():
        try:
            readiness = _read_json(readiness_path)
            readiness_gates = readiness.get("gates", readiness)
            required = _mapping(config.get("required_readiness_gates"), "required_readiness_gates")
            required_names = required.get("names", [])
            if not isinstance(required_names, list):
                errors.append("required_readiness_gates.names must be a list")
                required_names = []
            for name in required_names:
                value = readiness_gates.get(name) is True
                checks[f"readiness:{name}"] = value
                if not value:
                    errors.append(f"readiness gate is red: {name}")
        except (OSError, json.JSONDecodeError, BaselineSuiteError) as exc:
            errors.append(f"readiness cannot be parsed: {exc}")
    else:
        errors.append(f"readiness is missing: {readiness_path}")

    for key, expected_type in EXPECTED_AUDIT_TYPES.items():
        artifact, artifact_error = _artifact_check(
            resolved[key], label=key, expected_type=expected_type
        )
        checks[f"artifact:{key}"] = artifact_error is None
        if artifact_error:
            errors.append(artifact_error)
        elif artifact is not None and key == "preprocessing_parity":
            if artifact.get("samples", 0) < 100:
                errors.append("preprocessing parity has fewer than 100 samples")
        elif artifact is not None and key == "release_audit":
            release = _mapping(config.get("release"), "release")
            expected_commit = release.get("official_commit")
            release_checks = _mapping(artifact.get("checks", {}), "release checks")
            git_check = _mapping(release_checks.get("git", {}), "release git check")
            if git_check.get("commit") != expected_commit or not _status_passed(
                git_check.get("status")
            ):
                errors.append("release audit commit gate does not match the frozen commit")

    protocol_path = resolved["protocol"]
    protocol_hash = ""
    if protocol_path.is_file():
        try:
            protocol = _read_yaml(protocol_path)
            protocol_hash = _sha256(protocol_path)
            protocol_checks = {
                "id": protocol.get("protocol_id") == EXPECTED_PROTOCOL_ID,
                "seeds": protocol.get("seeds") == list(EXPECTED_SEEDS),
                "test_labels_available": protocol.get("test_labels_available") is False,
                "zero_shot_selection_uses_evttc": protocol.get("zero_shot_selection_uses_evttc")
                is False,
            }
            for name, valid in protocol_checks.items():
                checks[f"protocol:{name}"] = valid
                if not valid:
                    errors.append(f"protocol check failed: {name}")
        except (OSError, BaselineSuiteError, yaml.YAMLError) as exc:
            errors.append(f"protocol cannot be parsed: {exc}")
    else:
        errors.append(f"protocol is missing: {protocol_path}")

    manifest, cache_errors = _validate_cache(
        resolved["cache_manifest"], resolved["cache_audit"], config, protocol_hash
    )
    checks["cache:full_v4"] = not cache_errors
    errors.extend(cache_errors)
    if manifest is not None:
        checks["cache:split_counts"] = manifest.get("split_counts")

    release = _mapping(config.get("release"), "release")
    release_root = resolved["release_root"]
    for label, relative in {
        "release:train_entrypoint": release.get("entrypoint"),
        "release:official_config": release.get("config"),
    }.items():
        path = release_root / str(relative)
        checks[label] = path.is_file()
        if not path.is_file():
            errors.append(f"official release file is missing: {path}")
    variant_configs = _mapping(release.get("variant_configs"), "release.variant_configs")
    for variant, relative in variant_configs.items():
        path = release_root / str(relative)
        checks[f"release:variant_config:{variant}"] = path.is_file()
        if not path.is_file():
            errors.append(f"official variant config is missing: {path}")
    checkpoints = _mapping(release.get("checkpoints"), "release.checkpoints")
    for variant, item in checkpoints.items():
        if not isinstance(item, Mapping):
            errors.append(f"checkpoint spec is malformed: {variant}")
            continue
        checkpoint_path = release_root / str(item.get("path"))
        exists = checkpoint_path.is_file()
        checks[f"checkpoint:{variant}:exists"] = exists
        if not exists:
            errors.append(f"official checkpoint is missing: {checkpoint_path}")
            continue
        actual_hash = _sha256(checkpoint_path)
        expected_hash = str(item.get("sha256", "")).lower()
        checks[f"checkpoint:{variant}:sha256"] = actual_hash == expected_hash
        if actual_hash != expected_hash:
            errors.append(f"official checkpoint hash mismatch: {checkpoint_path}")

    checks["all_required_gates"] = not errors
    return {"checks": checks, "errors": errors, "protocol_sha256": protocol_hash}, errors


def _new_json(path: Path, payload: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    target = path
    if target.exists():
        suffix = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        target = path.with_name(f"{path.stem}_{suffix}{path.suffix}")
    with target.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, sort_keys=True)
        handle.write("\n")
    return target


def _failure(
    output_dir: Path,
    *,
    config_path: Path,
    errors: list[str],
    repo_root: Path,
    exception: Exception | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "artifact_type": "garl_baseline_suite_failure_v1",
        "schema_version": "garl_baseline_suite_v1",
        "status": "failed" if exception is not None else "blocked",
        "generated_at": datetime.now(UTC).isoformat(),
        "config_path": str(config_path),
        "config_sha256": _sha256(config_path) if config_path.is_file() else None,
        "git": _git_metadata(repo_root),
        "training_started": False,
        "metrics_available": False,
        "errors": errors + ([f"{type(exception).__name__}: {exception}"] if exception else []),
    }
    path = _new_json(output_dir / "FAILURE.json", payload)
    payload["failure_path"] = str(path)
    return payload


def _planned_command(
    *,
    entrypoint: Path,
    official_config: Path,
    eap_root: Path,
    garlttc_root: Path,
    output_dir: Path,
    seed: int,
    epochs: int,
    batch_size: int,
    workers: int,
) -> list[str]:
    return [
        "python",
        str(entrypoint),
        "--config",
        str(official_config),
        "--data-root",
        str(eap_root),
        "--garlttc-annotation-root",
        str(garlttc_root),
        "--output-dir",
        str(output_dir),
        "--epochs",
        str(epochs),
        "--batch-size",
        str(batch_size),
        "--num-workers",
        str(workers),
        "--seed",
        str(seed),
    ]


def build_plan(
    config: Mapping[str, Any],
    resolved: Mapping[str, Path],
    gate_report: Mapping[str, Any],
    *,
    output_dir: Path,
    repo_root: Path,
    config_path: Path,
) -> dict[str, Any]:
    """Build a plan without executing any command or fabricating any metric."""

    suite = _mapping(config["suite"], "suite")
    release = _mapping(config["release"], "release")
    training = _mapping(config["training"], "training")
    variant_specs = _mapping(config["variant_specs"], "variant_specs")
    checkpoint_specs = _mapping(release["checkpoints"], "release.checkpoints")
    entrypoint = resolved["release_root"] / str(release["entrypoint"])
    variant_configs = _mapping(release["variant_configs"], "release.variant_configs")
    runs: list[dict[str, Any]] = []
    for variant in suite["variants"]:
        spec = _mapping(variant_specs[variant], f"variant_specs.{variant}")
        checkpoint = _mapping(checkpoint_specs[variant], f"release.checkpoints.{variant}")
        official_config = resolved["release_root"] / str(variant_configs[variant])
        for seed in suite["seeds"]:
            run_dir = output_dir / variant / f"seed-{seed}"
            runs.append(
                {
                    "variant": variant,
                    "seed": seed,
                    "status": "planned_not_executed",
                    "official_dataset_mode": spec["official_dataset_mode"],
                    "branch_checkpoint": checkpoint["path"],
                    "official_config": str(variant_configs[variant]),
                    "config_overrides": spec.get("config_overrides", {}),
                    "config_materialization_required": True,
                    "command": _planned_command(
                        entrypoint=entrypoint,
                        official_config=official_config,
                        eap_root=resolved["eap_root"],
                        garlttc_root=resolved["garlttc_root"],
                        output_dir=run_dir,
                        seed=int(seed),
                        epochs=int(training["epochs"]),
                        batch_size=int(training["batch_size"]),
                        workers=int(training["workers"]),
                    ),
                    "expected_outputs": [
                        str(run_dir / "best.pt"),
                        str(run_dir / "last.pt"),
                        str(run_dir / "summary.json"),
                    ],
                }
            )

    evidence: dict[str, str] = {}
    for label, path in {
        "config": config_path,
        "protocol": resolved["protocol"],
        "cache_manifest": resolved["cache_manifest"],
        "cache_audit": resolved["cache_audit"],
        "readiness": resolved["readiness"],
        "release_audit": resolved["release_audit"],
        "preprocessing_parity": resolved["preprocessing_parity"],
        "model_parity": resolved["model_parity"],
    }.items():
        if path.is_file():
            evidence[label] = _sha256(path)
    return {
        "artifact_type": "garl_baseline_suite_plan_v1",
        "schema_version": "garl_baseline_suite_v1",
        "status": "validated",
        "generated_at": datetime.now(UTC).isoformat(),
        "git": _git_metadata(repo_root),
        "suite": dict(suite),
        "execution_mode": "plan_only",
        "training_started": False,
        "metrics_available": False,
        "selection_rule": training["selection_rule"],
        "evidence_sha256": evidence,
        "gate_report": dict(gate_report),
        "runs": runs,
        "negative_results_preserved": True,
    }


def run_suite(
    config_path: Path,
    output_dir: Path,
    *,
    repo_root: Path | None = None,
    overrides: Mapping[str, Path] | None = None,
) -> dict[str, Any]:
    """Validate the suite and write a plan or a blocking ``FAILURE.json``."""

    root = (repo_root or Path(__file__).resolve().parents[1]).resolve()
    config_path = config_path if config_path.is_absolute() else root / config_path
    config_path = config_path.resolve()
    output_dir = output_dir if output_dir.is_absolute() else root / output_dir
    output_dir = output_dir.resolve()
    try:
        config = _read_yaml(config_path)
        config_errors = validate_suite_config(config)
        if config_errors:
            return _failure(
                output_dir,
                config_path=config_path,
                errors=config_errors,
                repo_root=root,
            )
        resolved = _resolve_paths(config, root, overrides)
        gate_report, gate_errors = validate_gates(config, resolved)
        if gate_errors:
            return _failure(
                output_dir,
                config_path=config_path,
                errors=gate_errors,
                repo_root=root,
            )
        plan = build_plan(
            config,
            resolved,
            gate_report,
            output_dir=output_dir,
            repo_root=root,
            config_path=config_path,
        )
        plan_path = _new_json(output_dir / "baseline_plan.json", plan)
        plan["plan_path"] = str(plan_path)
        return plan
    except Exception as exc:
        return _failure(
            output_dir,
            config_path=config_path,
            errors=["baseline preparation raised an exception"],
            repo_root=root,
            exception=exc,
        )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/experiment/garl_baseline_suite_v1.yaml"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/runs/garl_baseline_suite_v1"),
    )
    parser.add_argument("--eap-root", type=Path)
    parser.add_argument("--garlttc-root", type=Path)
    parser.add_argument("--release-root", type=Path)
    parser.add_argument("--cache-manifest", type=Path)
    parser.add_argument("--cache-audit", type=Path)
    args = parser.parse_args(argv)
    overrides = {
        key: value
        for key, value in {
            "eap_root": args.eap_root,
            "garlttc_root": args.garlttc_root,
            "release_root": args.release_root,
            "cache_manifest": args.cache_manifest,
            "cache_audit": args.cache_audit,
        }.items()
        if value is not None
    }
    result = run_suite(args.config, args.output_dir, overrides=overrides)
    print(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True))
    return 0 if result.get("status") == "validated" else 1


if __name__ == "__main__":
    raise SystemExit(main())

"""Execute the frozen official Garl-TTC baseline matrix after all gates pass.

The existing ``run_garl_baseline_suite_v1.py`` remains preparation-only by
design.  This runner is the explicit execution counterpart: it validates that
the preparation contract is green, writes variant configs below the artifact
directory, invokes the audited release entrypoint, and records every terminal
state without fabricating metrics or mutating the release/source tree.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_garl_baseline_suite_v1 import (  # noqa: E402
    EXPECTED_SEEDS,
    EXPECTED_VARIANTS,
    run_suite,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _materialize_public_train_validation_split(
    *,
    garlttc_root: Path,
    cache_manifest: Path,
    destination: Path,
) -> dict[str, Any]:
    """Materialize the frozen sequence split used by official training/evaluation.

    The release trainer accepts parquet paths from its YAML config, so filtering
    the public train parquet into an artifact directory is sufficient to keep
    the official source tree untouched.  This is deliberately done before any
    training starts: otherwise the trainer would consume all 40 public
    sequences while validation is computed on eight of those same sequences.
    """

    manifest = json.loads(cache_manifest.read_text(encoding="utf-8"))
    split_value = manifest.get("split_path")
    if not isinstance(split_value, str) or not split_value:
        raise ValueError("Full cache manifest must declare split_path.")
    split_path = Path(split_value)
    if not split_path.is_absolute():
        split_path = (ROOT / split_path).resolve()
    split = json.loads(split_path.read_text(encoding="utf-8"))
    assignments = split.get("assignments")
    if not isinstance(assignments, dict):
        raise ValueError("Frozen split must contain assignments.")
    train_sequences = tuple(sorted(str(value) for value in assignments.get("train", [])))
    validation_sequences = tuple(sorted(str(value) for value in assignments.get("validation", [])))
    if not train_sequences or not validation_sequences:
        raise ValueError("Frozen split must contain non-empty train and validation roles.")
    if set(train_sequences) & set(validation_sequences):
        raise ValueError("Official baseline train/validation sequence groups overlap.")

    data_path = garlttc_root / "data" / "train.parquet"
    labels_path = garlttc_root / "annotations" / "train.parquet"
    if not data_path.is_file() or not labels_path.is_file():
        raise FileNotFoundError("Public Garl train data/annotations parquet is missing.")
    data = pd.read_parquet(data_path)
    labels = pd.read_parquet(labels_path)
    for frame, name in ((data, "data"), (labels, "labels")):
        if "sequence_id" not in frame.columns:
            raise ValueError(f"Garl {name} parquet has no sequence_id column.")

    selected = set(train_sequences) | set(validation_sequences)
    available = set(data["sequence_id"].astype(str).unique().tolist())
    unknown = sorted(available - selected)
    missing = sorted(selected - available)
    if unknown or missing:
        raise ValueError(
            "Frozen split does not exactly cover public Garl train sequences: "
            f"unknown={unknown[:5]}, missing={missing[:5]}"
        )

    destination = destination.resolve()
    report_path = destination / "split_manifest.json"
    existing = None
    if report_path.is_file():
        existing = json.loads(report_path.read_text(encoding="utf-8"))
        if (
            existing.get("split_sha256") == _sha256(split_path)
            and existing.get("source_data_sha256") == _sha256(data_path)
            and existing.get("source_labels_sha256") == _sha256(labels_path)
        ):
            return existing
        raise FileExistsError(f"Refusing to replace an incompatible protocol split: {report_path}")

    destination.mkdir(parents=True, exist_ok=True)
    split_roles: dict[str, dict[str, Any]] = {}
    for role, sequences in (("train", train_sequences), ("validation", validation_sequences)):
        role_data = data[data["sequence_id"].astype(str).isin(sequences)].copy()
        role_labels = labels[labels["sequence_id"].astype(str).isin(sequences)].copy()
        if role_data.empty or role_labels.empty:
            raise ValueError(f"Official split role {role!r} is empty.")
        role_dir = destination / role
        role_dir.mkdir(parents=True, exist_ok=True)
        role_data_path = role_dir / "data.parquet"
        role_labels_path = role_dir / "labels.parquet"
        role_asset_path = role_dir / "sequences.txt"
        role_data.to_parquet(role_data_path, index=False)
        role_labels.to_parquet(role_labels_path, index=False)
        role_asset_path.write_text("\n".join(sequences) + "\n", encoding="utf-8")
        split_roles[role] = {
            "sequences": list(sequences),
            "data_rows": int(len(role_data)),
            "label_rows": int(len(role_labels)),
            "data_parquet": role_data_path.as_posix(),
            "labels_parquet": role_labels_path.as_posix(),
            "asset_path": role_asset_path.as_posix(),
            "data_sha256": _sha256(role_data_path),
            "labels_sha256": _sha256(role_labels_path),
        }

    report = {
        "artifact_type": "garl_official_train_validation_split_v1",
        "schema_version": "v1",
        "status": "pass",
        "split_path": split_path.as_posix(),
        "split_sha256": _sha256(split_path),
        "source_data_path": data_path.as_posix(),
        "source_data_sha256": _sha256(data_path),
        "source_labels_path": labels_path.as_posix(),
        "source_labels_sha256": _sha256(labels_path),
        "train_validation_disjoint": True,
        "roles": split_roles,
    }
    _write_json(report_path, report)
    return report


def _materialize_variant_config(
    official_config: Path,
    destination: Path,
    *,
    variant: str,
    spec: dict[str, Any],
    snapshot_epochs: list[int],
    protocol_split: dict[str, Any] | None = None,
) -> str:
    """Write one variant config outside the release and return its hash."""

    payload = yaml.safe_load(official_config.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Official config is not a YAML mapping: {official_config}")
    dataset = payload.setdefault("dataset", {})
    model = payload.setdefault("model", {})
    if not isinstance(dataset, dict) or not isinstance(model, dict):
        raise ValueError("Official config dataset/model sections must be mappings.")
    dataset["mode"] = spec["official_dataset_mode"]
    if protocol_split is not None:
        roles = protocol_split.get("roles")
        if not isinstance(roles, dict) or "train" not in roles:
            raise ValueError("Protocol split is missing the train role.")
        train_role = roles["train"]
        if not isinstance(train_role, dict):
            raise ValueError("Protocol split train role must be a mapping.")
        dataset["annotation_format"] = "parquet"
        dataset["train"] = {
            "asset_path": train_role["asset_path"],
            "data_parquet": train_role["data_parquet"],
            "labels_parquet": train_role["labels_parquet"],
        }
    training_settings = payload.setdefault("training_settings", {})
    if not isinstance(training_settings, dict):
        raise ValueError("Official config training_settings section must be a mapping.")
    training_settings["snapshot_epochs"] = list(snapshot_epochs)
    for dotted_key, value in spec.get("config_overrides", {}).items():
        section, separator, key = str(dotted_key).partition(".")
        if not separator or section not in payload or not isinstance(payload[section], dict):
            raise ValueError(f"Unsupported variant override {dotted_key!r} for {variant}.")
        payload[section][key] = value
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return _sha256(destination)


def _existing_outputs(run_dir: Path) -> list[str]:
    return sorted(
        path.relative_to(run_dir).as_posix()
        for path in run_dir.rglob("*")
        if path.is_file() and path.name not in {"stdout.log", "stderr.log"}
    )


def _find_final_checkpoint(run_dir: Path) -> Path | None:
    """Find the official trainer's final checkpoint without selecting a snapshot."""

    candidates = sorted(
        (path for path in (run_dir / "release_output").rglob("ckpt.pth") if path.is_file()),
        key=lambda path: path.stat().st_mtime_ns,
    )
    return candidates[-1] if candidates else None


def _evaluate_validation_checkpoint(
    *,
    run_dir: Path,
    release_root: Path,
    config_path: Path,
    checkpoint: Path,
    eap_root: Path,
    validation_role: dict[str, Any],
    device: str,
    workers: int,
) -> dict[str, Any]:
    """Evaluate one completed official checkpoint on the disjoint validation role."""

    evaluation_dir = run_dir / "validation_evaluation"
    stdout_path = run_dir / "validation_stdout.log"
    stderr_path = run_dir / "validation_stderr.log"
    command = [
        sys.executable,
        str(ROOT / "scripts" / "evaluate_official_garl_validation.py"),
        "--release-root",
        str(release_root),
        "--config",
        str(config_path),
        "--checkpoint",
        str(checkpoint),
        "--dataset-root",
        str(eap_root),
        "--data-parquet",
        str(validation_role["data_parquet"]),
        "--labels-parquet",
        str(validation_role["labels_parquet"]),
        "--asset-list",
        str(validation_role["asset_path"]),
        "--output-dir",
        str(evaluation_dir),
        "--device",
        device,
        "--batch-size",
        "128",
        "--num-workers",
        str(max(0, workers // 2)),
    ]
    started = time.perf_counter()
    with (
        stdout_path.open("w", encoding="utf-8") as stdout,
        stderr_path.open("w", encoding="utf-8") as stderr,
    ):
        completed = subprocess.run(
            command,
            cwd=ROOT,
            stdout=stdout,
            stderr=stderr,
            check=False,
        )
    result: dict[str, Any] = {
        "artifact_type": "official_garl_validation_evaluation_run_v1",
        "status": "completed" if completed.returncode == 0 else "failed",
        "command": command,
        "checkpoint": checkpoint.as_posix(),
        "checkpoint_sha256": _sha256(checkpoint),
        "output_dir": evaluation_dir.as_posix(),
        "metrics_path": (evaluation_dir / "metrics.json").as_posix(),
        "metrics_sha256": (
            _sha256(evaluation_dir / "metrics.json")
            if (evaluation_dir / "metrics.json").is_file()
            else None
        ),
        "stdout_path": stdout_path.as_posix(),
        "stderr_path": stderr_path.as_posix(),
        "exit_code": int(completed.returncode),
        "elapsed_seconds": time.perf_counter() - started,
        "validation_sequences": validation_role["sequences"],
        "validation_data_sha256": _sha256(Path(validation_role["data_parquet"])),
        "validation_labels_sha256": _sha256(Path(validation_role["labels_parquet"])),
    }
    _write_json(run_dir / "validation_evaluation.json", result)
    if result["status"] != "completed":
        failure = dict(result)
        failure["artifact_type"] = "official_garl_validation_evaluation_failure_v1"
        failure["negative_result_preserved"] = True
        _write_json(run_dir / "VALIDATION_FAILURE.json", failure)
    return result


def _aggregate_signed_baseline_metrics(
    *,
    output_dir: Path,
    runs: list[dict[str, Any]],
    protocol_split: dict[str, Any],
) -> Path:
    """Aggregate only metrics actually emitted by the official evaluator."""

    records: list[dict[str, Any]] = []
    missing: list[str] = []
    for run in runs:
        evaluation = run.get("validation_evaluation")
        if not isinstance(evaluation, dict) or evaluation.get("status") != "completed":
            missing.append(f"{run.get('variant')}/seed-{run.get('seed')}: evaluation")
            continue
        metrics_path_value = evaluation.get("metrics_path")
        if not isinstance(metrics_path_value, str):
            missing.append(f"{run.get('variant')}/seed-{run.get('seed')}: metrics_path")
            continue
        metrics_path = Path(metrics_path_value)
        if not metrics_path.is_file():
            missing.append(f"{run.get('variant')}/seed-{run.get('seed')}: {metrics_path}")
            continue
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        signed = payload.get("signed_garl_metrics")
        macro = payload.get("sequence_macro_signed_metrics")
        if not isinstance(signed, dict) or not isinstance(macro, dict):
            missing.append(f"{run.get('variant')}/seed-{run.get('seed')}: signed metrics")
            continue
        records.append(
            {
                "variant": run.get("variant"),
                "seed": run.get("seed"),
                "checkpoint": run.get("checkpoint_path"),
                "checkpoint_sha256": run.get("checkpoint_sha256"),
                "metrics_path": metrics_path.as_posix(),
                "metrics_sha256": _sha256(metrics_path),
                "signed_garl_metrics": signed,
                "sequence_macro_signed_metrics": macro,
            }
        )
    expected_count = len(EXPECTED_VARIANTS) * len(EXPECTED_SEEDS)
    status = "pass" if not missing and len(records) == expected_count else "failed"
    artifact: dict[str, Any] = {
        "artifact_type": "garl_baseline_metrics_v1",
        "schema_version": "v1",
        "status": status,
        "protocol": "garl_signed_v1",
        "selection_protocol": "validation_sequence_macro_paper_MiD_overall_signed_v1",
        "generated_at": _now(),
        "matrix_output_dir": output_dir.resolve().as_posix(),
        "protocol_split_sha256": protocol_split.get("split_sha256"),
        "expected_run_count": expected_count,
        "observed_metric_count": len(records),
        "missing": missing,
        "runs": records,
        "test_used_for_selection": False,
        "evttc_used_for_selection": False,
        "negative_results_preserved": True,
    }
    path = ROOT / "artifacts" / "metrics" / "garl_baseline_training_v1_signed.json"
    _write_json(path, artifact)
    return path


def _run_one(
    *,
    release_root: Path,
    entrypoint: Path,
    eap_root: Path,
    run_dir: Path,
    config_path: Path,
    variant: str,
    seed: int,
    epochs: int,
    batch_size: int,
    workers: int,
    device: str,
    max_batches: int | None,
) -> dict[str, Any]:
    # The audited release runs with ``cwd=release_root``.  Resolve every
    # artifact/input path before crossing that process boundary; otherwise a
    # repository-relative config is incorrectly looked up below E:\\Garl-TTC.
    release_root = release_root.resolve()
    entrypoint = entrypoint.resolve()
    eap_root = eap_root.resolve()
    run_dir = run_dir.resolve()
    config_path = config_path.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = run_dir / "stdout.log"
    stderr_path = run_dir / "stderr.log"
    command = [
        sys.executable,
        str(entrypoint),
        "--config",
        str(config_path),
        "--data-root",
        str(eap_root),
        "--output-dir",
        str(run_dir / "release_output"),
        "--device",
        device,
        "--epochs",
        str(epochs),
        "--batch-size",
        str(batch_size),
        "--num-workers",
        str(workers),
        "--seed",
        str(seed),
    ]
    if max_batches is not None:
        command.extend(["--max-batches", str(max_batches)])
    started = time.perf_counter()
    started_at = _now()
    status = "failed"
    exit_code: int | None = None
    error_type: str | None = None
    try:
        with (
            stdout_path.open("w", encoding="utf-8") as stdout,
            stderr_path.open("w", encoding="utf-8") as stderr,
        ):
            completed = subprocess.run(
                command,
                cwd=release_root,
                stdout=stdout,
                stderr=stderr,
                check=False,
            )
        exit_code = int(completed.returncode)
        status = "completed" if exit_code == 0 else "failed"
    except KeyboardInterrupt:
        status = "interrupted"
        error_type = "KeyboardInterrupt"
    except OSError as error:
        status = "failed"
        error_type = type(error).__name__
    finished_at = _now()
    record: dict[str, Any] = {
        "artifact_type": "garl_baseline_training_run_v1",
        "schema_version": "v1",
        "variant": variant,
        "seed": seed,
        "status": status,
        "training_started": True,
        "metrics_available": False,
        "command": command,
        "release_root": str(release_root),
        "config_path": str(config_path),
        "config_sha256": _sha256(config_path),
        "started_at": started_at,
        "finished_at": finished_at,
        "elapsed_seconds": time.perf_counter() - started,
        "exit_code": exit_code,
        "error_type": error_type,
        "stdout_path": stdout_path.as_posix(),
        "stderr_path": stderr_path.as_posix(),
        "outputs": _existing_outputs(run_dir),
        "selection_protocol": "validation_sequence_macro_paper_MiD_overall_signed_v1",
        "test_used_for_selection": False,
        "evttc_used_for_selection": False,
    }
    checkpoint = _find_final_checkpoint(run_dir)
    record["checkpoint_path"] = checkpoint.as_posix() if checkpoint is not None else None
    record["checkpoint_sha256"] = _sha256(checkpoint) if checkpoint is not None else None
    _write_json(run_dir / "run.json", record)
    if status != "completed":
        failure = dict(record)
        failure["artifact_type"] = "garl_baseline_training_failure_v1"
        failure["negative_result_preserved"] = True
        _write_json(run_dir / "FAILURE.json", failure)
    return record


def execute(
    *,
    suite_config: Path,
    output_dir: Path,
    eap_root: Path,
    garlttc_root: Path,
    release_root: Path,
    cache_manifest: Path,
    cache_audit: Path,
    device: str,
    max_batches: int | None,
    epochs_override: int | None = None,
    batch_size_override: int | None = None,
    workers_override: int | None = None,
    variants: tuple[str, ...],
    seeds: tuple[int, ...],
) -> dict[str, Any]:
    """Validate gates and execute the requested official matrix."""

    suite_config = suite_config.resolve()
    output_dir = output_dir.resolve()
    eap_root = eap_root.resolve()
    garlttc_root = garlttc_root.resolve()
    release_root = release_root.resolve()
    cache_manifest = cache_manifest.resolve()
    cache_audit = cache_audit.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    prepared = run_suite(
        suite_config,
        output_dir,
        overrides={
            "eap_root": eap_root,
            "garlttc_root": garlttc_root,
            "release_root": release_root,
            "cache_manifest": cache_manifest,
            "cache_audit": cache_audit,
        },
    )
    if prepared.get("status") != "validated":
        failure = {
            "artifact_type": "garl_baseline_training_failure_v1",
            "status": "blocked_preflight",
            "training_started": False,
            "metrics_available": False,
            "negative_result_preserved": True,
            "preflight": prepared,
        }
        _write_json(output_dir / "FAILURE.json", failure)
        return failure

    if batch_size_override is not None and batch_size_override <= 0:
        raise ValueError("batch_size_override must be positive when provided.")
    if workers_override is not None and workers_override < 0:
        raise ValueError("workers_override must be non-negative when provided.")
    if epochs_override is not None:
        if epochs_override <= 0:
            raise ValueError("epochs_override must be positive when provided.")
        if max_batches is None:
            raise ValueError("epochs_override is only allowed for bounded smoke runs.")

    config = yaml.safe_load(suite_config.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("Baseline suite config must be a mapping.")
    release = config["release"]
    training = config["training"]
    execution = config["execution"]
    variant_specs = config["variant_specs"]
    variant_configs = release.get("variant_configs", {})
    if not isinstance(execution, dict):
        raise ValueError("Baseline suite execution section must be a mapping.")
    snapshot_epochs_value = execution.get("checkpoint_snapshot_epochs")
    if not isinstance(snapshot_epochs_value, list) or not all(
        isinstance(value, int) and value > 0 for value in snapshot_epochs_value
    ):
        raise ValueError("execution.checkpoint_snapshot_epochs must be positive integers.")
    snapshot_epochs = [int(value) for value in snapshot_epochs_value]
    entrypoint = release_root / str(release["entrypoint"])
    protocol_split = _materialize_public_train_validation_split(
        garlttc_root=garlttc_root,
        cache_manifest=cache_manifest,
        destination=output_dir / "protocol_split",
    )
    runs: list[dict[str, Any]] = []
    for variant in variants:
        if variant not in EXPECTED_VARIANTS:
            raise ValueError(f"Unknown baseline variant: {variant}")
        spec = variant_specs[variant]
        for seed in seeds:
            if seed not in EXPECTED_SEEDS:
                raise ValueError(f"Seed {seed} is outside the frozen matrix.")
            run_dir = output_dir / variant / f"seed-{seed}"
            materialized = run_dir / "config_materialized.yaml"
            source_config = release_root / str(variant_configs.get(variant, release["config"]))
            if not source_config.is_file():
                raise FileNotFoundError(f"Official variant config is missing: {source_config}")
            config_hash = _materialize_variant_config(
                source_config,
                materialized,
                variant=variant,
                spec=spec,
                snapshot_epochs=snapshot_epochs,
                protocol_split=protocol_split,
            )
            run = _run_one(
                release_root=release_root,
                entrypoint=entrypoint,
                eap_root=eap_root,
                run_dir=run_dir,
                config_path=materialized,
                variant=variant,
                seed=seed,
                epochs=(int(training["epochs"]) if epochs_override is None else epochs_override),
                batch_size=(
                    int(training["batch_size"])
                    if batch_size_override is None
                    else batch_size_override
                ),
                workers=(
                    int(training["workers"]) if workers_override is None else workers_override
                ),
                device=device,
                max_batches=max_batches,
            )
            if run["status"] == "completed" and max_batches is None:
                checkpoint_path = run.get("checkpoint_path")
                if not isinstance(checkpoint_path, str):
                    run["validation_evaluation"] = {
                        "status": "failed",
                        "reason": "official trainer completed without ckpt.pth",
                    }
                else:
                    run["validation_evaluation"] = _evaluate_validation_checkpoint(
                        run_dir=run_dir,
                        release_root=release_root,
                        config_path=materialized,
                        checkpoint=Path(checkpoint_path),
                        eap_root=eap_root,
                        validation_role=protocol_split["roles"]["validation"],
                        device=device,
                        workers=int(training["workers"]),
                    )
                run["metrics_available"] = run["validation_evaluation"].get("status") == "completed"
            elif run["status"] == "completed":
                run["validation_evaluation"] = {
                    "status": "not_run_bounded_smoke",
                    "reason": "validation is reserved for the unbounded matrix",
                }
            run["materialized_config_sha256"] = config_hash
            run["protocol_split_sha256"] = protocol_split["split_sha256"]
            run["protocol_train_sequences"] = protocol_split["roles"]["train"]["sequences"]
            run["protocol_validation_sequences"] = protocol_split["roles"]["validation"][
                "sequences"
            ]
            run["official_source_config"] = source_config.as_posix()
            run["official_source_config_sha256"] = _sha256(source_config)
            _write_json(run_dir / "run.json", run)
            runs.append(run)
    requested_keys = {(variant, seed) for variant in variants for seed in seeds}
    expected_keys = {(variant, seed) for variant in EXPECTED_VARIANTS for seed in EXPECTED_SEEDS}
    result = {
        "artifact_type": "garl_baseline_training_matrix_v1",
        "schema_version": "v1",
        "status": (
            "completed"
            if max_batches is None
            and all(
                run["status"] == "completed"
                and run.get("validation_evaluation", {}).get("status") == "completed"
                for run in runs
            )
            else (
                "completed_bounded_smoke"
                if max_batches is not None and all(run["status"] == "completed" for run in runs)
                else "failed"
            )
        ),
        "training_started": bool(runs),
        "full_matrix": max_batches is None and requested_keys == expected_keys,
        "metrics_available": any(run["metrics_available"] for run in runs),
        "variants": list(variants),
        "seeds": list(seeds),
        "max_batches": max_batches,
        "epochs_override": epochs_override,
        "batch_size_override": batch_size_override,
        "workers_override": workers_override,
        "checkpoint_snapshot_epochs": snapshot_epochs,
        "checkpoint_retention_note": execution.get("checkpoint_retention_note"),
        "selection_protocol": "validation_sequence_macro_paper_MiD_overall_signed_v1",
        "test_used_for_selection": False,
        "evttc_used_for_selection": False,
        "negative_results_preserved": True,
        "protocol_split": protocol_split,
        "signed_metrics_path": None,
        "runs": runs,
    }
    _write_json(output_dir / "matrix.json", result)
    if max_batches is None:
        signed_metrics_path = _aggregate_signed_baseline_metrics(
            output_dir=output_dir,
            runs=runs,
            protocol_split=protocol_split,
        )
        result["signed_metrics_path"] = signed_metrics_path.as_posix()
        _write_json(output_dir / "matrix.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--suite-config",
        type=Path,
        default=Path("configs/experiment/garl_baseline_suite_v1.yaml"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/runs/garl_baseline_training_v1"),
    )
    parser.add_argument("--eap-root", type=Path, default=Path(r"E:\eAP_dataset"))
    parser.add_argument("--garlttc-root", type=Path, default=Path(r"E:\GarlTTC_dataset"))
    parser.add_argument("--release-root", type=Path, default=Path(r"E:\Garl-TTC"))
    parser.add_argument("--cache-manifest", type=Path, required=True)
    parser.add_argument("--cache-audit", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-batches", type=int)
    parser.add_argument(
        "--epochs",
        type=int,
        dest="epochs_override",
        help="Override epochs only for bounded smoke runs (requires --max-batches).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        dest="batch_size_override",
        help="Optional release batch size override; use only for bounded smoke runs.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        dest="workers_override",
        help="Optional release DataLoader worker override; use 0 for bounded smoke runs.",
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=EXPECTED_VARIANTS,
        default=list(EXPECTED_VARIANTS),
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        choices=EXPECTED_SEEDS,
        default=list(EXPECTED_SEEDS),
    )
    args = parser.parse_args()
    result = execute(
        suite_config=args.suite_config,
        output_dir=args.output_dir,
        eap_root=args.eap_root,
        garlttc_root=args.garlttc_root,
        release_root=args.release_root,
        cache_manifest=args.cache_manifest,
        cache_audit=args.cache_audit,
        device=args.device,
        max_batches=args.max_batches,
        epochs_override=args.epochs_override,
        batch_size_override=args.batch_size_override,
        workers_override=args.workers_override,
        variants=tuple(args.variants),
        seeds=tuple(args.seeds),
    )
    print(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True))
    return 0 if result.get("status") in {"completed", "completed_bounded_smoke"} else 1


if __name__ == "__main__":
    raise SystemExit(main())

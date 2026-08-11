"""Train the immutable official Garl event-only model on the exact matched screen.

The adapter writes all configs and outputs in this repository, disables release
checkpoint initialization, and never modifies the official release checkout.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from e_jepa_ttc.artifacts.hashing import sign_artifact  # noqa: E402

EXPECTED_RELEASE_COMMIT = "256661242b8a7f5e56aa3c1c02348b30f6e89de6"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return value


def _git_release_state(release_root: Path) -> dict[str, Any]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=release_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=release_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return {"commit": commit, "dirty": bool(status)}


def materialize_config(
    *,
    official_config: Path,
    subset_manifest: Path,
    release_root: Path,
    eap_root: Path,
    output_config: Path,
    output_dir: Path,
    epochs: int,
    batch_size: int,
    num_workers: int,
    minimum_selection_epoch: int,
) -> dict[str, Any]:
    """Create an outside-release from-scratch config bound to exact subset files."""

    if epochs <= 0 or batch_size <= 0 or num_workers < 0:
        raise ValueError("epochs/batch_size must be positive and num_workers non-negative.")
    if not 1 <= minimum_selection_epoch <= epochs:
        raise ValueError("minimum_selection_epoch must lie in [1, epochs].")
    payload = yaml.safe_load(official_config.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("Official Garl config must be a YAML mapping.")
    manifest = _read_json(subset_manifest)
    if manifest.get("artifact_type") != "garl_event_only_matched_screen_subset_v1":
        raise ValueError("Matched subset manifest has the wrong artifact type.")
    roles = manifest.get("roles")
    if not isinstance(roles, dict):
        raise ValueError("Matched subset manifest has no roles mapping.")
    train = roles.get("train")
    validation = roles.get("validation")
    if not isinstance(train, dict) or not isinstance(validation, dict):
        raise ValueError("Matched subset requires train and validation roles.")
    train_rows = int(train.get("rows", -1))
    validation_rows = int(validation.get("rows", -1))
    if train_rows <= 0 or validation_rows <= 0:
        raise ValueError("Matched Garl config requires positive train/validation row counts.")
    if validation_rows != 2048:
        raise ValueError(
            f"Matched Garl public validation must remain the frozen 2048 rows; got {validation_rows}."
        )
    if train_rows not in {2048, 8192}:
        raise ValueError(
            f"Matched Garl train budget must be one of the preregistered 2048/8192 budgets; got {train_rows}."
        )

    subset_root = subset_manifest.parent.resolve()

    def role_config(role: dict[str, Any]) -> dict[str, str]:
        return {
            "asset_path": str((subset_root / role["assets"]["path"]).resolve()),
            "data_parquet": str((subset_root / role["data"]["path"]).resolve()),
            "labels_parquet": str((subset_root / role["labels"]["path"]).resolve()),
            "annotation_format": "parquet",
        }

    dataset = payload.setdefault("dataset", {})
    model = payload.setdefault("model", {})
    training = payload.setdefault("training_settings", {})
    sections_are_mappings = all(
        isinstance(section, dict) for section in (dataset, model, training)
    )
    if not sections_are_mappings:
        raise TypeError("Official config dataset/model/training_settings must be mappings.")
    dataset.update(
        {
            "root": str(eap_root.resolve()),
            "data_blob_dir": str((eap_root / "data_blobs").resolve()),
            "annotation_format": "parquet",
            "mode": "event_only",
            "db_sample_size": train_rows,
            "train": role_config(train),
            "test": role_config(validation),
        }
    )
    # Empty paths are deliberate: the paper checkpoint saw validation sequences.
    model["pretrained_ckpt_rgb"] = ""
    model["pretrained_ckpt_event"] = ""
    training.update(
        {
            "total_epochs": epochs,
            "batch_size": batch_size,
            "num_threads": num_workers,
            "shuffle": True,
            "resume": False,
            "ckpt_path": None,
            "snapshot_epochs": list(range(minimum_selection_epoch, epochs + 1)),
        }
    )
    payload.setdefault("dirs", {})["output"] = str(output_dir.resolve())
    payload["exp_type"] = "event_lhr_matched_screen"
    payload.setdefault("cudnn", {}).update(
        {"enabled": True, "deterministic": True, "benchmark": False}
    )
    payload["matched_protocol"] = {
        "subset_manifest": str(subset_manifest.resolve()),
        "subset_artifact_sha256": manifest.get("artifact_sha256"),
        "release_commit": EXPECTED_RELEASE_COMMIT,
        "from_scratch": True,
        "release_checkpoint_initialization": False,
        "selection_source": "public_validation_only",
        "minimum_selection_epoch": minimum_selection_epoch,
        "train_rows": train_rows,
        "validation_rows": validation_rows,
        "test_opened": False,
    }
    output_config.parent.mkdir(parents=True, exist_ok=True)
    output_config.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return payload


def _tracked_release_hashes(release_root: Path, official_config: Path) -> dict[str, str]:
    paths = {
        "official_config": official_config,
        "train_entrypoint": release_root / "tools" / "train.py",
        "trainer": release_root / "garl_ttc" / "engine" / "trainer.py",
        "dataset": release_root / "garl_ttc" / "datasets" / "ttc_dataset.py",
        "model": release_root / "garl_ttc" / "models" / "ttc_network.py",
    }
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(f"Official release file not found: {path}")
    return {name: _sha256(path) for name, path in paths.items()}


def _find_checkpoints(training_output: Path) -> list[Path]:
    checkpoints = sorted(training_output.rglob("*.pth"), key=lambda path: path.stat().st_mtime)
    if not checkpoints:
        raise RuntimeError("Official matched training produced no checkpoints.")
    return checkpoints


def run(
    *,
    release_root: Path,
    official_config: Path,
    subset_manifest: Path,
    eap_root: Path,
    output_dir: Path,
    device: str,
    seed: int,
    epochs: int,
    batch_size: int,
    num_workers: int,
    minimum_selection_epoch: int,
    max_batches: int | None,
    execute: bool,
    evaluate_final: bool,
    timeout_hours: float,
) -> dict[str, Any]:
    """Plan or execute official from-scratch training without touching release files."""

    if timeout_hours <= 0.0 or timeout_hours > 4.5:
        raise ValueError("timeout_hours must be in (0, 4.5] to preserve the 25% safety margin.")
    state_before = _git_release_state(release_root)
    if state_before != {"commit": EXPECTED_RELEASE_COMMIT, "dirty": False}:
        raise RuntimeError(
            f"Official release state is not the audited clean commit: {state_before}"
        )
    hashes_before = _tracked_release_hashes(release_root, official_config)
    output_dir.mkdir(parents=True, exist_ok=True)
    materialized_config = output_dir / "matched_event_lhr.yaml"
    training_output = output_dir / "release_output"
    materialize_config(
        official_config=official_config,
        subset_manifest=subset_manifest,
        release_root=release_root,
        eap_root=eap_root,
        output_config=materialized_config,
        output_dir=training_output,
        epochs=epochs,
        batch_size=batch_size,
        num_workers=num_workers,
        minimum_selection_epoch=minimum_selection_epoch,
    )
    command = [
        sys.executable,
        str((release_root / "tools" / "train.py").resolve()),
        "--config",
        str(materialized_config.resolve()),
        "--data-root",
        str(eap_root.resolve()),
        "--output-dir",
        str(training_output.resolve()),
        "--device",
        device,
        "--epochs",
        str(epochs),
        "--batch-size",
        str(batch_size),
        "--num-workers",
        str(num_workers),
        "--seed",
        str(seed),
    ]
    if max_batches is not None:
        if max_batches <= 0:
            raise ValueError("max_batches must be positive when provided.")
        command.extend(["--max-batches", str(max_batches)])
    report: dict[str, Any] = {
        "artifact_type": "garl_event_only_matched_screen_run_v1",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "status": "planned" if not execute else "running",
        "execute": execute,
        "smoke_only": max_batches is not None,
        "command": command,
        "protocol": {
            "seed": seed,
            "epochs": epochs,
            "batch_size": batch_size,
            "num_workers": num_workers,
            "minimum_selection_epoch": minimum_selection_epoch,
            "max_batches": max_batches,
            "from_scratch": True,
            "pretrained_release_checkpoint_used": False,
            "private_test_opened": False,
            "timeout_hours": timeout_hours,
        },
        "inputs": {
            "subset_manifest": {
                "path": str(subset_manifest.resolve()),
                "sha256": _sha256(subset_manifest),
                "artifact_sha256": _read_json(subset_manifest).get("artifact_sha256"),
            },
            "materialized_config": {
                "path": str(materialized_config.resolve()),
                "sha256": _sha256(materialized_config),
            },
            "release_commit": state_before["commit"],
            "release_hashes": hashes_before,
        },
    }
    if execute:
        stdout_path = output_dir / "training_stdout.log"
        stderr_path = output_dir / "training_stderr.log"
        started = time.perf_counter()
        with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open(
            "w", encoding="utf-8"
        ) as stderr:
            completed = subprocess.run(
                command,
                cwd=release_root,
                stdout=stdout,
                stderr=stderr,
                check=False,
                timeout=timeout_hours * 3600.0,
            )
        elapsed = time.perf_counter() - started
        report["training"] = {
            "returncode": completed.returncode,
            "elapsed_seconds_including_online_preprocessing": elapsed,
            "stdout": {"path": str(stdout_path.resolve()), "sha256": _sha256(stdout_path)},
            "stderr": {"path": str(stderr_path.resolve()), "sha256": _sha256(stderr_path)},
        }
        if completed.returncode != 0:
            report["status"] = "training_failed"
        else:
            checkpoints = _find_checkpoints(training_output)
            final_checkpoint = checkpoints[-1]
            report["status"] = "training_completed"
            report["checkpoints"] = [
                {
                    "path": str(path.resolve()),
                    "sha256": _sha256(path),
                    "size_bytes": path.stat().st_size,
                }
                for path in checkpoints
            ]
            if evaluate_final:
                evaluation_dir = output_dir / "validation_evaluation_final"
                evaluation_command = [
                    sys.executable,
                    str((ROOT / "scripts" / "evaluate_official_garl_validation.py").resolve()),
                    "--release-root",
                    str(release_root.resolve()),
                    "--config",
                    str(materialized_config.resolve()),
                    "--checkpoint",
                    str(final_checkpoint.resolve()),
                    "--dataset-root",
                    str(eap_root.resolve()),
                    "--data-parquet",
                    str((subset_manifest.parent / "validation_data.parquet").resolve()),
                    "--labels-parquet",
                    str((subset_manifest.parent / "validation_labels.parquet").resolve()),
                    "--asset-list",
                    str((subset_manifest.parent / "validation_assets.txt").resolve()),
                    "--output-dir",
                    str(evaluation_dir.resolve()),
                    "--device",
                    device,
                    "--batch-size",
                    str(batch_size),
                    "--num-workers",
                    str(num_workers),
                ]
                evaluation_started = time.perf_counter()
                evaluation = subprocess.run(
                    evaluation_command,
                    cwd=ROOT,
                    capture_output=True,
                    text=True,
                    check=False,
                    timeout=timeout_hours * 3600.0,
                )
                report["validation"] = {
                    "returncode": evaluation.returncode,
                    "elapsed_seconds_including_online_preprocessing": (
                        time.perf_counter() - evaluation_started
                    ),
                    "command": evaluation_command,
                    "stderr": evaluation.stderr[-4000:],
                }
                if evaluation.returncode == 0:
                    report["validation"]["metrics"] = _read_json(evaluation_dir / "metrics.json")
                    report["status"] = "completed_public_validation_only"
                else:
                    report["status"] = "validation_failed"

    state_after = _git_release_state(release_root)
    hashes_after = _tracked_release_hashes(release_root, official_config)
    report["release_unchanged"] = state_after == state_before and hashes_after == hashes_before
    if not report["release_unchanged"]:
        raise RuntimeError("Official release changed while running matched training.")
    sign_artifact(report)
    report_path = output_dir / "run.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-root", type=Path, default=Path(r"E:\Garl-TTC"))
    parser.add_argument("--official-config", type=Path)
    parser.add_argument("--subset-manifest", type=Path, required=True)
    parser.add_argument("--eap-root", type=Path, default=Path(r"E:\eAP_dataset"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--epochs", type=int, default=18)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--minimum-selection-epoch", type=int, default=8)
    parser.add_argument("--max-batches", type=int)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--evaluate-final", action="store_true")
    parser.add_argument("--timeout-hours", type=float, default=4.5)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    release_root = args.release_root
    official_config = args.official_config or (
        release_root / "configs" / "ablation" / "event_lhr.yaml"
    )
    try:
        report = run(
            release_root=release_root,
            official_config=official_config,
            subset_manifest=args.subset_manifest,
            eap_root=args.eap_root,
            output_dir=args.output_dir,
            device=args.device,
            seed=args.seed,
            epochs=args.epochs,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            minimum_selection_epoch=args.minimum_selection_epoch,
            max_batches=args.max_batches,
            execute=args.execute,
            evaluate_final=args.evaluate_final,
            timeout_hours=args.timeout_hours,
        )
    except Exception as error:
        print(f"matched Garl run failed: {type(error).__name__}: {error}", file=sys.stderr)
        return 1
    print(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True))
    return 0 if report["status"] not in {"training_failed", "validation_failed"} else 2


if __name__ == "__main__":
    raise SystemExit(main())

"""Execute and aggregate the official Garl model on the release-semantic cache."""

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

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_VARIANTS = ("event_only", "visual_only", "rgbe_late_fusion")
EXPECTED_SEEDS = (7, 13, 23)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _run_one(
    *,
    release_root: Path,
    cache_manifest: Path,
    output_dir: Path,
    variant: str,
    seed: int,
    epochs: int,
    batch_size: int,
    workers: int,
    device: str,
    max_batches: int | None,
    max_validation_batches: int | None,
) -> dict[str, Any]:
    run_dir = (output_dir / variant / f"seed-{seed}").resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(ROOT / "scripts" / "train_garl_release_cache.py"),
        "--release-root",
        str(release_root),
        "--cache-manifest",
        str(cache_manifest),
        "--output-dir",
        str(run_dir),
        "--variant",
        variant,
        "--seed",
        str(seed),
        "--epochs",
        str(epochs),
        "--batch-size",
        str(batch_size),
        "--num-workers",
        str(workers),
        "--device",
        device,
    ]
    if max_batches is not None:
        command.extend(["--max-batches", str(max_batches)])
    if max_validation_batches is not None:
        command.extend(["--max-validation-batches", str(max_validation_batches)])
    stdout_path = run_dir / "stdout.log"
    stderr_path = run_dir / "stderr.log"
    started = time.perf_counter()
    with (
        stdout_path.open("w", encoding="utf-8") as stdout,
        stderr_path.open("w", encoding="utf-8") as stderr,
    ):
        completed = subprocess.run(command, cwd=ROOT, stdout=stdout, stderr=stderr, check=False)
    run_json = run_dir / "run.json"
    if run_json.is_file():
        result = json.loads(run_json.read_text(encoding="utf-8"))
        if not isinstance(result, dict):
            raise ValueError(f"Cache training run is not a mapping: {run_json}")
        result["matrix_command"] = command
        result["matrix_exit_code"] = int(completed.returncode)
        result["matrix_elapsed_seconds"] = time.perf_counter() - started
        return result
    failure_path = run_dir / "FAILURE.json"
    failure = json.loads(failure_path.read_text(encoding="utf-8")) if failure_path.is_file() else {}
    return {
        "artifact_type": "garl_release_cache_training_matrix_run_v1",
        "status": "failed",
        "variant": variant,
        "seed": seed,
        "command": command,
        "exit_code": int(completed.returncode),
        "failure": failure,
        "stdout_path": stdout_path.as_posix(),
        "stderr_path": stderr_path.as_posix(),
        "elapsed_seconds": time.perf_counter() - started,
        "negative_results_preserved": True,
    }


def _require_readiness(path: Path, *, bounded: bool) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Readiness artifact is missing: {path}")
    readiness = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(readiness, dict):
        raise ValueError("Readiness artifact must be a JSON object.")
    if not bounded and readiness.get("long_training_authorized") is not True:
        raise RuntimeError("Official cache matrix is blocked by readiness gates.")
    return readiness


def execute_matrix(
    *,
    release_root: Path,
    cache_manifest: Path,
    output_dir: Path,
    readiness: Path,
    variants: tuple[str, ...],
    seeds: tuple[int, ...],
    epochs: int,
    batch_size: int,
    workers: int,
    device: str,
    max_batches: int | None,
    max_validation_batches: int | None,
) -> dict[str, Any]:
    bounded = max_batches is not None or max_validation_batches is not None
    readiness_payload = _require_readiness(readiness, bounded=bounded)
    for variant in variants:
        if variant not in EXPECTED_VARIANTS:
            raise ValueError(f"Unknown variant: {variant}")
    for seed in seeds:
        if seed not in EXPECTED_SEEDS:
            raise ValueError(f"Seed {seed} is outside the frozen matrix: {seed}")

    output_dir = output_dir.resolve()
    runs = [
        _run_one(
            release_root=release_root.resolve(),
            cache_manifest=cache_manifest.resolve(),
            output_dir=output_dir,
            variant=variant,
            seed=seed,
            epochs=epochs,
            batch_size=batch_size,
            workers=workers,
            device=device,
            max_batches=max_batches,
            max_validation_batches=max_validation_batches,
        )
        for variant in variants
        for seed in seeds
    ]
    expected = len(variants) * len(seeds)
    completed = [
        run for run in runs if run.get("status") in {"completed", "completed_bounded_smoke"}
    ]
    metrics = [
        {
            "variant": run.get("variant"),
            "seed": run.get("seed"),
            "checkpoint_path": run.get("checkpoint_path"),
            "checkpoint_sha256": run.get("checkpoint_sha256"),
            "validation_metrics": run.get("validation_metrics"),
        }
        for run in completed
        if isinstance(run.get("validation_metrics"), dict)
    ]
    result: dict[str, Any] = {
        "artifact_type": "garl_release_cache_training_matrix_v1",
        "schema_version": "v1",
        "status": (
            "completed_bounded_smoke"
            if bounded and len(completed) == expected
            else "completed"
            if not bounded and len(completed) == expected
            else "failed"
        ),
        "training_started": bool(runs),
        "full_matrix": not bounded
        and set((str(run.get("variant")), int(run.get("seed", -1))) for run in runs)
        == {(variant, seed) for variant in EXPECTED_VARIANTS for seed in EXPECTED_SEEDS},
        "variants": list(variants),
        "seeds": list(seeds),
        "epochs": epochs,
        "batch_size": batch_size,
        "workers": workers,
        "device": device,
        "git_commit": next(
            (run.get("git_commit") for run in runs if run.get("git_commit")),
            None,
        ),
        "protocol": "garl_signed_v1",
        "training_scope": "public_train40_retraining",
        "sampling_order_policy": next(
            (run.get("sampling_order_policy") for run in runs if run.get("sampling_order_policy")),
            None,
        ),
        "max_batches": max_batches,
        "max_validation_batches": max_validation_batches,
        "readiness_path": readiness.resolve().as_posix(),
        "readiness_sha256": _sha256(readiness.resolve()),
        "cache_manifest": cache_manifest.resolve().as_posix(),
        "cache_manifest_sha256": _sha256(cache_manifest.resolve()),
        "bbox_protocol": "P0_oracle_bbox_roi",
        "evaluator": "cache_backed_official_model_signed_v1",
        "input_parity_required": True,
        "test_used_for_selection": False,
        "evttc_used_for_selection": False,
        "negative_results_preserved": True,
        "expected_run_count": expected,
        "completed_run_count": len(completed),
        "signed_validation_metrics": metrics,
        "runs": runs,
        "generated_at": _now(),
        "readiness_snapshot": {
            "long_training_authorized": readiness_payload.get("long_training_authorized"),
            "release_input_cache_full_green": readiness_payload.get(
                "release_input_cache_full_green"
            ),
        },
    }
    if not bounded:
        signed_status = (
            "pass" if len(metrics) == expected and result["status"] == "completed" else "failed"
        )
        signed_payload = {
            "artifact_type": "garl_release_cache_training_metrics_v1",
            "schema_version": "v1",
            "status": signed_status,
            "protocol": "garl_signed_v1",
            "evaluator": "cache_backed_official_model_signed_v1",
            "git_commit": result["git_commit"],
            "training_scope": result["training_scope"],
            "sampling_order_policy": result["sampling_order_policy"],
            "expected_run_count": expected,
            "observed_metric_count": len(metrics),
            "missing": [] if len(metrics) == expected else ["one or more run validation metrics"],
            "runs": metrics,
            "test_used_for_selection": False,
            "evttc_used_for_selection": False,
            "negative_results_preserved": True,
        }
        signed_path = ROOT / "artifacts" / "metrics" / "garl_release_cache_training_v1_signed.json"
        _write_json(signed_path, signed_payload)
        result["signed_metrics_path"] = signed_path.as_posix()
    _write_json(output_dir / "matrix.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-root", type=Path, default=Path(r"E:\Garl-TTC"))
    parser.add_argument("--cache-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--readiness", type=Path, required=True)
    parser.add_argument(
        "--variants", nargs="+", choices=EXPECTED_VARIANTS, default=list(EXPECTED_VARIANTS)
    )
    parser.add_argument(
        "--seeds", nargs="+", type=int, choices=EXPECTED_SEEDS, default=list(EXPECTED_SEEDS)
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-batches", type=int)
    parser.add_argument("--max-validation-batches", type=int)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    result = execute_matrix(
        release_root=args.release_root,
        cache_manifest=args.cache_manifest,
        output_dir=args.output_dir,
        readiness=args.readiness,
        variants=tuple(args.variants),
        seeds=tuple(args.seeds),
        epochs=args.epochs,
        batch_size=args.batch_size,
        workers=args.num_workers,
        device=args.device,
        max_batches=args.max_batches,
        max_validation_batches=args.max_validation_batches,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] in {"completed", "completed_bounded_smoke"} else 1


if __name__ == "__main__":
    raise SystemExit(main())

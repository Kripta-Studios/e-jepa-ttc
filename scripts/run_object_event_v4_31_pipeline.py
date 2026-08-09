#!/usr/bin/env python3
"""Reproducible, CPU-safe v4.31 orchestration with per-stage UTF-8 logs.

The runner deliberately does not interpret a negative scientific gate as an
infrastructure failure: a stage returning zero is recorded as completed even when
the analyzer reports a negative causal result.  Infrastructure/precondition errors
return 2.  ``--dry-run`` writes only run metadata and printable argv arrays.
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src")]
from e_jepa_ttc.data.object_event_v4_31 import sha256_file, strict_json  # noqa: E402

SEEDS = (7, 13, 23)


class PipelineError(RuntimeError):
    """A missing or failed infrastructure prerequisite."""


@dataclass(frozen=True)
class Stage:
    """A stage name and an already-tokenized subprocess command."""

    name: str
    argv: list[str]


def _console_safe(text: str, encoding: str | None) -> str:
    """Escape characters unsupported by the parent console without losing logs."""
    target = encoding or "utf-8"
    return text.encode(target, errors="backslashreplace").decode(target)


def _terminate_child(process: subprocess.Popen[str]) -> None:
    """Ensure a stage cannot survive an exception in the parent runner."""
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def _timestamp() -> str:
    return dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")


def _git_value(*args: str) -> str | None:
    result = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=False)
    return result.stdout.strip() or None if result.returncode == 0 else None


def _environment() -> dict[str, object]:
    """Capture environment facts without allocating CUDA tensors."""
    result: dict[str, object] = {
        "python": sys.version,
        "git_commit": _git_value("rev-parse", "HEAD"),
        "git_status": _git_value("status", "--short"),
        "torch": None,
        "cuda": None,
        "gpu": None,
    }
    try:
        import torch

        result["torch"] = torch.__version__
        result["cuda"] = torch.version.cuda
        if torch.cuda.is_available():
            result["gpu"] = torch.cuda.get_device_name(0)
    except ImportError:
        pass
    return result


def _hash_if_file(path: Path) -> str | None:
    return sha256_file(path) if path.is_file() else None


def _stage2_sources(values: list[str]) -> dict[int, Path]:
    sources: dict[int, Path] = {}
    for value in values:
        try:
            raw_seed, raw_path = value.split("=", 1)
            seed = int(raw_seed)
        except ValueError as exc:
            raise PipelineError("--stage2-source must be SEED=PATH") from exc
        if seed in sources:
            raise PipelineError(f"duplicate stage2 seed: {seed}")
        sources[seed] = Path(raw_path)
    if sources and tuple(sorted(sources)) != SEEDS:
        raise PipelineError("stage2 sources must specify exactly seeds 7, 13, and 23")
    return sources


def _check_stage2(path: Path) -> None:
    if not (path / "manifest.json").is_file() or any(
        not (path / f"seed_{seed}.npz").is_file() for seed in SEEDS
    ):
        raise PipelineError("stage2 directory lacks manifest or seed_7/13/23 evidence")


def _overlap(left: Path, right: Path) -> bool:
    resolved_left, resolved_right = left.resolve(), right.resolve()
    return (
        resolved_left == resolved_right
        or resolved_left in resolved_right.parents
        or resolved_right in resolved_left.parents
    )


def _validate_disjoint_stage2(stage2: Path | None, args: argparse.Namespace) -> None:
    if stage2 is None:
        return
    for name, path in {
        "cache": args.cache,
        "output": args.output_dir,
        "log": args.log_root,
    }.items():
        if _overlap(stage2, path):
            raise PipelineError(f"stage2 directory must be disjoint from {name} root")


def _run_stage(stage: Stage, log_path: Path, *, dry_run: bool) -> dict[str, object]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = dt.datetime.now(dt.UTC).isoformat()
    command = subprocess.list2cmdline(stage.argv)
    if dry_run:
        text = f"[dry-run] {command}\n"
        log_path.write_text(text, encoding="utf-8")
        print(_console_safe(text, getattr(sys.stdout, "encoding", None)), end="")
        return {"name": stage.name, "argv": stage.argv, "started": started, "exit": 0}
    with log_path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(f"$ {command}\n")
        child_env = os.environ.copy()
        child_env["PYTHONIOENCODING"] = "utf-8"
        process = subprocess.Popen(
            stage.argv,
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=child_env,
        )
        try:
            assert process.stdout is not None
            for line in process.stdout:
                handle.write(line)
                print(_console_safe(line, getattr(sys.stdout, "encoding", None)), end="")
            exit_code = process.wait()
        finally:
            _terminate_child(process)
    return {
        "name": stage.name,
        "argv": stage.argv,
        "started": started,
        "exit": exit_code,
    }


def _python_stage(name: str, script: str, arguments: list[str]) -> Stage:
    return Stage(name, [sys.executable, str(ROOT / "scripts" / script), *arguments])


def build_stages(args: argparse.Namespace) -> tuple[list[Stage], Path | None]:
    """Build argv arrays, validating full-mode prerequisites before mutation."""
    full = args.mode == "full"
    mode_args = ["--full"] if full else []
    sources = _stage2_sources(args.stage2_source)
    stage2 = args.stage2_dir
    if sources:
        if args.stage2_output is None:
            raise PipelineError("--stage2-output is required with --stage2-source")
        stage2 = args.stage2_output
    if full and stage2 is None:
        raise PipelineError("full mode requires existing stage2 evidence or three sources")
    _validate_disjoint_stage2(stage2, args)
    if not args.dry_run and full and stage2 is not None and not sources:
        _check_stage2(stage2)

    stages: list[Stage] = []
    if args.build_cache:
        build_args = ["--config", str(args.config), "--output-dir", str(args.cache), *mode_args]
        if args.force:
            build_args.append("--force")
        stages.append(
            _python_stage("build", "build_object_event_v4_31_sanitized_cache.py", build_args)
        )
    elif not args.dry_run and not args.cache.is_dir():
        raise PipelineError("cache directory is absent; pass --build-cache to materialize it")

    # An existing cache is always checked before sanitizer/analyzer output mutation.
    preflight_args = ["--config", str(args.config), "--cache", str(args.cache), *mode_args]
    stages.append(_python_stage("preflight", "preflight_object_event_v4_31.py", preflight_args))
    if sources:
        sanitize_args = ["--output-dir", str(args.stage2_output)]
        for seed in SEEDS:
            sanitize_args.extend(["--source", f"{seed}={sources[seed]}"])
        if args.force:
            sanitize_args.append("--force")
        stages.append(
            _python_stage("sanitize", "sanitize_object_event_v4_30_stage2.py", sanitize_args)
        )
    analyze_args = [
        "--config",
        str(args.config),
        "--cache",
        str(args.cache),
        "--output-dir",
        str(args.output_dir),
        "--device",
        args.device,
        *mode_args,
    ]
    if stage2 is not None:
        analyze_args.extend(["--stage2", str(stage2)])
    if args.force:
        analyze_args.append("--force")
    stages.append(
        _python_stage("analyze", "analyze_object_event_v4_31_operator_audit.py", analyze_args)
    )
    return stages, stage2


def run(args: argparse.Namespace) -> int:
    """Run the requested pipeline, returning 2 only for infrastructure failures."""
    started = dt.datetime.now(dt.UTC).isoformat()
    if args.source_parquet is not None:
        os.environ["E_JEPA_TTC_SOURCE_PARQUET"] = str(args.source_parquet)
    if args.event_root is not None:
        os.environ["E_JEPA_TTC_EVENT_ROOT"] = str(args.event_root)
    run_dir = args.log_root / f"object_event_v4_31_{_timestamp()}"
    run_dir.mkdir(parents=True, exist_ok=False)
    manifest: dict[str, Any] = {
        "artifact_type": "object_event_v4_31_pipeline_run_v1",
        "started": started,
        "mode": args.mode,
        "device": args.device,
        "dry_run": args.dry_run,
        "paths": {
            "config": str(args.config),
            "cache": str(args.cache),
            "output_dir": str(args.output_dir),
            "log_dir": str(run_dir),
        },
        "hashes": {"config": _hash_if_file(args.config)},
        "environment": _environment(),
        "stages": [],
    }
    try:
        stages, stage2 = build_stages(args)
        if stage2 is not None:
            manifest["paths"]["stage2"] = str(stage2)
        for stage in stages:
            result = _run_stage(stage, run_dir / f"{stage.name}.log", dry_run=args.dry_run)
            manifest["stages"].append(result)
            if result["exit"] != 0:
                raise PipelineError(f"{stage.name} exited {result['exit']}")
        manifest["status"] = "completed"
        exit_code = 0
    except (OSError, PipelineError, subprocess.SubprocessError) as exc:
        manifest["status"] = "invalid_incomplete"
        manifest["error_type"] = type(exc).__name__
        manifest["error"] = str(exc)
        exit_code = 2
    manifest["finished"] = dt.datetime.now(dt.UTC).isoformat()
    manifest_path = run_dir / "run_manifest.json"
    manifest_path.write_text(strict_json(manifest), encoding="utf-8")
    summary = {
        "artifact_type": "object_event_v4_31_pipeline_summary_v1",
        "status": manifest["status"],
        "run_manifest": str(manifest_path),
        "run_manifest_sha256": sha256_file(manifest_path),
        "stage_exit_codes": [item["exit"] for item in manifest["stages"]],
    }
    summary_path = run_dir / "pipeline_summary.json"
    summary_path.write_text(strict_json(summary), encoding="utf-8")
    (run_dir / "pipeline_summary.json.sha256").write_text(
        f"{sha256_file(summary_path)}  pipeline_summary.json\n", encoding="utf-8"
    )
    return exit_code


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("mode", choices=("diagnostic", "full"))
    value.add_argument("--device", default="cpu")
    value.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/experiment/e_jepa_garl_object_event_operator_audit_v4_31.yaml",
    )
    value.add_argument("--cache", type=Path, required=True)
    value.add_argument("--source-parquet", type=Path)
    value.add_argument("--event-root", type=Path)
    value.add_argument("--output-dir", type=Path, required=True)
    value.add_argument("--log-root", type=Path, default=ROOT / "artifacts" / "logs")
    value.add_argument("--build-cache", action="store_true")
    value.add_argument("--stage2-dir", type=Path)
    value.add_argument("--stage2-source", action="append", default=[])
    value.add_argument("--stage2-output", type=Path)
    value.add_argument("--force", action="store_true")
    value.add_argument("--dry-run", action="store_true")
    return value


def main() -> int:
    return run(parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())

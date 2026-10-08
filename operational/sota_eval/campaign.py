"""Resume the frozen expanded EvTTC comparison from inference through scoring.

This is an orchestration layer only.  It performs no training and does not open
ground-truth data until both event-only and RGB+event prediction populations
have been sealed.  Every expensive child is resumable in its own output directory.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from operational.efficient_context.common import Lease, atomic_json, digest

ROOT = Path(__file__).resolve().parents[2]
MODELS = (
    "H8_seed7",
    "H8_seed13",
    "H8_seed23",
    "public_Garl_event_lhr",
    "public_Garl_rgb_event_full",
)
SOURCE_FILES = (
    "operational/sota_eval/campaign.py",
    "operational/sota_eval/scoring.py",
    "operational/evttc_transfer/inputs.py",
    "operational/evttc_transfer/models.py",
    "operational/evttc_transfer/run.py",
    "operational/evttc_transfer/score.py",
    "operational/evttc_rgb_transfer/inputs.py",
    "operational/evttc_rgb_transfer/model.py",
    "operational/evttc_rgb_transfer/run.py",
    "operational/evttc_rgb_transfer/score.py",
)
BASELINE_KEY_FILES = (
    "QUERY_MANIFEST.json",
    "INFERENCE_FREEZE.json",
    "PREDICTIONS_SEALED.json",
)
SCORE_FILES = ("RESULT.json", "SCORED_PREDICTIONS.csv", "PER_SEQUENCE.csv", "REPORT.md")
METRIC_FILES = {
    "REPORT.json",
    "METHOD_METRICS.csv",
    "PER_SEQUENCE.csv",
    "PAIRED.csv",
    "SCORED_ROWS.csv",
    "METADATA.json",
}
PUBLIC_FULL_FILES = (
    "paper_ours_full.pth",
    "configs/ablation/ours_full.yaml",
    "model_release_metadata.json",
)
HEAVY_ENTRYPOINTS = (
    "operational.sota_eval.campaign",
    "operational.evttc_transfer.run",
    "operational.evttc_rgb_transfer.run",
    "operational.sota_eval.prefetch",
    "operational.sota_eval.full_prefetch",
    "operational.efficient_context.r1_measure",
    "operational.efficient_context.r1_gib_measure",
    "operational.efficient_context.r1_profile",
    "operational.efficient_context.garl_train",
    "operational.simplex_t_h16_replication.train",
    "operational.train40_system.engine",
    "operational.train40_system.history_resources",
    "operational.train40_system.garl_predictions",
    "operational.train40_system.feature_resources",
    "train_baseline.py",
    "pretrain_jepa.py",
)

Invoker = Callable[[Sequence[str], Path, dict[str, str]], int]


def _stamp() -> str:
    return datetime.now(UTC).isoformat()


def _read(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def _immutable_json(path: Path, value: dict[str, Any]) -> None:
    if path.exists():
        if _read(path) != value:
            raise ValueError(f"Frozen campaign configuration changed: {path}")
        return
    atomic_json(path, value)


def _source_hashes(
    root: Path = ROOT, *, event_backend: str = "direct", full_backend: str = "direct"
) -> dict[str, str]:
    result: dict[str, str] = {}
    selected: list[str] = list(SOURCE_FILES)
    if event_backend == "prefetch" or full_backend == "prefetch":
        selected.append("operational/sota_eval/prefetch.py")
    if full_backend == "prefetch":
        selected.append("operational/sota_eval/full_prefetch.py")
    for relative in selected:
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        result[relative] = digest(path)
    return result


def _configuration(
    *,
    campaign: Path,
    baseline: Path,
    full: Path,
    data_repository: Path,
    code_root: Path,
    public_full_source: Path,
    metrics: Path,
    device: str,
    event_backend: str,
    full_backend: str,
    source_root: Path = ROOT,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "role": "ZERO_UPDATE_EXPANDED_EVTTC_CAMPAIGN",
        "campaign": str(campaign.resolve()),
        "baseline": str(baseline.resolve()),
        "full": str(full.resolve()),
        "data_repository": str(data_repository.resolve()),
        "inventory": str((source_root / "data/manifests/evttc_all32_local.yaml").resolve()),
        "code_root": str(code_root.resolve()),
        "public_full_source": str(public_full_source.resolve()),
        "metrics": str(metrics.resolve()),
        "device": device,
        "event_backend": event_backend,
        "full_backend": full_backend,
        "models": list(MODELS),
        "bootstrap_draws": 10_000,
        "seed": 20_261_008,
        "optimizer_updates": 0,
        "sources": _source_hashes(
            source_root, event_backend=event_backend, full_backend=full_backend
        ),
    }


def _active_heavy_pids() -> list[dict[str, Any]]:
    """Reject independent GPU/training owners while ignoring this process tree."""
    import psutil

    current = psutil.Process()
    related = {
        current.pid,
        *(process.pid for process in current.parents()),
        *(process.pid for process in current.children(recursive=True)),
    }
    found: list[dict[str, Any]] = []
    for process in psutil.process_iter(["pid", "name", "cmdline", "create_time"]):
        try:
            if process.pid in related:
                continue
            name = str(process.info.get("name") or "")
            if not name.lower().startswith("python"):
                continue
            command = [str(item) for item in (process.info.get("cmdline") or [])]
            line = " ".join(command).lower()
            if any(marker in line for marker in HEAVY_ENTRYPOINTS):
                found.append(
                    {
                        "pid": process.pid,
                        "create_time": process.info.get("create_time"),
                        "entrypoint": next(
                            marker for marker in HEAVY_ENTRYPOINTS if marker in line
                        ),
                    }
                )
        except (psutil.AccessDenied, psutil.NoSuchProcess, psutil.ZombieProcess):
            continue
    return found


def _invoke(command: Sequence[str], log: Path, environment: dict[str, str]) -> int:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8") as stream:
        stream.write(f"\n[{_stamp()}] {json.dumps(list(command), ensure_ascii=False)}\n")
        stream.flush()
        options: dict[str, Any] = {}
        if os.name == "nt":
            options["creationflags"] = subprocess.CREATE_NO_WINDOW
        completed = subprocess.run(
            list(command),
            cwd=ROOT,
            env=environment,
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=False,
            text=True,
            **options,
        )
    return completed.returncode


def _transient_state_replace_failure(log: Path) -> bool:
    if not log.is_file():
        return False
    tail = log.read_text(encoding="utf-8", errors="replace")[-16_384:].lower()
    return all(
        marker in tail
        for marker in ("permissionerror", "winerror 5", "os.replace", "state.json")
    )


def _run_child(
    phase: str,
    command: Sequence[str],
    output: Path,
    environment: dict[str, str],
    invoke: Invoker,
) -> tuple[int, int]:
    """Retry only the observed Windows atomic STATE receipt replacement race."""
    for attempt in range(1, 4):
        log = output / "logs" / f"{phase}.attempt_{attempt:02d}.log"
        return_code = invoke(command, log, environment)
        if return_code == 0:
            return return_code, attempt
        if attempt == 3 or not _transient_state_replace_failure(log):
            return return_code, attempt
    raise AssertionError("finite retry loop exhausted without returning")


def _snapshot_backend_execution(directory: Path, output: Path, label: str) -> None:
    source = directory / "BACKEND_EXECUTION.json"
    if not source.is_file():
        return
    checksum = digest(source)
    target = output / "backend_history" / f"{label}_{checksum}.json"
    if target.exists():
        if digest(target) != checksum:
            raise ValueError(f"Backend history snapshot differs: {target}")
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    if digest(target) != checksum:
        raise ValueError(f"Backend history snapshot copy failed: {target}")


def _backend_current(directory: Path, kind: str, backend: str) -> bool:
    path = directory / "BACKEND_FREEZE.json"
    if backend == "direct":
        return not path.exists()
    if backend != "prefetch":
        return False
    if kind == "event":
        from operational.sota_eval.prefetch import GIB, backend_binding
    else:
        from operational.sota_eval.full_prefetch import GIB, backend_binding
    expected = backend_binding(
        directory / "QUERY_MANIFEST.json",
        workers=2,
        max_ahead=2,
        min_available_bytes=2 * GIB,
    )
    return _read(path) == expected


def _event_binding_current(directory: Path, device: str, backend: str) -> bool:
    from operational.sota_eval.reuse import _verify_model_binding

    freeze = _read(directory / "INFERENCE_FREEZE.json")
    source_dir = ROOT / "operational/evttc_transfer"
    sources = {str(path): digest(path) for path in sorted(source_dir.glob("*.py"))}
    if (
        freeze.get("manifest_sha256") != digest(directory / "QUERY_MANIFEST.json")
        or freeze.get("device") != device
        or freeze.get("precision") != "float32"
        or freeze.get("optimizer_updates") != 0
        or freeze.get("no_label_reads") is not True
        or freeze.get("sources") != sources
        or not _backend_current(directory, "event", backend)
    ):
        return False
    _verify_model_binding(freeze["models"])
    return True


def _full_binding_current(directory: Path, device: str, backend: str) -> bool:
    from operational.evttc_rgb_transfer.run import verify_baseline

    freeze = _read(directory / "INFERENCE_FREEZE.json")
    source_paths = list((ROOT / "operational/evttc_rgb_transfer").glob("*.py"))
    source_paths += list((ROOT / "operational/evttc_transfer").glob("*.py"))
    sources = {str(path.resolve()): digest(path) for path in sorted(source_paths)}
    preservation_path = directory / "BASELINE_PRESERVATION.json"
    if (
        freeze.get("manifest_sha256") is None
        or freeze.get("device") != device
        or freeze.get("precision") != "float32"
        or freeze.get("optimizer_updates") != 0
        or freeze.get("labels_not_used_by_inference") is not True
        or freeze.get("sources") != sources
        or freeze.get("baseline_preservation_sha256") != digest(preservation_path)
        or not _backend_current(directory, "full", backend)
    ):
        return False
    model = freeze["model"]
    checkpoint = Path(model["checkpoint"])
    config = Path(model["config"])
    if (
        checkpoint.stat().st_size != model["checkpoint_bytes"]
        or digest(checkpoint) != model["checkpoint_sha256"]
        or digest(config) != model["config_sha256"]
        or digest(ROOT / "operational/evttc_rgb_transfer/model.py") != model["module_sha256"]
    ):
        return False
    native_root = Path(model["native_code_root"])
    for relative, expected in model["native_source_sha256"].items():
        if digest(native_root / relative) != expected:
            return False
    preservation = _read(preservation_path)
    baseline = Path(preservation["baseline"])
    verify_baseline(baseline, preservation)
    return freeze["manifest_sha256"] == digest(baseline / "QUERY_MANIFEST.json")


def _sealed(
    directory: Path,
    kind: str = "generic",
    expected_device: str | None = None,
    backend: str = "direct",
) -> bool:
    try:
        manifest = _read(directory / "QUERY_MANIFEST.json")
        seal = _read(directory / "PREDICTIONS_SEALED.json")
        rows = manifest.get("rows", [])
        fragments = seal.get("fragments", {})
        if not (
            seal.get("status") == "COMPLETE"
            and seal.get("optimizer_updates") == 0
            and bool(rows)
            and seal.get("queries") == len(rows)
            and seal.get("manifest_sha256") == digest(directory / "QUERY_MANIFEST.json")
            and seal.get("binding_sha256") == digest(directory / "INFERENCE_FREEZE.json")
            and len(fragments) == len(rows)
        ):
            return False
        expected = {f"predictions/query_{index:05d}.json" for index in range(len(rows))}
        if set(fragments) != expected:
            return False
        for index, row in enumerate(rows):
            name = f"predictions/query_{index:05d}.json"
            path = directory / name
            sidecar = path.with_suffix(".sha256")
            actual = digest(path)
            if fragments[name] != actual:
                return False
            if not sidecar.is_file() or sidecar.read_text(encoding="ascii").strip() != actual:
                return False
            saved = _read(path)
            if (
                saved.get("query_id") != row.get("query_id")
                or saved.get("sequence_id") != row.get("sequence_id")
                or saved.get("anchor_us") != row.get("anchor_us")
                or saved.get("binding_sha256") != seal.get("binding_sha256")
                or saved.get("optimizer_updates") != 0
            ):
                return False
        if kind == "event":
            return expected_device is not None and _event_binding_current(
                directory, expected_device, backend
            )
        if kind == "full":
            return expected_device is not None and _full_binding_current(
                directory, expected_device, backend
            )
        return True
    except (FileNotFoundError, KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
        return False


def _seal_score(directory: Path) -> None:
    result = _read(directory / "RESULT.json")
    if (
        result.get("status") != "COMPLETE"
        or result.get("prediction_seal_sha256")
        != digest(directory / "PREDICTIONS_SEALED.json")
        or result.get("manifest_sha256") != digest(directory / "QUERY_MANIFEST.json")
    ):
        raise ValueError("Scorer did not produce a complete result bound to predictions")
    value = {
        "status": "COMPLETE",
        "files": {name: digest(directory / name) for name in SCORE_FILES},
        "prediction_seal_sha256": digest(directory / "PREDICTIONS_SEALED.json"),
        "optimizer_updates": 0,
    }
    _immutable_json(directory / "SCORING_SEAL.json", value)


def _scored(directory: Path) -> bool:
    try:
        result = _read(directory / "RESULT.json")
        scoring_seal = _read(directory / "SCORING_SEAL.json")
        return bool(
            result.get("status") == "COMPLETE"
            and result.get("prediction_seal_sha256")
            == digest(directory / "PREDICTIONS_SEALED.json")
            and result.get("manifest_sha256") == digest(directory / "QUERY_MANIFEST.json")
            and scoring_seal.get("status") == "COMPLETE"
            and scoring_seal.get("optimizer_updates") == 0
            and scoring_seal.get("prediction_seal_sha256")
            == digest(directory / "PREDICTIONS_SEALED.json")
            and set(scoring_seal.get("files", {})) == set(SCORE_FILES)
            and all(
                digest(directory / name) == checksum
                for name, checksum in scoring_seal["files"].items()
            )
        )
    except (FileNotFoundError, OSError, ValueError, json.JSONDecodeError):
        return False


def _metrics_complete(directory: Path, input_csv: Path) -> bool:
    try:
        report = _read(directory / "REPORT.json")
        checksums = _read(directory / "SHA256.json")
        metadata = _read(directory / "METADATA.json")
        if not report.get("cohort") or set(checksums) != METRIC_FILES:
            return False
        if any(digest(directory / name) != checksum for name, checksum in checksums.items()):
            return False
        return (
            metadata.get("input_filename") == input_csv.name
            and input_csv.is_file()
            and digest(input_csv) == metadata.get("input_sha256")
        )
    except (FileNotFoundError, KeyError, OSError, ValueError, json.JSONDecodeError):
        return False


def _validated_event_command(
    command: list[str], campaign: Path, baseline: Path, device: str, backend: str
) -> None:
    module = (
        "operational.sota_eval.prefetch"
        if backend == "prefetch"
        else "operational.evttc_transfer.run"
    )
    expected = [str(Path(command[0]).resolve()), "-m", module]
    if backend == "prefetch":
        expected += ["run"]
    expected += [
        "--campaign",
        command[5] if backend == "prefetch" else command[4],
        "--output",
        command[7] if backend == "prefetch" else command[6],
        "--device",
        device,
    ]
    if backend == "prefetch":
        expected += ["--workers", "2", "--max-ahead", "2"]
    if command != expected:
        raise ValueError("Event-only launch receipt has an unexpected entrypoint")
    if Path(command[0]).resolve() != Path(sys.executable).resolve():
        raise ValueError("Event-only launch uses a different Python environment")
    campaign_value = command[5] if backend == "prefetch" else command[4]
    output_value = command[7] if backend == "prefetch" else command[6]
    if (
        (ROOT / campaign_value).resolve() != campaign.resolve()
        and Path(campaign_value).resolve() != campaign.resolve()
    ):
        raise ValueError("Event-only launch campaign differs")
    if (
        (ROOT / output_value).resolve() != baseline.resolve()
        and Path(output_value).resolve() != baseline.resolve()
    ):
        raise ValueError("Event-only launch output differs")


def _wait_for_event_only(
    receipt_path: Path,
    *,
    output: Path,
    campaign: Path,
    baseline: Path,
    device: str,
    backend: str,
    started: float,
    timeout_seconds: float = 6 * 60 * 60,
    interval_seconds: float = 5.0,
) -> str:
    """Wait only for the PID/create-time/command frozen in a launch receipt."""
    import psutil

    receipt = _read(receipt_path)
    command = receipt.get("command")
    if not isinstance(command, list) or not all(isinstance(item, str) for item in command):
        raise ValueError("Event-only launch receipt command must be a string list")
    _validated_event_command(command, campaign, baseline, device, backend)
    if receipt.get("optimizer_updates") != 0:
        raise ValueError("Event-only launch receipt is not a zero-update inference")
    pid = int(receipt["pid"])
    create_time = float(receipt["create_time"])
    receipt_sha = digest(receipt_path)
    waited = {
        "status": "VERIFIED_EXTERNAL_EVENT_ONLY_LAUNCH",
        "receipt": str(receipt_path.resolve()),
        "receipt_sha256": receipt_sha,
        "pid": pid,
        "create_time": create_time,
        "command": command,
        "optimizer_updates": 0,
    }
    _immutable_json(output / "WAITED_EVENT_ONLY.json", waited)
    deadline = time.monotonic() + timeout_seconds
    descendants: dict[int, float] = {}
    last_state = float("-inf")
    while True:
        if (output / "STOP_REQUEST").exists():
            _write_state(output, "wait_event_only", "PAUSED_PRESERVED", started)
            return "PAUSED_PRESERVED"
        parent_alive = False
        try:
            parent = psutil.Process(pid)
            if parent.create_time() != create_time:
                raise ValueError("Event-only PID was reused by another process")
            live_command = parent.cmdline()
            if live_command != command:
                raise ValueError("Event-only live command differs from its launch receipt")
            parent_alive = parent.is_running()
            for child in parent.children(recursive=True):
                descendants[child.pid] = child.create_time()
        except psutil.NoSuchProcess:
            parent_alive = False
        living_descendants = []
        for child_pid, child_created in descendants.items():
            try:
                child = psutil.Process(child_pid)
                if child.create_time() == child_created and child.is_running():
                    living_descendants.append(child_pid)
            except psutil.NoSuchProcess:
                continue
        if not parent_alive and not living_descendants:
            return "EXITED"
        now = time.monotonic()
        if now >= deadline:
            _write_state(
                output,
                "wait_event_only",
                "WAIT_TIMEOUT_PRESERVED",
                started,
                pid=pid,
                living_descendants=living_descendants,
            )
            return "WAIT_TIMEOUT_PRESERVED"
        if now - last_state >= 60:
            _write_state(
                output,
                "wait_event_only",
                "WAITING_VERIFIED_EVENT_ONLY",
                started,
                pid=pid,
                living_descendants=living_descendants,
            )
            last_state = now
        time.sleep(interval_seconds)


def _copy_public_full(source: Path, target: Path) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for relative in PUBLIC_FULL_FILES:
        original = source / relative
        destination = target / relative
        if not original.is_file():
            raise FileNotFoundError(original)
        expected = digest(original)
        if destination.exists():
            if not destination.is_file() or digest(destination) != expected:
                raise ValueError(f"Existing public model asset differs: {destination}")
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(original, destination)
            if digest(destination) != expected:
                raise ValueError(f"Copied public model asset failed verification: {destination}")
        hashes[relative] = expected
    return hashes


def _preserve_baseline(
    baseline: Path, full: Path, *, device: str, event_backend: str
) -> dict[str, Any]:
    if not _sealed(baseline, "event", device, event_backend):
        raise ValueError("Complete sealed event-only baseline required")
    value = {
        "status": "FROZEN",
        "baseline": str(baseline.resolve()),
        "key_files": {name: digest(baseline / name) for name in BASELINE_KEY_FILES},
        "optimizer_updates": 0,
    }
    _immutable_json(full / "BASELINE_PRESERVATION.json", value)
    return value


def _write_state(
    output: Path, phase: str, status: str, started: float, **extra: object
) -> None:
    atomic_json(
        output / "PIPELINE_STATE.json",
        {
            "phase": phase,
            "status": status,
            "elapsed_seconds": time.monotonic() - started,
            "optimizer_updates": 0,
            "checked_utc": _stamp(),
            **extra,
        },
    )


def run(
    *,
    output: Path,
    campaign: Path,
    baseline: Path,
    full: Path,
    data_repository: Path,
    code_root: Path,
    public_full_source: Path,
    device: str = "cuda",
    wait_for_event_only: Path | None = None,
    event_backend: str = "direct",
    full_backend: str = "direct",
    invoke: Invoker = _invoke,
    source_root: Path = ROOT,
) -> dict[str, Any]:
    """Run or resume each phase, advancing only across verified completion seals."""
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    metrics = output / "expanded_metrics"
    with Lease(output):
        if wait_for_event_only is not None:
            wait_status = _wait_for_event_only(
                wait_for_event_only.resolve(),
                output=output,
                campaign=campaign,
                baseline=baseline,
                device=device,
                backend=event_backend,
                started=started,
            )
            if wait_status != "EXITED":
                return {"status": wait_status, "phase": "wait_event_only"}
        active = _active_heavy_pids()
        if active:
            _write_state(output, "admission", "BLOCKED_ACTIVE_HEAVY", started, active=active)
            return {"status": "BLOCKED_ACTIVE_HEAVY", "active": active}
        config = _configuration(
            campaign=campaign,
            baseline=baseline,
            full=full,
            data_repository=data_repository,
            code_root=code_root,
            public_full_source=public_full_source,
            metrics=metrics,
            device=device,
            event_backend=event_backend,
            full_backend=full_backend,
            source_root=source_root,
        )
        freeze = output / "CAMPAIGN_FREEZE.json"
        _immutable_json(freeze, config)
        environment = dict(os.environ)
        environment.update(PYTHONUTF8="1", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2")
        inventory = Path(config["inventory"])
        python = sys.executable

        event_command = [python, "-m"]
        if event_backend == "prefetch":
            event_command += ["operational.sota_eval.prefetch", "run"]
        else:
            event_command += ["operational.evttc_transfer.run"]
        event_command += [
            "--campaign",
            str(campaign.resolve()),
            "--output",
            str(baseline.resolve()),
            "--device",
            device,
        ]
        if event_backend == "prefetch":
            event_command += ["--workers", "2", "--max-ahead", "2"]
        phases: list[tuple[str, list[str], Callable[[], bool]]] = [
            (
                "event_inference",
                event_command,
                lambda: _sealed(baseline, "event", device, event_backend),
            )
        ]
        completed: list[str] = []
        for phase, command, check in phases:
            if not check():
                _write_state(output, phase, "RUNNING", started, command=command)
                _snapshot_backend_execution(baseline, output, "event")
                return_code, attempts = _run_child(
                    phase, command, output, environment, invoke
                )
                if return_code != 0:
                    _write_state(
                        output,
                        phase,
                        "FAILED_PRESERVED",
                        started,
                        return_code=return_code,
                        attempts=attempts,
                    )
                    raise RuntimeError(f"{phase} failed with exit code {return_code}")
                if not check():
                    _write_state(output, phase, "PAUSED_PRESERVED", started, return_code=0)
                    return {"status": "PAUSED_PRESERVED", "phase": phase}
            completed.append(phase)

        full.mkdir(parents=True, exist_ok=True)
        _preserve_baseline(
            baseline, full, device=device, event_backend=event_backend
        )
        assets = _copy_public_full(public_full_source, full / "public_garl")
        _immutable_json(
            full / "PUBLIC_FULL_ASSETS.json",
            {
                "status": "VERIFIED_COPY",
                "source": str(public_full_source.resolve()),
                "files": assets,
                "optimizer_updates": 0,
            },
        )
        full_command = [python, "-m"]
        if full_backend == "prefetch":
            full_command += ["operational.sota_eval.full_prefetch", "run"]
        else:
            full_command += ["operational.evttc_rgb_transfer.run"]
        full_command += [
            "--output",
            str(full.resolve()),
            "--baseline",
            str(baseline.resolve()),
            "--code-root",
            str(code_root.resolve()),
            "--device",
            device,
        ]
        if full_backend == "prefetch":
            full_command += ["--workers", "2", "--max-ahead", "2"]
        later: list[tuple[str, list[str], Callable[[], bool]]] = [
            (
                "full_inference",
                full_command,
                lambda: _sealed(full, "full", device, full_backend),
            ),
            (
                "event_scoring",
                [
                    python,
                    "-m",
                    "operational.evttc_transfer.score",
                    "--output",
                    str(baseline.resolve()),
                    "--inventory",
                    str(inventory),
                    "--data-root",
                    str(data_repository.resolve()),
                ],
                lambda: _scored(baseline),
            ),
            (
                "full_scoring",
                [
                    python,
                    "-m",
                    "operational.evttc_rgb_transfer.score",
                    "--output",
                    str(full.resolve()),
                    "--baseline",
                    str(baseline.resolve()),
                ],
                lambda: _scored(full),
            ),
            (
                "expanded_metrics",
                [
                    python,
                    "-m",
                    "operational.sota_eval.scoring",
                    "--input",
                    str((full / "SCORED_PREDICTIONS.csv").resolve()),
                    "--output-dir",
                    str(metrics.resolve()),
                    "--models",
                    *MODELS,
                    "--bootstrap-draws",
                    "10000",
                    "--seed",
                    "20261008",
                ],
                lambda: _metrics_complete(metrics, full / "SCORED_PREDICTIONS.csv"),
            ),
        ]
        for phase, command, check in later:
            if not check():
                _write_state(output, phase, "RUNNING", started, command=command)
                if phase == "full_inference":
                    _snapshot_backend_execution(full, output, "full")
                return_code, attempts = _run_child(
                    phase, command, output, environment, invoke
                )
                if return_code != 0:
                    _write_state(
                        output,
                        phase,
                        "FAILED_PRESERVED",
                        started,
                        return_code=return_code,
                        attempts=attempts,
                    )
                    raise RuntimeError(f"{phase} failed with exit code {return_code}")
                if phase == "event_scoring":
                    _seal_score(baseline)
                elif phase == "full_scoring":
                    _seal_score(full)
                if not check():
                    _write_state(output, phase, "PAUSED_PRESERVED", started, return_code=0)
                    return {"status": "PAUSED_PRESERVED", "phase": phase}
            completed.append(phase)

        result = {
            "status": "COMPLETE",
            "phases": completed,
            "campaign_freeze_sha256": digest(freeze),
            "baseline_preservation_sha256": digest(full / "BASELINE_PRESERVATION.json"),
            "full_prediction_seal_sha256": digest(full / "PREDICTIONS_SEALED.json"),
            "scored_predictions_sha256": digest(full / "SCORED_PREDICTIONS.csv"),
            "metrics_sha256": digest(metrics / "SHA256.json"),
            "optimizer_updates": 0,
            "elapsed_seconds": time.monotonic() - started,
        }
        atomic_json(output / "CAMPAIGN_RESULT.json", result)
        _write_state(
            output,
            "complete",
            "COMPLETE",
            started,
            result_sha256=digest(output / "CAMPAIGN_RESULT.json"),
        )
        return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("artifacts/sota_campaign_20261008"))
    parser.add_argument("--campaign", type=Path, default=Path("artifacts/train40_system_20261005"))
    parser.add_argument("--data-repository", type=Path, default=Path("../e-jepa-ttc"))
    parser.add_argument("--code-root", type=Path, default=Path("E:/Garl-TTC"))
    parser.add_argument(
        "--public-full-source",
        type=Path,
        default=Path("artifacts/evttc_rgb_transfer_20261008/public_garl"),
    )
    parser.add_argument("--baseline-output", type=Path)
    parser.add_argument("--full-output", type=Path)
    parser.add_argument(
        "--wait-for-event-only",
        type=Path,
        help="Wait for the exact PID/create-time/command in this launch receipt",
    )
    parser.add_argument("--event-backend", choices=("direct", "prefetch"), default="direct")
    parser.add_argument("--full-backend", choices=("direct", "prefetch"), default="direct")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()
    baseline_output = args.baseline_output or args.output / "dev32_expanded"
    full_output = args.full_output or args.output / "dev32_expanded_rgb"
    result = run(
        output=args.output,
        campaign=args.campaign,
        baseline=baseline_output,
        full=full_output,
        data_repository=args.data_repository,
        code_root=args.code_root,
        public_full_source=args.public_full_source,
        device=args.device,
        wait_for_event_only=args.wait_for_event_only,
        event_backend=args.event_backend,
        full_backend=args.full_backend,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "COMPLETE" else 2


if __name__ == "__main__":
    raise SystemExit(main())

"""Ordered, zero-update R1 completion after the independent accuracy GPU queue."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import re
import subprocess
import sys
import time
import zipfile
from datetime import UTC, datetime
from pathlib import Path

from operational.efficient_context.common import ROOT, Lease, atomic_json, digest, read
from operational.efficient_context.r1_chain import DEADLINE
from operational.sota_eval import r1_resume

ARTIFACTS = r1_resume.ARTIFACTS / "supervisor"
OWNERSHIP = ROOT / "artifacts/sota_campaign_20261008/R1_PAUSE_OWNERSHIP.json"
SOURCE_FILES = (
    "operational/sota_eval/r1_resume.py",
    "operational/sota_eval/r1_resume_supervisor.py",
    "operational/efficient_context/r1_gib_bundle.py",
    "operational/efficient_context/common.py",
    "tests/unit/test_sota_r1_resume.py",
    "tests/unit/test_sota_r1_resume_supervisor.py",
)


def stamp() -> str:
    """Return UTC operational timestamps, separate from query measurements."""
    return datetime.now(UTC).isoformat()


def entrypoint(name: str, command: list[str]) -> str | None:
    """Extract real Python module/script identities rather than shell text."""
    if not name.lower().startswith("python"):
        return None
    if "-m" in command:
        i = command.index("-m") + 1
        return command[i].lower() if i < len(command) else None
    if len(command) > 1 and command[1].lower().endswith(".py"):
        return Path(command[1]).stem.lower()
    return None


def is_heavy(name: str, command: list[str]) -> bool:
    """Identify known GPU workers by their real entrypoint and execution mode."""
    module = entrypoint(name, command)
    if module is None:
        return False
    values = [str(value).replace("\\", "/").lower() for value in command]
    if "--device=cpu" in values or any(
        left == "--device" and right == "cpu"
        for left, right in zip(values, values[1:], strict=False)
    ):
        return False
    if module == "operational.sota_eval.fcwd_run" and "--score" in values:
        return False
    stage_pattern = r"(?:--?)?stage(?:70|71|72|73|74|75|76)"
    if re.search(r"stage(?:70|71|72|73|74|75|76)(?:\D|$)", module) or any(
        re.fullmatch(stage_pattern, value) for value in values[1:]
    ):
        return True
    exact_gpu_entrypoints = {
        "operational.sota_eval.fcwd_run",
        "operational.sota_eval.cost",
        "operational.sota_eval.prefetch",
        "operational.sota_eval.full_prefetch",
        "operational.sota_eval.campaign",
        "operational.evttc_transfer.run",
        "operational.evttc_rgb_transfer.run",
        "operational.efficient_context.r1_measure",
        "operational.efficient_context.r1_gib_measure",
        "operational.efficient_context.r1_profile",
        "operational.efficient_context.garl_train",
        "operational.train40_system.engine",
        "operational.simplex_t_h16_replication.train",
        "train_baseline",
        "pretrain_jepa",
        "finetune_ttc",
    }
    return module in exact_gpu_entrypoints or module.startswith(
        "operational.train40_system.engine_"
    )


def heavy_processes() -> list[dict]:
    """Ignore this coordinator's ancestry/child tree but reject unrelated owners."""
    import psutil

    current = psutil.Process()
    related = {
        current.pid,
        *(p.pid for p in current.parents()),
        *(p.pid for p in current.children(recursive=True)),
    }
    found = []
    for process in psutil.process_iter(["pid", "name", "cmdline", "create_time"]):
        try:
            if process.pid not in related and is_heavy(
                process.info["name"] or "", process.info["cmdline"] or []
            ):
                found.append(
                    dict(
                        pid=process.pid,
                        create_time=process.info["create_time"],
                        command=process.info["cmdline"],
                    )
                )
        except (psutil.AccessDenied, psutil.NoSuchProcess):
            continue
    return found


def same_process(identity: dict) -> bool:
    """Prevent a stale PID from conferring process ownership."""
    import psutil

    try:
        return psutil.Process(identity["pid"]).create_time() == identity["create_time"]
    except psutil.NoSuchProcess:
        return False


def owned_stop(output: Path, receipt: dict) -> Path:
    """Validate the exact root-owned marker before any ownership mutation."""
    path = output / "STOP_REQUEST"
    if (
        receipt["owner"] != "sota_campaign_20261008"
        or Path(receipt["path"]).resolve() != path.resolve()
    ):
        raise ValueError("stop receipt does not own this R1 marker")
    if not path.is_file() or digest(path) != receipt["sha256"]:
        raise ValueError("stop marker changed or belongs to another owner")
    return path


def retire_owner(process: object, expected: dict) -> None:
    """Terminate only the explicitly identified idle old control process."""
    from typing import cast

    import psutil

    owner = cast(psutil.Process, process)
    if owner.pid != expected["pid"] or owner.create_time() != expected["create_time"]:
        raise ValueError("old supervisor PID identity changed")
    if (
        entrypoint(owner.name(), owner.cmdline())
        != "operational.efficient_context.r1_gib_supervisor"
    ):
        raise ValueError("old process is not the admitted R1 supervisor")
    if owner.children(recursive=True):
        raise RuntimeError("old supervisor still has children; handoff must wait")
    owner.terminate()
    owner.wait(timeout=15)


def handoff(output: Path, artifacts: Path, config: dict) -> None:
    """Transfer only an explicitly authorized idle lease and exact owned marker."""
    import psutil

    path = artifacts / "HANDOFF.json"
    if path.exists() and read(path)["status"] == "COMPLETE":
        return
    if not config["handoff"]:
        return
    expected = config["old_owner"]
    receipt_path = Path(config["ownership_receipt"])
    if digest(receipt_path) != config["ownership_receipt_sha256"]:
        raise ValueError("root ownership receipt changed")
    receipt = read(receipt_path)
    if path.exists():
        prior = read(path)
        if prior["status"] == "OLD_OWNER_RETIRED" and not (output / "STOP_REQUEST").exists():
            if prior["expected_owner"] != expected or same_process(expected):
                raise ValueError("incomplete handoff owner identity changed")
            r1_resume.verify_snapshot(output, prior["receipt_snapshot"])
            atomic_json(
                path,
                prior
                | dict(status="COMPLETE", checked_utc=stamp(), original_receipts_preserved=True),
            )
            return
    marker = owned_stop(output, receipt)
    lease_path = output / "WRITER.lock"
    lease = read(lease_path)
    if lease != expected:
        raise ValueError("old R1 lease differs from explicit PID/create_time")
    snapshot = r1_resume.receipt_snapshot(output)
    if snapshot["count"] < receipt["preserve_pairs"]:
        raise ValueError("saved R1 work is missing")
    record: dict = dict(
        status="PREPARED",
        expected_owner=expected,
        stop_sha256=digest(marker),
        lease_sha256=digest(lease_path),
        stop_bytes_b64=base64.b64encode(marker.read_bytes()).decode(),
        lease=lease,
        receipt_snapshot=snapshot,
        checked_utc=stamp(),
    )
    atomic_json(path, record)
    if same_process(expected):
        owner = psutil.Process(expected["pid"])
        state = read(output / "CHAIN_STATE.json")
        if state["status"] != "WAITING" or not state.get("stop_requested"):
            raise RuntimeError("old control process has not reached its preserved wait boundary")
        owned_stop(output, receipt)
        retire_owner(owner, expected)
    elif psutil.pid_exists(expected["pid"]):
        raise ValueError("old PID was reused; do not terminate its new owner")
    r1_resume.verify_snapshot(output, snapshot)
    marker = owned_stop(output, receipt)
    record["status"] = "OLD_OWNER_RETIRED"
    atomic_json(path, record)
    marker.unlink()
    record.update(status="COMPLETE", checked_utc=stamp(), original_receipts_preserved=True)
    atomic_json(path, record)


def stop_child(output: Path, artifacts: Path, reason: str) -> None:
    """Create a marker only when absent; never overwrite someone else's pause."""
    path = output / "STOP_REQUEST"
    if path.exists():
        return
    payload = f"R1_RESUME_SUPERVISOR {reason}\n".encode()
    # Exclusive creation prevents racing another owner between exists and write.
    try:
        with path.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except FileExistsError:
        return
    atomic_json(
        artifacts / "OWN_STOP.json",
        dict(path=str(path), sha256=digest(path), reason=reason, checked_utc=stamp()),
    )


def clear_recovered_own_stop(output: Path, artifacts: Path, heavy: list[dict]) -> None:
    """Release only this coordinator's temporary exclusivity marker after recovery."""
    record = artifacts / "OWN_STOP.json"
    path = output / "STOP_REQUEST"
    if not record.exists() or not path.exists():
        return
    owned = read(record)
    if owned["reason"] != "EXTERNAL_HEAVY" or heavy or datetime.now(UTC) >= DEADLINE:
        return
    if Path(owned["path"]) == path and digest(path) == owned["sha256"]:
        path.unlink()
        atomic_json(artifacts / "OWN_STOP_RELEASED.json", owned | dict(released_utc=stamp()))


def classify_failure(text: str) -> str:
    """Use only this attempt's terminal exception; integrity never retries."""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    terminal = lines[-1] if lines else ""
    if terminal.startswith(("ValueError:", "AssertionError:")):
        return "INTEGRITY_PERMANENT"
    if any(word in terminal.lower() for word in ("checksum", "parity", "frozen", "changed")):
        return "INTEGRITY_PERMANENT"
    if terminal.startswith("InterruptedError:"):
        if "headroom" in terminal or "R1 needs" in terminal:
            return "RESOURCE_TRANSIENT"
        if "stop requested" in terminal:
            return "PRESERVED_STOP"
    if terminal.startswith(("OSError:", "PermissionError:", "MemoryError:")):
        return "RESOURCE_TRANSIENT"
    return "PERMANENT_FAILURE"


def configuration(args: argparse.Namespace) -> dict:
    """Pin explicit sources and original freeze without editing any old contract."""
    if args.handoff and (
        args.expected_owner_pid is None or args.expected_owner_create_time is None
    ):
        raise ValueError("--handoff requires explicit expected owner PID and create_time")
    old = read(args.output / "SUPERVISOR_GIB_FREEZE.json")
    for relative, expected in old["sources"].items():
        if digest(ROOT / relative) != expected:
            raise ValueError("old source freeze changed: " + relative)
    return dict(
        output=str(args.output),
        artifacts=str(args.artifacts),
        handoff=args.handoff,
        old_owner=dict(pid=args.expected_owner_pid, create_time=args.expected_owner_create_time),
        ownership_receipt=str(args.ownership_receipt),
        ownership_receipt_sha256=digest(args.ownership_receipt) if args.handoff else None,
        parent_freeze_sha256=digest(args.output / "SUPERVISOR_GIB_FREEZE.json"),
        sources={relative: digest(ROOT / relative) for relative in SOURCE_FILES},
        max_attempts=3,
        retry_seconds=1800,
        deadline_utc=DEADLINE.isoformat(),
        optimizer_updates=0,
        os_cache_history="Interrupted; OS cache history is not equivalent to uninterrupted replay.",
    )


def verify_sources(output: Path, config: dict) -> None:
    """Reject source edits between admission, retries, and final packaging."""
    if digest(output / "SUPERVISOR_GIB_FREEZE.json") != config["parent_freeze_sha256"]:
        raise ValueError("old freeze changed")
    sources = read(output / "SUPERVISOR_GIB_FREEZE.json")["sources"] | config["sources"]
    for relative, expected in sources.items():
        if digest(ROOT / relative) != expected:
            raise ValueError("admitted source changed: " + relative)


def reconcile_attempt(output: Path, path: Path, retry_seconds: int) -> dict:
    """Recover a dead coordinator's attempt from its own terminal log, fail closed."""
    record = read(path)
    if record["status"] not in ("STARTING", "RUNNING"):
        return record
    if record.get("child") and same_process(record["child"]):
        return record
    r1_resume.verify_snapshot(output, record["before_snapshot"])
    log = Path(record["log"])
    terminal = log.read_text(encoding="utf-8", errors="replace") if log.exists() else ""
    # An abruptly vanished process without a terminal transient exception is not
    # automatically restarted: its failure reason is unknown.
    record.update(
        status="PRESERVED_FAILURE",
        failure_kind=classify_failure(terminal),
        recovered_coordinator_utc=stamp(),
        retry_after_epoch=time.time() + retry_seconds,
    )
    atomic_json(path, record)
    return record


def finalize(output: Path, artifacts: Path, config: dict) -> dict:
    """Invoke the original bundle(output) and independently verify its ZIP members."""
    from operational.efficient_context.r1_gib_bundle import bundle

    verify_sources(output, config)
    result = read(output / "measurement/RESULT.json")
    if (
        result.get("status") != "COMPLETE"
        or result.get("pairs") != 768
        or not result.get("gpu_measured")
    ):
        raise ValueError("R1 measurement is not complete")
    admitted = r1_resume.plan(output)
    if admitted["completed_pairs"] != 768:
        raise ValueError("R1 result lacks all exact admitted receipts")
    atomic_json(
        output / "RESUME_RECOVERY_ADMISSION.json",
        dict(
            configuration=config,
            final_receipt_snapshot=admitted["receipt_snapshot"],
            resume_mode="completed stream groups skipped; partial group replay restored",
            os_cache_history=config["os_cache_history"],
            optimizer_updates=0,
            concurrency_exposures=[read(p) for p in sorted(artifacts.glob("EXPOSURE_*.json"))],
        ),
    )
    atomic_json(
        output / "RESUME_SOURCE_ARCHIVE.json",
        dict(
            sources=[
                dict(
                    path=relative,
                    sha256=digest(ROOT / relative),
                    bytes_base64=base64.b64encode((ROOT / relative).read_bytes()).decode(),
                )
                for relative in SOURCE_FILES
            ]
        ),
    )
    bundle(output)
    archive_path = output / "R1_ESSENTIAL.zip"
    verification = read(output / "BUNDLE_VERIFICATION.json")
    if verification["status"] != "PASSED" or digest(archive_path) != verification["sha256"]:
        raise ValueError("R1 bundle SHA verification failed")
    with zipfile.ZipFile(archive_path) as archive:
        manifest = json.loads(archive.read("SHA256_MANIFEST.json"))
        for name, expected in manifest.items():
            if hashlib.sha256(archive.read(name)).hexdigest() != expected:
                raise ValueError("R1 bundle member changed: " + name)
    atomic_json(
        artifacts / "FINAL_VERIFICATION.json",
        verification
        | dict(
            status="COMPLETE",
            receipts=768,
            optimizer_updates=0,
            interrupted_cache_history_declared=True,
            checked_utc=stamp(),
        ),
    )
    return verification


def publish(artifacts: Path, status: str, **details: object) -> None:
    """Publish current state and a concrete next action in the new supervisor root."""
    row = dict(status=status, checked_utc=stamp(), optimizer_updates=0, **details)
    atomic_json(artifacts / "STATE.json", row)
    atomic_json(artifacts / "NEXT_DECISION.json", row)


def run(args: argparse.Namespace, config: dict) -> None:
    """Wait for exclusive resources, transfer control if authorized, then complete R1."""
    import psutil

    output, artifacts = args.output, args.artifacts
    frozen = artifacts / "SUPERVISOR_FREEZE.json"
    if frozen.exists() and read(frozen) != config:
        raise ValueError("new R1 supervisor configuration/source freeze changed")
    if not frozen.exists():
        atomic_json(frozen, config)
    attempts_dir = artifacts / "attempts"
    attempts_dir.mkdir(parents=True, exist_ok=True)
    with Lease(artifacts):
        while True:
            verify_sources(output, config)
            if (output / "measurement/RESULT.json").exists():
                verification = finalize(output, artifacts, config)
                publish(artifacts, "COMPLETE", bundle_sha256=verification["sha256"])
                return
            if datetime.now(UTC) >= DEADLINE:
                publish(
                    artifacts,
                    "DEADLINE_PRESERVED",
                    next_action="Keep saved pairs; new authorization required",
                )
                return
            heavy = heavy_processes()
            clear_recovered_own_stop(output, artifacts, heavy)
            if heavy:
                publish(artifacts, "WAIT_GPU_QUEUE", owners=heavy)
                time.sleep(30)
                continue
            if config["handoff"]:
                try:
                    handoff(output, artifacts, config)
                except RuntimeError as exc:
                    publish(artifacts, "WAIT_HANDOFF_BOUNDARY", reason=str(exc))
                    time.sleep(30)
                    continue
            if (output / "STOP_REQUEST").exists():
                publish(
                    artifacts, "WAIT_OWNED_STOP", next_action="Marker owner must release its pause"
                )
                time.sleep(30)
                continue
            old_lease = output / "WRITER.lock"
            if old_lease.exists() and same_process(read(old_lease)):
                publish(artifacts, "WAIT_R1_OWNER", owner=read(old_lease))
                time.sleep(30)
                continue
            attempts = sorted(attempts_dir.glob("attempt_*.json"))
            if len(attempts) >= config["max_attempts"]:
                publish(artifacts, "ATTEMPTS_EXHAUSTED_PRESERVED", attempts=len(attempts))
                return
            if attempts:
                previous = reconcile_attempt(output, attempts[-1], config["retry_seconds"])
                if previous.get("child") and same_process(previous["child"]):
                    publish(artifacts, "WAIT_EXISTING_CHILD", child=previous["child"])
                    time.sleep(30)
                    continue
                if previous.get("failure_kind") not in ("RESOURCE_TRANSIENT", "PRESERVED_STOP"):
                    publish(artifacts, "PERMANENT_FAILURE_PRESERVED", previous=previous)
                    return
                wait = previous.get("retry_after_epoch", 0) - time.time()
                if wait > 0:
                    publish(artifacts, "WAIT_TRANSIENT_BACKOFF", remaining_seconds=wait)
                    time.sleep(min(30, wait))
                    continue
            if psutil.virtual_memory().available < 2 * 1024**3:
                publish(artifacts, "WAIT_RAM", available_bytes=psutil.virtual_memory().available)
                time.sleep(30)
                continue
            import shutil

            if shutil.disk_usage(output).free < 10_000_000_000:
                publish(artifacts, "WAIT_DISK")
                time.sleep(30)
                continue
            number = len(attempts) + 1
            record_path = attempts_dir / f"attempt_{number:03d}.json"
            log_path = attempts_dir / f"attempt_{number:03d}_{time.time_ns()}.log"
            child_artifacts = artifacts / f"attempt_{number:03d}_resume"
            command = [
                sys.executable,
                "-m",
                "operational.sota_eval.r1_resume",
                "--execute",
                "--output",
                str(output),
                "--artifacts",
                str(child_artifacts),
            ]
            snapshot = r1_resume.receipt_snapshot(output)
            record = dict(
                attempt=number,
                command=command,
                log=str(log_path),
                before_snapshot=snapshot,
                status="STARTING",
                started_utc=stamp(),
            )
            atomic_json(record_path, record)
            with log_path.open("xb", buffering=0) as log:
                child = subprocess.Popen(
                    command,
                    cwd=ROOT,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    creationflags=subprocess.CREATE_NO_WINDOW,
                )
                record.update(
                    status="RUNNING",
                    child=dict(pid=child.pid, create_time=psutil.Process(child.pid).create_time()),
                )
                atomic_json(record_path, record)
                while child.poll() is None:
                    heavy = heavy_processes()
                    if datetime.now(UTC) >= DEADLINE:
                        stop_child(output, artifacts, "DEADLINE")
                    elif heavy:
                        exposure = artifacts / f"EXPOSURE_{number:03d}.json"
                        if not exposure.exists():
                            atomic_json(
                                exposure,
                                dict(
                                    checked_utc=stamp(),
                                    owners=heavy,
                                    receipts_at_detection=r1_resume.receipt_snapshot(output),
                                    timing_caveat="An in-flight pair may overlap independent work.",
                                ),
                            )
                        stop_child(output, artifacts, "EXTERNAL_HEAVY")
                    publish(
                        artifacts,
                        "RUNNING",
                        attempt=number,
                        child=record["child"],
                        log=str(log_path),
                    )
                    time.sleep(15)
            r1_resume.verify_snapshot(output, snapshot)
            record.update(
                returncode=child.returncode, ended_utc=stamp(), log_sha256=digest(log_path)
            )
            if child.returncode == 0 and (output / "measurement/RESULT.json").exists():
                record["status"] = "COMPLETE"
                atomic_json(record_path, record)
                continue
            kind = classify_failure(log_path.read_text(encoding="utf-8", errors="replace"))
            record.update(
                status="PRESERVED_FAILURE",
                failure_kind=kind,
                retry_after_epoch=time.time() + config["retry_seconds"],
            )
            atomic_json(record_path, record)
            if kind not in ("RESOURCE_TRANSIENT", "PRESERVED_STOP"):
                publish(artifacts, "PERMANENT_FAILURE_PRESERVED", attempt=number, failure_kind=kind)
                return


def main() -> int:
    """Default is CPU planning; only --run can launch inference or --handoff act."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=r1_resume.OUTPUT)
    parser.add_argument("--artifacts", type=Path, default=ARTIFACTS)
    parser.add_argument("--ownership-receipt", type=Path, default=OWNERSHIP)
    parser.add_argument("--expected-owner-pid", type=int)
    parser.add_argument("--expected-owner-create-time", type=float)
    parser.add_argument("--handoff", action="store_true")
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    for name in ("output", "artifacts", "ownership_receipt"):
        setattr(args, name, getattr(args, name).resolve())
    config = configuration(args)
    atomic_json(args.artifacts / "PLAN.json", config)
    if args.run:
        try:
            run(args, config)
        except Exception as exc:
            publish(
                args.artifacts, "FAILED_PRESERVED", exception=type(exc).__name__, reason=str(exc)
            )
            raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Finite authorized nocturnal queue; one heavy child and durable state changes."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import psutil

ROOT = Path(__file__).resolve().parents[2]
NIGHT = ROOT / "artifacts/simplex_t/nocturnal_20261003"
H16 = ROOT / "artifacts/simplex_t/h16_replication_20261003"
sys.path.insert(0, str(ROOT / "operational/simplex_t_h16_replication"))
from common import atomic_json, digest, publish_json, record  # noqa: E402

CAMPAIGN = "SIMPLEX_T_NOCTURNAL_20261003"


def snapshot(phase: str, status: str) -> dict:
    """Summarize saved work, never substitute heartbeat for completed work."""
    saved = lost = endpoints = 0
    journals = {}
    for name, root in (("N1", H16), ("N2_N3", NIGHT / "execution")):
        path = root / "PHYSICAL_WORK.json"
        if path.exists():
            journal = record(path)
            a = journal["accounting"]
            saved += a["scientific_saved_updates"]
            lost += a["scientific_uncertain_lost_upper"]
            endpoints += sum(v["completed"] == 2500 for v in journal["fits"].values())
            journals[name] = dict(sha256=digest(path), accounting=a, fits=journal["fits"])
    active = {}
    for name, root in (("N1", H16), ("N2_N3", NIGHT / "execution")):
        path = root / "UPDATE_PROGRESS.json"
        if path.exists():
            active[name] = record(path)
    technical = NIGHT / "TECHNICAL_WORK.json"
    active_phase = "N1" if phase.startswith("N1") else "N2_N3"
    return dict(
        campaign=CAMPAIGN,
        phase=phase,
        status=status,
        endpoints=endpoints,
        scientific_saved_updates=saved,
        repeated_or_uncertain_updates_upper=lost,
        journals=journals,
        active=active,
        technical=record(technical) if technical.exists() else {},
        N0="COMPLETE" if (NIGHT / "BASELINE_REGISTRY.csv").exists() else "PENDING",
        deadline_utc=record(NIGHT / "WINDOW_AUTHORIZATION.json")["deadline_utc"],
        next_id=active.get(active_phase, {}).get("current_fit"),
    )


def persist(phase: str, status: str) -> None:
    state = snapshot(phase, status)
    path = NIGHT / "PROGRESS.json"
    if path.exists() and record(path) == state:
        return
    atomic_json(path, state)
    with (NIGHT / "ACTIVITY.md").open("a", encoding="utf-8") as stream:
        stream.write(
            f"\n- {datetime.now(UTC).isoformat()} {phase} {status}: "
            f"{state['endpoints']} endpoints; {state['scientific_saved_updates']} "
            f"saved updates; lost/uncertain upper {state['repeated_or_uncertain_updates_upper']}.\n"
        )
    print(json.dumps(state, ensure_ascii=False), flush=True)


def expired() -> bool:
    deadline = datetime.fromisoformat(record(NIGHT / "WINDOW_AUTHORIZATION.json")["deadline_utc"])
    return datetime.now(UTC) >= deadline


def run(phase: str, command: list[str], *, training: bool = False) -> int:
    attempts = NIGHT / "driver_receipts" / phase
    attempts.mkdir(parents=True, exist_ok=True)
    attempt = len(list(attempts.glob("attempt_*.json"))) + 1
    stem = attempts / f"attempt_{attempt:03d}"
    started = datetime.now(UTC).isoformat()
    with stem.with_suffix(".log").open("wb") as log:
        child = subprocess.Popen(
            command,
            cwd=ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            creationflags=subprocess.CREATE_NO_WINDOW,
            env={
                **os.environ,
                "PYTHONUTF8": "1",
                "PYTHONDONTWRITEBYTECODE": "1",
                "PYTHONPATH": str(ROOT) + os.pathsep + str(ROOT / "src"),
                "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
            },
        )
        atomic_json(
            NIGHT / "ACTIVE_CHILD.json",
            dict(
                phase=phase,
                pid=child.pid,
                create_time=psutil.Process(child.pid).create_time(),
                command=command,
                started_utc=started,
                training=training,
            ),
        )
        while child.poll() is None:
            persist(phase, "RUNNING")
            # The training resource callback owns a complete checkpoint at deadline;
            # never terminate a trainer or another owner's process from this driver.
            time.sleep(10)
        code = child.returncode
    atomic_json(
        stem.with_suffix(".json"),
        dict(
            phase=phase,
            command=command,
            started_utc=started,
            ended_utc=datetime.now(UTC).isoformat(),
            returncode=code,
            log_sha256=digest(stem.with_suffix(".log")),
        ),
    )
    persist(phase, "COMPLETED" if code == 0 else "PAUSED_OR_FAILED")
    return code


def command(script: str, *args: str) -> list[str]:
    if script.startswith("simplex_t_cost_context/"):
        module = "operational." + script.removesuffix(".py").replace("/", ".")
        return [sys.executable, "-B", "-m", module, *args]
    return [sys.executable, "-B", str(ROOT / "operational" / script), *args]


def train_until_sealed(phase: str, script: str, seal: Path, *args: str) -> bool:
    repeated = 0
    previous = None
    while not seal.exists() and not expired():
        code = run(phase, command(script, *args), training=True)
        if code == 0 and seal.exists():
            return True
        if phase == "N1_TRAIN" and code == 4:
            continue  # intentional first100 complete state restoration
        paths = (
            [H16 / "RESOURCES.json"]
            if phase == "N1_TRAIN"
            else [NIGHT / "execution/RESOURCES.json"]
        )
        observations = [record(p) for p in paths if p.exists()]
        attempt = sorted((NIGHT / "driver_receipts" / phase).glob("attempt_*.json"))[-1]
        receipt = record(attempt)
        started = datetime.fromisoformat(receipt["started_utc"]).timestamp()
        fresh = any(p.stat().st_mtime >= started for p in paths if p.exists())
        log = attempt.with_suffix(".log").read_text(encoding="utf-8", errors="replace")
        failure = tuple(tuple(v.get("reasons", [])) for v in observations)
        admitted_pause = code == 3 or (code == 1 and "PAUSED_RESOURCE" in log)
        if not any(failure) or not fresh or not admitted_pause:
            atomic_json(
                NIGHT / f"BLOCK_{phase}.json",
                dict(
                    returncode=code,
                    reason="NON_RESOURCE_ERROR_REQUIRES_DIAGNOSIS",
                    resources=observations,
                ),
            )
            return False
        repeated = repeated + 1 if failure == previous else 1
        previous = failure
        if repeated >= 3:
            atomic_json(
                NIGHT / f"BLOCK_{phase}.json",
                dict(
                    returncode=code,
                    reason="THREE_IDENTICAL_RESOURCE_REJECTIONS",
                    kind="RESOURCE_REJECTION",
                    resources=observations,
                ),
            )
            return False
        persist(phase, "WAITING_FOR_RESOURCE_CHANGE")
        time.sleep(30)
    return seal.exists()


def close_available_at_deadline() -> None:
    """Freeze the partial inventory before evaluating only complete three-fold arms."""
    if (H16 / "ENDPOINTS.json").exists() and not (H16 / "analysis/RESULTS.json").exists():
        if run("N1_DEADLINE_ANALYSIS", command("simplex_t_h16_replication/analyze.py")) == 0:
            run("N1_DEADLINE_DELIVERY", command("simplex_t_h16_replication/deliver.py"))
    progress = NIGHT / "execution/ENDPOINT_PROGRESS.json"
    if not progress.exists():
        return
    frozen = record(progress)
    for family in ("N2", "N3"):
        if (NIGHT / f"execution/analysis/{family}/RESULTS.json").exists():
            continue
        rows = [r for r in frozen["fits"] if r["family"] == family]
        complete = {
            r["arm"] for r in rows if {v["fold"] for v in rows if v["arm"] == r["arm"]} == {0, 1, 2}
        }
        rows = [r for r in rows if r["arm"] in complete]
        if not rows:
            continue
        seal = NIGHT / f"execution/{family}_ENDPOINTS.json"
        if seal.exists():
            run(
                family + "_DEADLINE_ANALYSIS",
                command("simplex_t_cost_context/analyze.py", "--family", family),
            )
        else:
            path = NIGHT / f"execution/{family}_DEADLINE_PARTIAL_INVENTORY.json"
            publish_json(
                path,
                dict(
                    family=family,
                    protocol_sha256=frozen["protocol_sha256"],
                    fits=rows,
                    frozen_before_first_evaluation=True,
                    incomplete_family=True,
                    original_endpoint_progress_sha256=digest(progress),
                ),
            )
            run(
                family + "_DEADLINE_ANALYSIS",
                command(
                    "simplex_t_cost_context/analyze.py",
                    "--family",
                    family,
                    "--partial-inventory",
                    str(path),
                ),
            )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", choices=[CAMPAIGN], required=True)
    parser.add_argument("--adopt-h16", action="store_true")
    args = parser.parse_args()
    lock = NIGHT / "DRIVER.lock"
    if lock.exists():
        prior = record(lock)
        if (
            psutil.pid_exists(prior["pid"])
            and psutil.Process(prior["pid"]).create_time() == prior["create_time"]
        ):
            raise RuntimeError("another live nocturnal driver owns the queue")
        lock.replace(NIGHT / f"STALE_DRIVER_{digest(lock)[:12]}.json")
    atomic_json(
        lock,
        dict(pid=os.getpid(), create_time=psutil.Process().create_time(), campaign=args.campaign),
    )
    try:
        if args.adopt_h16:
            owner = H16 / "WRITER.lock"
            while owner.exists():
                prior = record(owner)
                if (
                    not psutil.pid_exists(prior["pid"])
                    or psutil.Process(prior["pid"]).create_time() != prior["create_time"]
                ):
                    break
                persist("N1_TRAIN", "EXISTING_LIVE_OWNER_PRESERVED")
                time.sleep(10)
        h16_ready = train_until_sealed(
            "N1_TRAIN", "simplex_t_h16_replication/train.py", H16 / "ENDPOINTS.json"
        )
        if h16_ready and not expired():
            if run("N1_ANALYSIS", command("simplex_t_h16_replication/analyze.py")) == 0:
                run("N1_DELIVERY", command("simplex_t_h16_replication/deliver.py"))
        # New families are independent of an unfavorable or blocked H16 outcome.
        if not expired():
            engine = ROOT / "operational/simplex_t_cost_context/engine.py"
            technical = ROOT / "operational/simplex_t_cost_context/technical.py"
            register = ROOT / "operational/simplex_t_cost_context/register.py"
            if all(p.exists() for p in (engine, technical, register)):
                qa = run(
                    "N2_QA",
                    [
                        sys.executable,
                        "-B",
                        "-m",
                        "pytest",
                        "operational/simplex_t_cost_context/test_model.py",
                        "operational/simplex_t_cost_context/test_engine.py",
                        "-q",
                        "--import-mode=importlib",
                        "--junitxml=artifacts/simplex_t/nocturnal_20261003/verification/MODEL_ENGINE_QA.xml",
                    ],
                )
                if qa != 0:
                    atomic_json(
                        NIGHT / "BLOCK_N2_QA.json",
                        dict(reason="FOCUSED_CONTRACT_TEST_FAILED", returncode=qa),
                    )
                    persist("N2_QA", "BLOCKED_REQUIRES_DIAGNOSIS")
                    return run("N5_DELIVERY", command("simplex_t_cost_context/deliver.py"))
                if run("N2_REGISTER", command("simplex_t_cost_context/register.py")) == 0:
                    if run("N2_TECHNICAL", command("simplex_t_cost_context/technical.py")) == 0:
                        for family in ("N2", "N3"):
                            ready = train_until_sealed(
                                family + "_TRAIN",
                                "simplex_t_cost_context/engine.py",
                                NIGHT / f"execution/{family}_ENDPOINTS.json",
                                "--family",
                                family,
                            )
                            if ready and not expired():
                                run(
                                    family + "_ANALYSIS",
                                    command(
                                        "simplex_t_cost_context/analyze.py", "--family", family
                                    ),
                                )
                            if expired():
                                break
            else:
                atomic_json(
                    NIGHT / "BLOCK_N2.json",
                    dict(
                        reason="IMPLEMENTATION_NOT_READY",
                        missing=[str(p) for p in (engine, technical, register) if not p.exists()],
                    ),
                )
        if not expired():
            run("N4_PROFILE", command("simplex_t_cost_context/profile.py"))
        else:
            close_available_at_deadline()
        persist("N5", "DEADLINE_REACHED" if expired() else "QUEUE_FINISHED_OR_BRANCHES_BLOCKED")
        return run("N5_DELIVERY", command("simplex_t_cost_context/deliver.py"))
    finally:
        lock.unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())

"""Durable zero-update follow-up queue after the expanded EvTTC campaign.

The queue waits for one frozen campaign PID/create-time receipt, verifies the
complete campaign result, then runs FCWD inference/scoring, the fixed cost
benchmark, and finally the already-admitted R1 resume supervisor.  GPU work is
strictly sequential.  Branch failures are preserved independently; an R1
operational failure is never represented as a scientific negative result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
SCHEMA = "sota_zero_update_followups_v1"
SOURCE_FILES = (
    "operational/sota_eval/followups.py",
    "operational/sota_eval/cost.py",
    "operational/sota_eval/fcwd_inputs.py",
    "operational/sota_eval/fcwd_run.py",
    "operational/sota_eval/fcwd_score.py",
    "operational/sota_eval/r1_resume.py",
    "operational/sota_eval/r1_resume_supervisor.py",
)
EXPECTED_R1_PID = 44552
EXPECTED_R1_CREATE_TIME = "1791450044.5194364"


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def stamp() -> str:
    return datetime.now(UTC).isoformat()


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def python_entrypoint(command: Sequence[str]) -> str | None:
    normalized = [str(item).replace("\\", "/").lower() for item in command]
    if "-m" in normalized:
        index = normalized.index("-m") + 1
        return normalized[index] if index < len(normalized) else None
    if len(normalized) > 1 and normalized[1].endswith(".py"):
        return normalized[1]
    return None


def validate_wait_receipt(value: Mapping[str, Any]) -> tuple[int, float, list[str]]:
    """Validate the exact expanded-campaign process identity without starting work."""
    command = value.get("command")
    if not isinstance(command, list) or not all(isinstance(item, str) for item in command):
        raise ValueError("wait receipt command must be a string list")
    if python_entrypoint(command) != "operational.sota_eval.campaign":
        raise ValueError("wait receipt is not the expanded campaign runner")
    if value.get("optimizer_updates") != 0:
        raise ValueError("wait receipt is not a zero-update campaign")
    pid = int(value["pid"])
    created = float(value["create_time"])
    if pid <= 0 or not created > 0:
        raise ValueError("wait receipt process identity is invalid")
    return pid, created, list(command)


def wait_for_campaign(
    receipt_path: Path,
    campaign_root: Path,
    state: Path,
    timeout_seconds: float,
    interval_seconds: float = 5.0,
) -> None:
    """Wait only for the receipt owner and descendants, then require a valid result."""
    import psutil

    receipt = read(receipt_path)
    pid, created, command = validate_wait_receipt(receipt)
    deadline = time.monotonic() + timeout_seconds
    descendants: dict[int, float] = {}
    while True:
        if (state.parent / "STOP_REQUEST").exists():
            raise InterruptedError("follow-up STOP_REQUEST while waiting for campaign")
        parent_alive = False
        try:
            parent = psutil.Process(pid)
            if abs(float(parent.create_time()) - created) > 0.001:
                raise ValueError("campaign PID was reused")
            if [str(item) for item in parent.cmdline()] != command:
                raise ValueError("live campaign command differs from wait receipt")
            parent_alive = parent.is_running()
            for child in parent.children(recursive=True):
                descendants[int(child.pid)] = float(child.create_time())
        except psutil.NoSuchProcess:
            parent_alive = False
        living = []
        for child_pid, child_created in descendants.items():
            try:
                child = psutil.Process(child_pid)
                if abs(float(child.create_time()) - child_created) <= 0.001 and child.is_running():
                    living.append(child_pid)
            except psutil.NoSuchProcess:
                continue
        if not parent_alive and not living:
            break
        if time.monotonic() >= deadline:
            raise TimeoutError("expanded campaign wait timed out")
        atomic_json(
            state,
            {
                "schema": SCHEMA,
                "status": "WAIT_CAMPAIGN",
                "campaign_pid": pid,
                "living_descendants": living,
                "checked_utc": stamp(),
                "optimizer_updates": 0,
            },
        )
        time.sleep(interval_seconds)
    verify_campaign_result(campaign_root)


def _strong_campaign_current(root: Path, configuration: Mapping[str, Any]) -> bool:
    """Reuse the campaign's current source/model/backend/scoring validators."""
    from operational.sota_eval import campaign

    baseline = Path(str(configuration["baseline"]))
    full = Path(str(configuration["full"]))
    metrics = Path(str(configuration["metrics"]))
    device = str(configuration["device"])
    event_backend = str(configuration["event_backend"])
    full_backend = str(configuration["full_backend"])
    return bool(
        campaign._sealed(baseline, "event", device, event_backend)
        and campaign._sealed(full, "full", device, full_backend)
        and campaign._scored(baseline)
        and campaign._scored(full)
        and campaign._metrics_complete(metrics, full / "SCORED_PREDICTIONS.csv")
    )


def verify_campaign_result(
    root: Path,
    strong_validator: Callable[[Path, Mapping[str, Any]], bool] = _strong_campaign_current,
) -> dict[str, Any]:
    result_path = root / "CAMPAIGN_RESULT.json"
    result = read(result_path)
    if result.get("status") != "COMPLETE" or result.get("optimizer_updates") != 0:
        raise ValueError("expanded campaign result is not complete/zero-update")
    bindings = {
        "campaign_freeze_sha256": root / "CAMPAIGN_FREEZE.json",
        "baseline_preservation_sha256": root / "dev32_expanded_rgb/BASELINE_PRESERVATION.json",
        "full_prediction_seal_sha256": root / "dev32_expanded_rgb/PREDICTIONS_SEALED.json",
        "scored_predictions_sha256": root / "dev32_expanded_rgb/SCORED_PREDICTIONS.csv",
        "metrics_sha256": root / "expanded_metrics/SHA256.json",
    }
    for key, path in bindings.items():
        if not path.is_file() or result.get(key) != digest(path):
            raise ValueError(f"expanded campaign result binding failed: {key}")
    configuration = read(root / "CAMPAIGN_FREEZE.json")
    if not strong_validator(root, configuration):
        raise ValueError("expanded campaign current source/model/backend/scoring binding failed")
    return result


def verify_fcwd_population(manifest: Path) -> dict[str, Any]:
    population = read(manifest)
    freeze_path = manifest.with_name("FCWD_INPUT_FREEZE.json")
    freeze = read(freeze_path)
    rows = population.get("rows")
    if not isinstance(rows, list) or len(rows) != 630:
        raise ValueError("FCWD frozen population is not exactly 630 queries")
    if (
        population.get("status") != "FROZEN_LABEL_FREE"
        or population.get("query_count") != 630
        or population.get("ttc_targets_read") is not False
        or freeze.get("status") != "FROZEN_LABEL_FREE"
        or freeze.get("query_count") != 630
        or freeze.get("manifest_sha256") != digest(manifest)
        or freeze.get("input_adapter_sha256")
        != digest(ROOT / "operational/sota_eval/fcwd_inputs.py")
        or freeze.get("asset_manifest_sha256")
        != digest(Path(str(population["asset_manifest_path"])))
        or freeze.get("reference_contract_sha256")
        != digest(Path(str(population["reference_contract_path"])))
        or freeze.get("ttc_targets_read") is not False
        or freeze.get("optimizer_updates") != 0
    ):
        raise ValueError("FCWD population freeze/source/dependency binding failed")
    return freeze


def verify_fcwd(output: Path, manifest: Path) -> dict[str, Any]:
    seal = read(output / "PREDICTIONS_SEALED.json")
    rows = read(manifest).get("rows")
    if not isinstance(rows, list) or len(rows) != 630:
        raise ValueError("FCWD frozen population is not exactly 630 queries")
    if (seal.get("status"), seal.get("queries"), seal.get("manifest_sha256")) != (
        "COMPLETE",
        630,
        digest(manifest),
    ):
        raise ValueError("FCWD prediction seal is incomplete or unbound")
    scoring = read(output / "SCORING_COMPLETE.json")
    if scoring.get("status") != "COMPLETE" or scoring.get("queries") != 630:
        raise ValueError("FCWD score receipt is incomplete")
    if scoring.get("prediction_seal_sha256") != digest(output / "PREDICTIONS_SEALED.json"):
        raise ValueError("FCWD scoring is not bound to the prediction seal")
    return scoring


def verify_cost(output: Path) -> dict[str, Any]:
    result = read(output / "SYSTEM_COST_SUMMARY.json")
    if result.get("status") != "COMPLETE" or result.get("query_count") != 8:
        raise ValueError("system-cost result is incomplete")
    if result.get("optimizer_updates") != 0 or result.get("targets_read") is not False:
        raise ValueError("system-cost scientific boundary changed")
    if result.get("execution_freeze_sha256") != digest(output / "EXECUTION_FREEZE.json"):
        raise ValueError("system-cost execution freeze binding failed")
    if result.get("model_freeze_sha256") != digest(output / "MODEL_FREEZE.json"):
        raise ValueError("system-cost model freeze binding failed")
    return result


def verify_r1_state(path: Path) -> dict[str, Any]:
    value = read(path)
    if value.get("status") != "COMPLETE":
        raise RuntimeError(f"R1 supervisor ended without COMPLETE: {value.get('status')}")
    return value


def source_freeze(output: Path, arguments: argparse.Namespace) -> dict[str, Any]:
    value = {
        "schema": SCHEMA,
        "status": "FROZEN",
        "sources": {relative: digest(ROOT / relative) for relative in SOURCE_FILES},
        "wait_receipt": str(arguments.wait_receipt.resolve()),
        "wait_receipt_sha256": digest(arguments.wait_receipt.resolve()),
        "campaign_root": str(arguments.campaign_root.resolve()),
        "commands_use_current_python": str(Path(sys.executable).resolve()),
        "r1_expected_owner": {"pid": EXPECTED_R1_PID, "create_time": EXPECTED_R1_CREATE_TIME},
        "optimizer_updates": 0,
    }
    path = output / "SOURCE_FREEZE.json"
    if path.exists() and read(path) != value:
        raise ValueError("follow-up source/config freeze changed")
    if not path.exists():
        atomic_json(path, value)
    return value


def _run_child(name: str, command: Sequence[str], output: Path) -> dict[str, Any]:
    """Run one child invisibly and persist identity/log/return code before verification."""
    import psutil

    log = output / "logs" / f"{name}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    record_path = output / f"{name.upper()}_EXECUTION.json"
    with log.open("ab", buffering=0) as stream:
        child = subprocess.Popen(
            list(command),
            cwd=ROOT,
            stdout=stream,
            stderr=subprocess.STDOUT,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
        record = {
            "schema": SCHEMA,
            "status": "RUNNING",
            "name": name,
            "command": list(command),
            "child": {"pid": child.pid, "create_time": psutil.Process(child.pid).create_time()},
            "log": str(log),
            "started_utc": stamp(),
            "optimizer_updates": 0,
        }
        atomic_json(record_path, record)
        return_code = child.wait()
    record.update(
        status="EXITED", returncode=return_code, ended_utc=stamp(), log_sha256=digest(log)
    )
    atomic_json(record_path, record)
    if return_code != 0:
        raise RuntimeError(f"{name} exited {return_code}")
    return record


def _branch(
    output: Path,
    name: str,
    action: Callable[[], object],
    verify: Callable[[], object],
) -> dict[str, Any]:
    try:
        if (output / "STOP_REQUEST").exists():
            result = {
                "status": "PAUSED_PRESERVED",
                "reason": "STOP_REQUEST",
                "checked_utc": stamp(),
            }
        else:
            action()
            evidence = verify()
            result = {"status": "COMPLETE", "evidence": evidence, "checked_utc": stamp()}
    except Exception as exc:
        result = {
            "status": "FAILED_PRESERVED",
            "error_type": type(exc).__name__,
            "reason": str(exc),
            "scientific_negative": False,
            "checked_utc": stamp(),
        }
    atomic_json(output / f"{name.upper()}_BRANCH.json", result)
    return result


def run(arguments: argparse.Namespace) -> dict[str, Any]:
    output = arguments.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    source_freeze(output, arguments)
    state = output / "STATE.json"
    wait_for_campaign(
        arguments.wait_receipt.resolve(),
        arguments.campaign_root.resolve(),
        state,
        arguments.wait_timeout_seconds,
    )
    campaign_result = verify_campaign_result(arguments.campaign_root.resolve())
    py = str(Path(sys.executable).resolve())
    fcwd_manifest = arguments.fcwd_manifest.resolve()
    fcwd_output = arguments.fcwd_output.resolve()

    def fcwd_action() -> None:
        verify_fcwd_population(fcwd_manifest)
        if not (fcwd_output / "PREDICTIONS_SEALED.json").exists():
            from operational.sota_eval.cost import exclusive_gpu_preflight

            exclusive_gpu_preflight()
            _run_child(
                "fcwd_inference",
                [
                    py,
                    "-m",
                    "operational.sota_eval.fcwd_run",
                    "--manifest",
                    str(fcwd_manifest),
                    "--output",
                    str(fcwd_output),
                    "--campaign",
                    str(arguments.train40_campaign.resolve()),
                    "--public-full-dir",
                    str(arguments.public_full_dir.resolve()),
                    "--code-root",
                    str(arguments.code_root.resolve()),
                    "--device",
                    "cuda",
                ],
                output,
            )
        if not (fcwd_output / "SCORING_COMPLETE.json").exists():
            _run_child(
                "fcwd_score",
                [
                    py,
                    "-m",
                    "operational.sota_eval.fcwd_run",
                    "--score",
                    "--manifest",
                    str(fcwd_manifest),
                    "--output",
                    str(fcwd_output),
                    "--asset-manifest",
                    str(arguments.fcwd_asset_manifest.resolve()),
                ],
                output,
            )

    fcwd = _branch(output, "fcwd", fcwd_action, lambda: verify_fcwd(fcwd_output, fcwd_manifest))

    def cost_action() -> None:
        if not (arguments.cost_output / "SYSTEM_COST_SUMMARY.json").exists():
            _run_child(
                "cost",
                [
                    py,
                    "-m",
                    "operational.sota_eval.cost",
                    "--manifest",
                    str(arguments.expanded_manifest.resolve()),
                    "--campaign",
                    str(arguments.train40_campaign.resolve()),
                    "--public-full-dir",
                    str(arguments.public_full_dir.resolve()),
                    "--code-root",
                    str(arguments.code_root.resolve()),
                    "--output",
                    str(arguments.cost_output.resolve()),
                ],
                output,
            )

    cost = _branch(
        output, "cost", cost_action, lambda: verify_cost(arguments.cost_output.resolve())
    )

    r1_command = [
        py,
        "-m",
        "operational.sota_eval.r1_resume_supervisor",
        "--run",
        "--handoff",
        "--expected-owner-pid",
        str(EXPECTED_R1_PID),
        "--expected-owner-create-time",
        EXPECTED_R1_CREATE_TIME,
    ]
    r1 = _branch(
        output,
        "r1",
        lambda: _run_child("r1", r1_command, output),
        lambda: verify_r1_state(arguments.r1_state.resolve()),
    )
    final = {
        "schema": SCHEMA,
        "status": "COMPLETE_WITH_BRANCH_STATUS",
        "campaign_result_sha256": digest(
            arguments.campaign_root.resolve() / "CAMPAIGN_RESULT.json"
        ),
        "campaign": campaign_result,
        "branches": {"fcwd": fcwd, "cost": cost, "r1": r1},
        "r1_failure_is_scientific_negative": False,
        "optimizer_updates": 0,
        "completed_utc": stamp(),
    }
    atomic_json(output / "FOLLOWUPS_RESULT.json", final)
    atomic_json(state, final)
    return final


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    base = Path("artifacts/sota_campaign_20261008")
    value.add_argument("--wait-receipt", type=Path, required=True)
    value.add_argument("--output", type=Path, default=base / "followups")
    value.add_argument("--campaign-root", type=Path, default=base)
    value.add_argument(
        "--expanded-manifest", type=Path, default=base / "dev32_expanded/QUERY_MANIFEST.json"
    )
    value.add_argument(
        "--train40-campaign", type=Path, default=Path("artifacts/train40_system_20261005")
    )
    value.add_argument(
        "--public-full-dir",
        type=Path,
        default=Path("artifacts/evttc_rgb_transfer_20261008/public_garl"),
    )
    value.add_argument("--code-root", type=Path, default=Path("E:/Garl-TTC"))
    value.add_argument(
        "--fcwd-asset-manifest", type=Path, default=base / "fcwd/ASSET_MANIFEST.json"
    )
    value.add_argument(
        "--fcwd-manifest", type=Path, default=base / "fcwd_population/QUERY_MANIFEST.json"
    )
    value.add_argument("--fcwd-output", type=Path, default=base / "fcwd_inference")
    value.add_argument("--cost-output", type=Path, default=base / "cost")
    value.add_argument("--r1-state", type=Path, default=base / "r1_resume/supervisor/STATE.json")
    value.add_argument("--wait-timeout-seconds", type=float, default=6 * 60 * 60)
    return value


def main() -> int:
    arguments = parser().parse_args()
    try:
        result = run(arguments)
    except InterruptedError as exc:
        atomic_json(
            arguments.output.resolve() / "STATE.json",
            {
                "schema": SCHEMA,
                "status": "PAUSED_PRESERVED",
                "reason": str(exc),
                "checked_utc": stamp(),
                "optimizer_updates": 0,
            },
        )
        return 2
    except Exception as exc:
        atomic_json(
            arguments.output.resolve() / "STATE.json",
            {
                "schema": SCHEMA,
                "status": "FAILED_PRESERVED",
                "error_type": type(exc).__name__,
                "reason": str(exc),
                "checked_utc": stamp(),
                "optimizer_updates": 0,
            },
        )
        raise
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

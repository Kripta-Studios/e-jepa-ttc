"""Durable zero-update follow-up queue after the expanded EvTTC campaign.

The queue waits for one frozen campaign PID/create-time receipt, verifies the
complete campaign result, then runs FCWD inference/scoring, the fixed cost
benchmark, and finally the already-admitted R1 resume supervisor.  GPU work is
strictly sequential.  Branch failures are preserved independently; an R1
operational failure is never represented as a scientific negative result.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
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


def _verify_full_model_binding(model: Mapping[str, Any]) -> None:
    checkpoint, configuration = Path(str(model["checkpoint"])), Path(str(model["config"]))
    if (
        checkpoint.stat().st_size != model["checkpoint_bytes"]
        or digest(checkpoint) != model["checkpoint_sha256"]
        or digest(configuration) != model["config_sha256"]
        or digest(ROOT / "operational/evttc_rgb_transfer/model.py") != model["module_sha256"]
    ):
        raise ValueError("full Garl model/checkpoint/config binding changed")
    native = Path(str(model["native_code_root"]))
    for relative, expected in model["native_source_sha256"].items():
        if digest(native / relative) != expected:
            raise ValueError(f"full Garl native source changed: {relative}")


def verify_fcwd_inference(
    output: Path, manifest: Path, campaign_root: Path, public_full: Path, code_root: Path
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Verify current source/model/input lineage and every ordered FCWD fragment."""
    from operational.sota_eval import fcwd_run
    from operational.sota_eval.reuse import _verify_model_binding

    expected_source = fcwd_run.source_binding(
        manifest, campaign_root, public_full, code_root, "cuda"
    )
    if read(output / "SOURCE_FREEZE.json") != expected_source:
        raise ValueError("FCWD current source/campaign receipt freeze changed")
    inference = read(output / "INFERENCE_FREEZE.json")
    if (
        inference.get("source_freeze_sha256") != digest(output / "SOURCE_FREEZE.json")
        or inference.get("manifest_sha256") != digest(manifest)
        or inference.get("methods") != list(fcwd_run.METHODS)
        or inference.get("device") != "cuda"
        or inference.get("precision") != "float32_no_tf32"
        or inference.get("targets_read") is not False
        or inference.get("optimizer_updates") != 0
    ):
        raise ValueError("FCWD inference model/source/input freeze changed")
    _verify_model_binding(inference["own_models"])
    _verify_full_model_binding(inference["full_model"])
    manifest_value, predictions = fcwd_run.verify_seal(output, manifest)
    if [row["query_id"] for row in manifest_value["rows"]] != [
        prediction["query_id"] for prediction in predictions
    ]:
        raise ValueError("FCWD prediction order differs from frozen manifest")
    return manifest_value, predictions


def verify_fcwd(
    output: Path,
    manifest: Path,
    campaign_root: Path,
    public_full: Path,
    code_root: Path,
    inference_validator: Callable[
        [Path, Path, Path, Path, Path], tuple[dict[str, Any], list[dict[str, Any]]]
    ] = verify_fcwd_inference,
) -> dict[str, Any]:
    manifest_value, predictions = inference_validator(
        output, manifest, campaign_root, public_full, code_root
    )
    from operational.sota_eval import fcwd_run, scoring

    score_dir = output / "scoring"
    scoring_receipt = read(output / "SCORING_COMPLETE.json")
    coverage = read(score_dir / "MODEL_COVERAGE.json")
    target_join = read(score_dir / "TARGET_JOIN_CONTRACT.json")
    checksums = read(score_dir / "SHA256.json")
    expected_inventory = {
        "REPORT.json",
        "METHOD_METRICS.csv",
        "PER_SEQUENCE.csv",
        "PAIRED.csv",
        "SCORED_ROWS.csv",
        "METADATA.json",
    }
    if set(checksums) != expected_inventory or any(
        digest(score_dir / name) != expected for name, expected in checksums.items()
    ):
        raise ValueError("FCWD scorer SHA inventory changed or is incomplete")
    expected_reason = "MISSING_AUDITED_EVENT_TO_RGB_SPATIAL_MAPPING"
    expected_dependency = "Audited event-right to RGB-right spatial mapping without GT depth"
    gt_files = target_join.get("gt_files", {})
    gt_current = (
        isinstance(gt_files, dict)
        and set(gt_files) == {"FCWD1", "FCWD2", "FCWD3"}
        and all(
            digest(Path(str(receipt["path"]))) == receipt["sha256"]
            and Path(str(receipt["path"])).stat().st_size == receipt["bytes"]
            for receipt in gt_files.values()
        )
    )
    if (
        scoring_receipt.get("status") != "COMPLETE"
        or scoring_receipt.get("queries") != 630
        or scoring_receipt.get("methods") != list(fcwd_run.METHODS[:4])
        or scoring_receipt.get("planned_methods") != list(fcwd_run.METHODS)
        or scoring_receipt.get("optimizer_updates") != 0
        or scoring_receipt.get("prediction_seal_sha256")
        != digest(output / "PREDICTIONS_SEALED.json")
        or scoring_receipt.get("scoring_manifest_sha256") != digest(score_dir / "SHA256.json")
        or scoring_receipt.get("target_join_contract_sha256")
        != digest(score_dir / "TARGET_JOIN_CONTRACT.json")
        or coverage.get("planned_methods") != list(fcwd_run.METHODS)
        or coverage.get("scored_methods") != list(fcwd_run.METHODS[:4])
        or coverage.get("all_queries_retained") != 630
        or coverage.get("head_selection_performed") is not False
        or coverage.get("full_model", {}).get("status") != "DEPENDENCY_UNAVAILABLE"
        or coverage.get("full_model", {}).get("queries_with_missing_spatial_mapping") != 630
        or coverage.get("full_model", {}).get("not_a_negative_model_result") is not True
        or coverage.get("full_model", {}).get("dependency") != expected_dependency
        or any(
            prediction.get("unavailable_reasons", {}).get(fcwd_run.METHODS[4]) != expected_reason
            for prediction in predictions
        )
        or target_join.get("status") != "COMPLETE"
        or target_join.get("rows_retained") != 630
        or target_join.get("prediction_seal_sha256") != digest(output / "PREDICTIONS_SEALED.json")
        or target_join.get("source_freeze_sha256") != digest(output / "SOURCE_FREEZE.json")
        or target_join.get("asset_manifest_sha256")
        != digest(Path(str(manifest_value["asset_manifest_path"])))
        or target_join.get("reference_contract_sha256")
        != digest(Path(str(manifest_value["reference_contract_path"])))
        or target_join.get("model_coverage_sha256") != digest(score_dir / "MODEL_COVERAGE.json")
        or target_join.get("joined_predictions_sha256")
        != digest(score_dir / "SCORED_PREDICTIONS.csv")
        or target_join.get("scoring_manifest_sha256") != digest(score_dir / "SHA256.json")
        or target_join.get("scorer_module_sha256") != digest(Path(scoring.__file__))
        or target_join.get("query_manifest_sha256") != digest(manifest)
        or target_join.get("population_selection_used_gt") is not False
        or target_join.get("labels_opened_only_after_verified_prediction_seal") is not True
        or target_join.get("optimizer_updates") != 0
        or target_join.get("checkpoint_training_exclusion_certified") is not False
        or target_join.get("model_predictions_rerun") is not False
        or not gt_current
        or len(manifest_value["rows"]) != 630
    ):
        raise ValueError("FCWD scoring/coverage/target-join contract changed")
    return scoring_receipt


def verify_cost_fragments(
    output: Path, selected: Sequence[Mapping[str, Any]], execution_sha: str, model_sha: str
) -> list[dict[str, str]]:
    """Verify every cost fragment/raw row, fixed identity, order, and sample count."""
    from operational.sota_eval import cost

    expected_counts = {
        "cpu_prepare": (cost.CPU_WARMUPS + cost.CPU_MEASUREMENTS, cost.CPU_WARMUPS),
        "gpu_inference": (cost.GPU_WARMUPS + cost.GPU_MEASUREMENTS, cost.GPU_WARMUPS),
        "sequential_end_to_end": (cost.E2E_MEASUREMENTS, 0),
    }
    all_rows: list[dict[str, str]] = []
    for selected_row in selected:
        query_id = str(selected_row["query_id"])
        stem = hashlib.sha256(query_id.encode("utf-8")).hexdigest()[:16]
        fragment_path, raw_path = output / f"FRAGMENT_{stem}.json", output / f"RAW_{stem}.csv"
        fragment = read(fragment_path)
        cost.validate_resume(fragment, raw_path, execution_sha, model_sha, query_id)
        if (
            fragment.get("sequence_id") != selected_row["sequence_id"]
            or fragment.get("scenario_family") != selected_row["scenario_family"]
            or set(fragment.get("audit", {}).get("prediction_sha256", {})) != set(cost.SYSTEMS)
            or fragment.get("targets_read") is not False
            or fragment.get("optimizer_updates") != 0
            or fragment.get("raw_csv") != raw_path.name
        ):
            raise ValueError("cost fragment fixed identity/scientific boundary changed")
        with raw_path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames != list(cost.CSV_FIELDS):
                raise ValueError("cost raw CSV schema/order changed")
            rows = list(reader)
        for row in rows:
            if (
                row["query_id"] != query_id
                or row["sequence_id"] != selected_row["sequence_id"]
                or row["scenario_family"] != selected_row["scenario_family"]
                or row["status"] != "OK"
                or not math.isfinite(float(row["milliseconds"]))
                or float(row["milliseconds"]) < 0.0
            ):
                raise ValueError("cost raw row identity/status changed")
        for system in cost.SYSTEMS:
            for stage, (count, warmups) in expected_counts.items():
                group = [row for row in rows if row["system"] == system and row["stage"] == stage]
                if (
                    len(group) != count
                    or [int(row["iteration"]) for row in group] != list(range(count))
                    or sum(row["warmup"] == "true" for row in group) != warmups
                    or any(row["warmup"] not in {"true", "false"} for row in group)
                ):
                    raise ValueError("cost raw warmup/measurement inventory changed")
        if len(rows) != sum(count for count, _ in expected_counts.values()) * len(cost.SYSTEMS):
            raise ValueError("cost raw rows contain an undeclared system/stage")
        all_rows.extend(rows)
    return all_rows


def verify_cost(output: Path) -> dict[str, Any]:
    from importlib import metadata

    from operational.sota_eval import cost
    from operational.sota_eval.reuse import _verify_model_binding

    result = read(output / "SYSTEM_COST_SUMMARY.json")
    if result.get("status") != "COMPLETE" or result.get("query_count") != 8:
        raise ValueError("system-cost result is incomplete")
    if result.get("optimizer_updates") != 0 or result.get("targets_read") is not False:
        raise ValueError("system-cost scientific boundary changed")
    if result.get("execution_freeze_sha256") != digest(output / "EXECUTION_FREEZE.json"):
        raise ValueError("system-cost execution freeze binding failed")
    if result.get("model_freeze_sha256") != digest(output / "MODEL_FREEZE.json"):
        raise ValueError("system-cost model freeze binding failed")
    execution = read(output / "EXECUTION_FREEZE.json")
    model = read(output / "MODEL_FREEZE.json")
    manifest_path = Path(str(execution["manifest"]))
    manifest = read(manifest_path)
    selected = cost.select_fixed_queries(manifest)
    declared = [
        {
            key: row[key]
            for key in (
                "query_id",
                "sequence_id",
                "scenario_family",
                "anchor_us",
                "metadata_sha256",
            )
        }
        for row in selected
    ]
    if (
        execution.get("status") != "EXECUTION_FROZEN"
        or execution.get("source_sha256") != digest(Path(cost.__file__))
        or execution.get("manifest_sha256") != digest(manifest_path)
        or execution.get("selected_queries") != declared
        or execution.get("targets_read") is not False
        or execution.get("optimizer_updates") != 0
    ):
        raise ValueError("system-cost execution/input freeze changed")
    cost.validate_source_closure(cost.ROOT, execution["source_closure"])
    if (
        model.get("status") != "MODELS_FROZEN_BEFORE_MEASUREMENT"
        or model.get("execution_freeze_sha256") != digest(output / "EXECUTION_FREEZE.json")
        or model.get("targets_read") is not False
        or model.get("optimizer_updates") != 0
        or set(model.get("models", {})) != set(cost.SYSTEMS)
        or model["models"][cost.SYSTEMS[0]] != model["models"][cost.SYSTEMS[1]]
    ):
        raise ValueError("system-cost model freeze changed")
    _verify_model_binding(model["models"][cost.SYSTEMS[0]])
    _verify_full_model_binding(model["models"][cost.SYSTEMS[2]])
    runtime = model["runtime_identity"]
    current_nvidia = cost._nvidia(
        (
            "--query-gpu=name,driver_version,memory.total,compute_cap",
            "--format=csv,noheader,nounits",
        )
    )
    if (
        runtime.get("python_version") != sys.version
        or runtime.get("packages") != cost.python_cpu_identity()["packages"]
        or runtime.get("platform") != cost.python_cpu_identity()["platform"]
        or str(runtime.get("torch_version", "")).split("+")[0]
        != metadata.version("torch").split("+")[0]
        or runtime.get("nvidia_smi") != current_nvidia
        or result.get("preflight", {}).get("hardware_identity") != runtime.get("nvidia_smi")
        or result.get("targets_read") is not False
        or set(result.get("metrics", {})) != set(cost.SYSTEMS)
    ):
        raise ValueError("system-cost runtime/hardware/summary identity changed")
    raw_rows = verify_cost_fragments(
        output,
        selected,
        digest(output / "EXECUTION_FREEZE.json"),
        digest(output / "MODEL_FREEZE.json"),
    )
    if result.get("metrics") != cost._aggregate(raw_rows):
        raise ValueError("system-cost summary metrics differ from sealed raw rows")
    expected_measured = {"cpu_prepare": 24, "gpu_inference": 160, "sequential_end_to_end": 24}
    for system in cost.SYSTEMS:
        metrics = result["metrics"][system]
        if any(metrics[stage].get("count") != count for stage, count in expected_measured.items()):
            raise ValueError("system-cost summary measured counts changed")
        if set(metrics.get("by_sequence", {})) != {row["sequence_id"] for row in selected}:
            raise ValueError("system-cost per-sequence inventory changed")
        if any(item.get("count") != 3 for item in metrics["by_sequence"].values()):
            raise ValueError("system-cost per-sequence E2E counts changed")
    if any(
        float(result[key]) < 0
        for key in (
            "peak_process_rss_bytes_observed_50ms",
            "peak_cuda_allocated_bytes",
            "peak_cuda_reserved_bytes",
        )
    ) or any(
        not math.isfinite(float(result["setup_seconds"][system]))
        or float(result["setup_seconds"][system]) < 0
        for system in cost.SYSTEMS
    ):
        raise ValueError("system-cost setup/resource measurements are invalid")
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
        else:
            verify_fcwd_inference(
                fcwd_output,
                fcwd_manifest,
                arguments.train40_campaign.resolve(),
                arguments.public_full_dir.resolve(),
                arguments.code_root.resolve(),
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
        else:
            verify_fcwd(
                fcwd_output,
                fcwd_manifest,
                arguments.train40_campaign.resolve(),
                arguments.public_full_dir.resolve(),
                arguments.code_root.resolve(),
            )

    fcwd = _branch(
        output,
        "fcwd",
        fcwd_action,
        lambda: verify_fcwd(
            fcwd_output,
            fcwd_manifest,
            arguments.train40_campaign.resolve(),
            arguments.public_full_dir.resolve(),
            arguments.code_root.resolve(),
        ),
    )

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
        else:
            verify_cost(arguments.cost_output.resolve())

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

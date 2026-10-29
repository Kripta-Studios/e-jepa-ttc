"""Foreground, fail-closed R0 -> R1 local detection campaign queue.

This command is deliberately inert until called with a hash-pinned plan.  It
owns at most one child process, never retries, and never selects a recipe from
an AP value.  R1 starts cold only after the fixed R0 completion evidence passes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import psutil

SCRIPT_PATH = Path(__file__).resolve()
FROZEN_SIBLING_PRODUCER = SCRIPT_PATH.parents[2] / "producer"
PACKAGE_ROOT = (
    FROZEN_SIBLING_PRODUCER
    if (FROZEN_SIBLING_PRODUCER / "src/e_jepa_ttc/detection/campaign_training.py").is_file()
    else SCRIPT_PATH.parents[1]
)
try:
    ROOT = next(parent for parent in SCRIPT_PATH.parents if (parent / "pyproject.toml").is_file())
except StopIteration as error:
    raise RuntimeError("Could not locate the campaign workspace root") from error
if str(PACKAGE_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT / "src"))

from e_jepa_ttc.detection.campaign_training import (  # noqa: E402
    CampaignConfig,
    load_campaign_config,
)

SCHEMA = "phase_ttc_detection_campaign_queue_plan_v1"
RECEIPT_SCHEMA = "phase_ttc_detection_campaign_queue_receipt_v1"
RUN_SCHEMA = "eap_detection_campaign_run_v1"
EVALUATION_SCHEMA = "phase_ttc_detection_evaluation_result_v2"
MAXIMUM_DEADLINE_SECONDS = 24 * 60 * 60
MINIMUM_RESOURCE_GB = 8.0
ALLOWED_R1_DIFFERENCES = {
    "experiment_id",
    "optimization.scheduler",
    "optimization.warmup_fraction",
    "optimization.minimum_lr_ratio",
}


def sha256_file(path: Path) -> str:
    """Hash one file with bounded memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _resolve(value: object, name: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a nonempty path string")
    path = Path(value)
    return (ROOT / path).resolve() if not path.is_absolute() else path.resolve()


def _hex(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


def _binding(value: object, name: str) -> tuple[Path, str]:
    if not isinstance(value, Mapping) or set(value) != {"path", "sha256"}:
        raise ValueError(f"{name} must contain exactly path and sha256")
    return _resolve(value["path"], f"{name}.path"), _hex(value["sha256"], f"{name}.sha256")


def _verify_binding(value: object, name: str) -> dict[str, str]:
    path, expected = _binding(value, name)
    if not path.is_file():
        raise ValueError(f"Pinned file is unavailable: {name}: {path}")
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(f"Pinned file changed: {name}: {path}")
    return {"path": str(path), "sha256": actual}


def _require_keys(value: Mapping[str, object], expected: set[str], name: str) -> None:
    if set(value) != expected:
        raise ValueError(f"{name} fields differ: {sorted(set(value) ^ expected)}")


def load_plan(path: Path, expected_sha256: str) -> dict[str, Any]:
    """Load an exact plan and verify every immutable producer/config/source/data binding."""
    expected_sha256 = _hex(expected_sha256, "plan_sha256")
    if sha256_file(path) != expected_sha256:
        raise ValueError("Queue plan bytes differ from --plan-sha256")
    plan = _load_object(path)
    _require_keys(
        plan,
        {
            "schema",
            "python",
            "producer",
            "gpu",
            "r0",
            "r1",
            "source_bindings",
            "data_bindings",
        },
        "plan",
    )
    if plan["schema"] != SCHEMA:
        raise ValueError("Unsupported queue plan schema")
    if not isinstance(plan["gpu"], str) or not plan["gpu"]:
        raise ValueError("gpu must name exactly one CUDA_VISIBLE_DEVICES entry")
    if "," in plan["gpu"]:
        raise ValueError("The queue permits exactly one GPU")
    for role in ("r0", "r1"):
        raw = plan[role]
        if not isinstance(raw, Mapping):
            raise ValueError(f"{role} must be an object")
        expected = {"config", "output_1024", "output_8192"} if role == "r0" else {
            "config",
            "output",
        }
        _require_keys(raw, expected, role)
        _verify_binding(raw["config"], f"{role}.config")
        for key in expected - {"config"}:
            _resolve(raw[key], f"{role}.{key}")
    for group in ("source_bindings", "data_bindings"):
        values = plan[group]
        if not isinstance(values, list) or not values:
            raise ValueError(f"{group} must be a nonempty list")
        for index, item in enumerate(values):
            _verify_binding(item, f"{group}[{index}]")
    _verify_binding(plan["python"], "python")
    _verify_binding(plan["producer"], "producer")
    return plan


def _config(plan: Mapping[str, Any], role: str) -> tuple[Path, CampaignConfig]:
    path, _ = _binding(plan[role]["config"], f"{role}.config")
    config = load_campaign_config(path)
    if config.optimization.updates != 8192:
        raise ValueError(f"{role} must have exactly 8192 updates")
    if (
        config.resources.maximum_seconds <= 0
        or config.resources.maximum_seconds > MAXIMUM_DEADLINE_SECONDS
    ):
        raise ValueError(f"{role} maximum_seconds must be within (0,24h]")
    if (
        config.resources.minimum_available_ram_gb < MINIMUM_RESOURCE_GB
        or config.resources.minimum_available_vram_gb < MINIMUM_RESOURCE_GB
    ):
        raise ValueError(f"{role} must retain the frozen 8 GB RAM/VRAM gates")
    if tuple(config.evaluation.full_selection_updates) != (4096, 8192):
        raise ValueError(f"{role} must evaluate full selection at 4096 and 8192")
    if tuple(config.evaluation.updates) != tuple(range(1024, 8193, 1024)):
        raise ValueError(f"{role} selection update schedule changed")
    return path, config


def verify_disjoint_outputs(paths: Sequence[Path]) -> None:
    """Reject aliases, workspace-root targets and parent/child output containment."""
    resolved = [path.resolve() for path in paths]
    if len(set(resolved)) != len(resolved):
        raise ValueError("Queue and child outputs must be distinct")
    if any(path == ROOT or path == ROOT.parent for path in resolved):
        raise ValueError("A workspace or workspace parent cannot be an output")
    for index, left in enumerate(resolved):
        for right in resolved[index + 1 :]:
            if left.is_relative_to(right) or right.is_relative_to(left):
                raise ValueError("Queue and child outputs cannot contain one another")


def _flatten(value: object, prefix: str = "") -> dict[str, object]:
    if not isinstance(value, Mapping):
        return {prefix: value}
    result = {}
    for key, item in value.items():
        name = f"{prefix}.{key}" if prefix else str(key)
        result.update(_flatten(item, name))
    return result


def verify_recipe_pair(r0: CampaignConfig, r1: CampaignConfig) -> None:
    """Require the preregistered scheduler-only R1 intervention, with no hidden changes."""
    left, right = _flatten(asdict(r0)), _flatten(asdict(r1))
    differences = {key for key in set(left) | set(right) if left.get(key) != right.get(key)}
    if differences != ALLOWED_R1_DIFFERENCES:
        raise ValueError(f"R1 recipe differences changed: {sorted(differences)}")
    expected = {
        "optimization.scheduler": ("constant", "warmup_cosine"),
        "optimization.warmup_fraction": (0.0, 0.05),
        "optimization.minimum_lr_ratio": (1.0, 0.05),
    }
    if any((left[name], right[name]) != pair for name, pair in expected.items()):
        raise ValueError("R0/R1 scheduler intervention differs from preregistration")
    if r0.experiment_id == r1.experiment_id:
        raise ValueError("R0 and R1 experiment IDs must differ")


def verify_all_bindings(
    plan: Mapping[str, Any], plan_path: Path, plan_sha256: str
) -> dict[str, Any]:
    """Recheck every byte binding immediately before and after each child."""
    if sha256_file(plan_path) != plan_sha256:
        raise ValueError("Queue plan changed during execution")
    receipt = {
        "plan": {"path": str(plan_path), "sha256": plan_sha256},
        "python": _verify_binding(plan["python"], "python"),
        "producer": _verify_binding(plan["producer"], "producer"),
        "configs": {
            role: _verify_binding(plan[role]["config"], f"{role}.config")
            for role in ("r0", "r1")
        },
    }
    for group in ("source_bindings", "data_bindings"):
        receipt[group] = [
            _verify_binding(item, f"{group}[{index}]")
            for index, item in enumerate(plan[group])
        ]
    return receipt


def _finite_numbers(value: object, name: str) -> None:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return
    if isinstance(value, (int, float)):
        if not math.isfinite(float(value)):
            raise ValueError(f"Nonfinite numeric field: {name}")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            _finite_numbers(item, f"{name}.{key}")
        return
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for index, item in enumerate(value):
            _finite_numbers(item, f"{name}[{index}]")


def _resolved_report_path(value: object, name: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} is absent")
    path = Path(value)
    return (ROOT / path).resolve() if not path.is_absolute() else path.resolve()


def verify_evaluation(
    path: Path, *, protocol_sha256: str, expected_sample_count: int
) -> dict[str, object]:
    """Validate one complete v2 full-selection artifact without interpreting AP magnitude."""
    payload = _load_object(path)
    required = {"metrics_v2", "predictions", "ground_truth_v2", "decode_counts"}
    if not required <= set(payload):
        raise ValueError(f"Evaluation payload fields are incomplete: {path}")
    metrics = payload["metrics_v2"]
    predictions = payload["predictions"]
    truth = payload["ground_truth_v2"]
    if not all(isinstance(item, Mapping) for item in (metrics, predictions, truth)):
        raise ValueError(f"Evaluation payload types are invalid: {path}")
    if (
        metrics.get("schema") != EVALUATION_SCHEMA
        or metrics.get("protocol_sha256") != protocol_sha256
        or metrics.get("sample_count") != expected_sample_count
        or set(predictions) != set(truth)
        or len(truth) != expected_sample_count
    ):
        raise ValueError(f"Evaluation schema/population/protocol differs: {path}")
    population = metrics.get("population")
    hashes = (
        population.get("ground_truth_sha256_by_token")
        if isinstance(population, Mapping)
        else None
    )
    if (
        not isinstance(hashes, Mapping)
        or set(hashes) != set(truth)
        or any(
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
            for value in hashes.values()
        )
    ):
        raise ValueError(f"Evaluation ground-truth identity map differs: {path}")
    if any(
        not isinstance(frame, Mapping) or frame.get("protocol_sha256") != protocol_sha256
        for frame in truth.values()
    ):
        raise ValueError(f"Evaluation GT protocol differs: {path}")
    _finite_numbers(payload, path.name)
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "sample_count": expected_sample_count,
        "protocol_sha256": protocol_sha256,
        "ground_truth_sha256_by_token": dict(hashes),
    }


def verify_summary(
    output: Path,
    config: CampaignConfig,
    *,
    expected_status: str,
    expected_final_update: int,
    expected_start_update: int,
    resume_sha256: str | None,
) -> dict[str, Any]:
    """Verify one completed segment and its selected/last checkpoint lineage."""
    summary_path = output / "summary.json"
    summary = _load_object(summary_path)
    if (
        summary.get("schema") != RUN_SCHEMA
        or summary.get("status") != expected_status
        or summary.get("final_update") != expected_final_update
        or summary.get("start_update") != expected_start_update
        or summary.get("config_sha256") != config.sha256
    ):
        raise ValueError(f"Campaign summary state differs: {summary_path}")
    resource = summary.get("resource_admission")
    if (
        not isinstance(resource, Mapping)
        or resource.get("status") != "PASS"
        or isinstance(resource.get("available_ram_bytes"), bool)
        or isinstance(resource.get("available_vram_bytes"), bool)
        or not isinstance(resource.get("available_ram_bytes"), int)
        or not isinstance(resource.get("available_vram_bytes"), int)
        or resource["available_ram_bytes"] < int(MINIMUM_RESOURCE_GB * 2**30)
        or resource["available_vram_bytes"] < int(MINIMUM_RESOURCE_GB * 2**30)
    ):
        raise ValueError(f"Frozen runner resource admission is absent/invalid: {summary_path}")
    metric = summary.get("best_selection_metric")
    if (
        isinstance(metric, bool)
        or not isinstance(metric, (int, float))
        or not math.isfinite(metric)
    ):
        raise ValueError(f"Campaign selected metric is absent/nonfinite: {summary_path}")
    _finite_numbers(summary, "summary")
    checkpoint = output / "checkpoints/last.pt"
    checkpoint_sha256 = sha256_file(checkpoint)
    if (
        summary.get("last_checkpoint_sha256") != checkpoint_sha256
        or _resolved_report_path(summary.get("checkpoint_path"), "checkpoint_path") != checkpoint
        or summary.get("resumed_from_sha256") != resume_sha256
    ):
        raise ValueError(f"Last/resume checkpoint identity differs: {summary_path}")
    selected = summary.get("selected_checkpoint")
    if not isinstance(selected, Mapping) or selected.get("status") != "AVAILABLE":
        raise ValueError(f"Selected checkpoint is unavailable: {summary_path}")
    selected_path = _resolved_report_path(selected.get("path"), "selected_checkpoint.path")
    if (
        selected_path != output / "checkpoints/selected.pt"
        or selected.get("sha256") != sha256_file(selected_path)
        or selected.get("best_metric") != metric
    ):
        raise ValueError(f"Selected checkpoint identity differs: {summary_path}")
    if resume_sha256 is None:
        if summary.get("parent_run") is not None or summary.get("parent_run_chain") not in (
            [],
            None,
        ):
            raise ValueError(f"Cold run unexpectedly has parent lineage: {summary_path}")
    else:
        parent = summary.get("parent_run")
        if (
            not isinstance(parent, Mapping)
            or parent.get("resume_checkpoint_sha256") != resume_sha256
        ):
            raise ValueError(f"Resume parent lineage differs: {summary_path}")
    return {
        "summary": summary,
        "summary_path": str(summary_path),
        "summary_sha256": sha256_file(summary_path),
        "last_checkpoint": str(checkpoint),
        "last_checkpoint_sha256": checkpoint_sha256,
        "selected_checkpoint_sha256": selected["sha256"],
    }


def verify_full_evaluations(
    output: Path, summary: Mapping[str, Any]
) -> dict[str, dict[str, object]]:
    protocol = summary.get("evaluation_protocol_sha256")
    count = summary.get("full_selection_count")
    if (
        not isinstance(protocol, str)
        or len(protocol) != 64
        or not isinstance(count, int)
        or count < 1
    ):
        raise ValueError("Summary lacks protocol/full-selection population identity")
    return {
        str(update): verify_evaluation(
            output / f"evaluation/full_selection_update_{update:06d}.json",
            protocol_sha256=protocol,
            expected_sample_count=count,
        )
        for update in (4096, 8192)
    }


def verify_r0_to_r1_gate(r0: Mapping[str, Any], r1_config: CampaignConfig) -> None:
    summary = r0["summary"]
    if summary.get("status") != "COMPLETED" or summary.get("final_update") != 8192:
        raise ValueError("R0 has not completed the fixed 8192-update budget")
    if summary.get("evaluation_protocol_sha256") is None:
        raise ValueError("R0 protocol identity is absent")
    if summary.get("dataset_manifest_hash") != r1_config.data.expected_manifest_sha256:
        raise ValueError("R0/R1 manifest identity differs")
    if summary.get("split_hash") != r1_config.data.expected_split_sha256:
        raise ValueError("R0/R1 split identity differs")
    if summary.get("full_selection_sha256") != r1_config.data.expected_full_selection_sha256:
        raise ValueError("R0/R1 full-selection identity differs")


def verify_population_same(
    r0_evaluations: Mapping[str, Mapping[str, object]],
    r1_evaluations: Mapping[str, Mapping[str, object]],
) -> None:
    """Require identical detector-independent GT identities at both fixed updates."""
    for update in ("4096", "8192"):
        if (
            r0_evaluations[update]["protocol_sha256"]
            != r1_evaluations[update]["protocol_sha256"]
            or r0_evaluations[update]["ground_truth_sha256_by_token"]
            != r1_evaluations[update]["ground_truth_sha256_by_token"]
        ):
            raise ValueError(f"R0/R1 GT population or protocol differs at update {update}")


def _terminate_owned_tree(process: psutil.Popen) -> None:
    owned = [*process.children(recursive=True), process]
    for item in reversed(owned):
        try:
            item.terminate()
        except psutil.NoSuchProcess:
            pass
    _, alive = psutil.wait_procs(owned, timeout=10)
    for item in alive:
        try:
            item.kill()
        except psutil.NoSuchProcess:
            pass
    psutil.wait_procs(alive, timeout=10)


def resource_preflight(config: CampaignConfig, gpu: str) -> dict[str, object]:
    """Check the same 8 GiB physical-RAM/VRAM gates before each frozen runner child."""
    available_ram = int(psutil.virtual_memory().available)
    command = [
        "nvidia-smi",
        f"--id={gpu}",
        "--query-gpu=memory.free,memory.total",
        "--format=csv,noheader,nounits",
    ]
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    lines = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    if len(lines) != 1:
        raise RuntimeError("GPU preflight must resolve exactly one device")
    fields = [field.strip() for field in lines[0].split(",")]
    if len(fields) != 2 or any(not field.isdigit() for field in fields):
        raise RuntimeError("Could not parse nvidia-smi physical-memory receipt")
    free_mib, total_mib = (int(field) for field in fields)
    available_vram = free_mib * 2**20
    minimum_ram = int(config.resources.minimum_available_ram_gb * 2**30)
    minimum_vram = int(config.resources.minimum_available_vram_gb * 2**30)
    receipt = {
        "status": "PASS",
        "available_ram_bytes": available_ram,
        "available_vram_bytes": available_vram,
        "total_vram_bytes": total_mib * 2**20,
        "minimum_available_ram_bytes": minimum_ram,
        "minimum_available_vram_bytes": minimum_vram,
        "gpu": gpu,
        "command": command,
    }
    if available_ram < minimum_ram or available_vram < minimum_vram:
        raise RuntimeError("Queue resource headroom is below the frozen runner 8 GiB gates")
    return receipt


def run_child(
    *,
    name: str,
    python: Path,
    producer: Path,
    config_path: Path,
    config: CampaignConfig,
    output: Path,
    queue_output: Path,
    gpu: str,
    resume: Path | None = None,
    stop_after_update: int | None = None,
) -> dict[str, object]:
    """Run exactly one foreground child with durable PID/log/source receipts."""
    if output.exists():
        raise FileExistsError(f"Child output must be fresh: {output}")
    command = [str(python), str(producer), "--config", str(config_path), "--output", str(output)]
    if resume is not None:
        command.extend(("--resume", str(resume)))
    if stop_after_update is not None:
        command.extend(("--stop-after-update", str(stop_after_update)))
    stdout_path = queue_output / f"{name}.stdout.log"
    stderr_path = queue_output / f"{name}.stderr.log"
    receipt_path = queue_output / f"{name}.json"
    receipt: dict[str, object] = {
        "schema": "phase_ttc_detection_campaign_queue_child_v1",
        "status": "STARTING",
        "name": name,
        "command": command,
        "producer_sha256": sha256_file(producer),
        "config_file_sha256": sha256_file(config_path),
        "config_canonical_sha256": config.sha256,
        "resume": str(resume) if resume else None,
        "resume_sha256": sha256_file(resume) if resume else None,
        "output": str(output),
        "stdout": str(stdout_path),
        "stderr": str(stderr_path),
        "deadline_seconds": config.resources.maximum_seconds,
        "started_at_utc": datetime.now(UTC).isoformat(),
    }
    _atomic_json(receipt_path, receipt)
    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = gpu
    creationflags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    with stdout_path.open("x", encoding="utf-8") as stdout, stderr_path.open(
        "x", encoding="utf-8"
    ) as stderr:
        process = psutil.Popen(
            command,
            cwd=ROOT,
            env=environment,
            stdout=stdout,
            stderr=stderr,
            creationflags=creationflags,
        )
        receipt.update(status="RUNNING", pid=process.pid)
        _atomic_json(receipt_path, receipt)
        timed_out = False
        try:
            returncode = process.wait(timeout=config.resources.maximum_seconds)
        except psutil.TimeoutExpired:
            timed_out = True
            _terminate_owned_tree(process)
            returncode = process.returncode if process.returncode is not None else -1
    receipt.update(
        status="FAILED" if timed_out or returncode else "EXITED_ZERO",
        returncode=returncode,
        timed_out=timed_out,
        ended_at_utc=datetime.now(UTC).isoformat(),
        stdout_sha256=sha256_file(stdout_path),
        stderr_sha256=sha256_file(stderr_path),
    )
    _atomic_json(receipt_path, receipt)
    if timed_out:
        raise TimeoutError(f"Child exceeded its fixed deadline: {name}")
    if returncode:
        raise RuntimeError(f"Child failed with return code {returncode}: {name}")
    return receipt


def run_queue(plan_path: Path, plan_sha256: str, queue_output: Path) -> dict[str, object]:
    """Run R0 segments then cold R1, stopping durably on the first failure."""
    plan_path = plan_path.resolve()
    queue_output = queue_output.resolve()
    if queue_output.exists():
        raise FileExistsError(f"Queue receipt output must be fresh: {queue_output}")
    plan = load_plan(plan_path, plan_sha256)
    r0_path, r0_config = _config(plan, "r0")
    r1_path, r1_config = _config(plan, "r1")
    verify_recipe_pair(r0_config, r1_config)
    r0_1024 = _resolve(plan["r0"]["output_1024"], "r0.output_1024")
    r0_8192 = _resolve(plan["r0"]["output_8192"], "r0.output_8192")
    r1_output = _resolve(plan["r1"]["output"], "r1.output")
    verify_disjoint_outputs((r0_1024, r0_8192, r1_output, queue_output))
    if any(path.exists() for path in (r0_1024, r0_8192, r1_output)):
        raise FileExistsError("Every child output must be fresh before queue start")
    queue_output.mkdir(parents=True)
    python, _ = _binding(plan["python"], "python")
    producer, _ = _binding(plan["producer"], "producer")
    segments: dict[str, Any] = {}
    report: dict[str, Any] = {
        "schema": RECEIPT_SCHEMA,
        "status": "RUNNING",
        "plan_sha256": plan_sha256,
        "queue_script_sha256": sha256_file(Path(__file__)),
        "gpu": plan["gpu"],
        "started_at_utc": datetime.now(UTC).isoformat(),
        "ap_threshold_selection": False,
        "automatic_retries": 0,
        "maximum_concurrent_children": 1,
        "execution_mode": "foreground_wait_each_child",
        "segments": segments,
    }
    receipt_path = queue_output / "queue_receipt.json"
    _atomic_json(receipt_path, report)
    try:
        for segment_name in ("r0_1024", "r0_8192", "r1_8192"):
            report["last_binding_receipt"] = verify_all_bindings(
                plan, plan_path, plan_sha256
            )
            if segment_name == "r0_1024":
                child_output, config_path, config = r0_1024, r0_path, r0_config
                resume, stop = None, 1024
            elif segment_name == "r0_8192":
                child_output, config_path, config = r0_8192, r0_path, r0_config
                resume, stop = r0_1024 / "checkpoints/last.pt", 8192
            else:
                child_output, config_path, config = r1_output, r1_path, r1_config
                resume, stop = None, None
            report["active_segment"] = segment_name
            report["resource_preflight"] = resource_preflight(config, plan["gpu"])
            _atomic_json(receipt_path, report)
            child = run_child(
                name=segment_name,
                python=python,
                producer=producer,
                config_path=config_path,
                config=config,
                output=child_output,
                queue_output=queue_output,
                gpu=plan["gpu"],
                resume=resume,
                stop_after_update=stop,
            )
            report["last_binding_receipt"] = verify_all_bindings(
                plan, plan_path, plan_sha256
            )
            verified: dict[str, Any]
            if segment_name == "r0_1024":
                verified = verify_summary(
                    child_output,
                    config,
                    expected_status="BUDGET_STOPPED",
                    expected_final_update=1024,
                    expected_start_update=0,
                    resume_sha256=None,
                )
            elif segment_name == "r0_8192":
                resume_sha256 = sha256_file(r0_1024 / "checkpoints/last.pt")
                verified = verify_summary(
                    child_output,
                    config,
                    expected_status="COMPLETED",
                    expected_final_update=8192,
                    expected_start_update=1024,
                    resume_sha256=resume_sha256,
                )
                verified["full_evaluations"] = verify_full_evaluations(
                    child_output, verified["summary"]
                )
                verify_r0_to_r1_gate(verified, r1_config)
            else:
                verified = verify_summary(
                    child_output,
                    config,
                    expected_status="COMPLETED",
                    expected_final_update=8192,
                    expected_start_update=0,
                    resume_sha256=None,
                )
                r1_evaluations = verify_full_evaluations(
                    child_output, verified["summary"]
                )
                verified["full_evaluations"] = r1_evaluations
                r0_evaluations = segments["r0_8192"]["verified"]["full_evaluations"]
                verify_population_same(r0_evaluations, r1_evaluations)
            segments[segment_name] = {"child": child, "verified": verified}
            report["active_segment"] = None
            _atomic_json(receipt_path, report)
        report.update(status="COMPLETED", ended_at_utc=datetime.now(UTC).isoformat())
        _atomic_json(receipt_path, report)
        return report
    except Exception as error:
        report.update(
            status="FAILED",
            error_type=type(error).__name__,
            error=str(error),
            failed_at_utc=datetime.now(UTC).isoformat(),
        )
        _atomic_json(receipt_path, report)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        run_queue(args.plan, args.plan_sha256, args.output)
    except Exception as error:  # noqa: BLE001 - queue receipt is the durable error report
        output = args.output.resolve()
        if not output.exists():
            output.mkdir(parents=True)
            _atomic_json(
                output / "queue_receipt.json",
                {
                    "schema": RECEIPT_SCHEMA,
                    "status": "FAILED",
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "failed_at_utc": datetime.now(UTC).isoformat(),
                    "plan": str(args.plan.resolve()),
                    "requested_plan_sha256": args.plan_sha256,
                },
            )
        print(f"{type(error).__name__}: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

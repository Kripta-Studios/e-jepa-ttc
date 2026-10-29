"""CPU contracts and evidence checks for a serial GPU engineering admission."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import torch

from e_jepa_ttc.detection.campaign_checkpoint_migration import component_sha256
from e_jepa_ttc.detection.campaign_training import (
    canonical_sha256,
    load_campaign_config,
    sha256_file,
)

SCHEMA = "eap_detection_gpu_admission_plan_v1"
ARM_NAMES = ("old_A", "old_B", "optimized")
REQUIRED_BINDING_ROLES = frozenset(
    {
        "config",
        "dataset_manifest",
        "media_admission",
        "old_runtime_preregistration",
        "old_producer",
        "optimized_runtime_preregistration",
        "optimized_producer",
        "performance_config",
        "python_executable",
    }
)


@dataclass(frozen=True)
class Tolerances:
    """Preregistered numerical and performance admission ceilings."""

    first_loss_absolute_floor: float = 1e-7
    first_loss_absolute_cap: float = 1e-6
    repeat_multiplier: float = 4.0
    later_loss_absolute_floor: float = 5e-6
    later_loss_absolute_cap: float = 1e-2
    gradient_absolute_floor: float = 1e-5
    gradient_absolute_cap: float = 1e-3
    minimum_preparation_speedup_each_half: float = 1.05

    def validate(self) -> None:
        values = tuple(self.__dict__.values())
        if not all(
            isinstance(value, float) and math.isfinite(value) and value > 0
            for value in values
        ):
            raise ValueError("Every admission tolerance must be finite and positive")
        if self.repeat_multiplier < 1:
            raise ValueError("repeat_multiplier must be at least one")


def _digest(value: object, field: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError(f"{field} must be a lowercase SHA-256")
    return value


def _flag_value(command: Sequence[str], flag: str) -> str:
    if command.count(flag) != 1:
        raise ValueError(f"Command must contain exactly one {flag}")
    index = command.index(flag)
    if index + 1 >= len(command):
        raise ValueError(f"Command flag lacks value: {flag}")
    return command[index + 1]


def load_plan(path: Path) -> dict[str, Any]:
    """Load a strict, self-hashed admission plan."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or raw.get("schema") != SCHEMA:
        raise ValueError("Unknown GPU admission plan schema")
    declared = raw.pop("plan_sha256", None)
    if declared != canonical_sha256(raw):
        raise ValueError("GPU admission plan self-hash differs")
    raw["plan_sha256"] = declared
    if raw.get("arm_order") != list(ARM_NAMES):
        raise ValueError("Admission arm order must be old_A, old_B, optimized")
    if raw.get("updates") != 16:
        raise ValueError("Admission budget must be exactly 16 updates")
    if isinstance(raw.get("batch_size"), bool) or not isinstance(raw.get("batch_size"), int):
        raise ValueError("batch_size must be an integer")
    if raw["batch_size"] < 1:
        raise ValueError("batch_size must be positive")
    resources = raw.get("resources")
    expected_resource_keys = {
        "minimum_available_physical_ram_bytes",
        "minimum_free_vram_bytes",
    }
    if not isinstance(resources, dict) or set(resources) != expected_resource_keys:
        raise ValueError("Resource gate fields differ from the exact admission contract")
    for field in expected_resource_keys:
        value = resources[field]
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"Resource gate must be a positive integer: {field}")
    if raw.get("candidate_policy") != "ENGINEERING_NON_CANDIDATE_DISCARD_OUTPUTS":
        raise ValueError("Engineering outputs must be declared non-candidates")
    if not isinstance(raw.get("execution_receipt_path"), str) or not raw[
        "execution_receipt_path"
    ]:
        raise ValueError("execution_receipt_path must be preregistered")
    _digest(raw.get("config_sha256"), "config_sha256")
    _digest(raw.get("protocol_sha256"), "protocol_sha256")
    _digest(raw.get("dataset_manifest_sha256"), "dataset_manifest_sha256")
    graphics = raw.get("graphics_baseline")
    if not isinstance(graphics, list):
        raise ValueError("graphics_baseline must be captured in the plan")
    graphic_identities = []
    for row in graphics:
        if not isinstance(row, dict) or row.get("type") != "C+G":
            raise ValueError("Graphics baseline may contain only C+G process identities")
        identity = (row.get("pid"), row.get("creation_time_us"))
        if not all(isinstance(value, int) and value >= 0 for value in identity):
            raise ValueError("Graphics baseline PID/creation identity is malformed")
        graphic_identities.append(identity)
    if len(graphic_identities) != len(set(graphic_identities)):
        raise ValueError("Graphics baseline contains duplicate process identities")
    tolerances = Tolerances(**raw["tolerances"])
    tolerances.validate()
    arms = raw.get("arms")
    if not isinstance(arms, dict) or tuple(arms) != ARM_NAMES:
        raise ValueError("Admission plan arms differ from the fixed serial order")
    for name, arm in arms.items():
        if not isinstance(arm, dict) or arm.get("name") != name:
            raise ValueError(f"Malformed admission arm: {name}")
        for artifact in ("resume_checkpoint", "selected_checkpoint"):
            item = arm.get(artifact)
            if not isinstance(item, dict):
                raise ValueError(f"{name}.{artifact} must be an artifact binding")
            source = Path(str(item.get("path"))).resolve(strict=True)
            if sha256_file(source) != _digest(item.get("sha256"), f"{name}.{artifact}"):
                raise ValueError(f"{name}.{artifact} bytes differ from plan")
        resume_path = Path(arm["resume_checkpoint"]["path"]).resolve()
        selected_path = Path(arm["selected_checkpoint"]["path"]).resolve()
        if selected_path.parent != resume_path.parent or selected_path.name != "selected.pt":
            raise ValueError(f"{name} selected checkpoint is not the resume sibling selected.pt")
        command = arm.get("command")
        if not isinstance(command, list) or not command or not all(
            isinstance(value, str) and value for value in command
        ):
            raise ValueError(f"{name}.command must be a nonempty argv list")
        if int(arm["stop_after_update"]) - int(arm["start_update"]) != int(raw["updates"]):
            raise ValueError(f"{name} update budget differs from the plan")
        for field in ("output", "metrics_path", "summary_path", "evaluation_path"):
            if not isinstance(arm.get(field), str) or not arm[field]:
                raise ValueError(f"{name}.{field} must be a nonempty path")
        output = Path(arm["output"]).resolve()
        for field in ("metrics_path", "summary_path", "evaluation_path"):
            if not Path(arm[field]).resolve().is_relative_to(output):
                raise ValueError(f"{name}.{field} escapes its new arm output")
    bindings = raw.get("bindings", [])
    roles = [binding.get("role") for binding in bindings if isinstance(binding, dict)]
    if len(roles) != len(set(roles)) or set(roles) != REQUIRED_BINDING_ROLES:
        raise ValueError("Runtime binding roles differ from the required admission closure")
    for binding in bindings:
        if not isinstance(binding, dict):
            raise ValueError("Plan bindings must be objects")
        source = Path(str(binding.get("path"))).resolve(strict=True)
        if sha256_file(source) != _digest(binding.get("sha256"), "binding.sha256"):
            raise ValueError(f"Bound runtime artifact changed: {source}")
    binding_by_role = {binding["role"]: binding for binding in bindings}
    config = load_campaign_config(Path(binding_by_role["config"]["path"]))
    if config.sha256 != raw["config_sha256"]:
        raise ValueError("Bound config canonical hash differs from the plan")
    if config.data.expected_manifest_sha256 != raw["dataset_manifest_sha256"]:
        raise ValueError("Bound config dataset manifest hash differs from the plan")
    if binding_by_role["dataset_manifest"]["sha256"] != raw["dataset_manifest_sha256"]:
        raise ValueError("Dataset-manifest binding bytes differ from configured hash")
    dataset_manifest = json.loads(
        Path(binding_by_role["dataset_manifest"]["path"]).read_text(encoding="utf-8")
    )
    for runtime_role, producer_role in (
        ("old_runtime_preregistration", "old_producer"),
        ("optimized_runtime_preregistration", "optimized_producer"),
    ):
        prereg = json.loads(Path(binding_by_role[runtime_role]["path"]).read_text(encoding="utf-8"))
        if (
            prereg.get("config_sha256") != raw["config_sha256"]
            or prereg.get("protocol_sha256") != raw["protocol_sha256"]
            or prereg.get("dataset_media_sha256") != dataset_manifest.get("source_media_sha256")
        ):
            raise ValueError(f"{runtime_role} scientific bindings differ from the plan")
        code = prereg.get("code_sha256")
        producer_key = f"scripts/{Path(binding_by_role[producer_role]['path']).name}"
        if not isinstance(code, dict) or code.get(producer_key) != binding_by_role[producer_role][
            "sha256"
        ]:
            raise ValueError(f"{runtime_role} does not bind its declared producer bytes")
    for name in ARM_NAMES:
        command = arms[name]["command"]
        expected_producer = binding_by_role[
            "optimized_producer" if name == "optimized" else "old_producer"
        ]["path"]
        expected_python = Path(binding_by_role["python_executable"]["path"]).resolve()
        if Path(command[0]).resolve() != expected_python:
            raise ValueError(f"{name} command Python executable differs from binding")
        if Path(command[1]).resolve() != Path(expected_producer).resolve():
            raise ValueError(f"{name} command producer differs from binding")
        expected_flags = {
            "--config": binding_by_role["config"]["path"],
            "--resume": arms[name]["resume_checkpoint"]["path"],
            "--output": arms[name]["output"],
            "--stop-after-update": str(arms[name]["stop_after_update"]),
        }
        if name == "optimized":
            expected_flags["--performance-config"] = binding_by_role["performance_config"][
                "path"
            ]
        for flag, expected in expected_flags.items():
            actual = _flag_value(command, flag)
            if flag == "--stop-after-update":
                if actual != expected:
                    raise ValueError(f"{name} command {flag} differs from plan")
            elif Path(actual).resolve() != Path(expected).resolve():
                raise ValueError(f"{name} command {flag} differs from binding")
    return raw


def validate_checkpoint_lineage(plan: Mapping[str, Any]) -> None:
    """Bind each CPU checkpoint to config, protocol and its reviewed runtime."""
    binding_by_role = {binding["role"]: binding for binding in plan["bindings"]}
    prereg_by_arm = {
        "old_A": binding_by_role["old_runtime_preregistration"],
        "old_B": binding_by_role["old_runtime_preregistration"],
        "optimized": binding_by_role["optimized_runtime_preregistration"],
    }
    for name in ARM_NAMES:
        for artifact in ("resume_checkpoint", "selected_checkpoint"):
            checkpoint = torch.load(
                Path(plan["arms"][name][artifact]["path"]),
                map_location="cpu",
                weights_only=True,
            )
            if checkpoint.get("config_sha256") != plan["config_sha256"]:
                raise ValueError(f"{name}.{artifact} config differs from the plan")
            if checkpoint.get("protocol_sha256") != plan["protocol_sha256"]:
                raise ValueError(f"{name}.{artifact} protocol differs from the plan")
            provenance = checkpoint.get("runtime_provenance")
            if not isinstance(provenance, dict) or provenance.get(
                "runtime_preregistration_sha256"
            ) != prereg_by_arm[name]["sha256"]:
                raise ValueError(f"{name}.{artifact} runtime preregistration differs")
            if artifact == "resume_checkpoint" and checkpoint.get("update") != plan["arms"][
                name
            ]["start_update"]:
                raise ValueError(f"{name} resume update differs from preregistered start_update")


def checkpoint_components(path: Path) -> dict[str, str]:
    """Load one trusted CPU checkpoint and fingerprint all continuation state."""
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, Mapping):
        raise TypeError("Campaign checkpoint must be a mapping")
    return component_sha256(payload)


def validate_initial_state(plan: Mapping[str, Any]) -> dict[str, object]:
    """Require all three arms to begin from byte-semantic identical CPU state."""
    components = {
        name: checkpoint_components(Path(plan["arms"][name]["resume_checkpoint"]["path"]))
        for name in ARM_NAMES
    }
    baseline = components["old_A"]
    if components["old_B"] != baseline or components["optimized"] != baseline:
        raise ValueError("Admission arms do not share exact initial continuation state")
    selected = {
        name: checkpoint_components(Path(plan["arms"][name]["selected_checkpoint"]["path"]))
        for name in ARM_NAMES
    }
    if selected["old_B"] != selected["old_A"] or selected["optimized"] != selected["old_A"]:
        raise ValueError("Admission arms do not share exact selected-checkpoint state")
    return {"resume_components": baseline, "selected_components": selected["old_A"]}


def _records(path: Path, expected: int) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if len(rows) != expected:
        raise ValueError(f"Expected {expected} metric records, found {len(rows)}")
    if any(not isinstance(row, dict) for row in rows):
        raise TypeError("Metric records must be JSON objects")
    return rows


def _finite(row: Mapping[str, Any], field: str) -> float:
    value = row.get(field)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"Metric {field} must be finite")
    return float(value)


def _ceiling(repeat_delta: float, floor: float, cap: float, multiplier: float) -> float:
    return min(cap, max(floor, repeat_delta * multiplier))


def verify_numerical_metrics(plan: Mapping[str, Any]) -> dict[str, object]:
    """Verify exposure identity and bounded numerical drift only.

    This deliberately makes no timing, memory, speed, or candidate decision.  It is
    suitable for recovering numerical evidence after a resource-admission failure,
    while the ordinary admission remains non-passing.
    """
    count = int(plan["updates"])
    rows = {
        name: _records(Path(plan["arms"][name]["metrics_path"]), count)
        for name in ARM_NAMES
    }
    for index in range(count):
        reference = rows["old_A"][index]
        expected_update = int(plan["arms"]["old_A"]["start_update"]) + index + 1
        if reference.get("update") != expected_update:
            raise ValueError("Metric updates are not the exact contiguous admission interval")
        if len(reference.get("sample_tokens", ())) != int(plan["batch_size"]):
            raise ValueError("Metric exposure count differs from preregistered batch size")
        for name in ARM_NAMES[1:]:
            candidate = rows[name][index]
            if candidate.get("update") != expected_update:
                raise ValueError(f"Metric update differs at relative update {index + 1}: {name}")
            if candidate.get("sample_tokens") != reference.get("sample_tokens"):
                raise ValueError(f"Sample order differs at relative update {index + 1}: {name}")
            if candidate.get("learning_rates") != reference.get("learning_rates"):
                raise ValueError(f"Learning rates differ at relative update {index + 1}: {name}")
    tolerance = Tolerances(**plan["tolerances"])
    numerical: list[dict[str, float | int]] = []
    for index in range(count):
        old_a = _finite(rows["old_A"][index], "loss")
        old_b = _finite(rows["old_B"][index], "loss")
        optimized = _finite(rows["optimized"][index], "loss")
        repeat = abs(old_a - old_b)
        floor = (
            tolerance.first_loss_absolute_floor
            if index == 0
            else tolerance.later_loss_absolute_floor
        )
        cap = (
            tolerance.first_loss_absolute_cap
            if index == 0
            else tolerance.later_loss_absolute_cap
        )
        allowed = _ceiling(repeat, floor, cap, tolerance.repeat_multiplier)
        observed = abs(old_a - optimized)
        if observed > allowed:
            raise ValueError(f"Optimized loss drift exceeds preregistered ceiling at {index + 1}")
        head_losses = {
            name: rows[name][index].get("losses") for name in ARM_NAMES
        }
        if not all(isinstance(value, dict) for value in head_losses.values()):
            raise ValueError("Per-head losses are required for admission")
        typed_head_losses = {
            name: cast(Mapping[str, Any], value) for name, value in head_losses.items()
        }
        head_keys = set(typed_head_losses["old_A"])
        if not head_keys or any(
            set(typed_head_losses[name]) != head_keys for name in ARM_NAMES[1:]
        ):
            raise ValueError("Per-head loss keys differ across arms")
        maximum_head_delta = 0.0
        for head in sorted(head_keys):
            head_a = _finite(typed_head_losses["old_A"], head)
            head_b = _finite(typed_head_losses["old_B"], head)
            head_optimized = _finite(typed_head_losses["optimized"], head)
            head_allowed = _ceiling(abs(head_a - head_b), floor, cap, tolerance.repeat_multiplier)
            head_delta = abs(head_a - head_optimized)
            if head_delta > head_allowed:
                raise ValueError(
                    f"Optimized per-head loss drift exceeds ceiling at {index + 1}: {head}"
                )
            maximum_head_delta = max(maximum_head_delta, head_delta)
        repeat_gradient = abs(
            _finite(rows["old_A"][index], "gradient_norm_before_clip")
            - _finite(rows["old_B"][index], "gradient_norm_before_clip")
        )
        allowed_gradient = _ceiling(
            repeat_gradient,
            tolerance.gradient_absolute_floor,
            tolerance.gradient_absolute_cap,
            tolerance.repeat_multiplier,
        )
        observed_gradient = abs(
            _finite(rows["old_A"][index], "gradient_norm_before_clip")
            - _finite(rows["optimized"][index], "gradient_norm_before_clip")
        )
        if observed_gradient > allowed_gradient:
            raise ValueError(f"Optimized gradient drift exceeds ceiling at {index + 1}")
        numerical.append(
            {
                "relative_update": index + 1,
                "repeat_loss_delta": repeat,
                "optimized_loss_delta": observed,
                "loss_ceiling": allowed,
                "gradient_ceiling": allowed_gradient,
                "maximum_per_head_loss_delta": maximum_head_delta,
            }
        )
    return {
        "status": "PASS_NUMERICAL_PARITY_ONLY",
        "candidate_decision": None,
        "performance_comparison": "INCONCLUSIVE",
        "memory_comparison": "INCONCLUSIVE",
        "numerical": numerical,
        "exposures_per_arm": count * int(plan["batch_size"]),
    }


def verify_metrics(plan: Mapping[str, Any]) -> dict[str, object]:
    """Compare exact inputs, bounded drift, and preregistered performance gates."""
    numerical_result = verify_numerical_metrics(plan)
    count = int(plan["updates"])
    rows = {
        name: _records(Path(plan["arms"][name]["metrics_path"]), count)
        for name in ARM_NAMES
    }
    tolerance = Tolerances(**plan["tolerances"])
    halves = (range(0, count // 2), range(count // 2, count))
    speedups = []
    for half in halves:
        old = sum(
            _finite(rows[name][index], "preprocessing_seconds")
            for name in ("old_A", "old_B")
            for index in half
        ) / (2 * len(half))
        optimized = sum(
            _finite(rows["optimized"][index], "preprocessing_seconds") for index in half
        ) / len(half)
        speedup = old / optimized if optimized > 0 else math.inf
        if speedup < tolerance.minimum_preparation_speedup_each_half:
            raise ValueError("Preparation speedup is not consistent across both halves")
        speedups.append(speedup)
    performance: dict[str, object] = {}
    binding_by_role = {binding["role"]: binding for binding in plan["bindings"]}
    for name in ARM_NAMES:
        summary = json.loads(Path(plan["arms"][name]["summary_path"]).read_text(encoding="utf-8"))
        expected_prereg = binding_by_role[
            "optimized_runtime_preregistration"
            if name == "optimized"
            else "old_runtime_preregistration"
        ]["sha256"]
        if (
            summary.get("schema") != "eap_detection_campaign_run_v1"
            or summary.get("status") != "BUDGET_STOPPED"
            or summary.get("final_update") != plan["arms"][name]["stop_after_update"]
            or summary.get("external_execution_stop_requested")
            != plan["arms"][name]["stop_after_update"]
            or summary.get("stop_reason")
            != "external_execution_stop_after_completed_update"
            or summary.get("config_sha256") != plan["config_sha256"]
            or summary.get("evaluation_protocol_sha256") != plan["protocol_sha256"]
            or not isinstance(summary.get("runtime_provenance"), dict)
            or summary["runtime_provenance"].get("runtime_preregistration_sha256")
            != expected_prereg
        ):
            raise ValueError(f"{name} summary lineage/status differs from the plan")
        for field in (
            "preprocessing_seconds_median",
            "step_seconds_median",
            "stage_training_seconds",
            "cumulative_training_seconds",
        ):
            value = summary.get(field)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value < 0
            ):
                raise ValueError(f"{name} summary timing is malformed: {field}")
        for field in ("peak_allocated_bytes", "peak_reserved_bytes"):
            value = summary.get(field)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} summary memory is malformed: {field}")
        evaluation_path = Path(plan["arms"][name]["evaluation_path"])
        if not evaluation_path.is_file():
            raise ValueError(f"{name} required final evaluation is absent")
        evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
        metrics_v2 = evaluation.get("metrics_v2")
        if (
            not isinstance(metrics_v2, dict)
            or metrics_v2.get("schema") != "phase_ttc_detection_evaluation_result_v2"
            or metrics_v2.get("protocol_sha256") != plan["protocol_sha256"]
            or metrics_v2.get("official_eap_score") is not False
            or isinstance(metrics_v2.get("sample_count"), bool)
            or not isinstance(metrics_v2.get("sample_count"), int)
            or metrics_v2["sample_count"] <= 0
        ):
            raise ValueError(f"{name} final evaluation schema/protocol is invalid")
        for field in ("inference_seconds_median", "inference_seconds_total"):
            value = metrics_v2.get(field)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value < 0
            ):
                raise ValueError(f"{name} final evaluation timing is malformed: {field}")
        performance[name] = {
            "preprocessing_seconds_median": summary.get("preprocessing_seconds_median"),
            "step_seconds_median": summary.get("step_seconds_median"),
            "peak_allocated_bytes": summary.get("peak_allocated_bytes"),
            "peak_reserved_bytes": summary.get("peak_reserved_bytes"),
            "performance_metrics": summary.get("performance_metrics"),
            "evaluation_sha256": sha256_file(evaluation_path),
        }
    return {
        "status": "PASS_ENGINEERING_ADMISSION_ONLY",
        "candidate_decision": None,
        "extrapolation_to_8192_updates": None,
        "numerical": numerical_result["numerical"],
        "preparation_speedup_by_half": speedups,
        "performance": performance,
        "exposures_per_arm": count * int(plan["batch_size"]),
    }


def bind_plan(raw: dict[str, Any]) -> dict[str, Any]:
    """Return a self-hashed plan after strict JSON finiteness validation."""
    if "plan_sha256" in raw:
        raise ValueError("Unbound plan must not already contain plan_sha256")
    json.dumps(raw, allow_nan=False)
    return raw | {"plan_sha256": canonical_sha256(raw)}


__all__ = [
    "ARM_NAMES",
    "REQUIRED_BINDING_ROLES",
    "SCHEMA",
    "Tolerances",
    "bind_plan",
    "checkpoint_components",
    "load_plan",
    "validate_initial_state",
    "validate_checkpoint_lineage",
    "verify_numerical_metrics",
    "verify_metrics",
]

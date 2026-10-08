"""Label-free, fixed-query system-cost benchmark for H8 and full Garl.

The module is deliberately import-light.  Torch, NumPy, model wrappers, and
dataset preparation code are loaded only inside :func:`run`, after the
manifest, source freeze, output lease, and exclusive-GPU preflight pass.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import importlib
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from importlib import metadata
from pathlib import Path
from typing import Any, Protocol

ROOT = Path(__file__).resolve().parents[2]
SCHEMA = "sota_system_cost_fixed8_v1"
COST_SOURCE_FILES = (
    "operational/sota_eval/cost.py",
    "operational/evttc_transfer/inputs.py",
    "operational/evttc_transfer/models.py",
    "operational/evttc_rgb_transfer/inputs.py",
    "operational/evttc_rgb_transfer/model.py",
    "operational/efficient_context/common.py",
    "operational/simplex_t_shared_route/adapter.py",
    "operational/train40_system/contracts.py",
    "operational/train40_system/garl_predictions.py",
    "src/e_jepa_ttc/data/annotations.py",
    "src/e_jepa_ttc/data/event_v4_geometry.py",
    "src/e_jepa_ttc/data/evttc.py",
    "src/e_jepa_ttc/data/evttc_object_cache.py",
    "src/e_jepa_ttc/data/garl_official_preprocessing.py",
    "src/e_jepa_ttc/data/types.py",
    "src/e_jepa_ttc/efficient_context/garl_input.py",
    "src/e_jepa_ttc/efficient_context/mapped_union.py",
)
CPU_WARMUPS = 1
CPU_MEASUREMENTS = 3
GPU_WARMUPS = 5
GPU_MEASUREMENTS = 20
E2E_MEASUREMENTS = 3
PREPARE_CAP_SECONDS = 120.0
SYSTEMS = (
    "h8_system_three_heads",
    "garl_event_only_shared_preparation",
    "garl_full_rgb_event",
)
FORBIDDEN = {"ttc.csv", "gt.hdf5", "distance", "depth", "navigation"}
CSV_FIELDS = (
    "query_id",
    "sequence_id",
    "scenario_family",
    "system",
    "stage",
    "iteration",
    "warmup",
    "milliseconds",
    "status",
)


class _OwnModels(Protocol):
    bindings: dict[str, Any]

    def predict(
        self,
        own_events: Any,  # noqa: ANN401 - lazy NumPy wrapper boundary
        delta_t_s: Any,  # noqa: ANN401 - lazy NumPy wrapper boundary
        valid: Any,  # noqa: ANN401 - lazy NumPy wrapper boundary
    ) -> dict[str, float]: ...

    def garl_predict(
        self,
        sensor: Any,  # noqa: ANN401 - lazy NumPy wrapper boundary
    ) -> dict[str, float | list[float]]: ...


class _FullModel(Protocol):
    bindings: dict[str, Any]

    def predict_sensor(
        self,
        sensor: Any,  # noqa: ANN401 - lazy NumPy wrapper boundary
    ) -> dict[str, float | list[float]]: ...


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha(value: object) -> str:
    raw = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def source_closure(root: Path) -> dict[str, str]:
    """Hash every local input/model helper whose executable bytes affect cost."""
    pending: list[str] = list(COST_SOURCE_FILES)
    result: dict[str, str] = {}
    while pending:
        relative = pending.pop()
        if relative in result:
            continue
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        result[relative] = _sha256(path)
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        module = relative[:-3].replace("/", ".")
        if module.startswith("src."):
            module = module[4:]
        if module.endswith(".__init__"):
            module = module[: -len(".__init__")]
        imports: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    base = module.split(".")[:-1]
                    keep = max(0, len(base) - node.level + 1)
                    prefix: list[str] = list(base[:keep])
                    if node.module:
                        prefix.extend(node.module.split("."))
                    imported = ".".join(prefix)
                else:
                    imported = node.module or ""
                if imported:
                    imports.add(imported)
                    imports.update(f"{imported}.{alias.name}" for alias in node.names)
        for imported in imports:
            if not imported.startswith(("operational.", "e_jepa_ttc.")):
                continue
            roots = (root,) if imported.startswith("operational.") else (root / "src", root)
            for base in roots:
                candidate = base.joinpath(*imported.split(".")).with_suffix(".py")
                package = base.joinpath(*imported.split("."), "__init__.py")
                for resolved in (candidate, package):
                    if resolved.is_file():
                        child = resolved.relative_to(root).as_posix()
                        if child not in result:
                            pending.append(child)
    return result


def validate_source_closure(root: Path, expected: Mapping[str, str]) -> None:
    if source_closure(root) != dict(expected):
        raise ValueError("system-cost executable source closure changed")


def python_cpu_identity() -> dict[str, Any]:
    """Return stable interpreter/package/host identity affecting CPU preparation."""
    return {
        "python_version": sys.version,
        "packages": {
            distribution: metadata.version(distribution)
            for distribution in ("numpy", "h5py", "PyYAML")
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "cpu_count": os.cpu_count(),
        },
    }


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _stamp() -> str:
    from datetime import UTC, datetime

    return datetime.now(UTC).isoformat()


def percentile(values: Sequence[float], fraction: float) -> float:
    """Return a linearly interpolated percentile for finite non-negative values."""
    checked = [float(value) for value in values]
    if not checked or not 0.0 <= fraction <= 1.0:
        raise ValueError("percentile needs values and a fraction in [0,1]")
    if any(not math.isfinite(value) or value < 0.0 for value in checked):
        raise ValueError("timings must be finite and non-negative")
    ordered = sorted(checked)
    position = fraction * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def summarize_ms(values: Sequence[float]) -> dict[str, float | int | None]:
    """Summarize raw milliseconds without discarding slow observations."""
    checked = [float(value) for value in values]
    p50 = percentile(checked, 0.50)
    mean = statistics.fmean(checked)
    return {
        "count": len(checked),
        "mean_ms": mean,
        "p50_ms": p50,
        "p95_ms": percentile(checked, 0.95),
        "minimum_ms": min(checked),
        "maximum_ms": max(checked),
        "samples_per_second_from_mean": 1000.0 / mean if mean > 0.0 else None,
    }


def select_fixed_queries(document: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Select the first manifest row from each of exactly eight scenario families."""
    if document.get("status") != "LABEL_FREE_MANIFEST":
        raise ValueError("manifest must be LABEL_FREE_MANIFEST")
    forbidden = {str(item).lower() for item in document.get("forbidden_assets", [])}
    if not FORBIDDEN.issubset(forbidden):
        raise ValueError("manifest does not forbid all target-bearing assets")
    rows = document.get("rows")
    if not isinstance(rows, list):
        raise ValueError("manifest rows are missing")
    selected: dict[str, dict[str, Any]] = {}
    for candidate in rows:
        if not isinstance(candidate, dict):
            raise ValueError("manifest row is not an object")
        family = candidate.get("scenario_family")
        if isinstance(family, str) and family and family not in selected:
            selected[family] = dict(candidate)
    if len(selected) != 8:
        raise ValueError(
            f"cost protocol requires exactly eight scenario families, got {len(selected)}"
        )
    result = list(selected.values())
    if len({str(row.get("query_id")) for row in result}) != 8:
        raise ValueError("fixed cost queries are not unique")
    return result


def _safe_name(query_id: str) -> str:
    return hashlib.sha256(query_id.encode("utf-8")).hexdigest()[:16]


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(dict(row))  # type: ignore[arg-type]
    os.replace(temporary, path)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def validate_resume(
    fragment: Mapping[str, Any],
    csv_path: Path,
    freeze_sha256: str,
    model_freeze_sha256: str,
    query_id: str,
) -> None:
    """Fail closed unless a completed fragment binds exact protocol and raw bytes."""
    if fragment.get("status") != "COMPLETE" or fragment.get("query_id") != query_id:
        raise ValueError("resume fragment identity/status mismatch")
    if fragment.get("execution_freeze_sha256") != freeze_sha256:
        raise ValueError("resume fragment freeze mismatch")
    if fragment.get("model_freeze_sha256") != model_freeze_sha256:
        raise ValueError("resume fragment model freeze mismatch")
    if fragment.get("raw_csv_sha256") != _sha256(csv_path):
        raise ValueError("resume raw CSV bytes changed")


class _Lease:
    def __init__(self, path: Path) -> None:
        self.path = path

    def __enter__(self) -> _Lease:
        try:
            descriptor = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError as exc:
            raise RuntimeError(f"cost benchmark lease already exists: {self.path}") from exc
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump({"pid": os.getpid(), "created_utc": _stamp()}, handle)
        return self

    def __exit__(self, *_: object) -> None:
        self.path.unlink(missing_ok=True)


def _nvidia(arguments: Sequence[str]) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            ["nvidia-smi", *arguments], check=False, capture_output=True, text=True, timeout=2.0
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"available": False, "error_type": type(exc).__name__}
    return {
        "available": completed.returncode == 0,
        "returncode": completed.returncode,
        "rows": [line.strip() for line in completed.stdout.splitlines() if line.strip()],
    }


def _python_entrypoint(name: str, arguments: Sequence[str]) -> str | None:
    """Extract the executed Python module/script, ignoring text in wrapper arguments."""
    if not name.lower().startswith("python"):
        return None
    normalized = [str(item).replace("\\", "/").lower() for item in arguments]
    if "-m" in normalized:
        index = normalized.index("-m") + 1
        return normalized[index] if index < len(normalized) else None
    if len(normalized) > 1 and normalized[1].endswith(".py"):
        return normalized[1]
    return None


def exclusive_gpu_preflight() -> dict[str, Any]:
    """Reject existing compute clients before this process imports Torch."""
    psutil: Any = importlib.import_module("psutil")
    module_roles = {
        "operational.sota_eval.prefetch": "event_only_prefetch",
        "operational.sota_eval.full_prefetch": "full_prefetch",
        "operational.sota_eval.campaign": "expanded_campaign",
        "operational.sota_eval.fcwd_run": "fcwd",
        "operational.sota_eval.r1_resume": "r1",
        "operational.evttc_transfer.run": "transfer",
        "operational.evttc_rgb_transfer.run": "rgb_transfer",
        "operational.efficient_context.r1_gib_measure": "r1",
        "operational.efficient_context.r1_measure": "r1",
        "operational.efficient_context.r1_profile": "r1",
    }
    conflicts: list[dict[str, int | str]] = []
    for process in psutil.process_iter(["pid", "name"]):
        try:
            name = str(process.info.get("name") or "").lower()
            if process.pid == os.getpid() or "python" not in name:
                continue
            arguments = [str(item) for item in process.cmdline()]
            command = " ".join(arguments).replace("\\", "/").lower()
            if "--device cpu" in command or "--device=cpu" in command:
                continue
            entrypoint = _python_entrypoint(name, arguments)
            role = module_roles.get(entrypoint or "")
            if role is None and entrypoint:
                if entrypoint.startswith("operational.train40_system.engine"):
                    role = "train40"
                else:
                    script_roles = {
                        "operational/sota_eval/prefetch.py": "event_only_prefetch",
                        "operational/sota_eval/full_prefetch.py": "full_prefetch",
                        "operational/sota_eval/campaign.py": "expanded_campaign",
                        "operational/sota_eval/fcwd_run.py": "fcwd",
                        "operational/sota_eval/r1_resume.py": "r1",
                        "operational/evttc_transfer/run.py": "transfer",
                        "operational/evttc_rgb_transfer/run.py": "rgb_transfer",
                        "operational/efficient_context/r1_gib_measure.py": "r1",
                        "operational/efficient_context/r1_measure.py": "r1",
                        "operational/efficient_context/r1_profile.py": "r1",
                    }
                    role = next(
                        (
                            value
                            for suffix, value in script_roles.items()
                            if entrypoint.endswith(suffix)
                        ),
                        None,
                    )
            if role is not None:
                conflicts.append({"pid": int(process.pid), "role": role})
        except psutil.NoSuchProcess:
            continue
        except psutil.AccessDenied as exc:
            raise RuntimeError("cannot inspect a Python process for exclusive GPU use") from exc
    if conflicts:
        raise RuntimeError(f"project GPU/R1 conflict count: {len(conflicts)}")
    compute = _nvidia(
        ("--query-compute-apps=pid,process_name,used_gpu_memory", "--format=csv,noheader,nounits")
    )
    if not compute["available"]:
        raise RuntimeError("nvidia-smi compute-process preflight unavailable")
    rows = list(compute["rows"])
    if rows:
        raise RuntimeError(
            f"exclusive benchmark requires zero existing GPU compute clients; got {len(rows)}"
        )
    hardware = _nvidia(
        (
            "--query-gpu=name,driver_version,memory.total,memory.free,temperature.gpu,"
            "power.draw,clocks.sm,clocks.mem",
            "--format=csv,noheader,nounits",
        )
    )
    identity = _nvidia(
        (
            "--query-gpu=name,driver_version,memory.total,compute_cap",
            "--format=csv,noheader,nounits",
        )
    )
    if not identity["available"] or len(identity["rows"]) != 1:
        raise RuntimeError("stable NVIDIA hardware identity unavailable")
    return {
        "compute_clients": 0,
        "project_gpu_conflicts": 0,
        "hardware_identity": identity,
        "hardware_telemetry": hardware,
        "checked_utc": _stamp(),
    }


def _timed(
    call: Callable[[], Any], synchronize: Callable[[], None] | None = None
) -> tuple[float, Any]:
    if synchronize is not None:
        synchronize()
    begun = time.perf_counter()
    value = call()
    if synchronize is not None:
        synchronize()
    return (time.perf_counter() - begun) * 1000.0, value


def _output_digest(value: object) -> str:
    def safe(item: object) -> object:
        if isinstance(item, float) and not math.isfinite(item):
            if math.isnan(item):
                return {"nonfinite_float": "nan"}
            return {"nonfinite_float": "+inf" if item > 0 else "-inf"}
        if isinstance(item, Mapping):
            return {str(key): safe(nested) for key, nested in item.items()}
        if isinstance(item, (list, tuple)):
            return [safe(nested) for nested in item]
        return item

    return _canonical_sha(safe(value))


def _record(
    rows: list[dict[str, Any]],
    identity: Mapping[str, str],
    system: str,
    stage: str,
    iteration: int,
    warmup: bool,
    milliseconds: float,
) -> None:
    if not math.isfinite(milliseconds) or milliseconds < 0.0:
        raise ValueError("invalid measured duration")
    rows.append(
        {
            **identity,
            "system": system,
            "stage": stage,
            "iteration": iteration,
            "warmup": str(warmup).lower(),
            "milliseconds": f"{milliseconds:.9f}",
            "status": "OK",
        }
    )


def _measure_query(
    row: Mapping[str, Any],
    prepare_h8: Callable[[Mapping[str, Any]], Any],
    prepare_full: Callable[[Mapping[str, Any]], Any],
    own: _OwnModels,
    full: _FullModel,
    synchronize: Callable[[], None],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    identity = {key: str(row[key]) for key in ("query_id", "sequence_id", "scenario_family")}
    raw: list[dict[str, Any]] = []
    prepared: dict[str, Any] = {}
    for system, prepare in ((SYSTEMS[0], prepare_h8), (SYSTEMS[2], prepare_full)):
        for index in range(CPU_WARMUPS + CPU_MEASUREMENTS):
            milliseconds, value = _timed(lambda prepare=prepare: prepare(row))
            if milliseconds > PREPARE_CAP_SECONDS * 1000.0:
                raise TimeoutError(f"{system} preparation exceeded {PREPARE_CAP_SECONDS}s cap")
            _record(raw, identity, system, "cpu_prepare", index, index < CPU_WARMUPS, milliseconds)
            prepared[system] = value
            if system == SYSTEMS[0]:
                _record(
                    raw,
                    identity,
                    SYSTEMS[1],
                    "cpu_prepare",
                    index,
                    index < CPU_WARMUPS,
                    milliseconds,
                )
                prepared[SYSTEMS[1]] = value
    full_sensor = prepared[SYSTEMS[2]].get("sensor")
    if full_sensor is None:
        reason = prepared[SYSTEMS[2]].get("unavailable_reason", "unknown")
        raise ValueError(f"full Garl input unavailable without reselection: {reason}")
    if prepared[SYSTEMS[1]].get("garl_events") is None:
        raise ValueError("event-only Garl input unavailable without reselection")
    calls = {
        SYSTEMS[0]: lambda: own.predict(
            prepared[SYSTEMS[0]]["own_events"],
            prepared[SYSTEMS[0]]["delta_t_s"],
            prepared[SYSTEMS[0]]["valid"],
        ),
        SYSTEMS[1]: lambda: own.garl_predict(prepared[SYSTEMS[1]]["garl_events"]),
        SYSTEMS[2]: lambda: full.predict_sensor(full_sensor),
    }
    prediction_sha: dict[str, str] = {}
    for system, call in calls.items():
        observed: list[str] = []
        for index in range(GPU_WARMUPS + GPU_MEASUREMENTS):
            milliseconds, prediction = _timed(call, synchronize)
            _record(
                raw, identity, system, "gpu_inference", index, index < GPU_WARMUPS, milliseconds
            )
            observed.append(_output_digest(prediction))
        if len(set(observed)) != 1:
            raise ValueError(f"{system} prediction bytes changed across fixed-input repeats")
        prediction_sha[system] = observed[0]
    e2e_sha: dict[str, list[str]] = {name: [] for name in SYSTEMS}
    for system, prepare, predict in (
        (
            SYSTEMS[0],
            prepare_h8,
            lambda value: own.predict(value["own_events"], value["delta_t_s"], value["valid"]),
        ),
        (SYSTEMS[1], prepare_h8, lambda value: own.garl_predict(value["garl_events"])),
        (SYSTEMS[2], prepare_full, lambda value: full.predict_sensor(value["sensor"])),
    ):
        for index in range(E2E_MEASUREMENTS):
            synchronize()
            begun = time.perf_counter()
            value = prepare(row)
            if system == SYSTEMS[2] and value.get("sensor") is None:
                raise ValueError("full Garl became unavailable during E2E measurement")
            prediction = predict(value)
            synchronize()
            milliseconds = (time.perf_counter() - begun) * 1000.0
            if milliseconds > PREPARE_CAP_SECONDS * 1000.0:
                raise TimeoutError(f"{system} sequential E2E exceeded preparation cap")
            _record(raw, identity, system, "sequential_end_to_end", index, False, milliseconds)
            e2e_sha[system].append(_output_digest(prediction))
        if any(item != prediction_sha[system] for item in e2e_sha[system]):
            raise ValueError(f"{system} E2E output differs from inference-only output")
    return raw, {
        "prediction_sha256": prediction_sha,
        "h8_native_contexts": 16,
        "h8_reported_heads": [7, 13, 23],
        "h8_single_head": "NOT_MEASURED_OPTIONAL_NO_BACKBONE_SLICING",
        "garl_event_only_native_context": "2 event endpoints",
        "garl_full_native_context": "2 RGB endpoints + 2 event endpoints",
    }


def _aggregate(rows: Sequence[Mapping[str, str]]) -> dict[str, Any]:
    measured = [row for row in rows if row["warmup"] == "false" and row["status"] == "OK"]
    result: dict[str, Any] = {}
    for system in SYSTEMS:
        result[system] = {}
        for stage in ("cpu_prepare", "gpu_inference", "sequential_end_to_end"):
            values = [
                float(row["milliseconds"])
                for row in measured
                if row["system"] == system and row["stage"] == stage
            ]
            result[system][stage] = summarize_ms(values)
        result[system]["by_sequence"] = {
            sequence: summarize_ms(
                [
                    float(row["milliseconds"])
                    for row in measured
                    if row["system"] == system
                    and row["stage"] == "sequential_end_to_end"
                    and row["sequence_id"] == sequence
                ]
            )
            for sequence in sorted(
                {row["sequence_id"] for row in measured if row["system"] == system}
            )
        }
    return result


def run(
    manifest: Path, campaign: Path, public_full_dir: Path, code_root: Path, output: Path
) -> None:
    """Run or resume the frozen eight-query CUDA cost protocol."""
    manifest, campaign = manifest.resolve(), campaign.resolve()
    public_full_dir, code_root, output = (
        public_full_dir.resolve(),
        code_root.resolve(),
        output.resolve(),
    )
    document = json.loads(manifest.read_text(encoding="utf-8"))
    selected = select_fixed_queries(document)
    output.mkdir(parents=True, exist_ok=True)
    source = Path(__file__).resolve()
    freeze: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "EXECUTION_FROZEN",
        "source": str(source),
        "source_sha256": _sha256(source),
        "source_closure": source_closure(ROOT),
        "manifest": str(manifest),
        "manifest_sha256": _sha256(manifest),
        "campaign": str(campaign),
        "public_full_dir": str(public_full_dir),
        "code_root": str(code_root),
        "selected_before_latency_and_without_targets": True,
        "selected_queries": [
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
        ],
        "protocol": {
            "batch_size": 1,
            "cpu_warmups": CPU_WARMUPS,
            "cpu_measurements": CPU_MEASUREMENTS,
            "gpu_warmups": GPU_WARMUPS,
            "gpu_measurements": GPU_MEASUREMENTS,
            "sequential_e2e_measurements": E2E_MEASUREMENTS,
            "h8_history": 8,
            "h8_native_producer_dispatch": 16,
            "h8_prediction": "system_three_heads",
            "precision": "float32_no_tf32",
            "cache_state": "warm_after_explicit_per-query_warmups",
        },
        "preprocessing_disclosure": {
            SYSTEMS[0]: "native frozen H8 history8 preparation; existing helper also prepares the "
            "event-only Garl tensor, so this is a conservative H8-system cost",
            SYSTEMS[1]: "the exact same measured H8+event-only bundle preparation as H8, shared "
            "without duplicate CPU execution; includes unused H8 history and is conservative",
            SYSTEMS[2]: "published full Garl RGB+event sensor preparation",
        },
        "targets_read": False,
        "optimizer_updates": 0,
    }
    freeze_path = output / "EXECUTION_FREEZE.json"
    if freeze_path.exists():
        if json.loads(freeze_path.read_text(encoding="utf-8")) != freeze:
            raise ValueError("execution freeze changed; use a new output directory")
    else:
        _atomic_json(freeze_path, freeze)
    freeze_sha = _sha256(freeze_path)
    with _Lease(output / "WRITER.lock"):
        preflight = exclusive_gpu_preflight()
        from operational.evttc_rgb_transfer.inputs import prepare as prepare_full
        from operational.evttc_rgb_transfer.model import FullGarl
        from operational.evttc_transfer.inputs import prepare as prepare_h8
        from operational.evttc_transfer.models import FrozenModels

        psutil: Any = importlib.import_module("psutil")
        torch: Any = importlib.import_module("torch")

        torch.set_num_threads(2)
        torch.set_num_interop_threads(2)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA cost benchmark requested but unavailable")
        synchronize = torch.cuda.synchronize
        process = psutil.Process()
        rss_samples: list[int] = []
        stop_sample = threading.Event()

        def sample_rss() -> None:
            while not stop_sample.wait(0.05):
                try:
                    rss_samples.append(int(process.memory_info().rss))
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    return

        sampler = threading.Thread(target=sample_rss, daemon=True)
        sampler.start()
        try:
            torch.cuda.reset_peak_memory_stats()
            begun = time.perf_counter()
            own = FrozenModels(campaign, device="cuda")
            own_setup = time.perf_counter() - begun
            begun = time.perf_counter()
            full = FullGarl(public_full_dir, code_root, device="cuda")
            full_setup = time.perf_counter() - begun
            setup = {
                SYSTEMS[0]: own_setup,
                SYSTEMS[1]: own_setup,
                SYSTEMS[2]: full_setup,
                "excluded_from_latency": True,
                "model_bindings": {
                    SYSTEMS[0]: own.bindings,
                    SYSTEMS[1]: own.bindings,
                    SYSTEMS[2]: full.bindings,
                },
            }
            model_freeze = {
                "schema": SCHEMA,
                "status": "MODELS_FROZEN_BEFORE_MEASUREMENT",
                "execution_freeze_sha256": freeze_sha,
                "models": setup["model_bindings"],
                "runtime_identity": {
                    **python_cpu_identity(),
                    "torch_version": str(torch.__version__),
                    "torch_cuda_version": str(torch.version.cuda),
                    "cudnn_version": int(torch.backends.cudnn.version()),
                    "cuda_device_name": str(torch.cuda.get_device_name(0)),
                    "cuda_device_capability": list(torch.cuda.get_device_capability(0)),
                    "cuda_total_memory_bytes": int(
                        torch.cuda.get_device_properties(0).total_memory
                    ),
                    "nvidia_smi": preflight["hardware_identity"],
                },
                "targets_read": False,
                "optimizer_updates": 0,
            }
            model_freeze_path = output / "MODEL_FREEZE.json"
            if model_freeze_path.exists():
                if json.loads(model_freeze_path.read_text(encoding="utf-8")) != model_freeze:
                    raise ValueError("model freeze changed; preserved timings cannot be reused")
            else:
                _atomic_json(model_freeze_path, model_freeze)
            model_freeze_sha = _sha256(model_freeze_path)
            validate_source_closure(ROOT, freeze["source_closure"])
            all_rows: list[dict[str, str]] = []
            for row in selected:
                validate_source_closure(ROOT, freeze["source_closure"])
                stem = _safe_name(str(row["query_id"]))
                csv_path = output / f"RAW_{stem}.csv"
                fragment_path = output / f"FRAGMENT_{stem}.json"
                if fragment_path.exists():
                    fragment = json.loads(fragment_path.read_text(encoding="utf-8"))
                    validate_resume(
                        fragment, csv_path, freeze_sha, model_freeze_sha, str(row["query_id"])
                    )
                    all_rows.extend(_read_csv(csv_path))
                    continue
                try:
                    raw, audit = _measure_query(
                        row, prepare_h8, prepare_full, own, full, synchronize
                    )
                    _write_csv(csv_path, raw)
                    fragment = {
                        "schema": SCHEMA,
                        "status": "COMPLETE",
                        "query_id": row["query_id"],
                        "sequence_id": row["sequence_id"],
                        "scenario_family": row["scenario_family"],
                        "execution_freeze_sha256": freeze_sha,
                        "model_freeze_sha256": model_freeze_sha,
                        "raw_csv": csv_path.name,
                        "raw_csv_sha256": _sha256(csv_path),
                        "audit": audit,
                        "completed_utc": _stamp(),
                        "targets_read": False,
                        "optimizer_updates": 0,
                    }
                    _atomic_json(fragment_path, fragment)
                    all_rows.extend(
                        [{key: str(value) for key, value in item.items()} for item in raw]
                    )
                except Exception as exc:
                    _atomic_json(
                        output / f"FAILURE_{stem}.json",
                        {
                            "schema": SCHEMA,
                            "status": "FAILED_PRESERVED",
                            "query_id": row["query_id"],
                            "error_type": type(exc).__name__,
                            "error": str(exc),
                            "execution_freeze_sha256": freeze_sha,
                            "model_freeze_sha256": model_freeze_sha,
                            "targets_read": False,
                            "optimizer_updates": 0,
                            "failed_utc": _stamp(),
                        },
                    )
                    raise
            summary = {
                "schema": SCHEMA,
                "status": "COMPLETE",
                "query_count": len(selected),
                "execution_freeze_sha256": freeze_sha,
                "model_freeze_sha256": model_freeze_sha,
                "setup_seconds": setup,
                "metrics": _aggregate(all_rows),
                "preflight": preflight,
                "peak_process_rss_bytes_observed_50ms": max(rss_samples, default=0),
                "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated()),
                "peak_cuda_reserved_bytes": int(torch.cuda.max_memory_reserved()),
                "post_hardware": _nvidia(
                    (
                        "--query-gpu=name,memory.used,temperature.gpu,"
                        "power.draw,clocks.sm,clocks.mem",
                        "--format=csv,noheader,nounits",
                    )
                ),
                "interpretation": [
                    "Measured on this host only; no comparison to published hardware latency.",
                    "Architecture contexts differ by design and are reported, not normalized.",
                    "Sequential execution excludes model setup and includes warm-cache "
                    "preparation.",
                    "This is observational system cost, not causal architecture efficiency.",
                ],
                "targets_read": False,
                "optimizer_updates": 0,
                "completed_utc": _stamp(),
            }
            validate_source_closure(ROOT, freeze["source_closure"])
            _atomic_json(output / "SYSTEM_COST_SUMMARY.json", summary)
        finally:
            stop_sample.set()
            sampler.join(timeout=1.0)


def main() -> None:
    """CLI for the controlled final system-cost run."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--public-full-dir", type=Path, required=True)
    parser.add_argument("--code-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    run(
        arguments.manifest,
        arguments.campaign,
        arguments.public_full_dir,
        arguments.code_root,
        arguments.output,
    )


if __name__ == "__main__":
    main()

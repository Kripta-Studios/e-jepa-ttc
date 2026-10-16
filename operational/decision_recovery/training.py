"""V19 FULL-L0/LW: isolated durable training on immutable V18 sensor inputs."""

from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import shutil
import subprocess
import time
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import patch

import numpy as np
import pandas as pd
import psutil

from . import common
from .losses import MASSES, training_mass
from .runtime_performance import POLICY as PERFORMANCE_POLICY

if TYPE_CHECKING:
    from operational.joint_measurement.geometry_model import GeometricPairMeasurement
    from operational.joint_measurement.persistent_adaptation import Session as BaseSession

ENDPOINT = 12000
GPU_CAP = 60000
UPDATE_CAP = 100000
TECHNICAL_CAP = 200
RECOVERY_CAP = 1000
PATCH_TARGETS = (
    "operational.joint_measurement.persistent_adaptation.accounting",
    "operational.joint_measurement.persistent_adaptation.require_resources",
    "operational.joint_measurement.persistent_adaptation.backward",
    "operational.joint_measurement.data.cache_root",
    "operational.joint_measurement.data.role_rows",
)


def destination(relative: str) -> Path:
    """Resolve all mutable campaign files inside the new campaign only."""
    path = (common.RUN / relative).resolve()
    if not path.is_relative_to(common.RUN.resolve()):
        raise ValueError("V19 output escaped its campaign")
    if path.is_relative_to(common.SOURCE_ROOT.resolve()):
        raise ValueError("V18 is read-only")
    return path


def source_run() -> Path:
    return common.SOURCE_ROOT / "artifacts/joint_measurement_20261012"


def recipe() -> dict:
    return common.read_json(common.SOURCE_ROOT / "configs/joint_measurement/primary.json")


def parent_path() -> Path:
    return (
        source_run()
        / "adaptive/B1_correspondence/fits/B1-CORRESPONDENCE-STAGE1-S7/checkpoint_last.pt"
    )


def cache_root(role: str) -> Path:
    if role not in {"P", "H"}:
        raise ValueError("V19 training and development cannot open V")
    return Path(recipe()["cache_root"]) / "sensor" / role


def manifest_path(role: str) -> Path:
    if role not in {"P", "H"}:
        raise ValueError("V is excluded")
    return (common.SOURCE_ROOT / recipe()["owner_relative"]).resolve() / f"{role}_MANIFEST.json"


def role_rows(role: str) -> pd.DataFrame:
    manifest = common.read_json(manifest_path(role))
    path = Path(manifest["rows_path"])
    if common.sha(path) != manifest["rows_sha256"]:
        raise ValueError("Role metadata hash changed")
    rows = pd.read_parquet(path)
    if len(rows) != manifest["population_size"] or set(rows.role) != {role}:
        raise ValueError("Role population changed")
    return rows


def _freeze(relative: str, value: dict) -> dict:
    path = destination(relative)
    if path.exists():
        if common.read_json(path) != value:
            raise ValueError(f"Registered evidence changed: {relative}")
    else:
        common.write_json(path, value)
    return value


def _sources() -> dict:
    """Bind imported model, loss, loader, checkpoint and orchestration sources."""
    paths = list((common.ROOT / "src/e_jepa_ttc").rglob("*.py"))
    for folder in ("joint_measurement", "rgb_port", "rgb_port_revision", "range_transfer"):
        paths.extend((common.ROOT / "operational" / folder).glob("*.py"))
    paths += [Path(__file__), Path(__file__).with_name("losses.py"), Path(common.__file__)]
    paths += [Path(__file__).with_name("runtime_performance.py")]
    paths += [Path(__file__).with_name("sensor_runtime.py")]
    paths += [common.ROOT / "configs/rgb_port/producers.json"]
    paths += [common.ROOT / "tests/unit/test_decision_recovery_training.py"]
    paths += [common.ROOT / "tests/unit/test_decision_recovery_runtime_performance.py"]
    paths += [common.ROOT / "tests/unit/test_decision_recovery_sensor_runtime.py"]
    return {p.relative_to(common.ROOT).as_posix(): common.sha(p) for p in sorted(set(paths))}


def committed_sources(sources: dict) -> str:
    """A source hash must name committed bytes, not merely an unrelated HEAD."""
    tracked = set(
        subprocess.check_output(["git", "ls-files", "-z"], cwd=common.ROOT).decode().split("\0")
    )
    changed = set(
        subprocess.check_output(["git", "diff", "HEAD", "--name-only", "-z"], cwd=common.ROOT)
        .decode()
        .split("\0")
    )
    if set(sources) - tracked or set(sources) & changed:
        raise ValueError("Commit all bound V19 sources before training registration")
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=common.ROOT).decode().strip()


def prepare_registration() -> dict:
    """Freeze the prospective recipe without constructing a CUDA model or starting fits."""
    from operational.range_transfer.common import digest

    from .sensor_runtime import runtime_contract

    if destination("training/REGISTRATION.json").exists():
        return registered()
    main = destination("REGISTRATION.json")
    if not main.is_file():
        raise ValueError("The root V19 campaign must be registered first")
    parent = parent_path()
    receipt = common.read_json(parent.with_name("CHECKPOINT_RECEIPT.json"))
    if (
        receipt["status"] != "COMPLETE"
        or receipt["completed_updates"] != 2000
        or not receipt["complete_state"]
        or common.sha(parent) != receipt["checkpoint_sha256"]
    ):
        raise ValueError("Complete immutable B1 stage1 parent required")
    rows = role_rows("P")
    if len(rows) != 53890 or common.read_json(cache_root("P") / "COMPLETE.json")["rows"] != len(
        rows
    ):
        raise ValueError("All 53890 native P queries required")
    sources = _sources()
    source_commit = committed_sources(sources)
    value = {
        "schema": "v19_full_loss_screen_v1",
        "main_registration_sha256": common.sha(main),
        "parent_sha256": receipt["checkpoint_sha256"],
        "parent_receipt_sha256": common.sha(parent.with_name("CHECKPOINT_RECEIPT.json")),
        "architecture": "GeometricPairMeasurement",
        "parameter_policy": "FULL",
        "arms": list(MASSES),
        "band_masses": {k: list(v) for k, v in MASSES.items()},
        "mass_sha256": {
            arm: digest(
                training_mass(rows.target_ttc.to_numpy(), rows.group_id.to_numpy(), arm).tolist()
            )
            for arm in MASSES
        },
        "optimizer": recipe()["optimizer"],
        "seed": 7,
        "updates_per_arm": ENDPOINT,
        "batch_size": 32,
        "microbatch_size": 4,
        "precision": "FP32",
        "tf32": False,
        "sampler": "uniform_with_replacement_shared_seed",
        "objective": "native emitted phase plus 0.1 raw phase; only global masses differ",
        "historical_control_reused": False,
        "new_scientific_updates": 24000,
        "gpu_cap_seconds": GPU_CAP,
        "physical_update_cap": UPDATE_CAP,
        "technical_update_cap": TECHNICAL_CAP,
        "recovery_update_cap": RECOVERY_CAP,
        "diagnostic_checkpoints": [0, 2000, 6000, 12000],
        "source_commit": source_commit,
        "sources": sources,
        "source_recipe_sha256": common.sha(
            common.SOURCE_ROOT / "configs/joint_measurement/primary.json"
        ),
        "P_manifest_sha256": common.sha(manifest_path("P")),
        "sensor_complete_sha256": common.sha(cache_root("P") / "COMPLETE.json"),
        "patch_targets": list(PATCH_TARGETS),
        "runtime_performance": PERFORMANCE_POLICY,
        "sensor_runtime": runtime_contract(),
        "prior_geometry_capture_admission": {
            name: common.sha(source_run() / name)
            for name in (
                "qa/geometry_capture_v3/RESULT.json",
                "runtime/A1_loss_graph_v1/ADMISSION.json",
            )
        },
        "qa": {
            "cpu": "native L0 weights and batch transport including duplicate draws",
            "native": (
                "zero-update native phase loss versus captured loss, both with captured model: "
                "exact gradients, loss, weights, CPU/CUDA RNG for both masses; "
                "prior V18 geometry capture admission bound separately"
            ),
            "cuda": (
                "each arm continuous4, persistent2+2, resumed2+2, "
                "all state and curves exact; 24 technical updates"
            ),
        },
    }
    return _freeze("training/REGISTRATION.json", value)


def registered() -> dict:
    value = common.read_json(destination("training/REGISTRATION.json"))
    if (
        value["sources"] != _sources()
        or common.sha(parent_path()) != value["parent_sha256"]
        or common.sha(destination("REGISTRATION.json")) != value["main_registration_sha256"]
        or common.sha(manifest_path("P")) != value["P_manifest_sha256"]
        or common.sha(cache_root("P") / "COMPLETE.json") != value["sensor_complete_sha256"]
        or common.sha(common.SOURCE_ROOT / "configs/joint_measurement/primary.json")
        != value["source_recipe_sha256"]
    ):
        raise ValueError("Training registration or external inputs changed")
    if any(
        common.sha(source_run() / name) != expected
        for name, expected in value["prior_geometry_capture_admission"].items()
    ):
        raise ValueError("Historical geometry capture admission changed")
    return value


def accounting() -> dict:
    counts = {"scientific": 0, "technical": 0, "recovery_upper": 0, "pending_upper": 0}
    for path in destination("training").rglob("UPDATE_JOURNAL.json"):
        row = common.read_json(path)
        key = "technical" if "qa" in path.relative_to(common.RUN).parts else "scientific"
        counts[key] += row["completed_updates"]
        counts["recovery_upper"] += row["recovery_upper"]
        counts["pending_upper"] += row["pending_update_upper"]
    counts["physical_upper"] = sum(counts.values())
    if (
        counts["physical_upper"] > UPDATE_CAP
        or counts["technical"] > TECHNICAL_CAP
        or counts["recovery_upper"] > RECOVERY_CAP
    ):
        raise RuntimeError("V19 update budget exceeded")
    common.write_json(destination("training/ACCOUNTING.json"), counts)
    return counts


def require_resources(*, starting: bool = False) -> None:
    from operational.range_transfer.common import windows_commit_headroom

    minimum = (4 if starting else 2) * 1024**3
    commit = windows_commit_headroom()
    rss = 0
    for process in psutil.process_iter(["name", "memory_info"]):
        try:
            if "python" in (process.info["name"] or "").lower() and Path(
                process.cwd()
            ).is_relative_to(common.ROOT.parent):
                rss += process.info["memory_info"].rss
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    values = {
        "available_bytes": psutil.virtual_memory().available,
        "commit_headroom_bytes": commit,
        "project_rss_bytes": rss,
        "output_free_bytes": shutil.disk_usage(common.ROOT).free,
        "cache_free_bytes": shutil.disk_usage(cache_root("P")).free,
    }
    allowed = (
        values["available_bytes"] >= minimum
        and (commit is None or commit >= minimum)
        and rss <= 23_000_000_000
        and values["output_free_bytes"] >= 14_000_000_000
        and values["cache_free_bytes"] >= 10_000_000_000
    )
    common.write_json(destination("training/RESOURCE_LATEST.json"), {**values, "allowed": allowed})
    if not allowed:
        raise RuntimeError("RESOURCE_WAIT: V19 resource guard")


@contextmanager
def file_lock(path: Path, *, existing: bool = False) -> Iterator[None]:
    """Lock the existing V18 byte without changing it; new locks belong to V19."""
    if not existing:
        path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("r+b" if existing else "a+b") as stream:
        if not existing and stream.tell() == 0:
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == "nt":
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream, fcntl.LOCK_UN)


def guard_idle() -> None:
    own = {os.getpid(), *[p.pid for p in psutil.Process().parents()]}
    for process in psutil.process_iter(["name", "cmdline"]):
        try:
            if process.pid in own or "python" not in (process.info["name"] or "").lower():
                continue
            command = process.info["cmdline"] or []
            if "operational.joint_measurement.hourly_observe" in command:
                continue
            if Path(process.cwd()).is_relative_to(common.ROOT.parent):
                raise RuntimeError(f"COMPUTE_WAIT: project Python {process.pid} remains live")
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    raw = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], timeout=15
    ).decode()
    if any(int(line.strip()) not in own for line in raw.splitlines() if line.strip().isdigit()):
        raise RuntimeError("GPU_OWNER_WAIT: another GPU process remains live")


@contextmanager
def gpu_charge(scope: str, *, reserve: float = 900) -> Iterator[float]:
    import torch

    with ExitStack() as stack:
        stack.enter_context(
            file_lock(common.ROOT.parent / "E_JEPA_SHARED_COORDINATION/V19_GPU.lock")
        )
        stack.enter_context(file_lock(source_run() / "V18_GPU.lock", existing=True))
        guard_idle()
        require_resources(starting=not torch.cuda.is_initialized())
        path = destination("training/GPU_BUDGET.json")
        book = common.read_json(path) if path.exists() else {"cap_s": GPU_CAP, "entries": []}
        if book["cap_s"] != GPU_CAP:
            raise ValueError("V19 GPU cap changed")
        reserve = min(reserve, GPU_CAP - sum(row["charged_s"] for row in book["entries"]))
        if reserve < 60:
            raise RuntimeError("GPU_BUDGET_EXHAUSTED")
        row = {"scope": scope, "pid": os.getpid(), "charged_s": reserve, "status": "RESERVED"}
        book["entries"].append(row)
        common.write_json(path, book)
        start = time.perf_counter()
        try:
            yield start + reserve - 30
        finally:
            if torch.cuda.is_initialized():
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            row.update(charged_s=elapsed, status="SETTLED")
            common.write_json(path, book)
            if elapsed > reserve:
                raise RuntimeError("V19 GPU reservation overrun")


@contextmanager
def runtime_context(transfer: str = "legacy") -> Iterator[None]:
    """Only process-local callbacks change; every historical source stays byte-identical."""
    from operational.joint_measurement.captured_phase_loss import backward

    from .runtime_performance import batch_device_backward

    if transfer not in {"legacy", "batch"}:
        raise ValueError("Unknown transfer runtime")
    if transfer == "batch":
        backward = batch_device_backward
    replacements = (accounting, require_resources, backward, cache_root, role_rows)
    with ExitStack() as stack:
        for target, value in zip(PATCH_TARGETS, replacements, strict=True):
            stack.enter_context(patch(target, value))
        yield


def make_model() -> GeometricPairMeasurement:
    import torch

    from e_jepa_ttc.models.causal_scale_ttc import CausalScaleTTC
    from operational.joint_measurement.adaptation_depth import configure
    from operational.joint_measurement.geometry_model import GeometricPairMeasurement
    from operational.joint_measurement.train import native_config

    model = GeometricPairMeasurement(CausalScaleTTC(native_config()[0]))
    payload = torch.load(parent_path(), map_location="cpu", weights_only=False)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    configure(model, "FULL")
    return model


def session_class() -> type[BaseSession]:
    """Defer torch and historical runtime imports until a caller needs model execution."""
    import torch

    from operational.joint_measurement.persistent_adaptation import Session as Base
    from operational.joint_measurement.train import make_optimizer, setup
    from operational.range_transfer.common import digest
    from operational.rgb_port.train_producers import ProducerCheckpoint

    class Session(Base):
        def __init__(
            self, directory: Path, arm: str, masses: np.ndarray, *, synthetic: bool = False
        ) -> None:
            if not directory.resolve().is_relative_to(destination("training")):
                raise ValueError("Session writes must remain inside V19 training")
            record = registered()
            setup(7)
            self.model = make_model().cuda().train()
            self.directory, self.mode, self.masses = directory, "FULL", masses
            self.endpoint = 4 if synthetic else ENDPOINT
            self.arm = arm
            identity = {
                "fit_id": directory.name,
                "arm": arm,
                "mode": "FULL",
                "seed": 7,
                "registration_sha256": common.sha(destination("training/REGISTRATION.json")),
                "parent_sha256": record["parent_sha256"],
                "endpoint": self.endpoint,
                "population": "synthetic8" if synthetic else "P",
                "mass_sha256": digest(masses.tolist()),
                "qa_sha256": None
                if synthetic
                else common.sha(destination("training/qa/RESULT.json")),
            }
            identity["identity_sha256"] = digest(identity)
            _freeze(str((directory / "IDENTITY.json").relative_to(common.RUN)), identity)
            # Native optimizer math is reused with the frozen V19 recipe injected locally.
            with patch(
                "operational.joint_measurement.train.config",
                return_value={"optimizer": record["optimizer"]},
            ):
                self.optimizer, self.scheduler = make_optimizer(self.model, updates=ENDPOINT)
            self.sampler = torch.Generator().manual_seed(7)
            self.state = ProducerCheckpoint(directory, identity, self.endpoint)
            cursor = (
                self.state.restore(self.model, self.optimizer, self.scheduler, self.sampler)
                if self.state.pointer.exists()
                else None
            )
            self.curve, self.timings = (cursor["curve"], cursor["timings"]) if cursor else ([], [])
            self.profile = []
            self.scope_count = self.capture_count = 0
            self._graph_identity = None
            provenance = directory / "RUN_PROVENANCE.json"
            if not provenance.exists():
                common.write_json(
                    provenance,
                    {
                        "experiment_id": "V19_FULL_LOSS_SCREEN",
                        "run_name": directory.name,
                        "source_commit": record["source_commit"],
                        "registration_sha256": identity["registration_sha256"],
                        "parent_sha256": record["parent_sha256"],
                        "seed": 7,
                        "python_version": platform.python_version(),
                        "torch_version": str(torch.__version__),
                        "cuda_version": torch.version.cuda,
                        "gpu_name": torch.cuda.get_device_name(0),
                        "host": platform.node(),
                        "start_time_unix": time.time(),
                        "end_time_status": "CHECKPOINT_RECEIPT.json and RUNNER_STATE.json",
                    },
                )
            if cursor is None:
                self.save("RUNNING")

        def publish_runtime(self, sensors: object) -> None:
            super().publish_runtime(sensors)
            loss = getattr(self.model, "_v18_phase_loss", None)
            identity = (tuple(id(p) for p in self.model.parameters()), id(loss))
            if self._graph_identity is None and loss is not None:
                self._graph_identity = identity
            if loss is not None and (
                self.capture_count != 1
                or loss.capture_count != 1
                or self._graph_identity != identity
            ):
                raise ValueError(
                    "Persistent model/loss graph recaptured or parameter identity changed"
                )
            common.write_json(
                self.directory / "GRAPH_RUNTIME.json",
                {
                    "completed_updates": self.state.completed,
                    "model_captures_this_process": self.capture_count,
                    "loss_captures_this_process": loss.capture_count if loss is not None else 0,
                    "scopes_this_process": self.scope_count,
                    "same_parameter_and_loss_graph_ids": self._graph_identity == identity,
                    "transfer": sensors.counters() if hasattr(sensors, "device_batches") else {},
                },
            )

        def save(self, status: str) -> None:
            super().save(status)
            if self.state.completed in {0, 2000, 6000, 12000}:
                snapshot = (
                    self.directory / "diagnostic_checkpoints" / f"{self.state.completed:06d}.pt"
                )
                snapshot.parent.mkdir(parents=True, exist_ok=True)
                current = self.directory / "checkpoint_last.pt"
                if not snapshot.exists():
                    try:
                        os.link(current, snapshot)
                    except OSError:
                        shutil.copyfile(current, snapshot)
                elif common.sha(snapshot) != common.sha(current):
                    # Same weights/state may have a later PAUSED/COMPLETE envelope; retain first.
                    prior = torch.load(snapshot, map_location="cpu", weights_only=False)
                    if prior["completed_updates"] != self.state.completed:
                        raise ValueError("Diagnostic checkpoint cursor changed")
            parent = torch.load(parent_path(), map_location="cpu", weights_only=False)[
                "model_state_dict"
            ]
            values = {
                name: float((value.detach().cpu() - parent[name]).abs().max())
                for name, value in self.model.state_dict().items()
            }
            if not all(np.isfinite(v) for v in values.values()):
                raise ValueError("Nonfinite durable weight witness")
            if self.state.completed and not any(
                v > 0 for k, v in values.items() if k.startswith("encoder.")
            ):
                raise ValueError("FULL must update its encoder")
            common.write_json(
                self.directory / "WEIGHT_CHANGE.json",
                {
                    "completed_updates": self.state.completed,
                    "checkpoint_sha256": common.sha(self.directory / "checkpoint_last.pt"),
                    "parent_sha256": common.sha(parent_path()),
                    "maxabs": values,
                },
            )

    return Session


def qa() -> dict:
    """Run CPU transport, zero-update native parity and counted CUDA resume proofs."""
    import torch

    from e_jepa_ttc.joint_measurement.loss import training_mass as native_mass
    from operational.joint_measurement.captured_phase_loss import backward as candidate_backward
    from operational.joint_measurement.data import SensorDataset
    from operational.joint_measurement.geometry_capture import contiguous_subset
    from operational.joint_measurement.geometry_capture_qa import STATE_KEYS
    from operational.joint_measurement.geometry_train import SyntheticSensors
    from operational.joint_measurement.phase_transport_loss import backward as reference_backward
    from operational.joint_measurement.qa import exact
    from operational.joint_measurement.train import setup

    from .runtime_performance import BatchDeviceSensors, native_transfer_probe
    from .sensor_runtime import make_training_sensors

    record = registered()
    output = destination("training/qa/RESULT.json")
    if output.exists():
        value = common.read_json(output)
        _admit_qa(value, record)
        return value
    checks = {}
    performance = {}
    with file_lock(destination("training/DRIVER.lock")), runtime_context():
        rows = role_rows("P")
        checks["L0_exact_native_masses"] = np.array_equal(
            training_mass(rows.target_ttc.to_numpy(), rows.group_id.to_numpy(), "L0"),
            native_mass(rows.target_ttc.to_numpy(), rows.group_id.to_numpy(), role="P"),
        )
        sensors = make_training_sensors(qa=True)
        serial = SensorDataset("P")
        try:
            ids = torch.randint(
                len(rows), (32,), generator=torch.Generator().manual_seed(7)
            ).tolist()
            ids[1] = ids[0]
            before = torch.get_rng_state().clone()
            batch, other = sensors.load(ids), serial.batch(ids)
            checks["CPU_native_transport_duplicate_draw"] = exact(vars(batch), vars(other))
            checks["CPU_transport_preserves_rng"] = torch.equal(before, torch.get_rng_state())
            if not all(checks.values()):
                raise ValueError("CPU admission failed")
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA QA unavailable; no PASS receipt written")
            with gpu_charge("V19_native_zero_update_parity"):
                batches = [
                    contiguous_subset(batch, list(range(i, i + 4))).to(torch.device("cuda"))
                    for i in range(0, 32, 4)
                ]
                for arm in MASSES:
                    weights = torch.from_numpy(
                        training_mass(rows.target_ttc.to_numpy(), rows.group_id.to_numpy(), arm)[
                            ids
                        ]
                    ).float()
                    observed = []
                    for fn in (reference_backward, candidate_backward):
                        setup(7)
                        model = make_model().cuda().train()
                        initial = {
                            k: v.detach().cpu().clone() for k, v in model.state_dict().items()
                        }
                        model.zero_grad(set_to_none=True)
                        loss = fn(model, batches, weights, len(rows))
                        observed.append(
                            {
                                "loss": loss,
                                "gradients": {
                                    k: p.grad.detach().cpu().clone()
                                    for k, p in model.named_parameters()
                                    if p.grad is not None
                                },
                                "cpu_rng": torch.get_rng_state(),
                                "cuda_rng": torch.cuda.get_rng_state_all(),
                            }
                        )
                        checks[f"{arm}_zero_update_weights_{len(observed)}"] = exact(
                            initial, {k: v.detach().cpu() for k, v in model.state_dict().items()}
                        )
                        del model
                        gc.collect()
                        torch.cuda.empty_cache()
                    checks[f"{arm}_native_loss_gradient_rng_exact"] = exact(*observed)
                # Match production pinned B32 transport; original values and IDs are unchanged.
                from dataclasses import fields

                for field in fields(batch):
                    value = getattr(batch, field.name)
                    if isinstance(value, torch.Tensor):
                        setattr(batch, field.name, value.pin_memory())
                for arm in MASSES:
                    weights = torch.from_numpy(
                        training_mass(rows.target_ttc.to_numpy(), rows.group_id.to_numpy(), arm)[
                            ids
                        ]
                    ).float()
                    performance[arm] = native_transfer_probe(batch, weights, len(rows), make_model)
            if not all(checks.values()):
                raise ValueError("Native zero-update parity failed")
        finally:
            sensors.shutdown()
            serial.close()
        transfer = (
            "batch"
            if all(p["selected_transfer"] == "batch" for p in performance.values())
            else "legacy"
        )
        session_type = session_class()
        for arm in MASSES:
            synthetic = SyntheticSensors()
            masses = training_mass(
                synthetic.batch(list(range(8))).target_ttc_s.numpy(),
                np.array(["synthetic"] * 8),
                arm,
            )
            if transfer == "batch":
                synthetic = BatchDeviceSensors(synthetic)
            payloads = []
            for trajectory in ("continuous", "persistent", "resumed"):
                path = destination(f"training/qa/{arm}/{trajectory}")
                if (path / "UPDATE_JOURNAL.json").exists():
                    raise ValueError("Partial QA exists; do not silently repeat physical updates")
                with (
                    gpu_charge(f"V19_QA_{arm}_{trajectory}") as deadline,
                    runtime_context(transfer),
                ):
                    session = session_type(path, arm, masses, synthetic=True)
                    if trajectory == "continuous":
                        session.advance(synthetic, limit=4, deadline=deadline)
                    else:
                        session.advance(synthetic, limit=2, deadline=deadline)
                        if trajectory == "resumed":
                            del session
                            gc.collect()
                            torch.cuda.empty_cache()
                            session = session_type(path, arm, masses, synthetic=True)
                        session.advance(synthetic, limit=4, deadline=deadline)
                    if session.state.completed != 4:
                        raise RuntimeError("QA scope ended before all four updates")
                    if trajectory == "persistent":
                        checks[f"{arm}_persistent_single_capture"] = session.capture_count == 1
                        checks[f"{arm}_persistent_single_loss_capture"] = (
                            session.model._v18_phase_loss.capture_count == 1
                        )
                    del session
                    gc.collect()
                    torch.cuda.empty_cache()
                payloads.append(
                    torch.load(path / "checkpoint_last.pt", map_location="cpu", weights_only=False)
                )
            for name, other in zip(("persistent", "resumed"), payloads[1:], strict=True):
                checks[f"{arm}_{name}_complete_state"] = all(
                    exact(payloads[0][key], other[key]) for key in STATE_KEYS
                )
                checks[f"{arm}_{name}_curve"] = exact(
                    payloads[0]["cursor"]["curve"], other["cursor"]["curve"]
                )
            synthetic.close()
        value = {
            "status": "PASS" if all(checks.values()) else "FAIL",
            "checks": checks,
            "registration_sha256": common.sha(destination("training/REGISTRATION.json")),
            "accounting": accounting(),
            "physical_technical_updates": 24,
            "scientific_updates": 0,
            "runtime": {"selected_transfer": transfer, "per_arm": performance},
            "source_commit": record["source_commit"],
        }
        common.write_json(output, value)
        if value["status"] != "PASS":
            raise ValueError("V19 exact resume QA failed")
        _admit_qa(value, record)
        return value


def validate_qa_proof(proof: dict, record: dict, registration_sha: str) -> str:
    """Recompute runtime selection and require the complete exact-state QA contract."""
    from .runtime_performance import select_transfer_mode

    expected = {
        "L0_exact_native_masses",
        "CPU_native_transport_duplicate_draw",
        "CPU_transport_preserves_rng",
        *{
            f"{arm}_{name}"
            for arm in MASSES
            for name in (
                "zero_update_weights_1",
                "zero_update_weights_2",
                "native_loss_gradient_rng_exact",
                "persistent_single_capture",
                "persistent_single_loss_capture",
                "persistent_complete_state",
                "persistent_curve",
                "resumed_complete_state",
                "resumed_curve",
            )
        },
    }
    if (
        proof["status"] != "PASS"
        or proof["registration_sha256"] != registration_sha
        or proof["source_commit"] != record["source_commit"]
        or set(proof["checks"]) != expected
        or not all(value is True for value in proof["checks"].values())
        or proof["physical_technical_updates"] != 24
        or proof["scientific_updates"] != 0
        or proof["accounting"]["technical"] != 24
        or proof["accounting"]["scientific"] != 0
        or proof["accounting"]["recovery_upper"] != 0
        or proof["accounting"]["pending_upper"] != 0
    ):
        raise ValueError("Complete bound exact QA checks and 24 technical updates required")
    per_arm = proof["runtime"]["per_arm"]
    if set(per_arm) != set(MASSES):
        raise ValueError("Both arms require native transfer performance evidence")
    selections = []
    for item in per_arm.values():
        if (
            item["per_trial_exact"] != [True] * PERFORMANCE_POLICY["native_trials"]
            or item["weights_unchanged"] is not True
        ):
            raise ValueError("Native transfer parity or unchanged weight proof failed")
        recomputed = select_transfer_mode(
            item["trials"], exact_candidate=True, device_total_bytes=item["device_total_bytes"]
        )
        if any(item[key] != value for key, value in recomputed.items()):
            raise ValueError("Saved native transfer performance decision changed")
        selections.append(recomputed["selected_transfer"])
    selected = "batch" if all(value == "batch" for value in selections) else "legacy"
    if proof["runtime"]["selected_transfer"] != selected:
        raise ValueError("Runtime selection does not match the registered paired evidence")
    return selected


def _admit_qa(proof: dict, record: dict) -> dict:
    registration_sha = common.sha(destination("training/REGISTRATION.json"))
    selected = validate_qa_proof(proof, record, registration_sha)
    return _freeze(
        "training/ADMISSION.json",
        {
            "status": "ADMITTED",
            "registration_sha256": registration_sha,
            "qa_sha256": common.sha(destination("training/qa/RESULT.json")),
            "source_commit": record["source_commit"],
            "selected_transfer": selected,
        },
    )


def run(arms: tuple[str, ...] = ("L0", "LW"), seed: int = 7) -> dict:
    """Run fixed endpoints sequentially; no historical writer or counter is reused."""
    import torch

    from operational.range_transfer.common import digest

    from .runtime_performance import BatchDeviceSensors
    from .sensor_runtime import make_training_sensors

    if seed != 7 or not arms or len(set(arms)) != len(arms) or any(a not in MASSES for a in arms):
        raise ValueError("Only registered unique seed7 L0/LW arms are admitted")
    record = registered()
    proof = common.read_json(destination("training/qa/RESULT.json"))
    transfer = validate_qa_proof(
        proof, record, common.sha(destination("training/REGISTRATION.json"))
    )
    admission = common.read_json(destination("training/ADMISSION.json"))
    if admission != {
        "status": "ADMITTED",
        "registration_sha256": proof["registration_sha256"],
        "qa_sha256": common.sha(destination("training/qa/RESULT.json")),
        "source_commit": record["source_commit"],
        "selected_transfer": transfer,
    }:
        raise ValueError("Frozen QA admission changed")
    session_type = session_class()
    receipts = {}
    with file_lock(destination("training/DRIVER.lock")), runtime_context(transfer):
        for arm in arms:
            directory = destination(f"training/fits/FULL-{arm}-S7")
            sensors = make_training_sensors()
            if transfer == "batch":
                sensors = BatchDeviceSensors(sensors)
            session = None
            try:
                masses = training_mass(
                    sensors.rows.target_ttc.to_numpy(), sensors.rows.group_id.to_numpy(), arm
                )
                if digest(masses.tolist()) != record["mass_sha256"][arm]:
                    raise ValueError("Registered population weights changed")
                while session is None or session.state.completed < ENDPOINT:
                    with gpu_charge(f"V19_FULL_{arm}_S7") as deadline:
                        if session is None:
                            session = session_type(directory, arm, masses)
                        before = session.state.completed
                        counts = accounting()
                        remaining = UPDATE_CAP - counts["physical_upper"]
                        limit = min(ENDPOINT, before + remaining)
                        if remaining <= 0:
                            raise RuntimeError("V19 physical update budget exhausted")
                        session.advance(sensors, limit=limit, deadline=deadline)
                        if session.state.completed == before and before != ENDPOINT:
                            raise RuntimeError(
                                "No progress; inspect resource guards before resuming"
                            )
                    common.write_json(
                        destination("training/RUNNER_STATE.json"),
                        {
                            "status": "TRAINING",
                            "arm": arm,
                            "completed_updates": session.state.completed,
                            "pid": os.getpid(),
                        },
                    )
                receipts[arm] = common.read_json(directory / "CHECKPOINT_RECEIPT.json")
            finally:
                sensors.shutdown()
                del session
                gc.collect()
                torch.cuda.empty_cache()
        value = {
            "status": "ENDPOINTS_COMPLETE"
            if set(receipts) == set(MASSES)
            else "SELECTED_ENDPOINTS_COMPLETE",
            "requested_arms": list(arms),
            "receipts": receipts,
            "accounting": accounting(),
            "evaluation_status": "PENDING_COMPLETE_P_H",
            "system_promotion": False,
        }
        common.write_json(destination("training/RUNNER_STATE.json"), value)
        return value


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("register", "qa", "run"))
    args = parser.parse_args()
    print(
        json.dumps(
            {"register": prepare_registration, "qa": qa, "run": run}[args.action](), default=str
        )
    )

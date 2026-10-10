"""Durable fixed-endpoint training for matched RGB-PORT causal-scale producers."""

from __future__ import annotations

import argparse
import ctypes
import errno
import importlib
import json
import os
import platform
import random
import shutil
import time
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, cast

import numpy as np
import torch
from torch import Tensor, nn

from e_jepa_ttc.data.object_event_v4 import (
    ObjectEventV4Batch,
    box_geometry_targets,
)
from e_jepa_ttc.distillation.dinov3_relational import local_relational_distillation_loss
from e_jepa_ttc.losses.causal_scale_ttc import (
    CausalScaleTTCLossConfig,
    causal_scale_ttc_loss,
)
from e_jepa_ttc.models.causal_scale_ttc import CausalScaleTTC, CausalScaleTTCConfig
from e_jepa_ttc.reproducibility import seed_everything
from e_jepa_ttc.simplex_t.expert_phase import expert_phase_from_ttc
from operational.rgb_port.accounting import (
    TRANSIENT_WINERRORS,
    atomic_write_json,
    read_json_shared,
)
from operational.rgb_port.recipe import (
    ProducerRecipe,
    canonical_sha256,
    file_sha256,
    resolved_recipe,
    verify_authority,
)

if TYPE_CHECKING:
    from operational.rgb_port_revision.cache import GroupRowCache
    from operational.rgb_port_revision.loss import EffectiveBatch


class ProducerSource(Protocol):
    """Random-access P-role source; targets remain outside model inputs."""

    population_size: int
    identity: Mapping[str, Any]
    frame_counts: Sequence[int]
    input_span_us: Sequence[int]

    def batch(
        self, indices: Sequence[int], modality: str
    ) -> ObjectEventV4Batch | EffectiveBatch: ...


def _role_input_spans(index_path: Path, tokens: Sequence[str]) -> list[int]:
    """Read each role row's complete input span from the original sensor-clock index."""
    with np.load(index_path, allow_pickle=False) as archive:
        if "tokens" not in archive or "windows_us" not in archive:
            raise ValueError("TRAIN40 index lacks token/window identity")
        index_tokens = np.asarray(archive["tokens"]).astype(str)
        windows = np.asarray(archive["windows_us"])
    if (
        windows.dtype.kind not in {"i", "u"}
        or windows.shape != (len(index_tokens), 3, 2)
        or len(set(index_tokens.tolist())) != len(index_tokens)
    ):
        raise ValueError("TRAIN40 sensor-clock windows differ from [N,3,2] int64 identity")
    starts, ends = windows[:, :, 0], windows[:, :, 1]
    if (
        np.any(ends <= starts)
        or np.any(np.diff(starts, axis=1) < 0)
        or np.any(np.diff(ends, axis=1) < 0)
        or np.any(starts[:, 1:] < ends[:, :-1])
    ):
        raise ValueError("TRAIN40 windows do not preserve one monotonic sensor clock")
    spans = ends[:, -1] - starts[:, 0]
    if np.any(spans <= 0) or np.any(spans > 650_000):
        raise ValueError("TRAIN40 input span must be in (0,650000] microseconds")
    lookup = {token: index for index, token in enumerate(index_tokens.tolist())}
    missing = [token for token in tokens if token not in lookup]
    if missing:
        raise ValueError(f"role tokens absent from TRAIN40 index: {missing[:3]}")
    return [int(spans[lookup[token]]) for token in tokens]


class EventProducerSource:
    """Role-manifest token view over the audited lossless TRAIN40 event cache."""

    def __init__(self, manifest_path: Path, event_source_root: Path) -> None:
        import pandas as pd

        from operational.train40_system.models import Inputs

        manifest_path = manifest_path.resolve(strict=True)
        event_source_root = event_source_root.resolve(strict=True)
        manifest = read_json_shared(manifest_path)
        if manifest.get("status") != "COMPLETE" or manifest.get("role") not in {"P", "H", "V"}:
            raise ValueError("Event producer source requires a complete P/H/V manifest")
        rows_path = Path(str(manifest["rows_path"])).resolve(strict=True)
        if file_sha256(rows_path) != manifest.get("rows_sha256"):
            raise ValueError("P rows changed before event producer construction")
        rows = pd.read_parquet(rows_path, columns=["ordinal", "sample_token"])
        self._tokens = rows["sample_token"].astype(str).tolist()
        self.population_size = len(self._tokens)
        self.frame_counts = [3] * self.population_size
        if self.population_size != int(manifest["population_size"]):
            raise ValueError("P event ordinal population differs")
        required = [
            event_source_root / "PREPARE_FREEZE.json",
            event_source_root / "TRAIN40_ROWS.parquet",
            event_source_root / "TRAIN40_INDEX.npz",
            event_source_root / "TEACHER_FREEZE.json",
            event_source_root / "TEACHER_MANIFEST.json",
        ]
        if any(not path.is_file() for path in required):
            raise FileNotFoundError("Audited TRAIN40 event cache metadata is incomplete")
        self._inputs = Inputs(event_source_root)
        # Retain the current and next logical group without accumulating an 8 GiB
        # historical cache that can exhaust Windows commit before the RSS limit.
        # This changes only eviction; shard contents and sample order are unchanged.
        self._inputs.limit = 2 * 1024**3
        token_to_index = {
            str(row["sample_token"]): index for index, row in enumerate(self._inputs.metadata)
        }
        if len(token_to_index) != len(self._inputs.metadata):
            raise ValueError("TRAIN40 event source sample tokens are not unique")
        missing = [token for token in self._tokens if token not in token_to_index]
        if missing:
            raise ValueError(f"P tokens absent from TRAIN40 event source: {missing[:3]}")
        self._event_indices = [token_to_index[token] for token in self._tokens]
        self.input_span_us = _role_input_spans(required[2], self._tokens)
        self.identity = {
            "schema": "rgb_port_event_producer_source_v1",
            "role": manifest["role"],
            "role_manifest_path": str(manifest_path),
            "role_manifest_sha256": file_sha256(manifest_path),
            "rows_sha256": manifest["rows_sha256"],
            "split_assignment_sha256": manifest["split_assignment_sha256"],
            "population_size": self.population_size,
            "decoded_event_cache_limit_bytes": self._inputs.limit,
            "event_source_root": str(event_source_root),
            "prepare_freeze_sha256": file_sha256(required[0]),
            "train40_rows_sha256": file_sha256(required[1]),
            "train40_index_sha256": file_sha256(required[2]),
            "input_span_clock": "TRAIN40_INDEX.windows_us sensor microseconds",
            "input_span_definition": "windows_us[-1,1]-windows_us[0,0]",
            "dino_teacher_freeze_sha256": file_sha256(required[3]),
            "dino_targets_sha256": file_sha256(required[4]),
            "target_anchor": "query_time_us",
        }

    _audit_row_cache: GroupRowCache | None = None

    def batch(self, indices: Sequence[int], modality: str) -> ObjectEventV4Batch:
        if modality != "event" or not indices:
            raise ValueError("EventProducerSource only serves nonempty event batches")
        local = [int(index) for index in indices]
        cache = getattr(self, "_audit_row_cache", None)
        if cache is not None:
            return cache.batch(local, modality)
        batch = self._inputs.batch([self._event_indices[index] for index in local])
        expected = [self._tokens[index] for index in local]
        if batch.sample_tokens != expected:
            raise ValueError("P event source token order differs")
        return batch

    def close(self) -> None:
        self._inputs.close()


def make_event_producer_source(
    manifest_path: str | Path, *, event_source_root: str | Path
) -> EventProducerSource:
    """Build a frozen role view; the audited cache root must be explicit."""
    return EventProducerSource(Path(manifest_path), Path(event_source_root))


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    atomic_write_json(path, value)


def _durable_replace(source: Path, target: Path, *, retries: int = 10) -> None:
    """Replace a file despite transient Windows sharing from live monitors."""
    for attempt in range(retries):
        try:
            os.replace(source, target)
            return
        except OSError as error:
            if (
                getattr(error, "winerror", None) not in TRANSIENT_WINERRORS
                or attempt + 1 == retries
            ):
                raise
            time.sleep(min(0.025 * 2**attempt, 0.75))


def _safe_unlink(path: Path, *, retries: int = 8) -> None:
    for attempt in range(retries):
        try:
            path.unlink(missing_ok=True)
            return
        except OSError as error:
            if (
                getattr(error, "winerror", None) not in TRANSIENT_WINERRORS
                or attempt + 1 == retries
            ):
                return
            time.sleep(min(0.025 * 2**attempt, 0.5))


def _transient_input_error(error: OSError) -> bool:
    return isinstance(error, FileNotFoundError) or (
        getattr(error, "winerror", None) in TRANSIENT_WINERRORS or error.errno == errno.ENOTCONN
    )


def build_producer(recipe: ProducerRecipe) -> CausalScaleTTC:
    """Construct a fresh seed-7 model; no event checkpoint adaptation is accepted."""
    seed_everything(recipe.seed, deterministic=True)
    model = CausalScaleTTC(CausalScaleTTCConfig(**recipe.model_config))
    if not all(parameter.requires_grad for parameter in model.parameters()):
        raise ValueError("Every producer parameter must be trainable")
    return model


def producer_features(
    model: CausalScaleTTC,
    inputs: Tensor,
    delta_t_s: Tensor,
    *,
    anchor_us: Tensor | None = None,
    available_us: Tensor | None = None,
) -> dict[str, Tensor]:
    """Emit the common label-free endpoint contract consumed by PAIR and heads."""
    if inputs.dtype != torch.float32 or delta_t_s.dtype != torch.float32:
        raise ValueError("Producer feature extraction requires FP32 sensor/time inputs")
    if model.config.modality == "rgb" and (
        not bool(torch.isfinite(inputs).all())
        or bool((inputs < 0).any())
        or bool((inputs > 1).any())
    ):
        raise ValueError("RGB producer inputs must be raw [0,1], never ImageNet-normalized")
    measured_delta = _pair_intervals(delta_t_s, batch_size=len(inputs), steps=int(inputs.shape[1]))
    output = model(inputs, measured_delta, return_dense_features=False)
    ttc = output.ttc_mean_seconds.float()
    phase_np = expert_phase_from_ttc(ttc.detach().cpu().numpy())
    phase = torch.from_numpy(phase_np).to(device=ttc.device, dtype=torch.float32)
    margin = torch.minimum(
        output.log_height_ratio[:, -1].abs().float() / 0.002,
        output.sensor_support[:, -1].float() / 0.0001,
    )
    result = {
        "token128": output.pair_tokens[:, -1].float(),
        "prediction_ttc": ttc,
        "point_phase": phase,
        "flow": output.diagnostics["transport_flow_magnitude"][:, -1].float(),
        "margin": margin,
        "log_variance": output.ttc_log_variance.float(),
        "support": output.sensor_support.float(),
        "known": output.known_mask,
    }
    batch = len(inputs)
    for name, value in (("anchor_us", anchor_us), ("available_us", available_us)):
        if value is not None:
            if value.shape != (batch,) or value.dtype != torch.int64:
                raise ValueError(f"{name} must be int64[B]")
            result[name] = value
    return result


class ProducerCheckpoint:
    """Atomic optimizer-boundary state with queue-compatible receipts."""

    schema = "rgb_port_producer_checkpoint_v1"

    def __init__(self, directory: Path, identity: Mapping[str, Any], updates: int) -> None:
        self.directory = directory
        self.path = directory / "checkpoint_last.pt"
        self.pointer = directory / "CHECKPOINT_POINTER.json"
        self.versions = directory / "checkpoint_versions"
        self.receipt = directory / "CHECKPOINT_RECEIPT.json"
        self.journal = directory / "UPDATE_JOURNAL.json"
        self.identity = dict(identity)
        self.identity_sha256 = cast(str, identity["identity_sha256"])
        self.updates = int(updates)
        self.completed = 0
        self.durable = 0
        self.recovery_upper = 0
        self.pending = False
        directory.mkdir(parents=True, exist_ok=True)
        self.versions.mkdir(parents=True, exist_ok=True)
        if not self.journal.exists():
            self._write_journal()

    def _write_journal(self) -> None:
        _atomic_json(
            self.journal,
            {
                "schema": "rgb_port_update_journal_v1",
                "identity_sha256": self.identity_sha256,
                "completed_updates": self.completed,
                "durable_updates": self.durable,
                "recovery_upper": self.recovery_upper,
                "pending_update_upper": int(self.pending),
            },
        )

    def begin(self) -> None:
        if self.pending or self.completed >= self.updates:
            raise RuntimeError("Invalid producer optimizer boundary")
        self.pending = True
        self._write_journal()

    def commit(self) -> None:
        if not self.pending:
            raise RuntimeError("No producer update is pending")
        self.completed += 1
        self.pending = False
        self._write_journal()

    def save(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        generator: torch.Generator,
        cursor: Mapping[str, Any],
        *,
        status: str,
    ) -> None:
        if self.pending:
            raise RuntimeError("Cannot save inside an optimizer update")
        model_on_cuda = any(parameter.device.type == "cuda" for parameter in model.parameters())
        payload = {
            "schema": self.schema,
            "code_migration_sha256": (
                file_sha256(self.directory.parent.parent / "AUDIT_CODE_MIGRATION.json")
                if (self.directory.parent.parent / "AUDIT_CODE_MIGRATION.json").is_file()
                else None
            ),
            "identity": self.identity,
            "identity_sha256": self.identity_sha256,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "sampler_generator_state": generator.get_state(),
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state_all": torch.cuda.get_rng_state_all() if model_on_cuda else None,
            "numpy_random_state": np.random.get_state(),
            "python_random_state": random.getstate(),
            "cursor": dict(cursor),
            "completed_updates": self.completed,
            "recovery_upper": self.recovery_upper,
            "accumulation_index": 0,
            "status": status,
        }
        version = self.versions / f"checkpoint_{self.completed:06d}.pt"
        pending = self.versions / f".{version.name}.{os.getpid()}.pending"
        torch.save(payload, pending)
        with pending.open("rb+") as stream:
            os.fsync(stream.fileno())
        _durable_replace(pending, version)
        checkpoint_sha256 = file_sha256(version)
        _atomic_json(
            self.pointer,
            {
                "schema": "rgb_port_checkpoint_pointer_v1",
                "identity_sha256": self.identity_sha256,
                "completed_updates": self.completed,
                "version": version.name,
                "checkpoint_sha256": checkpoint_sha256,
            },
        )
        # Queue consumers require checkpoint_last.pt. A hard link gives them stable
        # bytes without duplicating a potentially large checkpoint.
        alias = self.directory / f".checkpoint_last.{os.getpid()}.pending.pt"
        _safe_unlink(alias)
        try:
            os.link(version, alias)
        except OSError:
            shutil.copyfile(version, alias)
            with alias.open("rb+") as stream:
                os.fsync(stream.fileno())
        _durable_replace(alias, self.path)
        self.durable = self.completed
        self._write_journal()
        _atomic_json(
            self.receipt,
            {
                "schema": "rgb_port_checkpoint_receipt_v1",
                "fit_id": self.identity["fit_id"],
                "status": status,
                "completed_updates": self.completed,
                "scientific_endpoint": status == "COMPLETE" and self.completed == self.updates,
                "identity_sha256": self.identity_sha256,
                "checkpoint_path": str(self.path.resolve()),
                "checkpoint_sha256": checkpoint_sha256,
                "complete_state": True,
                "accumulation_index": 0,
                "recovery_upper": self.recovery_upper,
            },
        )
        keep = {version.name}
        prior = sorted(self.versions.glob("checkpoint_*.pt"), reverse=True)
        keep.update(item.name for item in prior[:2])
        for stale in prior:
            if stale.name not in keep:
                _safe_unlink(stale)

    def _recover_version(self) -> tuple[Path, dict[str, Any]]:
        pointer = read_json_shared(self.pointer)
        if (
            pointer.get("schema") != "rgb_port_checkpoint_pointer_v1"
            or pointer.get("identity_sha256") != self.identity_sha256
        ):
            raise ValueError("Producer checkpoint pointer identity differs")
        version = self.versions / str(pointer["version"])
        if not version.is_file() or pointer.get("checkpoint_sha256") != file_sha256(version):
            raise ValueError("Producer checkpoint transaction is incomplete")
        return version, pointer

    def restore(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        generator: torch.Generator,
    ) -> dict[str, Any]:
        version, pointer = self._recover_version()
        payload = torch.load(version, map_location="cpu", weights_only=False)
        migration = payload.get("code_migration_sha256")
        if migration is not None:
            from operational.rgb_port_revision.migration import validate

            run = self.directory.parent.parent
            validate(run)
            if file_sha256(run / "AUDIT_CODE_MIGRATION.json") != migration:
                raise ValueError("Checkpoint code migration differs")
        if (
            payload.get("schema") != self.schema
            or payload.get("identity") != self.identity
            or payload.get("identity_sha256") != self.identity_sha256
            or int(payload.get("accumulation_index", -1)) != 0
        ):
            raise ValueError("Producer checkpoint identity/state differs")
        # Receipt/alias may lag the atomic pointer after a crash. The immutable
        # pointed version is authoritative and repairs both without losing work.
        receipt = read_json_shared(self.receipt) if self.receipt.is_file() else {}
        journal = read_json_shared(self.journal)
        if int(pointer["completed_updates"]) != int(payload["completed_updates"]):
            raise ValueError("Producer checkpoint pointer cursor differs")
        self.completed = int(payload["completed_updates"])
        self.durable = self.completed
        self.recovery_upper = (
            max(int(payload["recovery_upper"]), int(journal["recovery_upper"]))
            + max(0, int(journal["completed_updates"]) - self.completed)
            + int(journal["pending_update_upper"])
        )
        if self.recovery_upper > 11_092:
            raise RuntimeError("RGB-PORT recovery reserve exceeded")
        model.load_state_dict(payload["model_state_dict"], strict=True)
        optimizer.load_state_dict(payload["optimizer_state_dict"])
        scheduler.load_state_dict(payload["scheduler_state_dict"])
        generator.set_state(payload["sampler_generator_state"])
        torch.set_rng_state(payload["torch_rng_state"])
        if payload["cuda_rng_state_all"] is not None:
            if not torch.cuda.is_available():
                raise ValueError("CUDA checkpoint cannot resume on a CPU-only host")
            torch.cuda.set_rng_state_all(payload["cuda_rng_state_all"])
        np.random.set_state(payload["numpy_random_state"])
        random.setstate(payload["python_random_state"])
        self.pending = False
        self._write_journal()
        expected_sha = cast(str, pointer["checkpoint_sha256"])
        if not self.path.is_file() or file_sha256(self.path) != expected_sha:
            alias = self.directory / f".checkpoint_last.{os.getpid()}.repair.pt"
            _safe_unlink(alias)
            try:
                os.link(version, alias)
            except OSError:
                shutil.copyfile(version, alias)
            _durable_replace(alias, self.path)
        if (
            receipt.get("checkpoint_sha256") != expected_sha
            or int(receipt.get("completed_updates", -1)) != self.completed
        ):
            _atomic_json(
                self.receipt,
                {
                    "schema": "rgb_port_checkpoint_receipt_v1",
                    "fit_id": self.identity["fit_id"],
                    "status": payload["status"],
                    "completed_updates": self.completed,
                    "scientific_endpoint": payload["status"] == "COMPLETE"
                    and self.completed == self.updates,
                    "identity_sha256": self.identity_sha256,
                    "checkpoint_path": str(self.path.resolve()),
                    "checkpoint_sha256": expected_sha,
                    "complete_state": True,
                    "accumulation_index": 0,
                    "recovery_upper": self.recovery_upper,
                },
            )
        return cast(dict[str, Any], payload["cursor"])


def _windows_commit_headroom() -> int | None:
    if os.name != "nt":
        return None

    class Performance(ctypes.Structure):
        _fields_ = [
            ("cb", ctypes.c_ulong),
            *[
                (name, ctypes.c_size_t)
                for name in (
                    "CommitTotal",
                    "CommitLimit",
                    "CommitPeak",
                    "PhysicalTotal",
                    "PhysicalAvailable",
                    "SystemCache",
                    "KernelTotal",
                    "KernelPaged",
                    "KernelNonpaged",
                    "PageSize",
                )
            ],
            ("HandleCount", ctypes.c_ulong),
            ("ProcessCount", ctypes.c_ulong),
            ("ThreadCount", ctypes.c_ulong),
        ]

    counter = Performance()
    counter.cb = ctypes.sizeof(counter)
    if not ctypes.windll.psapi.GetPerformanceInfo(ctypes.byref(counter), counter.cb):
        raise OSError("Windows commit counters unavailable")
    return int((counter.CommitLimit - counter.CommitTotal) * counter.PageSize)


def _resource_guard(run_root: Path, *, checkpoint_reservation: int) -> tuple[bool, dict[str, Any]]:
    import psutil

    repository = Path(__file__).resolve().parents[2]
    project_markers = (str(repository).lower(), repository.name.lower(), "operational.rgb_port")
    aggregate = 0
    project_processes: list[dict[str, Any]] = []
    for process in psutil.process_iter(["pid", "cmdline", "memory_info"]):
        try:
            command = " ".join(process.info.get("cmdline") or []).lower()
            if any(marker in command for marker in project_markers) and "pytest" not in command:
                rss = int(process.info["memory_info"].rss)
                aggregate += rss
                project_processes.append({"pid": process.pid, "rss": rss})
        except (psutil.AccessDenied, psutil.NoSuchProcess, OSError):
            continue
    commit = _windows_commit_headroom()
    disk_free = int(psutil.disk_usage(str(run_root.resolve().anchor)).free)
    values = {
        "host_available_bytes": int(psutil.virtual_memory().available),
        "project_tree_rss_bytes": aggregate,
        "project_processes": project_processes,
        "windows_commit_headroom_bytes": commit,
        "disk_free_bytes": disk_free,
        "checkpoint_reservation_bytes": checkpoint_reservation,
        "disk_free_after_reservation_bytes": disk_free - checkpoint_reservation,
    }
    allowed = (
        values["host_available_bytes"] >= 2 * 1024**3
        and values["project_tree_rss_bytes"] <= 23_000_000_000
        and (commit is None or commit >= 3 * 1024**3)
        and values["disk_free_after_reservation_bytes"] >= 10_000_000_000
    )
    return allowed, values


def _fast_resource_guard(
    run_root: Path, *, checkpoint_reservation: int
) -> tuple[bool, dict[str, Any]]:
    """Cheap per-boundary guard; aggregate process scans run less often."""
    import psutil

    available = int(psutil.virtual_memory().available)
    disk = int(psutil.disk_usage(str(run_root.resolve().anchor)).free)
    commit = _windows_commit_headroom()
    values = {
        "host_available_bytes": available,
        "windows_commit_headroom_bytes": commit,
        "disk_free_bytes": disk,
        "checkpoint_reservation_bytes": checkpoint_reservation,
        "disk_free_after_reservation_bytes": disk - checkpoint_reservation,
    }
    return (
        available >= 2 * 1024**3
        and (commit is None or commit >= 3 * 1024**3)
        and disk - checkpoint_reservation >= 10_000_000_000,
        values,
    )


def _global_recovery_upper(run_root: Path) -> int:
    total = 0
    for journal in (run_root / "fits").glob("*/UPDATE_JOURNAL.json"):
        value = read_json_shared(journal)
        total += int(value.get("recovery_upper", 0)) + int(value.get("pending_update_upper", 0))
    return total


def _assert_recovery_budget(run_root: Path, *, prospective_pending: int) -> None:
    if _global_recovery_upper(run_root) + prospective_pending > 11_092:
        raise RuntimeError("RGB-PORT global recovery reserve exceeded")


def _homogeneous_microbatches(
    source: ProducerSource, indices: Sequence[int], maximum: int
) -> list[list[int]]:
    """Group T2/T3 without changing one effective-batch optimizer boundary."""
    from operational.rgb_port_revision.loss import EffectiveSource

    if isinstance(source, EffectiveSource):
        return [list(indices)]
    counts = getattr(source, "frame_counts", None)
    if counts is None:
        return [list(indices[start : start + maximum]) for start in range(0, len(indices), maximum)]
    groups: dict[int, list[int]] = {}
    for index in indices:
        groups.setdefault(int(counts[index]), []).append(index)
    return [
        group[start : start + maximum]
        for _, group in sorted(groups.items())
        for start in range(0, len(group), maximum)
    ]


def _epoch_order(source: ProducerSource, generator: torch.Generator) -> Tensor:
    """Shuffle cache-local groups and rows with one saved RNG stream."""
    supplied = getattr(source, "sampling_groups", None)
    groups = (
        tuple(tuple(map(int, group)) for group in supplied)
        if supplied is not None
        else tuple(
            tuple(range(start, min(start + 256, source.population_size)))
            for start in range(0, source.population_size, 256)
        )
    )
    flattened = [index for group in groups for index in group]
    if sorted(flattened) != list(range(source.population_size)):
        raise ValueError("Producer sampling groups must partition P exactly")
    order: list[int] = []
    for group_index in torch.randperm(len(groups), generator=generator).tolist():
        group = groups[group_index]
        row_order = torch.randperm(len(group), generator=generator).tolist()
        order.extend(group[index] for index in row_order)
    return torch.tensor(order, dtype=torch.int64)


def _to_device(
    batch: ObjectEventV4Batch | EffectiveBatch, device: torch.device
) -> ObjectEventV4Batch | EffectiveBatch:
    return batch.to(device)


def _validate_batch(batch: ObjectEventV4Batch | EffectiveBatch, recipe: ProducerRecipe) -> None:
    from operational.rgb_port_revision.loss import EffectiveBatch

    if isinstance(batch, EffectiveBatch):
        for part in batch.parts:
            _validate_batch(part, recipe)
        return
    events = cast(Tensor, batch.events)
    if events.ndim != 5 or events.shape[2] != int(recipe.model_config["in_channels"]):
        raise ValueError("Producer batch sensor shape differs from recipe")
    if recipe.modality == "rgb" and (
        events.dtype != torch.float32
        or not bool(torch.isfinite(events).all())
        or bool((events < 0).any())
        or bool((events > 1).any())
    ):
        raise ValueError("RGB model path requires raw unit-interval FP32 frames")
    if batch.dinov3_relation_targets is None or batch.dinov3_relation_valid is None:
        raise ValueError("Both matched modalities require frozen P-only DINO targets")


def _pair_intervals(delta: Tensor, *, batch_size: int, steps: int) -> Tensor:
    """Preserve measured RGB intervals and repeat only legacy event scalar cadence."""
    if delta.shape == (batch_size,):
        delta = delta[:, None].expand(-1, steps - 1)
    if delta.shape != (batch_size, steps - 1):
        raise ValueError("RGB-PORT delta_t_s must be measured [B,T-1]")
    return delta


def _producer_loss(
    model: CausalScaleTTC,
    batch: ObjectEventV4Batch | EffectiveBatch,
    config: CausalScaleTTCLossConfig,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Historical objective with the RGB port's measured per-pair intervals."""
    from operational.rgb_port_revision.loss import EffectiveBatch, effective_loss

    if isinstance(batch, EffectiveBatch):
        if len(batch.parts) == 1:
            return _producer_loss(model, batch.parts[0], config)
        return effective_loss(model, batch, config)
    batch_size, steps = batch.events.shape[:2]
    delta = _pair_intervals(batch.delta_t_s, batch_size=batch_size, steps=steps)
    endpoint_valid = torch.ones(
        batch.boxes_xyxy.shape[:2], device=batch.boxes_xyxy.device, dtype=torch.bool
    )
    endpoint_valid[:, 0] = False
    target_valid = torch.isfinite(batch.target_ttc_s) & (batch.target_ttc_s != 0.0)
    geometry = box_geometry_targets(
        batch.boxes_xyxy,
        height=int(batch.events.shape[-2]),
        width=int(batch.events.shape[-1]),
        endpoint_valid=endpoint_valid & target_valid[:, None],
    )
    output = model(batch.events, delta, return_dense_features=True)
    base = causal_scale_ttc_loss(
        output,
        target_ttc_seconds=batch.target_ttc_s,
        delta_t_s=delta,
        risk_thresholds_s=model.config.risk_thresholds_s,
        target_valid=target_valid,
        target_geometry=geometry,
        config=config,
    )
    if batch.dinov3_relation_targets is None or batch.dinov3_relation_valid is None:
        raise ValueError("Frozen DINO relation targets are required")
    dense = output.endpoint_dense_features
    if dense is None or dense.shape[1] < 2:
        raise RuntimeError("DINO relational distillation requires two dense endpoints")
    student = dense[:, :2] if dense.shape[1] == 2 else dense[:, 1:3]
    relational = local_relational_distillation_loss(
        student,
        batch.dinov3_relation_targets,
        batch.dinov3_relation_valid,
    )
    weighted = 8.0 * relational
    components = dict(base.components)
    components["dinov3_relational_raw"] = relational.detach()
    components["dinov3_relational_weighted"] = weighted.detach()
    return base.total + weighted, components


def fit_producer(
    source: ProducerSource,
    recipe: ProducerRecipe,
    run_root: Path,
    *,
    device: str = "cuda",
    max_updates_this_call: int | None = None,
) -> dict[str, Any]:
    """Run/resume one producer to its fixed epoch-18 endpoint."""
    from e_jepa_ttc.training.causal_scale_eap import (
        _autocast,
        _foreground_only_loss_config,
    )
    from operational.rgb_port_revision.cache import GroupRowCache
    from operational.rgb_port_revision.loss import EffectiveSource
    from operational.rgb_port_revision.migration import bind_runtime

    revision_sha = bind_runtime(run_root, recipe.fit_id, rgb=recipe.modality == "rgb")
    if recipe.modality == "rgb":
        source = EffectiveSource(source, recipe, revision_sha)
    elif isinstance(source, EventProducerSource):
        source._audit_row_cache = GroupRowCache(source)
    if source.population_size != recipe.producer_population:
        raise ValueError("P source population differs from the frozen recipe")
    if dict(source.identity).get("role_manifest_sha256") != recipe.role_manifest_sha256:
        raise ValueError("P source role identity differs")
    if dict(source.identity).get("role") != "P":
        raise ValueError("Producer fitting is restricted to role P")
    source_freeze = run_root / "SOURCE_FREEZE.json"
    if not source_freeze.is_file():
        raise FileNotFoundError("RGB-PORT SOURCE_FREEZE.json is required before producer training")
    source_freeze_sha256 = file_sha256(source_freeze)
    source_freeze_record = read_json_shared(source_freeze)
    preflight_ok, preflight = _resource_guard(run_root, checkpoint_reservation=1 * 1024**3)
    if not preflight_ok:
        raise RuntimeError(f"RGB-PORT producer resource preflight denied: {preflight}")
    target_device = torch.device(device)
    if target_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA producer requested but unavailable")
    model = build_producer(recipe).to(target_device)
    loss_config = CausalScaleTTCLossConfig(**recipe.loss_config)
    warmup = _foreground_only_loss_config(loss_config)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(recipe.training_config["learning_rate"]),
        weight_decay=float(recipe.training_config["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=recipe.epochs,
        eta_min=float(recipe.training_config["minimum_learning_rate"]),
    )
    generator = torch.Generator().manual_seed(recipe.seed)
    fit_directory = run_root / "fits" / recipe.fit_id
    identity = {
        **recipe.identity,
        "recipe_identity_sha256": recipe.identity_sha256,
        "source_identity": dict(source.identity),
        "teacher_binding": {
            "audited_config_field": recipe.training_config.get(
                "representation_teacher_cache_artifact_sha256"
            ),
            "actual_p_target_sha256": dict(source.identity).get(
                "dino_targets_sha256",
                dict(source.identity).get("prepare_freeze_sha256"),
            ),
            "scope": "P_only",
        },
        "source_freeze_sha256": source_freeze_sha256,
        "environment": {
            "host": platform.node(),
            "python": platform.python_version(),
            "torch": str(torch.__version__),
            "cuda": torch.version.cuda,
        },
        "device": device,
    }
    identity["identity_sha256"] = canonical_sha256(identity)
    state = ProducerCheckpoint(fit_directory, identity, recipe.updates)
    cursor: dict[str, Any] = {
        "epoch": 1,
        "position": 0,
        "order": _epoch_order(source, generator),
        "loss_sums": {},
        "examples": 0,
        "pending_curve": [],
    }
    if state.pointer.is_file():
        cursor = state.restore(model, optimizer, scheduler, generator)
    else:
        state.save(model, optimizer, scheduler, generator, cursor, status="READY")
    provenance_path = fit_directory / "RUN_PROVENANCE.json"
    if not provenance_path.exists():
        _atomic_json(
            provenance_path,
            {
                "schema": "rgb_port_run_provenance_v1",
                "experiment_id": "RGB_PORT_V1",
                "run_id": recipe.fit_id,
                "fit_id": recipe.fit_id,
                "git_commit": source_freeze_record.get("base_commit"),
                "config_hash": recipe.identity_sha256,
                "dataset_manifest_hash": recipe.role_manifest_sha256,
                "split_version": dict(source.identity).get("split_assignment_sha256"),
                "identity_sha256": state.identity_sha256,
                "recipe_identity_sha256": recipe.identity_sha256,
                "source_freeze_sha256": source_freeze_sha256,
                "source_identity": dict(source.identity),
                "ancestry": {
                    "historical_authority_sha256": recipe.authority_sha256,
                    "source_freeze_sha256": source_freeze_sha256,
                    "initialization": "fresh_seed7",
                    "generic_dino": dict(source.identity).get("dino_targets_sha256"),
                },
                "precision": recipe.training_config["precision"],
                "context": "P_only_fixed_epoch18",
                "seed": recipe.seed,
                "host": platform.node(),
                "python_version": platform.python_version(),
                "torch_version": torch.__version__,
                "cuda_version": torch.version.cuda,
                "gpu_name": (
                    torch.cuda.get_device_name(target_device)
                    if target_device.type == "cuda"
                    else None
                ),
                "device": device,
                "start_time": datetime.now(UTC).isoformat(),
                "end_time": None,
                "status": "RUNNING",
                "checkpoint_path": None,
                "metrics_path": None,
            },
        )
    call_updates = 0
    last_losses: dict[str, float] = {}
    checkpoint_reservation = max(1 * 1024**3, 2 * state.path.stat().st_size)
    last_resource_check = -25
    resources: dict[str, Any] = {}
    model.train()
    while int(cursor["epoch"]) <= recipe.epochs:
        allowed, fast_resources = _fast_resource_guard(
            run_root, checkpoint_reservation=checkpoint_reservation
        )
        resources.update(fast_resources)
        if state.completed - last_resource_check >= 25:
            full_allowed, resources = _resource_guard(
                run_root, checkpoint_reservation=checkpoint_reservation
            )
            allowed = allowed and full_allowed
            last_resource_check = state.completed
        if not allowed or (fit_directory / "STOP_REQUEST").is_file():
            state.save(model, optimizer, scheduler, generator, cursor, status="PAUSED_RESOURCE")
            _atomic_json(fit_directory / "RESOURCE_PAUSE.json", resources)
            break
        epoch = int(cursor["epoch"])
        start = int(cursor["position"])
        stop = min(start + recipe.effective_batch_size, source.population_size)
        indices = cast(Tensor, cursor["order"])[start:stop].tolist()
        optimizer.zero_grad(set_to_none=True)
        batch_losses: dict[str, float] = {}
        input_paused = False
        for micro_ids in _homogeneous_microbatches(source, indices, recipe.microbatch_size):
            try:
                host_batch = source.batch(micro_ids, recipe.modality)
            except OSError as error:
                if not _transient_input_error(error):
                    raise
                optimizer.zero_grad(set_to_none=True)
                state.save(model, optimizer, scheduler, generator, cursor, status="PAUSED_RESOURCE")
                _atomic_json(
                    fit_directory / "RESOURCE_PAUSE.json",
                    {
                        "kind": "transient_input_io",
                        "errno": error.errno,
                        "winerror": getattr(error, "winerror", None),
                        "message": str(error),
                    },
                )
                input_paused = True
                break
            batch = _to_device(host_batch, target_device)
            _validate_batch(batch, recipe)
            with _autocast(target_device, str(recipe.training_config["precision"])):
                total, components = _producer_loss(
                    model, batch, warmup if epoch <= 3 else loss_config
                )
            if not bool(torch.isfinite(total)):
                raise FloatingPointError("Nonfinite RGB-PORT producer loss")
            weight = len(micro_ids) / len(indices)
            (total * weight).backward()
            for name, value in components.items():
                batch_losses[name] = (
                    batch_losses.get(name, 0.0) + float(value.detach().float().cpu()) * weight
                )
            batch_losses["total"] = (
                batch_losses.get("total", 0.0) + float(total.detach().float().cpu()) * weight
            )
        if input_paused:
            break
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            float(recipe.training_config["grad_clip_norm"]),
            error_if_nonfinite=True,
        )
        _assert_recovery_budget(run_root, prospective_pending=1)
        state.begin()
        optimizer.step()
        state.commit()
        call_updates += 1
        last_losses = batch_losses
        for name, value in batch_losses.items():
            cursor["loss_sums"][name] = cursor["loss_sums"].get(name, 0.0) + value * len(indices)
        cursor["examples"] += len(indices)
        cursor["position"] = stop
        cursor["pending_curve"].append(
            {
                "update": state.completed,
                "epoch": epoch,
                "examples": len(indices),
                "losses": batch_losses,
                "learning_rate": optimizer.param_groups[0]["lr"],
            }
        )
        if stop == source.population_size:
            _atomic_json(
                fit_directory / f"epoch_{epoch:02d}.json",
                {
                    "epoch": epoch,
                    "completed_updates": state.completed,
                    "examples": cursor["examples"],
                    "losses": {
                        name: value / cursor["examples"]
                        for name, value in cursor["loss_sums"].items()
                    },
                },
            )
            scheduler.step()
            cursor.update(
                {
                    "epoch": epoch + 1,
                    "position": 0,
                    "order": _epoch_order(source, generator),
                    "loss_sums": {},
                    "examples": 0,
                }
            )
        complete = int(cursor["epoch"]) > recipe.epochs
        requested_pause = (
            max_updates_this_call is not None and call_updates >= max_updates_this_call
        )
        if state.completed % 100 == 0 or complete or requested_pause:
            if cursor["pending_curve"]:
                _atomic_json(
                    fit_directory / f"curve_{state.completed:06d}.json",
                    {"updates": cursor["pending_curve"]},
                )
                cursor["pending_curve"] = []
            status = (
                "COMPLETE" if complete else "PAUSED_REQUESTED" if requested_pause else "RUNNING"
            )
            state.save(model, optimizer, scheduler, generator, cursor, status=status)
            if complete or requested_pause:
                break
    receipt = read_json_shared(state.receipt)
    provenance = read_json_shared(provenance_path)
    provenance.update(
        {
            "end_time": datetime.now(UTC).isoformat(),
            "status": receipt["status"],
            "completed_updates": receipt["completed_updates"],
            "checkpoint_path": str(state.path.resolve()),
            "checkpoint_sha256": receipt["checkpoint_sha256"],
            "metrics_path": str(fit_directory.resolve()),
            "last_losses": last_losses,
        }
    )
    _atomic_json(provenance_path, provenance)
    cache = getattr(source, "_audit_row_cache", None)
    if cache is not None:
        runtime_path = fit_directory / "AUDIT_REVISION_RUNTIME.json"
        runtime = read_json_shared(runtime_path)
        runtime["cache"] = {
            "shard_reads": cache.reads,
            "row_hits": cache.hits,
            "peak_retained_bytes": cache.peak_bytes,
            "limit_bytes": cache.inputs.limit,
        }
        _atomic_json(runtime_path, runtime)
    return receipt


def load_producer_endpoint(
    fit_directory: Path, recipe: ProducerRecipe, *, device: str = "cpu"
) -> CausalScaleTTC:
    """Strictly load one complete fixed endpoint for feature production."""
    receipt = read_json_shared(fit_directory / "CHECKPOINT_RECEIPT.json")
    checkpoint = fit_directory / "checkpoint_last.pt"
    if (
        receipt.get("status") != "COMPLETE"
        or int(receipt.get("completed_updates", -1)) != recipe.updates
        or receipt.get("checkpoint_sha256") != file_sha256(checkpoint)
    ):
        raise ValueError("Producer endpoint is incomplete or changed")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if payload.get("identity", {}).get("recipe_identity_sha256") != recipe.identity_sha256:
        raise ValueError("Producer endpoint recipe differs")
    if recipe.modality == "rgb":
        from operational.rgb_port_revision.migration import validate_rgb_endpoint

        validate_rgb_endpoint(fit_directory, payload)
    model = build_producer(recipe)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    return model.to(device).float().eval().requires_grad_(False)


def _load_source(
    factory: str, manifest: Path, *, event_source_root: Path | None = None
) -> ProducerSource:
    module_name, separator, function_name = factory.partition(":")
    if not separator:
        raise ValueError("Source factory must be module:function")
    factory_fn = getattr(importlib.import_module(module_name), function_name)
    if event_source_root is not None:
        return cast(ProducerSource, factory_fn(manifest, event_source_root=event_source_root))
    return cast(ProducerSource, factory_fn(manifest))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--authority", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--fit-id", required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--source-factory", required=True)
    parser.add_argument("--event-source-root", type=Path)
    parser.add_argument("--role-manifest-sha256", required=True)
    parser.add_argument("--population", type=int, required=True)
    parser.add_argument("--microbatch", type=int)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-updates-this-call", type=int)
    args = parser.parse_args(argv)
    verify_authority(args.config, args.authority)
    recipe = resolved_recipe(
        args.config,
        fit_id=args.fit_id,
        producer_population=args.population,
        role_manifest_sha256=args.role_manifest_sha256,
        microbatch_size=args.microbatch,
    )
    source = _load_source(
        args.source_factory,
        args.source_manifest,
        event_source_root=args.event_source_root,
    )
    try:
        result = fit_producer(
            source,
            recipe,
            args.run,
            device=args.device,
            max_updates_this_call=args.max_updates_this_call,
        )
    finally:
        close = getattr(source, "close", None)
        if close is not None:
            close()
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "COMPLETE" else 3


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ProducerCheckpoint",
    "ProducerSource",
    "EventProducerSource",
    "build_producer",
    "fit_producer",
    "load_producer_endpoint",
    "make_event_producer_source",
    "main",
    "producer_features",
]

"""Concrete, label-free media adapter for the detected-PHASE TTC panel.

Configuration and candidate verification are CPU-only.  CUDA models and sensor
readers are created lazily, once per condition/query lifetime, by
``runtime_factory``.  Source event intervals are read exactly once and remain
contiguous even though RGB exposure endpoints are not aligned to TRAIN40's
20 ms integral-history cadence.
"""

from __future__ import annotations

import hashlib
import json
import tarfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, cast

import numpy as np

if TYPE_CHECKING:
    from operational.train40_runtime.multiobject import ObjectRegion, RawRGBHistory

ROOT = Path(__file__).resolve().parents[2]
UPDATE_US = 20_000
HISTORY_US = 600_000


class _History(Protocol):
    def initialize(self, raw: Mapping[str, np.ndarray], start_us: int, end_us: int) -> None: ...


class _SealedRuntime(Protocol):
    def rebase_history(
        self, raw_history: Mapping[str, np.ndarray], start_us: int, end_us: int
    ) -> None: ...

    def update(
        self,
        raw_packet: Mapping[str, np.ndarray],
        start_us: int,
        end_us: int,
        objects: Sequence[ObjectRegion],
        rgb_history: RawRGBHistory,
    ) -> dict[str, Any]: ...

    def close(self) -> None: ...


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def _runtime_sources() -> dict[str, Path]:
    return {
        "adapter": Path(__file__),
        "phase_service": ROOT / "operational/detection_phase_service/service.py",
        "chunked_runtime": ROOT / "operational/detection_phase_service/chunked_runtime.py",
        "phase_context": ROOT / "operational/detection_phase_service/phase_context.py",
        "multiobject": ROOT / "operational/train40_runtime/multiobject.py",
        "integral_history": ROOT / "operational/train40_runtime/integral.py",
        "replay_media_geometry": ROOT / "scripts/replay_detection_phase_service.py",
        "candidate_loader": ROOT / "scripts/final_train40_transfer.py",
        "event_reader": ROOT / "src/e_jepa_ttc/data/eap.py",
        "rgb_reader": ROOT / "src/e_jepa_ttc/rgb_port/data.py",
    }


def _exact(raw: Mapping[str, Any], expected: set[str], source: str) -> None:
    if set(raw) != expected:
        raise ValueError(
            f"{source} fields differ: missing={sorted(expected - set(raw))}, "
            f"extra={sorted(set(raw) - expected)}"
        )


@dataclass(frozen=True)
class PanelAdapterConfig:
    """Strict CPU-validated configuration for one operational adapter."""

    candidate_root: Path
    eap_root: Path
    device: str
    max_objects: int
    logical_max_objects: int
    rgb_workers: int
    rgb_object_chunk_size: int
    phase_bounds: bool
    max_frame_cache_bytes: int
    event_stream_sha256: Mapping[str, str]


def load_config(path: Path) -> tuple[PanelAdapterConfig, dict[str, Any]]:
    """Read a strict JSON config without constructing any CUDA object."""
    config_path = path.resolve(strict=True)
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise TypeError("Adapter config must be a JSON object")
    expected = {
        "schema",
        "candidate_root",
        "eap_root",
        "device",
        "max_objects",
        "logical_max_objects",
        "rgb_workers",
        "rgb_object_chunk_size",
        "phase_bounds",
        "max_frame_cache_bytes",
        "event_stream_sha256",
    }
    _exact(raw, expected, "adapter config")
    if raw["schema"] != "detected_phase_panel_adapter_config_v1":
        raise ValueError("Unknown adapter config schema")
    candidate_value = Path(raw["candidate_root"]).expanduser()
    eap_value = Path(raw["eap_root"]).expanduser()
    candidate = (
        candidate_value if candidate_value.is_absolute() else config_path.parent / candidate_value
    ).resolve(strict=True)
    eap = (eap_value if eap_value.is_absolute() else config_path.parent / eap_value).resolve(
        strict=True
    )
    integers = {
        key: raw[key]
        for key in (
            "max_objects",
            "logical_max_objects",
            "rgb_workers",
            "rgb_object_chunk_size",
            "max_frame_cache_bytes",
        )
    }
    if any(isinstance(value, bool) or not isinstance(value, int) for value in integers.values()):
        raise TypeError("Adapter integer settings must be integers, not booleans")
    if raw["device"] != "cuda":
        raise ValueError("The sealed operational adapter requires device='cuda'")
    if not 1 <= integers["max_objects"] <= 20:
        raise ValueError("max_objects must be in [1,20]")
    if integers["logical_max_objects"] < integers["max_objects"]:
        raise ValueError("logical_max_objects cannot be below max_objects")
    if not 1 <= integers["rgb_workers"] <= 8:
        raise ValueError("rgb_workers must be in [1,8]")
    if not 1 <= integers["rgb_object_chunk_size"] <= integers["max_objects"]:
        raise ValueError("rgb_object_chunk_size must fit max_objects")
    if integers["max_frame_cache_bytes"] < 0:
        raise ValueError("max_frame_cache_bytes must be nonnegative")
    if not isinstance(raw["phase_bounds"], bool):
        raise TypeError("phase_bounds must be boolean")
    event_hashes = raw["event_stream_sha256"]
    if (
        not isinstance(event_hashes, dict)
        or not event_hashes
        or any(
            not isinstance(relative, str)
            or Path(relative).is_absolute()
            or ".." in Path(relative).parts
            or not isinstance(digest, str)
            or len(digest) != 64
            for relative, digest in event_hashes.items()
        )
    ):
        raise ValueError("event_stream_sha256 must bind relative paths to SHA-256 values")
    for relative in event_hashes:
        event_path = (eap / relative).resolve(strict=True)
        if not event_path.is_relative_to(eap) or not event_path.is_file():
            raise ValueError("Configured event stream escapes the eAP root or is not a file")
    manifest = eap / "data/train.parquet"
    if not manifest.is_file():
        raise FileNotFoundError(manifest)
    return (
        PanelAdapterConfig(
            candidate,
            eap,
            raw["device"],
            integers["max_objects"],
            integers["logical_max_objects"],
            integers["rgb_workers"],
            integers["rgb_object_chunk_size"],
            raw["phase_bounds"],
            integers["max_frame_cache_bytes"],
            dict(event_hashes),
        ),
        {
            "path": str(config_path),
            "sha256": _sha256(config_path),
            "canonical_sha256": _canonical(raw),
        },
    )


def _validate_raw(raw: Mapping[str, np.ndarray], start_us: int, end_us: int) -> None:
    if set(raw) != {"x", "y", "t", "p"}:
        raise ValueError("Event packet must contain exactly x/y/t/p")
    lengths = {len(raw[key]) for key in raw}
    if len(lengths) != 1:
        raise ValueError("Event packet arrays must have equal lengths")
    timestamps = np.asarray(raw["t"], dtype=np.int64)
    if len(timestamps) and (
        np.any(np.diff(timestamps) < 0)
        or int(timestamps[0]) < start_us
        or int(timestamps[-1]) >= end_us
    ):
        raise ValueError("Event packet violates its exact half-open interval")


def _slice(raw: Mapping[str, np.ndarray], start_us: int, end_us: int) -> dict[str, np.ndarray]:
    timestamps = np.asarray(raw["t"], dtype=np.int64)
    left, right = np.searchsorted(timestamps, (start_us, end_us), side="left")
    return {key: np.asarray(value[left:right]) for key, value in raw.items()}


def _concatenate(parts: Sequence[Mapping[str, np.ndarray]]) -> dict[str, np.ndarray]:
    dtypes = {"x": np.int32, "y": np.int32, "t": np.int64, "p": np.int8}
    return {
        key: np.concatenate([np.asarray(part[key], dtype=dtype) for part in parts])
        if parts
        else np.empty(0, dtype=dtype)
        for key, dtype in dtypes.items()
    }


class PanelCadenceRuntime:
    """Bridge irregular RGB cutoffs to the sealed 20 ms history without rereads."""

    supports_empty_updates = True

    def __init__(
        self,
        sealed: object,
        history: object,
        *,
        max_objects: int,
        on_close: Callable[[], None],
    ) -> None:
        self.sealed = cast(_SealedRuntime, sealed)
        self.history = cast(_History, history)
        self.max_objects = max_objects
        self._on_close = on_close
        self._parts: list[dict[str, np.ndarray]] = []
        self._first_start: int | None = None
        self._last_end: int | None = None
        self._closed = False

    def update(
        self,
        raw_packet: Mapping[str, np.ndarray],
        start_us: int,
        end_us: int,
        objects: Sequence[ObjectRegion],
        rgb_history: RawRGBHistory,
    ) -> dict[str, Any]:
        if self._closed:
            raise RuntimeError("Panel cadence runtime is closed")
        if self._last_end is not None and start_us != self._last_end:
            raise ValueError("Source event intervals must be contiguous and unique")
        _validate_raw(raw_packet, start_us, end_us)
        if self._first_start is None:
            self._first_start = start_us
        self._parts.append({key: np.asarray(value) for key, value in raw_packet.items()})
        self._last_end = end_us
        combined = _concatenate(self._parts)
        retain_from = end_us - HISTORY_US - UPDATE_US
        combined = _slice(combined, retain_from, end_us)
        self._parts = [combined]
        if not objects:
            return {
                "schema": "ttc_panel_source_packet_v1",
                "objects": [],
                "object_count": 0,
                "packet_ingested_count": 1,
                "source_interval_us": [start_us, end_us],
                "targets_read": False,
                "optimizer_updates": 0,
            }
        if start_us > end_us - UPDATE_US or not self._parts:
            raise ValueError("A mature query lacks the final 20 ms source packet")
        history_start = end_us - HISTORY_US - UPDATE_US
        if history_start < 0 or self._first_start is None or self._first_start > history_start:
            raise ValueError("Invalid causal history boundary")
        old = _slice(combined, history_start, end_us - UPDATE_US)
        current = _slice(combined, end_us - UPDATE_US, end_us)
        self.sealed.rebase_history(old, history_start, end_us - UPDATE_US)
        result = self.sealed.update(current, end_us - UPDATE_US, end_us, objects, rgb_history)
        result = dict(result)
        result.update(
            packet_ingested_count=1,
            source_interval_us=[start_us, end_us],
            source_packet_semantics="unique_interval_buffered_once",
            sealed_history_materialization_us=[history_start, end_us],
        )
        return result

    def close(self) -> None:
        if not self._closed:
            try:
                self.sealed.close()
            finally:
                self._on_close()
                self._parts.clear()
                self._closed = True


class PanelAdapter:
    """Stateful factory whose active media lifetime follows one service."""

    def __init__(self, config: PanelAdapterConfig, binding: dict[str, Any]) -> None:
        self.config = config
        self.binding = binding
        self._active: dict[str, Any] | None = None

    def _preflight_prefix_media(
        self, condition: str, query_id: str, prefix: Sequence[Mapping[str, Any]]
    ) -> dict[str, Any]:
        """Hash every admitted sensor byte before any Torch/CUDA construction."""
        event_paths = {str(frame["events_path"]) for frame in prefix}
        if not event_paths or not event_paths.issubset(self.config.event_stream_sha256):
            raise ValueError("Prefix event streams are not bound by adapter configuration")
        event_actual = {}
        for relative in sorted(event_paths):
            path = (self.config.eap_root / relative).resolve(strict=True)
            if not path.is_relative_to(self.config.eap_root):
                raise ValueError("Event stream escapes the configured eAP root")
            actual = _sha256(path)
            if actual != self.config.event_stream_sha256[relative]:
                raise ValueError("Event stream changed before CUDA construction")
            event_actual[relative] = actual
        by_shard: dict[str, list[Mapping[str, Any]]] = {}
        for frame in prefix:
            digest = frame.get("rgb_member_sha256")
            if not isinstance(digest, str) or len(digest) != 64:
                raise ValueError("Every prefix RGB member needs a prepared SHA-256")
            by_shard.setdefault(str(frame["rgb_shard_path"]), []).append(frame)
        rgb_actual = {}
        for relative, frames in sorted(by_shard.items()):
            archive = (self.config.eap_root / relative).resolve(strict=True)
            if not archive.is_relative_to(self.config.eap_root):
                raise ValueError("RGB archive escapes the configured eAP root")
            with tarfile.open(archive, mode="r:") as tar:
                names = tar.getnames()
                if len(names) != len(set(names)):
                    raise ValueError("RGB TAR contains duplicate member names")
                for frame in frames:
                    member = str(frame["rgb_member_path"])
                    extracted = tar.extractfile(tar.getmember(member))
                    if extracted is None:
                        raise FileNotFoundError(member)
                    with extracted:
                        actual = hashlib.sha256(extracted.read()).hexdigest()
                    if actual != frame["rgb_member_sha256"]:
                        raise ValueError("RGB member changed before CUDA construction")
                    rgb_actual[str(frame["sample_token"])] = actual
        return {
            "condition": condition,
            "query_id": query_id,
            "event_stream_sha256": event_actual,
            "rgb_member_sha256": rgb_actual,
            "targets_read": False,
        }

    def _close_media(self) -> None:
        active, self._active = self._active, None
        if active is not None:
            active["events"].close()
            active["frames"].close()

    def runtime_factory(self, condition: str, query: Mapping[str, Any]) -> PanelCadenceRuntime:
        """Construct the real CUDA runtime only when the run command calls it."""
        if self._active is not None:
            raise RuntimeError("Previous condition/query media lifetime was not closed")
        current_sources = {name: _sha256(path) for name, path in _runtime_sources().items()}
        if current_sources != self.binding["source_sha256"]:
            raise ValueError("Runtime sources changed after adapter validation")
        if (
            _sha256(self.config.eap_root / "data/train.parquet")
            != self.binding["media_manifest_sha256"]
        ):
            raise ValueError("eAP manifest changed after adapter validation")
        if condition == "R3" and "oracle" in query:
            raise ValueError("R3 adapter view must not contain oracle data")
        view = query.get("r3_label_free")
        if not isinstance(view, Mapping) or not isinstance(view.get("prefix"), list):
            raise ValueError("Adapter query lacks a label-free prefix")
        prefix = view["prefix"]
        if not prefix:
            raise ValueError("Adapter prefix cannot be empty")
        sequence = str(query["sequence_id"])
        if any(str(frame.get("sequence_id")) != sequence for frame in prefix):
            raise ValueError("Every prefix frame must belong to the query sequence")
        ends = [int(frame["exposure_end_us"]) for frame in prefix]
        if any(left >= right for left, right in zip(ends[:-1], ends[1:], strict=True)):
            raise ValueError("Prefix exposure endpoints must increase strictly")

        preflight = self._preflight_prefix_media(condition, str(query["query_id"]), prefix)
        self.binding["media_preflight_receipts"].append(preflight)

        import torch

        from e_jepa_ttc.data.eap import EAPEventReader
        from e_jepa_ttc.rgb_port.data import TarFrameReader
        from operational.detection_phase_service.chunked_runtime import ChunkedTrain40Runtime
        from operational.detection_phase_service.phase_context import BatchedPhaseContextRuntime
        from operational.train40_runtime.integral import IntegralEventHistory
        from operational.train40_runtime.multiobject import MultiObjectTrain40Service
        from scripts.final_train40_transfer import load_models, verify_candidate

        cuda = torch.cuda
        backends = torch.backends
        if not cuda.is_available():
            raise RuntimeError("CUDA is unavailable; adapter validation itself remains CPU-only")
        backends.cuda.matmul.allow_tf32 = False
        backends.cudnn.allow_tf32 = False
        paths, candidate_binding = verify_candidate(self.config.candidate_root)
        if candidate_binding != self.binding["candidate_binding"]:
            raise ValueError("Candidate binding changed after CPU validation")
        models, normalizers = load_models(paths, self.config.device)
        history = IntegralEventHistory(
            device=self.config.device,
            max_objects=self.config.max_objects,
            roi_batch_size=min(4, self.config.max_objects),
        )
        backend = MultiObjectTrain40Service(
            models,
            normalizers,
            history,
            device=self.config.device,
            graphs=False,
            parallel_branches=False,
            max_objects=self.config.max_objects,
            rgb_workers=self.config.rgb_workers,
            rgb_object_chunk_size=self.config.rgb_object_chunk_size,
            overlap_event_with_rgb_preparation=False,
            rgb_uint8=False,
        )
        if self.config.phase_bounds:
            backend.context = BatchedPhaseContextRuntime(
                models,
                normalizers,
                self.config.device,
                graphs=False,
                max_objects=self.config.max_objects,
            )
        sealed = ChunkedTrain40Runtime(
            backend,
            max_objects=self.config.logical_max_objects,
            chunk_size=self.config.max_objects,
        )
        event_paths = {str(frame["events_path"]) for frame in prefix}
        if len(event_paths) != 1:
            sealed.close()
            raise ValueError("One query must use one immutable event stream")
        event_path = (self.config.eap_root / next(iter(event_paths))).resolve(strict=True)
        if not event_path.is_relative_to(self.config.eap_root):
            sealed.close()
            raise ValueError("Event stream escapes the configured eAP root")
        events = EAPEventReader(event_path)
        events.open()
        frames = TarFrameReader(
            self.config.eap_root, max_frame_cache_bytes=self.config.max_frame_cache_bytes
        )
        self._active = {
            "condition": condition,
            "query_id": str(query["query_id"]),
            "prefix": prefix,
            "cursor": 0,
            "previous_end": None,
            "events": events,
            "frames": frames,
            "images": {},
            "timestamps": {},
        }
        return PanelCadenceRuntime(
            sealed, history, max_objects=self.config.logical_max_objects, on_close=self._close_media
        )

    def frame_loader(
        self, condition: str, query: Mapping[str, Any], frame: Mapping[str, Any]
    ) -> tuple[Mapping[str, np.ndarray], RawRGBHistory, int, int]:
        """Read the next unique event interval and only causal prefix RGB frames."""
        active = self._active
        if (
            active is None
            or active["condition"] != condition
            or active["query_id"] != query["query_id"]
        ):
            raise RuntimeError("Frame loader does not own this condition/query lifetime")
        cursor = active["cursor"]
        prefix = active["prefix"]
        if cursor >= len(prefix) or frame is not prefix[cursor]:
            raise ValueError("Frames must be loaded once in the sealed prefix order")
        end_us = int(frame["exposure_end_us"])
        start_us = (
            end_us - UPDATE_US if active["previous_end"] is None else int(active["previous_end"])
        )
        if end_us <= start_us:
            raise ValueError("Prefix event intervals must be positive and contiguous")
        packet = active["events"].read_window(start_us, end_us)
        _validate_raw(packet, start_us, end_us)
        image = active["frames"].read_uint8(frame["rgb_shard_path"], frame["rgb_member_path"])
        expected_hash = frame.get("rgb_member_sha256")
        if expected_hash is not None:
            archive = (self.config.eap_root / frame["rgb_shard_path"]).resolve(strict=True)
            if not archive.is_relative_to(self.config.eap_root):
                raise ValueError("RGB archive escapes the configured eAP root")
            with tarfile.open(archive, mode="r") as tar:
                extracted = tar.extractfile(frame["rgb_member_path"])
                if extracted is None:
                    raise FileNotFoundError(frame["rgb_member_path"])
                with extracted:
                    actual_hash = hashlib.sha256(extracted.read()).hexdigest()
            if actual_hash != expected_hash:
                raise ValueError("RGB member changed after panel preparation")
        active["images"][cursor] = image
        active["timestamps"][cursor] = end_us
        from operational.train40_runtime.multiobject import RawRGBHistory
        from scripts.replay_detection_phase_service import causal_rgb_groups

        rgb = RawRGBHistory(
            end_us,
            dict(active["images"]),
            dict(active["timestamps"]),
            causal_rgb_groups(dict(active["timestamps"]), end_us),
            "" if cursor >= 2 else "INSUFFICIENT_CAUSAL_RGB_HISTORY",
        )
        active["cursor"] = cursor + 1
        active["previous_end"] = end_us
        return packet, rgb, start_us, end_us


def create_panel_adapter(config_path: Path) -> PanelAdapter:
    """CPU-validate configuration, candidate seal and source identities."""
    config, config_binding = load_config(config_path)
    from scripts.final_train40_transfer import verify_candidate

    _, candidate_binding = verify_candidate(config.candidate_root)
    source_paths = _runtime_sources()
    binding = {
        "schema": "detected_phase_panel_adapter_binding_v1",
        "config": config_binding,
        "candidate_binding": candidate_binding,
        "media_manifest_sha256": _sha256(config.eap_root / "data/train.parquet"),
        "source_sha256": {name: _sha256(path) for name, path in source_paths.items()},
        "cuda_constructed_during_validation": False,
        "source_packet_semantics": "unique_contiguous_half_open_intervals",
        "history_materialization": "last_620ms_from_once-read_source_buffer",
        "configured_event_stream_sha256": dict(config.event_stream_sha256),
        "media_preflight_receipts": [],
    }
    return PanelAdapter(config, binding)


__all__ = [
    "PanelAdapter",
    "PanelAdapterConfig",
    "PanelCadenceRuntime",
    "create_panel_adapter",
    "load_config",
]

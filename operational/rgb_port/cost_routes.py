"""Measure honest batch-1 RGB-PORT route costs after every endpoint is frozen."""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn

from e_jepa_ttc.rgb_port.data import (
    TarFrameReader,
    decode_query,
    history_observations,
    make_rgb_inference_source,
)
from e_jepa_ttc.rgb_port.features import (
    EVENT_PHASE17_SHA256,
    RGB_PHASE17_SHA256,
    ProducerObservation,
    event_phase17,
    event_statistics_from_channels12,
    rgb_phase17,
)
from e_jepa_ttc.rgb_port.normalization import FrozenNormalizer
from e_jepa_ttc.simplex_t.phase import emitted_phase, phase_to_ttc
from operational.rgb_port.accounting import atomic_write_json, read_json_shared, sha256_file
from operational.rgb_port.infer_experts import (
    _parents,
    _producer,
    clock_features,
    delta_matrix,
)
from operational.rgb_port.predict_heads import _load_cache, _load_inputs
from operational.rgb_port.profile import profile_callable
from operational.rgb_port.train_heads import load_head_endpoint, pair_point_phase
from operational.rgb_port.train_producers import (
    make_event_producer_source,
    producer_features,
)
from operational.rgb_port_revision.raw_events import RawEventSource

PARENT_IDS = {
    "E_A5_MATCHED",
    "E_C2F_MATCHED",
    "PAIR_E_MATCHED",
    "R_A5",
    "R_C2F",
    "PAIR_R",
}
HEAD_IDS = {"E_H1_MATCHED", "E_CTX_MATCHED", "R_H1", "R_CTX", "F_TRUE", "F_ZERO"}


def _normalizers(path: Path) -> dict[str, FrozenNormalizer]:
    value = read_json_shared(path)
    if set(value) != {"event", "rgb"} or any(not isinstance(item, str) for item in value.values()):
        raise ValueError("normalizers JSON must map event/rgb to frozen JSON paths")
    result = {
        name: FrozenNormalizer.load(Path(source).resolve(strict=True))
        for name, source in value.items()
    }
    result["event"].validate_endpoint(
        modality="event",
        fit_role="H",
        schema_sha256=EVENT_PHASE17_SHA256,
        producer_sha256=result["event"].producer_sha256,
    )
    result["rgb"].validate_endpoint(
        modality="rgb",
        fit_role="H",
        schema_sha256=RGB_PHASE17_SHA256,
        producer_sha256=result["rgb"].producer_sha256,
    )
    return result


def _event_histories(path: Path, rows: Sequence[Mapping[str, Any]]) -> list[list[int]]:
    """Resolve the frozen native event HISTORY mapping to source row indices."""
    with np.load(path, allow_pickle=False) as archive:
        required = {"query_id", "observation_id", "valid"}
        if not required.issubset(archive.files):
            raise ValueError(
                f"event history misses fields: {sorted(required - set(archive.files))}"
            )
        query = np.asarray(archive["query_id"]).astype(str)
        observation = np.asarray(archive["observation_id"]).astype(str)
        valid = np.asarray(archive["valid"], dtype=np.bool_)
    if observation.ndim != 2 or valid.shape != observation.shape or query.shape != (len(rows),):
        raise ValueError("event HISTORY query/observation/valid shapes differ")
    row_tokens = [str(row["sample_token"]) for row in rows]
    if len(set(row_tokens)) != len(row_tokens) or len(set(query)) != len(query):
        raise ValueError("event route sample/query identities must be unique")
    source_lookup = {token: index for index, token in enumerate(row_tokens)}
    history_lookup = {identity: index for index, identity in enumerate(query)}
    if set(history_lookup) != set(source_lookup):
        raise ValueError("event HISTORY query population differs from the V manifest")
    result: list[list[int]] = []
    for token in row_tokens:
        history_row = history_lookup[token]
        identities = observation[history_row][valid[history_row]].tolist()
        if not identities or len(identities) > 8 or len(set(identities)) != len(identities):
            raise ValueError("event HISTORY must contain one to eight distinct observations")
        try:
            result.append([source_lookup[identity] for identity in identities])
        except KeyError as error:
            raise ValueError(
                f"event HISTORY references unknown observation {error.args[0]}"
            ) from error
    return result


def _head_tensor(
    model: nn.Module, cache: Mapping[str, np.ndarray], fit_id: str, row: int
) -> Tensor:
    device = next(model.parameters()).device
    features = torch.as_tensor(cache["features"][row : row + 1], device=device).float()
    timing = torch.as_tensor(cache["timing"][row : row + 1], device=device).float()
    valid = torch.as_tensor(cache["valid"][row : row + 1], device=device).bool()
    experts = torch.as_tensor(cache["expert_phase"][row : row + 1], device=device).float()
    if fit_id in {"E_H1_MATCHED", "R_H1"}:
        features, timing, valid = features[:, -1:], timing[:, -1:], valid[:, -1:]
    with torch.inference_mode():
        return phase_to_ttc(model(features, timing, valid, experts)["point_phase"])


class RouteRuntime:
    """Loaded weights and sources; construction/loading is outside every timer."""

    def __init__(
        self,
        *,
        run: Path,
        config: Path,
        p_manifest: Path,
        manifest: Path,
        event_source_root: Path,
        event_history: Path,
        normalizers: Mapping[str, FrozenNormalizer],
        device: str,
    ) -> None:
        self.run = run
        self.device = torch.device(device)
        self.rgb_source = make_rgb_inference_source(manifest)
        self.event_source = make_event_producer_source(
            manifest, event_source_root=event_source_root
        )
        self.raw_event_source = RawEventSource(self.event_source, self.rgb_source.dataset.eap_root)
        self.rows = self.rgb_source.dataset.rows
        self.normalizers = dict(normalizers)
        self.producers = {
            fit_id: _producer(run, config, p_manifest, fit_id, device)
            for fit_id in ("E_A5_MATCHED", "E_C2F_MATCHED", "R_A5", "R_C2F")
        }
        self.pairs = {
            fit_id: load_head_endpoint(run / "fits" / fit_id, fit_id, self.device)
            for fit_id in ("PAIR_E_MATCHED", "PAIR_R")
        }
        self.heads = {
            fit_id: load_head_endpoint(run / "fits" / fit_id, fit_id, self.device)
            for fit_id in HEAD_IDS
        }
        self.reader = TarFrameReader(self.rgb_source.dataset.eap_root)
        self.event_histories = _event_histories(event_history, self.rows)

    def close(self) -> None:
        self.raw_event_source.close()
        self.reader.close()
        dataset_reader = getattr(self.rgb_source.dataset, "reader", None)
        if dataset_reader is not None:
            dataset_reader.close()
        getattr(self.rgb_source, "close", lambda: None)()
        getattr(self.event_source, "close", lambda: None)()

    def raw_call(self, index: int, method: str, *args: str) -> Tensor:
        prepared = self.event_source
        self.event_source = self.raw_event_source
        try:
            return getattr(self, method)(index, *args)
        finally:
            self.event_source = prepared

    def raw_preprocess(self, index: int) -> Tensor:
        return self.raw_event_source.batch([index], "event").events

    def event_producer(self, index: int, fit_id: str = "E_A5_MATCHED") -> Tensor:
        batch = self.event_source.batch([index], "event")
        sensor = batch.events.to(self.device).float()
        delta = delta_matrix(batch.delta_t_s, sensor.shape[1]).to(self.device)
        with torch.inference_mode():
            return producer_features(self.producers[fit_id], sensor, delta)["prediction_ttc"]

    def rgb_producer(self, index: int, fit_id: str = "R_A5") -> Tensor:
        # Source.batch performs the real TAR read, shared-ROI crop, resize and FP32 conversion.
        batch = self.rgb_source.batch([index], "rgb")
        sensor = batch.events.to(self.device).float()
        delta = delta_matrix(batch.delta_t_s, sensor.shape[1]).to(self.device)
        with torch.inference_mode():
            return producer_features(self.producers[fit_id], sensor, delta)["prediction_ttc"]

    def _event_observation(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        batch = self.event_source.batch([index], "event")
        sensor = batch.events.to(self.device).float()
        delta = delta_matrix(batch.delta_t_s, sensor.shape[1]).to(self.device)
        with torch.inference_mode():
            outputs = [
                producer_features(self.producers[name], sensor, delta)
                for name in ("E_A5_MATCHED", "E_C2F_MATCHED")
            ]
            pair_input = ProducerObservation.from_output(outputs[0]).pair_input(delta[:, -1])
            pair = emitted_phase(pair_point_phase(self.pairs["PAIR_E_MATCHED"], pair_input))
            phase = torch.stack((outputs[0]["point_phase"], outputs[1]["point_phase"], pair), -1)
            features = event_phase17(
                event_statistics_from_channels12(sensor[:, -1]),
                torch.stack(
                    (outputs[0]["flow"], outputs[0]["margin"], outputs[0]["log_variance"]), -1
                ),
                torch.stack(
                    (outputs[1]["flow"], outputs[1]["margin"], outputs[1]["log_variance"]), -1
                ),
                phase,
            )
        return features[0].cpu().numpy(), phase[0].cpu().numpy()

    def event_context(self, index: int) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        row = self.rows[index]
        history = self.event_histories[index]
        observations = [self._event_observation(item) for item in history]
        raw = np.stack([item[0] for item in observations]).astype(np.float32)
        normalized = self.normalizers["event"].transform(raw)
        anchors = [int(self.rows[item]["query_time_us"]) for item in history]
        timing = clock_features(anchors, anchors, int(row["query_time_us"]))
        return (
            torch.from_numpy(normalized)[None].to(self.device),
            torch.from_numpy(timing)[None].to(self.device),
            torch.ones((1, len(history)), dtype=torch.bool, device=self.device),
            torch.from_numpy(observations[-1][1])[None].to(self.device),
        )

    def _rgb_observation(self, item: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
        decoded = decode_query(item, eap_root=self.rgb_source.dataset.eap_root, reader=self.reader)
        sensor = torch.from_numpy(decoded["rgb"])[None].to(self.device)
        delta = torch.from_numpy(decoded["delta_t_s"])[None].to(self.device)
        with torch.inference_mode():
            outputs = [
                producer_features(self.producers[name], sensor, delta) for name in ("R_A5", "R_C2F")
            ]
            pair_input = ProducerObservation.from_output(outputs[0]).pair_input(delta[:, -1])
            pair = emitted_phase(pair_point_phase(self.pairs["PAIR_R"], pair_input))
            phase = torch.stack((outputs[0]["point_phase"], outputs[1]["point_phase"], pair), -1)
            features = rgb_phase17(
                sensor[:, -1],
                torch.stack(
                    (outputs[0]["flow"], outputs[0]["margin"], outputs[0]["log_variance"]), -1
                ),
                torch.stack(
                    (outputs[1]["flow"], outputs[1]["margin"], outputs[1]["log_variance"]), -1
                ),
                phase,
            )
        return features[0].cpu().numpy(), phase[0].cpu().numpy()

    def rgb_context(self, index: int) -> tuple[Tensor, Tensor, Tensor, Tensor] | None:
        row = self.rows[index]
        history = history_observations(row)[-8:]
        if not history:
            return None
        observations = [self._rgb_observation(item) for item in history]
        raw = np.stack([item[0] for item in observations]).astype(np.float32)
        normalized = self.normalizers["rgb"].transform(raw)
        anchors = [int(item["anchor_us"]) for item in history]
        available = [
            int(item.get("available_us", max(item["producer_available_us"]))) for item in history
        ]
        timing = clock_features(anchors, available, int(row["query_time_us"]))
        return (
            torch.from_numpy(normalized)[None].to(self.device),
            torch.from_numpy(timing)[None].to(self.device),
            torch.ones((1, len(history)), dtype=torch.bool, device=self.device),
            torch.from_numpy(observations[-1][1])[None].to(self.device),
        )

    def full_event(self, index: int) -> Tensor:
        features, timing, valid, experts = self.event_context(index)
        with torch.inference_mode():
            return phase_to_ttc(
                self.heads["E_CTX_MATCHED"](features, timing, valid, experts)["point_phase"]
            )

    def full_rgb(self, index: int) -> Tensor:
        context = self.rgb_context(index)
        if context is None:
            return torch.full((1,), float("nan"), device=self.device)
        with torch.inference_mode():
            return phase_to_ttc(self.heads["R_CTX"](*context)["point_phase"])

    def full_fusion(self, index: int, fit_id: str = "F_TRUE") -> Tensor:
        event = self.event_context(index)
        rgb = self.rgb_context(index)
        if rgb is None:
            with torch.inference_mode():
                return phase_to_ttc(
                    self.heads["E_CTX_MATCHED"](*event)["point_phase"]
                )  # No RGB decode/encoder call.
        with torch.inference_mode():
            output = self.heads[fit_id](
                event[0], event[1], event[2], event[3], rgb[0], rgb[1], rgb[2]
            )
            return output["ttc"]


class RotatingBatchOne:
    def __init__(self, function: Callable[[int], Tensor], indices: Sequence[int]) -> None:
        self.function = function
        self.indices = list(indices)
        self.cursor = 0

    def __call__(self) -> Tensor:
        index = self.indices[self.cursor % len(self.indices)]
        self.cursor += 1
        return self.function(index)


def _select_queries(
    rows: Sequence[Mapping[str, Any]], caches: Mapping[str, Mapping[str, np.ndarray]]
) -> list[int]:
    available = set(np.asarray(caches["R_CTX"]["sample_token"]).astype(str))
    available &= set(np.asarray(caches["F_TRUE"]["sample_token"]).astype(str))
    ranked = sorted(
        (
            (sha256_file_token(str(row["sample_token"])), index)
            for index, row in enumerate(rows)
            if str(row["sample_token"]) in available
        )
    )
    if len(ranked) < 8:
        raise ValueError("fewer than eight label-blind queries have complete event/RGB routes")
    return [index for _, index in ranked[:8]]


def sha256_file_token(value: str) -> str:
    import hashlib

    return hashlib.sha256(value.encode()).hexdigest()


def _process_rss() -> int:
    import psutil

    return int(psutil.Process().memory_info().rss)


def _prepared_cache_inventory(root: Path) -> dict[str, Any]:
    files = [path for path in root.rglob("*") if path.is_file()]
    progress = root / "PROGRESS.json"
    return {
        "path": str(root.resolve()),
        "file_count": len(files),
        "total_bytes": sum(path.stat().st_size for path in files),
        "progress": read_json_shared(progress) if progress.is_file() else None,
        "preparation_cost": "NOT_RECORDED_AS_A_ROUTE_LATENCY",
    }


def profile_routes(args: argparse.Namespace) -> dict[str, Any]:
    """Load all weights first, then measure fixed label-blind batch-1 routes."""
    parent_hashes = _parents(args.run, sorted(PARENT_IDS | HEAD_IDS))
    inputs = _load_inputs(args.head_inputs)
    caches = {fit_id: _load_cache(path, fit_id) for fit_id, path in inputs.items()}
    normalizers = _normalizers(args.normalizers)
    runtime = RouteRuntime(
        run=args.run,
        config=args.config,
        p_manifest=args.p_manifest,
        manifest=args.manifest,
        event_source_root=args.event_source_root,
        event_history=args.event_history,
        normalizers=normalizers,
        device=args.device,
    )
    try:
        bindings = {
            "source_sha256": sha256_file(Path(__file__)),
            "config_sha256": sha256_file(args.config),
            "p_manifest_sha256": sha256_file(args.p_manifest),
            "manifest_sha256": sha256_file(args.manifest),
            "event_history_sha256": sha256_file(args.event_history),
            "head_inputs_sha256": sha256_file(args.head_inputs),
            "normalizers_sha256": sha256_file(args.normalizers),
            "head_cache_sha256": {
                fit_id: sha256_file(path) for fit_id, path in sorted(inputs.items())
            },
            "normalizer_file_sha256": {
                name: sha256_file(Path(source).resolve(strict=True))
                for name, source in sorted(read_json_shared(args.normalizers).items())
            },
            "parents": parent_hashes,
            "event_source_identity": dict(runtime.event_source.identity),
            "raw_event_source_identity": dict(runtime.raw_event_source.identity),
            "device": args.device,
        }
        if args.output.is_file():
            existing = read_json_shared(args.output)
            if existing.get("status") != "COMPLETE" or existing.get("bindings") != bindings:
                raise ValueError("existing route-cost receipt has different frozen inputs")
            return existing
        selected = _select_queries(runtime.rows, caches)
        # Fail before timing if native raw preprocessing differs from the frozen cache.
        for index in selected:
            prepared = runtime.event_source.batch([index], "event")
            raw = runtime.raw_event_source.batch([index], "event")
            if not torch.equal(prepared.events, raw.events) or not torch.equal(
                prepared.delta_t_s, raw.delta_t_s
            ):
                raise ValueError("Raw event preprocessing differs from frozen prepared tensors")
        tokens = [str(runtime.rows[index]["sample_token"]) for index in selected]
        head_lookup = {
            fit_id: {
                token: index
                for index, token in enumerate(np.asarray(cache["sample_token"]).astype(str))
            }
            for fit_id, cache in caches.items()
        }
        routes: dict[str, Callable[[int], Tensor]] = {
            "RAW_EVENT_H5_TO_VOXELS_CPU": runtime.raw_preprocess,
            "E_A5_RAW_EVENT_H5_TO_TTC": lambda index: runtime.raw_call(
                index, "event_producer", "E_A5_MATCHED"
            ),
            "E_C2F_RAW_EVENT_H5_TO_TTC": lambda index: runtime.raw_call(
                index, "event_producer", "E_C2F_MATCHED"
            ),
            "FULL_RAW_EVENT_H5_TO_E_CTX_TTC": lambda index: runtime.raw_call(index, "full_event"),
            "FULL_RAW_EVENT_H5_RGBTAR_TO_F_TRUE_TTC": lambda index: runtime.raw_call(
                index, "full_fusion", "F_TRUE"
            ),
            "FULL_RAW_EVENT_H5_RGBTAR_TO_F_ZERO_TTC": lambda index: runtime.raw_call(
                index, "full_fusion", "F_ZERO"
            ),
            "E_A5_PREPARED_EVENT_TO_TTC": lambda index: runtime.event_producer(
                index, "E_A5_MATCHED"
            ),
            "E_C2F_PREPARED_EVENT_TO_TTC": lambda index: runtime.event_producer(
                index, "E_C2F_MATCHED"
            ),
            "R_A5_RGBTAR_TO_TTC": lambda index: runtime.rgb_producer(index, "R_A5"),
            "R_C2F_RGBTAR_TO_TTC": lambda index: runtime.rgb_producer(index, "R_C2F"),
            "PREPARED_EVENT_TO_E_CTX_TTC": runtime.full_event,
            "FULL_OWN_RGBTAR_TO_R_CTX_TTC": runtime.full_rgb,
            "PREPARED_EVENT_RGBTAR_TO_F_TRUE_TTC": lambda index: runtime.full_fusion(
                index, "F_TRUE"
            ),
            "PREPARED_EVENT_RGBTAR_TO_F_ZERO_TTC": lambda index: runtime.full_fusion(
                index, "F_ZERO"
            ),
        }
        for fit_id in ("E_CTX_MATCHED", "R_CTX"):
            routes[f"{fit_id}_NORMALIZED_PHASE17_TO_TTC"] = lambda index, fit_id=fit_id: (
                _head_tensor(
                    runtime.heads[fit_id],
                    caches[fit_id],
                    fit_id,
                    head_lookup[fit_id][str(runtime.rows[index]["sample_token"])],
                )
            )
        profiles: dict[str, Any] = {}
        synchronize = torch.cuda.synchronize if args.device.startswith("cuda") else None
        for scope, function in routes.items():
            measured = profile_callable(
                RotatingBatchOne(function, selected),
                scope=scope,
                device=args.device,
                warm_iterations=20,
                synchronize=synchronize,
                memory_reader=_process_rss,
            )
            measured.update(
                batch_semantics="one_real_query_per_call",
                fixed_label_blind_query_tokens=tokens,
                weights_loaded_outside_timer=True,
                memory_measurement="process_RSS_and_CUDA_peak_allocated_reserved_when_CUDA",
            )
            profiles[scope] = measured
        cache_files = {
            fit_id: {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}
            for fit_id, path in inputs.items()
        }
        result = {
            "schema": "rgb_port_route_costs_v1",
            "revision": "audit_20261009",
            "raw_to_prepared_exact_parity_queries": len(selected),
            "status": "COMPLETE",
            "optimizer_updates": 0,
            "bindings": bindings,
            "parents": parent_hashes,
            "profiles": profiles,
            "historical_three_h8_heads": {
                "status": "BLOCKED_NOT_REMEASURED",
                "reason": (
                    "No current-role adapter with three real historical seed endpoints was "
                    "declared; no new replicas were fabricated."
                ),
            },
            "offline_exclusions": {
                "bulk_cache_generation": "offline_only; raw_H5_to_voxels_IS_INCLUDED_in_RAW_routes",
                "weights_loading": "EXCLUDED_AND_NOT_SUMMED_WITH_ROUTE_LATENCY",
                "teacher_fit": "EXCLUDED_NOT_AN_INFERENCE_OPERATION",
                "prepared_event_cache": _prepared_cache_inventory(args.event_source_root),
                "prepared_cache_files": cache_files,
            },
            "clock_contract": {
                "event": (
                    "frozen prepared HISTORY native mapping; capacity8 but no synthetic "
                    "50ms windows"
                ),
                "event_history_length_counts": {
                    str(length): sum(len(history) == length for history in runtime.event_histories)
                    for length in sorted({len(history) for history in runtime.event_histories})
                },
                "rgb": "sensor deltas only within producer; query age uses annotation clock",
                "rgb_history": (
                    "actual distinct T3 triplets plus genuine T2 cold observations, "
                    "maximum8 within650ms"
                ),
            },
        }
        atomic_write_json(args.output, result)
        return result
    finally:
        runtime.close()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--p-manifest", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--event-source-root", type=Path, required=True)
    parser.add_argument("--event-history", type=Path, required=True)
    parser.add_argument("--head-inputs", type=Path, required=True)
    parser.add_argument("--normalizers", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args(argv)
    for name in (
        "run",
        "config",
        "p_manifest",
        "manifest",
        "event_source_root",
        "event_history",
        "head_inputs",
        "normalizers",
    ):
        setattr(args, name, getattr(args, name).resolve(strict=True))
    args.output = args.output.resolve()
    result = profile_routes(args)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["RouteRuntime", "RotatingBatchOne", "profile_routes", "main"]

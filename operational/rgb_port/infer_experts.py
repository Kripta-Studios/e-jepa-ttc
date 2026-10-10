"""Materialize RGB-PORT features by executing the campaign's frozen experts."""

from __future__ import annotations

import argparse
import errno
import io
import json
from bisect import bisect_right
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch

from e_jepa_ttc.models.causal_scale_ttc import CausalScaleTTC
from e_jepa_ttc.rgb_port.data import (
    TarFrameReader,
    decode_query,
    history_observations,
    make_rgb_inference_source,
)
from e_jepa_ttc.rgb_port.features import (
    ProducerObservation,
    event_statistics_from_channels12,
    raw_rgb_statistics,
)
from e_jepa_ttc.simplex_t.phase import emitted_phase
from operational.rgb_port.accounting import (
    atomic_write_bytes,
    atomic_write_json,
    read_json_shared,
    sha256_file,
)
from operational.rgb_port.recipe import canonical_sha256, resolved_recipe
from operational.rgb_port.train_heads import load_head_endpoint, pair_point_phase
from operational.rgb_port.train_producers import (
    _resource_guard,
    load_producer_endpoint,
    make_event_producer_source,
    producer_features,
)


def save_npz(path: Path, values: Mapping[str, np.ndarray]) -> None:
    """Atomically publish a small numeric fragment without pickle."""
    buffer = io.BytesIO()
    if "allow_pickle" in values:
        raise ValueError("Reserved NPZ control keyword in numeric schema")
    cast(Callable[..., None], np.savez)(buffer, **dict(values))
    atomic_write_bytes(path, buffer.getvalue())


def delta_matrix(delta: torch.Tensor, steps: int) -> torch.Tensor:
    """Expand only the historical event scalar; retain measured RGB intervals."""
    if delta.ndim == 1:
        return delta[:, None].expand(-1, steps - 1).contiguous().float()
    if delta.ndim != 2 or delta.shape[1] != steps - 1:
        raise ValueError("Expert interval shape differs from its real observations")
    return delta.float()


def _parents(run: Path, ids: Sequence[str]) -> dict[str, str]:
    parents: dict[str, str] = {}
    for fit_id in ids:
        receipt = read_json_shared(run / "fits" / fit_id / "CHECKPOINT_RECEIPT.json")
        checkpoint = Path(receipt["checkpoint_path"])
        expected_root = (run / "fits" / fit_id).resolve()
        if (
            receipt.get("status") != "COMPLETE"
            or receipt.get("fit_id") != fit_id
            or receipt.get("scientific_endpoint") is not True
            or not checkpoint.resolve().is_relative_to(expected_root)
            or receipt.get("checkpoint_sha256") != sha256_file(checkpoint)
        ):
            raise ValueError(f"Expert endpoint is incomplete or changed: {fit_id}")
        parents[fit_id] = receipt["checkpoint_sha256"]
    return parents


def _producer(
    run: Path, config: Path, p_manifest: Path, fit_id: str, device: str
) -> CausalScaleTTC:
    p = read_json_shared(p_manifest)
    recipe = resolved_recipe(
        config,
        fit_id=fit_id,
        producer_population=int(p["population_size"]),
        role_manifest_sha256=sha256_file(p_manifest),
    )
    return load_producer_endpoint(run / "fits" / fit_id, recipe, device=device)


class Fragments:
    """Resume hash-bound inference fragments, preserving completed model calls."""

    def __init__(self, root: Path, binding: Mapping[str, Any]) -> None:
        self.root = root
        root.mkdir(parents=True, exist_ok=True)
        self.binding = dict(binding)
        self.digest = canonical_sha256(binding)
        path = root / "BINDING.json"
        if path.exists() and read_json_shared(path) != self.binding:
            raise ValueError("Inference fragment identity changed")
        if not path.exists():
            atomic_write_json(path, self.binding)

    def read(self, index: int) -> dict[str, np.ndarray] | None:
        path = self.root / f"part_{index:06d}.npz"
        receipt_path = path.with_suffix(".json")
        if not receipt_path.exists():
            return None
        receipt = read_json_shared(receipt_path)
        if receipt.get("binding_sha256") != self.digest or receipt.get("sha256") != sha256_file(
            path
        ):
            raise ValueError("Completed inference fragment differs from its receipt")
        with np.load(path, allow_pickle=False) as archive:
            return {name: archive[name] for name in archive.files}

    def write(self, index: int, values: Mapping[str, np.ndarray]) -> None:
        path = self.root / f"part_{index:06d}.npz"
        save_npz(path, values)
        atomic_write_json(
            path.with_suffix(".json"),
            {
                "binding_sha256": self.digest,
                "sha256": sha256_file(path),
                "optimizer_updates": 0,
            },
        )


def _admit(run: Path) -> None:
    allowed, snapshot = _resource_guard(run, checkpoint_reservation=256 * 1024**2)
    if not allowed or (run / "PAUSE").exists():
        atomic_write_json(run / "INFERENCE_RESOURCE_PAUSE.json", snapshot)
        raise InterruptedError("Recoverable inference resource pause")


def _binding(
    manifest: Path, config: Path, parents: Mapping[str, str], stage: str
) -> dict[str, Any]:
    metadata = read_json_shared(manifest)
    rows = Path(metadata["rows_path"])
    if sha256_file(rows) != metadata["rows_sha256"]:
        raise ValueError("Sensor metadata changed after role admission")
    root = Path(__file__).resolve().parents[2]
    source_paths = [
        Path(__file__),
        root / "src/e_jepa_ttc/rgb_port/data.py",
        root / "src/e_jepa_ttc/rgb_port/history.py",
        root / "src/e_jepa_ttc/rgb_port/features.py",
        root / "src/e_jepa_ttc/rgb_port/normalization.py",
        root / "src/e_jepa_ttc/models/causal_scale_ttc.py",
        root / "src/e_jepa_ttc/simplex_t/phase.py",
        root / "src/e_jepa_ttc/simplex_t/expert_phase.py",
        root / "src/e_jepa_ttc/training/stage61_pair_head.py",
        root / "operational/rgb_port/train_heads.py",
        root / "operational/rgb_port/train_producers.py",
        root / "operational/rgb_port/recipe.py",
    ]
    for fit_id in parents:
        endpoint = manifest.parent / "fits" / fit_id / "ENDPOINT_FREEZE.json"
        if endpoint.exists():
            source_paths.append(endpoint)
    freeze = manifest.parent / "SOURCE_FREEZE.json"
    if freeze.exists():
        source_paths.append(freeze)
    return {
        "schema": "rgb_port_real_expert_inference_v1",
        "stage": stage,
        "manifest_sha256": sha256_file(manifest),
        "config_sha256": sha256_file(config),
        "parents": dict(parents),
        "source_sha256": sha256_file(Path(__file__)),
        "executed_source_sha256": {str(path.resolve()): sha256_file(path) for path in source_paths},
        "precision": "float32",
        "optimizer_updates": 0,
        "targets_are_model_inputs": False,
    }


def pair_source(args: argparse.Namespace) -> dict[str, Any]:
    """Execute a frozen own A5 on P, keeping each row's true final interval."""
    if read_json_shared(args.manifest)["role"] != "P":
        raise ValueError("PAIR features can only be produced for P")
    rgb = make_rgb_inference_source(args.manifest)
    source = (
        rgb
        if args.modality == "rgb"
        else make_event_producer_source(args.manifest, event_source_root=args.event_source_root)
    )
    fit_id = "R_A5" if args.modality == "rgb" else "E_A5_MATCHED"
    parent = _parents(args.run, [fit_id])
    store = Fragments(
        args.output.with_suffix(".parts"),
        _binding(args.manifest, args.config, parent, "pair_source"),
    )
    model = _producer(args.run, args.config, args.p_manifest, fit_id, args.device)
    parts: list[dict[str, np.ndarray]] = []
    device = torch.device(args.device)
    try:
        with torch.inference_mode():
            for part, start in enumerate(range(0, source.population_size, args.batch_size)):
                saved = store.read(part)
                if saved is not None:
                    parts.append(saved)
                    continue
                _admit(args.run)
                indices = list(range(start, min(start + args.batch_size, source.population_size)))
                result: dict[int, dict[str, np.ndarray]] = {}
                for count in (2, 3):
                    selected = [
                        index
                        for index in indices
                        if (int(source.frame_counts[index]) if args.modality == "rgb" else 3)
                        == count
                    ]
                    if not selected:
                        continue
                    batch = source.batch(selected, args.modality)
                    inputs = batch.events.to(device).float()
                    delta = delta_matrix(batch.delta_t_s, inputs.shape[1]).to(device)
                    output = producer_features(model, inputs, delta)
                    for local, index in enumerate(selected):
                        result[index] = {
                            "token128": output["token128"][local].cpu().numpy(),
                            "delta_t_s": delta[local, -1].cpu().numpy(),
                            "support": output["support"][local, -2:].cpu().numpy(),
                        }
                rows = rgb.dataset.rows
                values = {
                    name: np.stack([result[index][name] for index in indices])
                    for name in ("token128", "delta_t_s", "support")
                }
                for name in ("target_phase", "mass"):
                    values[name] = np.asarray([rows[index][name] for index in indices], np.float32)
                for name in ("sequence_id", "sample_token", "group_id", "role"):
                    values[name] = np.asarray([rows[index][name] for index in indices])
                store.write(part, values)
                parts.append(values)
                atomic_write_json(
                    store.root / "PROGRESS.json",
                    {
                        "completed_rows": indices[-1] + 1,
                        "total_rows": source.population_size,
                        "optimizer_updates": 0,
                    },
                )
        values = {name: np.concatenate([part[name] for part in parts]) for name in parts[0]}
        save_npz(args.output, values)
        receipt = {
            "status": "COMPLETE",
            "rows": source.population_size,
            "output": str(args.output.resolve()),
            "sha256": sha256_file(args.output),
            "parents": parent,
            "optimizer_updates": 0,
        }
        atomic_write_json(args.output.with_suffix(".receipt.json"), receipt)
        return receipt
    finally:
        getattr(source, "close", lambda: None)()


def _observation_signature(row: Mapping[str, Any]) -> str:
    keys = (
        "producer_times_us",
        "producer_sensor_times_us",
        "producer_available_us",
        "producer_shards",
        "producer_members",
        "producer_boxes_xyxy",
        "roi_xyxy",
    )
    return canonical_sha256({key: row[key] for key in keys if key in row})


def clock_features(anchors: Sequence[int], available: Sequence[int], query_us: int) -> np.ndarray:
    """Four timing fields subtract integer clocks before conversion to seconds."""
    values = np.zeros((len(anchors), 4), np.float32)
    for index, (anchor, arrival) in enumerate(zip(anchors, available, strict=True)):
        if anchor > query_us or arrival > query_us or arrival < anchor:
            raise ValueError("Expert observation is unavailable at the query")
        values[index] = [
            (query_us - anchor) / 1e6,
            (query_us - arrival) / 1e6,
            0.0 if index == 0 else (anchor - anchors[index - 1]) / 1e6,
            (arrival - anchor) / 1e6,
        ]
    return values


def bounded_event_history(
    timeline: Sequence[tuple[int, str]],
    anchors: Sequence[int],
    observations: Mapping[str, Mapping[str, Any]],
    query_us: int,
) -> list[str]:
    """Bound total sensor information, including each expert's input duration."""
    stop = bisect_right(anchors, query_us)
    selected = [
        identity
        for anchor, identity in timeline[max(0, stop - 8) : stop]
        if 0 < int(observations[identity]["input_span_us"]) <= 650000
        and query_us - anchor + int(observations[identity]["input_span_us"]) <= 650000
    ]
    if not selected:
        raise ValueError("Event query has no observation within the total 650 ms budget")
    return selected


def observations(args: argparse.Namespace) -> dict[str, Any]:
    """Execute own A5/C2F/PAIR once for each actual H/V expert observation."""
    manifest = read_json_shared(args.manifest)
    role = str(manifest["role"])
    if role not in {"H", "V"}:
        raise ValueError("Context feature production requires H or V")
    rgb = make_rgb_inference_source(args.manifest)
    rows = rgb.dataset.rows
    modality = args.modality
    prefix = "R" if modality == "rgb" else "E"
    fit_ids = (
        ["R_A5", "R_C2F", "PAIR_R"]
        if prefix == "R"
        else ["E_A5_MATCHED", "E_C2F_MATCHED", "PAIR_E_MATCHED"]
    )
    parents = _parents(args.run, fit_ids)
    models = [
        _producer(args.run, args.config, args.p_manifest, fit_id, args.device)
        for fit_id in fit_ids[:2]
    ]
    device = torch.device(args.device)
    pair = load_head_endpoint(args.run / "fits" / fit_ids[2], fit_ids[2], device)
    all_observations: dict[str, dict[str, Any]] = {}
    histories: list[list[str]] = []
    if modality == "rgb":
        for row in rows:
            selected = history_observations(row)
            if not selected:
                raise ValueError("Available RGB query has no genuine T2/T3 observation")
            ids: list[str] = []
            for item in selected:
                identity = str(item["observation_id"])
                if identity in all_observations and _observation_signature(
                    item
                ) != _observation_signature(all_observations[identity]):
                    raise ValueError("Repeated RGB frame/ROI identity changed")
                all_observations.setdefault(identity, item)
                ids.append(identity)
            histories.append(ids[-8:])
        event_source = None
    else:
        event_source = make_event_producer_source(
            args.manifest, event_source_root=args.event_source_root
        )
        by_track: dict[str, list[tuple[int, str]]] = {}
        for index, row in enumerate(rows):
            identity = str(row["sample_token"])
            anchor = int(row["query_time_us"])
            all_observations[identity] = {
                "index": index,
                "anchor_us": anchor,
                "available_us": anchor,
                "input_span_us": int(event_source.input_span_us[index]),
            }
            by_track.setdefault(str(row["track_id"]), []).append((anchor, identity))
        for timeline in by_track.values():
            timeline.sort()
        track_times = {
            track: [anchor for anchor, _ in timeline] for track, timeline in by_track.items()
        }
        for row in rows:
            cutoff = int(row["query_time_us"])
            track = str(row["track_id"])
            timeline, anchors = by_track[track], track_times[track]
            histories.append(bounded_event_history(timeline, anchors, all_observations, cutoff))
    ordered = sorted(all_observations)
    store = Fragments(
        args.output_dir / "parts",
        _binding(args.manifest, args.config, parents, f"{role}_{modality}_observations"),
    )
    reader = TarFrameReader(rgb.dataset.eap_root)
    parts: list[dict[str, np.ndarray]] = []
    try:
        with torch.inference_mode():
            for part, start in enumerate(range(0, len(ordered), args.batch_size)):
                saved = store.read(part)
                if saved is not None:
                    parts.append(saved)
                    continue
                _admit(args.run)
                ids = ordered[start : start + args.batch_size]
                samples = [all_observations[identity] for identity in ids]
                result: dict[int, dict[str, np.ndarray]] = {}
                for count in (2, 3):
                    selected = [
                        i
                        for i, item in enumerate(samples)
                        if (len(item["producer_members"]) if modality == "rgb" else 3) == count
                    ]
                    if not selected:
                        continue
                    if modality == "rgb":
                        decoded = [
                            decode_query(
                                samples[index], eap_root=rgb.dataset.eap_root, reader=reader
                            )
                            for index in selected
                        ]
                        sensor = torch.from_numpy(np.stack([item["rgb"] for item in decoded])).to(
                            device
                        )
                        delta = torch.from_numpy(
                            np.stack([item["delta_t_s"] for item in decoded])
                        ).to(device)
                        base = raw_rgb_statistics(sensor[:, -1])
                    else:
                        assert event_source is not None
                        batch = event_source.batch(
                            [int(samples[index]["index"]) for index in selected], "event"
                        )
                        sensor = batch.events.to(device).float()
                        delta = delta_matrix(batch.delta_t_s, sensor.shape[1]).to(device)
                        base = event_statistics_from_channels12(sensor[:, -1])
                    outputs = [producer_features(model, sensor, delta) for model in models]
                    pair_features = ProducerObservation.from_output(outputs[0]).pair_input(
                        delta[:, -1]
                    )
                    pair_phase = emitted_phase(pair_point_phase(pair, pair_features))
                    expert_phase = torch.stack(
                        (outputs[0]["point_phase"], outputs[1]["point_phase"], pair_phase), -1
                    )
                    diagnostics = [
                        torch.stack((output["flow"], output["margin"], output["log_variance"]), -1)
                        for output in outputs
                    ]
                    for local, index in enumerate(selected):
                        result[index] = {
                            "base_statistics2": base[local].cpu().numpy(),
                            "a5_diagnostics3": diagnostics[0][local].cpu().numpy(),
                            "c2f_diagnostics3": diagnostics[1][local].cpu().numpy(),
                            "expert_phase": expert_phase[local].cpu().numpy(),
                        }
                values = {
                    name: np.stack([result[index][name] for index in range(len(ids))])
                    for name in result[0]
                }
                values["observation_id"] = np.asarray(ids)
                values["anchor_us"] = np.asarray(
                    [int(item["anchor_us"]) for item in samples], np.int64
                )
                values["available_us"] = np.asarray(
                    [
                        int(item.get("available_us", max(item["producer_available_us"])))
                        if modality == "rgb"
                        else int(item["available_us"])
                        for item in samples
                    ],
                    np.int64,
                )
                values["role"] = np.full(len(ids), role)
                store.write(part, values)
                parts.append(values)
                atomic_write_json(
                    store.root / "PROGRESS.json",
                    {
                        "completed_observations": start + len(ids),
                        "total_observations": len(ordered),
                        "optimizer_updates": 0,
                    },
                )
        blocks = {name: np.concatenate([part[name] for part in parts]) for name in parts[0]}
        lookup = {
            identity: index for index, identity in enumerate(blocks["observation_id"].astype(str))
        }
        n = len(rows)
        requested = np.full((n, 8), "", dtype=f"U{max(map(len, ordered))}")
        valid = np.zeros((n, 8), np.bool_)
        timing = np.zeros((n, 8, 4), np.float32)
        current = np.empty((n, 3), np.float32)
        for index, (row, history) in enumerate(zip(rows, histories, strict=True)):
            anchors = [int(blocks["anchor_us"][lookup[identity]]) for identity in history]
            arrivals = [int(blocks["available_us"][lookup[identity]]) for identity in history]
            if len(set(anchors)) != len(anchors):
                raise ValueError("Context contains duplicated observation timestamps")
            requested[index, -len(history) :] = history
            valid[index, -len(history) :] = True
            timing[index, -len(history) :] = clock_features(
                anchors, arrivals, int(row["query_time_us"])
            )
            current[index] = blocks["expert_phase"][lookup[history[-1]]]
        history_values = {
            "query_id": np.asarray([row["sample_token"] for row in rows]),
            "observation_id": requested,
            "timing": timing,
            "valid": valid,
            "current_expert_phase": current,
        }
        for name in ("target_phase", "mass"):
            history_values[name] = np.asarray([row[name] for row in rows], np.float32)
        for name in ("sample_token", "group_id", "role"):
            history_values[name] = np.asarray([row[name] for row in rows])
        save_npz(args.output_dir / "blocks.npz", blocks)
        save_npz(args.output_dir / "history.npz", history_values)
        receipt = {
            "status": "COMPLETE",
            "modality": modality,
            "role": role,
            "queries": n,
            "observations": len(ordered),
            "parents": parents,
            "optimizer_updates": 0,
            "query_label_anchor": "query_time_us",
            "model_RGB_intervals": "same_sensor_clock_integer_difference",
            "history_clock": "annotation_frame_clock_public_mapping",
            "context_total_information_limit_us": 650000,
            "context_length_counts": {
                str(length): sum(len(history) == length for history in histories)
                for length in sorted({len(history) for history in histories})
            },
            "event_input_span_policy": (
                "same_event_sensor_clock_duration_plus_annotation_anchor_lag"
                if modality == "event"
                else None
            ),
            "claims_eight_independent_observations": False,
            "files": {
                name: sha256_file(args.output_dir / name) for name in ("blocks.npz", "history.npz")
            },
        }
        atomic_write_json(args.output_dir / "REAL_EXPERT_RECEIPT.json", receipt)
        return receipt
    finally:
        reader.close()
        if event_source is not None:
            event_source.close()


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    commands = result.add_subparsers(dest="command", required=True)
    for name in ("pair-source", "observations"):
        item = commands.add_parser(name)
        item.add_argument("--run", type=Path, required=True)
        item.add_argument("--config", type=Path, required=True)
        item.add_argument("--p-manifest", type=Path, required=True)
        item.add_argument("--manifest", type=Path, required=True)
        item.add_argument("--modality", choices=("event", "rgb"), required=True)
        item.add_argument("--event-source-root", type=Path, required=True)
        item.add_argument("--batch-size", type=int, default=32)
        item.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
        item.add_argument(
            "--output" if name == "pair-source" else "--output-dir", type=Path, required=True
        )
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.batch_size < 1:
        raise ValueError("Inference batch size must be positive")
    torch.set_num_threads(2)
    try:
        receipt = pair_source(args) if args.command == "pair-source" else observations(args)
    except (InterruptedError, OSError) as error:
        if isinstance(error, OSError) and not isinstance(error, InterruptedError):
            transient_errno = {errno.EIO, errno.EBUSY, errno.ENOMEM, errno.EINVAL}
            transient_windows = {5, 32, 33, 21, 1167}
            if (
                error.errno not in transient_errno
                and getattr(error, "winerror", None) not in transient_windows
            ):
                raise
        atomic_write_json(
            args.run / "EXPERT_INFERENCE_PAUSE.json",
            {"status": "PAUSED_RESOURCE", "reason": str(error), "optimizer_updates": 0},
        )
        return 3
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Bounded FP32 expert extraction for the unchanged query-conditioned H8 context."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch

from operational.efficient_context.common import Lease, digest
from operational.train40_system.contracts import read, verified_endpoint, verify_sources
from operational.train40_system.data_audit import OUTPUT
from operational.train40_system.durable_io import atomic_json
from operational.train40_system.models import resource_guard

if TYPE_CHECKING:
    from e_jepa_ttc.simplex_t.cache import CachedQueries

_reader_pool = None


def worker_init() -> None:
    """Keep one persistent HDF5 reader and one compute thread per raw worker."""
    global _reader_pool
    from e_jepa_ttc.simplex_t.cached_event_reader import ReaderPool

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    _reader_pool = ReaderPool()


def prepare(job: dict) -> np.ndarray:
    """Read causal sensor context only; no labels or model state are accepted."""
    from e_jepa_ttc.efficient_context.mapped_union import encode_union

    if _reader_pool is None:
        raise RuntimeError("H8 raw worker is not initialized")
    tensor = encode_union(
        _reader_pool.get(job["path"]),
        job["windows"],
        np.arange(15, -1, -1, dtype=np.int64) * 50_000,
        np.concatenate((np.zeros(8, dtype=bool), job["valid"])),
        tuple(job["square"]),
        sequence_id=job["sequence"],
        roi_size=128,
        event_pixel_diff=5,
    )
    return tensor.numpy()


def load_producers(output: Path, *, pair: bool) -> tuple[dict, dict]:
    """Load only newly trained complete endpoints, never historical initializers."""
    from e_jepa_ttc.models.causal_scale_ttc import CausalScaleTTC, CausalScaleTTCConfig
    from e_jepa_ttc.training.stage61_pair_head import CachedPairDirectPhase

    protocol = read(output / "TRAINING_PROTOCOL.json")
    models, parents = {}, {}
    for arm in ("a5", "c2f", "pair") if pair else ("a5",):
        path, receipt = verified_endpoint(output, f"{arm}_seed7", 6840 if arm == "pair" else 49932)
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if (
            payload["status"] != "COMPLETE"
            or payload["committed_updates"] != receipt["committed_updates"]
        ):
            raise ValueError("Producer checkpoint does not contain a complete endpoint")
        model = (
            CachedPairDirectPhase()
            if arm == "pair"
            else CausalScaleTTC(CausalScaleTTCConfig(**protocol["producers"][arm]["model_config"]))
        )
        model.load_state_dict(payload["model_state_dict"], strict=True)
        models[arm.upper()] = model.float().eval().requires_grad_(False).to("cuda")
        parents[arm] = {
            "checkpoint_sha256": receipt["sha256"],
            "updates": receipt["committed_updates"],
        }
    return models, parents


def load_source(output: Path, manifest: dict) -> CachedQueries:
    """Open the deduplicated small head cache, keeping raw events outside the fitter."""
    from e_jepa_ttc.simplex_t.cache import CachedQueries, Normalizer

    with np.load(output / "H8_FEATURES.npz", allow_pickle=False) as stored:
        arrays = {key: stored[key] for key in stored.files}
    return CachedQueries(
        features=arrays["features"],
        anchor_us=arrays["anchor_us"],
        available_us=arrays["available_us"],
        history=arrays["history"],
        target_phase=arrays["target_phase"],
        mass=arrays["mass"],
        normalizer=Normalizer(arrays["mean"], arrays["scale"], manifest["consumed_ids_sha256"]),
        identity_sha256=manifest["manifest_sha256"],
        length=8,
    )


def pair_cache(output: Path) -> None:
    """Extract the original shared-A5 133 coordinates in canonical FP32 batches."""
    models, parents = load_producers(output, pair=False)
    freeze = read(output / "PREPARE_FREEZE.json")
    index_path = output / "TRAIN40_INDEX.npz"
    with np.load(index_path, allow_pickle=False) as stored:
        tokens, sequences, target = (
            stored["tokens"],
            stored["sequences"],
            stored["phase"].astype(np.float32),
        )
    directory = output / "pair_feature_fragments"
    directory.mkdir(exist_ok=True)
    binding = {
        "parents": parents,
        "index_sha256": digest(index_path),
        "source_sha256": digest(Path(__file__)),
    }
    binding_path = directory / "BINDING.json"
    if binding_path.exists() and read(binding_path) != binding:
        raise ValueError("PAIR feature parents or implementation changed")
    atomic_json(binding_path, binding)
    all_features = np.empty((88744, 133), np.float32)
    with torch.inference_mode():
        for number, start in enumerate(range(0, 88744, 32)):
            allowed, resources = resource_guard(output)
            if not allowed:
                atomic_json(output / "PAIR_FEATURE_PAUSE.json", resources)
                return
            stop = min(start + 32, 88744)
            path = directory / f"shard_{number:05d}.npz"
            receipt_path = path.with_suffix(".json")
            if receipt_path.exists():
                receipt = read(receipt_path)
                if (
                    receipt["binding_sha256"] != digest(binding_path)
                    or digest(path) != receipt["sha256"]
                ):
                    raise ValueError("PAIR feature fragment changed")
                with np.load(path, allow_pickle=False) as stored:
                    value = stored["features"]
            else:
                raw_path = Path(freeze["cache_root"]) / path.name
                raw_receipt = read(raw_path.with_suffix(".json"))
                if digest(raw_path) != raw_receipt["sha256"]:
                    raise ValueError("Canonical current TRAIN events changed")
                with np.load(raw_path, allow_pickle=False) as stored:
                    if not np.array_equal(stored["tokens"], tokens[start:stop]):
                        raise ValueError("PAIR feature population misaligned")
                    events = torch.from_numpy(stored["events"]).to("cuda")
                    elapsed = torch.from_numpy(stored["delta"]).to("cuda")
                delta = elapsed[:, None].expand(-1, 2)
                result = models["A5"](events, delta, return_dense_features=True)
                support = result.sensor_support
                value = (
                    torch.cat(
                        (
                            result.pair_tokens[:, -1].float(),
                            torch.stack(
                                (elapsed, torch.log(elapsed + 1e-8), elapsed.reciprocal()), -1
                            ),
                            support[:, -1:],
                            torch.minimum(support[:, -2], support[:, -1]).unsqueeze(-1),
                        ),
                        -1,
                    )
                    .float()
                    .cpu()
                    .numpy()
                )
                if value.shape != (stop - start, 133) or not np.isfinite(value).all():
                    raise ValueError("PAIR feature contract failed")
                temporary = path.with_suffix(".pending.npz")
                np.savez_compressed(temporary, features=value, ordinals=np.arange(start, stop))
                os.replace(temporary, path)
                atomic_json(
                    receipt_path,
                    {
                        "sha256": digest(path),
                        "binding_sha256": digest(binding_path),
                        "optimizer_updates": 0,
                    },
                )
            all_features[start:stop] = value
            atomic_json(
                output / "PAIR_FEATURE_PROGRESS.json",
                {"status": "RUNNING", "completed_rows": stop, "total_rows": 88744},
            )
    path = output / "PAIR_FEATURES.npz"
    temporary = path.with_suffix(".pending.npz")
    np.savez_compressed(
        temporary, features=all_features, target_phase=target, sequences=sequences, tokens=tokens
    )
    os.replace(temporary, path)
    atomic_json(
        output / "PAIR_FEATURE_MANIFEST.json",
        {
            "status": "COMPLETE_VERIFIED",
            "row_count": 88744,
            "parents": parents,
            "files": [{"path": path.name, "sha256": digest(path)}],
            "binding_sha256": digest(binding_path),
            "optimizer_updates": 0,
        },
    )


def history_cache(output: Path, raw_root: Path) -> None:
    """Generate eight fixed slots with full supplied-ROI exposure availability and SHA fragments."""
    import h5py
    import pyarrow.dataset as ds

    from e_jepa_ttc.data.eap import _require_hdf5plugin
    from e_jepa_ttc.simplex_t.cache import fit_normalizer, training_mass
    from e_jepa_ttc.simplex_t.context_dedup import ContextSource, observation_key
    from e_jepa_ttc.simplex_t.query_context import QueryContextInput
    from operational.simplex_t_shared_route.adapter import extract

    models, parents = load_producers(output, pair=True)
    _require_hdf5plugin()
    audit = read(output / "DATA_AUDIT.json")
    with np.load(output / "TRAIN40_INDEX.npz", allow_pickle=False) as stored:
        names = (
            "tokens",
            "sequences",
            "windows_us",
            "square_xyxy",
            "delta_t_s",
            "rgb_members",
            "phase",
            "ttc_s",
        )
        index = {key: stored[key] for key in names}
    media_path = raw_root / "data/train.parquet"
    media = (
        ds.dataset(media_path)
        .to_table(
            columns=[
                "sequence_id",
                "rgb_member_path",
                "rgb_exposure_start_timestamp_us",
                "rgb_exposure_end_timestamp_us",
            ],
            filter=ds.field("sequence_id").isin(audit["sequences"]),
        )
        .to_pylist()
    )
    frames = {(row["sequence_id"], row["rgb_member_path"]): row for row in media}
    if len(frames) != len(media):
        raise ValueError("Ambiguous TRAIN exposure reference")
    raw_pins = {item["sequence_id"]: item for item in read(output / "RAW_PLAN.json")["files"]}
    bounds = {}
    for sequence in audit["sequences"]:
        with h5py.File(raw_root / "data/train" / sequence / "events.h5", "r") as handle:
            timestamps = handle["events/t"]
            if not isinstance(timestamps, h5py.Dataset):
                raise ValueError("Raw timestamps must be a dataset")
            bounds[sequence] = (int(timestamps[0]), int(timestamps[-1]))
    family_sha = hashlib.sha256(json.dumps(parents, sort_keys=True).encode()).hexdigest()
    preprocessing_sha = digest(output / "PREPARE_FREEZE.json")
    directory = output / "h8_feature_fragments"
    directory.mkdir(exist_ok=True)
    binding = {
        "parents": parents,
        "index_sha256": audit["index_sha256"],
        "exposure_metadata_sha256": digest(media_path),
        "source_sha256": digest(Path(__file__)),
        "lags_us": list(range(350000, -1, -50000)),
        "precision": "float32",
    }
    binding_path = directory / "BINDING.json"
    if binding_path.exists() and read(binding_path) != binding:
        raise ValueError("H8 feature lineage changed")
    atomic_json(binding_path, binding)
    jobs, histories, positions, values, anchors, availability = [], [], {}, [], [], []
    for row in range(88744):
        sequence = str(index["sequences"][row])
        windows = index["windows_us"][row]
        start_bound, end_bound = bounds[sequence]
        endpoints = [frames[(sequence, str(member))] for member in index["rgb_members"][row]]
        if [int(frame["rgb_exposure_start_timestamp_us"]) for frame in endpoints] != windows[
            1:, 1
        ].tolist():
            raise ValueError("TRAIN exposure and event clock mapping differs")
        available = max(int(frame["rgb_exposure_end_timestamp_us"]) for frame in endpoints)
        if available < windows[-1, 1] or windows.max() > end_bound:
            raise ValueError("Current exposure or complete raw support missing")
        current = QueryContextInput(
            query_token=f"{row}:{index['tokens'][row]}",
            producer_family_sha256=family_sha,
            anchor_us=int(windows[-1, 1]),
            roi_available_us=available,
            allowed_cutoff_us=available,
            source_start_us=start_bound,
            windows_us=(
                (int(windows[0, 0]), int(windows[0, 1])),
                (int(windows[1, 0]), int(windows[1, 1])),
                (int(windows[2, 0]), int(windows[2, 1])),
            ),
            square_xyxy=(
                float(index["square_xyxy"][row, 0]),
                float(index["square_xyxy"][row, 1]),
                float(index["square_xyxy"][row, 2]),
                float(index["square_xyxy"][row, 3]),
            ),
        )
        source = ContextSource(sequence, raw_pins[sequence]["sha256"], preprocessing_sha, current)
        valid = windows.min() - np.arange(7, -1, -1) * 50000 >= start_bound
        if not valid[-1]:
            raise ValueError("Current TRAIN context lacks causal raw support")
        keys = [
            observation_key(source, int(lag)) if present else None
            for lag, present in zip(range(350000, -1, -50000), valid, strict=True)
        ]
        histories.append(keys)
        jobs.append(
            {
                "path": str(raw_root / "data/train" / sequence / "events.h5"),
                "sequence": sequence,
                "windows": windows,
                "square": index["square_xyxy"][row],
                "valid": valid,
                "delta": float(index["delta_t_s"][row]),
                "available": available,
                "anchor": int(windows[-1, 1]),
            }
        )
    with ProcessPoolExecutor(max_workers=8, initializer=worker_init) as pool:
        pending = {}
        with torch.inference_mode():
            for row, job in enumerate(jobs):
                allowed, resources = resource_guard(output)
                if not allowed:
                    atomic_json(output / "H8_FEATURE_PAUSE.json", resources)
                    return
                path = directory / f"query_{row:05d}.npz"
                receipt_path = path.with_suffix(".json")
                if receipt_path.exists():
                    receipt = read(receipt_path)
                    if (
                        receipt["binding_sha256"] != digest(binding_path)
                        or digest(path) != receipt["sha256"]
                    ):
                        raise ValueError("H8 feature fragment changed")
                    with np.load(path, allow_pickle=False) as stored:
                        features = stored["features"]
                else:
                    for next_row in range(row, min(row + 8, len(jobs))):
                        if (
                            next_row not in pending
                            and not (directory / f"query_{next_row:05d}.json").exists()
                        ):
                            pending[next_row] = pool.submit(prepare, jobs[next_row])
                    started = time.perf_counter()
                    tensor = torch.from_numpy(pending.pop(row).result()).to("cuda")
                    delta = torch.full((16, 2), job["delta"], dtype=torch.float32, device="cuda")
                    features = extract("H8_SEED7", models, tensor, delta)[-8:]
                    features[~job["valid"]] = 0
                    temporary = path.with_suffix(".pending.npz")
                    np.savez_compressed(temporary, features=features, valid=job["valid"])
                    os.replace(temporary, path)
                    atomic_json(
                        receipt_path,
                        {
                            "sha256": digest(path),
                            "binding_sha256": digest(binding_path),
                            "seconds": time.perf_counter() - started,
                            "optimizer_updates": 0,
                        },
                    )
                for slot, key in enumerate(histories[row]):
                    if key is None:
                        continue
                    if key in positions:
                        if not np.array_equal(values[positions[key]], features[slot]):
                            raise ValueError(
                                "Identical H8 observations have differing expert features"
                            )
                    else:
                        positions[key] = len(values)
                        values.append(features[slot].copy())
                        anchors.append(job["anchor"] - (7 - slot) * 50000)
                        availability.append(job["available"])
                atomic_json(
                    output / "H8_FEATURE_PROGRESS.json",
                    {"status": "RUNNING", "completed_rows": row + 1, "total_rows": 88744},
                )
    features = np.asarray(values, np.float32)
    history = np.asarray(
        [[positions[key] if key is not None else -1 for key in keys] for keys in histories],
        np.int64,
    )
    normalizer = fit_normalizer(features, history, np.ones(len(features), bool))
    mass = training_mass(index["ttc_s"], index["sequences"])
    path = output / "H8_FEATURES.npz"
    temporary = path.with_suffix(".pending.npz")
    np.savez_compressed(
        temporary,
        features=features,
        history=history,
        anchor_us=np.asarray(anchors, np.int64),
        available_us=np.asarray(availability, np.int64),
        mean=normalizer.mean,
        scale=normalizer.scale,
        mass=mass,
        target_phase=index["phase"].astype(np.float32),
    )
    os.replace(temporary, path)
    atomic_json(
        output / "H8_FEATURE_MANIFEST.json",
        {
            "status": "COMPLETE_VERIFIED",
            "row_count": 88744,
            "parents": parents,
            "files": [{"path": path.name, "sha256": digest(path)}],
            "binding_sha256": digest(binding_path),
            "consumed_ids_sha256": normalizer.consumed_ids_sha256,
            "unique_observations": len(features),
            "optimizer_updates": 0,
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--kind", choices=("H8",), required=True)
    args = parser.parse_args()
    with Lease(args.output.resolve()):
        freeze = read(args.output / "FEATURES_EIGHT_WORKER_FREEZE.json")
        verify_sources(freeze)
        if digest(Path(__file__)) != freeze["source_sha256"]:
            raise ValueError("Feature engine changed after QA")
        torch.set_num_threads(4)
        torch.set_num_interop_threads(2)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(False)
        if args.kind == "PAIR":
            pair_cache(args.output.resolve())
        else:
            history_cache(args.output.resolve(), args.raw_root.resolve())

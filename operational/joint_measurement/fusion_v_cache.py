"""Frozen-pair V feature extraction, with original H feature semantics and clocks."""

from __future__ import annotations

import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from e_jepa_ttc.simplex_t.model import TemporalConfig, TemporalRefiner
from operational.range_transfer.common import freeze, read, sha
from operational.rgb_port.predict_heads import _forward

from .common import RUN, gpu_charge, owner, require_resources
from .data import SensorDataset, cache_root
from .fusion_cache import FusionCacheWriter
from .fusion_features import FeatureBlock, FeatureIdentity
from .fusion_history_extract import extract_history
from .fusion_pair_cache import extract_shards
from .fusion_pair_extract import clock_pairs
from .fusion_train import DIRECTORY, freeze_endpoints, inputs
from .geometry_evaluate import load_model
from .geometry_graph_probe import no_live_training_driver
from .rgb_run import require_authorization
from .train import committed_sources, setup

OUT = DIRECTORY / "inputs/V"


def register() -> dict:
    """Require the two fixed scientific endpoints before admitting any V arrays."""
    endpoints = freeze_endpoints()
    blocks, _, _ = inputs()
    root = owner()
    historical = read(RUN / "verification/HISTORICAL_MULTIMODAL_ADMISSION.json")
    manifest = read(root / "V_MANIFEST.json")
    history = root / "features/V/R_HISTORY.npz"
    receipt = read(history.with_suffix(".json"))
    observations = read(root / "features/V/R_OBSERVATIONS.json")
    if (
        sha(root / "V_MANIFEST.json") != historical["manifest_hashes"]["V"]
        or sha(Path(manifest["rows_path"])) != manifest["rows_sha256"]
        or receipt["status"] != "COMPLETE"
        or receipt["role"] != "V"
        or receipt["modality"] != "rgb"
        or receipt["queries"] != manifest["population_size"]
        or receipt["output_sha256"] != sha(history)
        or receipt["normalizer_sha256"] != historical["heads"]["R_CTX"]["normalizer_sha256"]
        or receipt["source_sha256"] != observations["output_sha256"]
        or observations["output_sha256"] != sha(root / "features/V/R_OBSERVATIONS.npz")
        or observations["endpoint_hashes"] != historical["expert_endpoints"]
        or read(cache_root("V") / "BINDING.json")["manifest_sha256"]
        != historical["manifest_hashes"]["V"]
        or read(cache_root("V") / "COMPLETE.json")["rows"] != manifest["population_size"]
    ):
        raise ValueError("V native history, parent, normalizer or cohort admission changed")
    sources = {"operational/joint_measurement/fusion_v_cache.py": sha(Path(__file__))}
    committed_sources(sources)
    record = {
        "endpoints": endpoints,
        "role": "V",
        "identities": {name: asdict(block.identity) for name, block in blocks.items()},
        "manifest_sha256": sha(root / "V_MANIFEST.json"),
        "history_sha256": receipt["output_sha256"],
        "sensor_binding_sha256": sha(cache_root("V") / "BINDING.json"),
        "sensor_complete_sha256": sha(cache_root("V") / "COMPLETE.json"),
        "sources": sources,
        "feature_semantics": "Identical admitted H extractors and parents; new V caller bound here",
        "rgb_batch": 128,
        "event_batch": 8,
        "chunk_size": 256,
        "precision": "FP32; event CUDA no TF32; RGB native CPU",
        "normalization": "Existing H transforms only; no V fit",
        "optimizer_updates": 0,
        "interpretation": "Exposed internal development; never official test",
    }
    freeze(OUT / "REGISTRATION.json", record)
    return record


def rgb(record: dict) -> dict:
    """Preserve missing RGB explicitly and verify every available native prediction."""
    root = owner()
    manifest = read(root / "V_MANIFEST.json")
    rows = pd.read_parquet(
        manifest["rows_path"], columns=["sample_token", "rgb_available", "producer_sensor_times_us"]
    )
    with np.load(root / "features/V/R_HISTORY.npz", allow_pickle=False) as archive:
        if not np.all(archive["role"] == "V"):
            raise ValueError("V history role changed")
        values = {
            key: archive[key].copy()
            for key in ("features", "timing", "valid", "expert_phase", "sample_token")
        }
    tokens = rows.sample_token.to_numpy(dtype=str)
    if not np.array_equal(tokens, values["sample_token"]):
        raise ValueError("V history must preserve every query in manifest order")
    identity = FeatureIdentity(**record["identities"]["rgb"])
    sink = FusionCacheWriter(
        OUT / "R_CTX",
        identity,
        role="V",
        manifest_sha256=record["manifest_sha256"],
        sample_tokens=tokens,
        feature_count=28,
        chunk_size=record["chunk_size"],
        input_binding={"registration_sha256": sha(OUT / "REGISTRATION.json")},
    )
    model = TemporalRefiner(TemporalConfig(hidden=160))
    payload = torch.load(
        root / "fits/R_CTX/checkpoint_last.pt", map_location="cpu", weights_only=False
    )
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.eval().requires_grad_(False)
    original = {key: value.clone() for key, value in model.state_dict().items()}
    rng = torch.get_rng_state().clone()
    for index in range(sink.chunks):
        if sink.read_chunk(index) is not None:
            continue
        require_resources()
        start, stop = sink.bounds(index)
        available = rows.iloc[start:stop].rgb_available.to_numpy(dtype=np.bool_)
        features = np.full((stop - start, 28), np.nan, np.float32)
        prediction = np.full(stop - start, np.nan, np.float32)
        local = np.flatnonzero(available)
        for offset in range(0, len(local), record["rgb_batch"]):
            ids = local[offset : offset + record["rgb_batch"]]
            batch = {key: value[start + ids] for key, value in values.items()}
            args = [
                torch.from_numpy(batch[key])
                for key in ("features", "timing", "valid", "expert_phase")
            ]
            features[ids], prediction[ids] = extract_history(model, *args)
            np.testing.assert_array_equal(
                prediction[ids].astype(np.float64),
                _forward(model, "R_CTX", batch)["prediction_ttc"],
            )
        delta = np.full(stop - start, np.nan, np.float32)
        for i in local:
            times = rows.iloc[start + i].producer_sensor_times_us
            delta[i] = (times[-1] - times[-2]) / 1_000_000
        clocks = clock_pairs(
            rows.iloc[start:stop], modality="rgb", delta_t_s=delta, available=available
        )
        sink.write_chunk(
            index,
            FeatureBlock(
                identity,
                "V",
                record["manifest_sha256"],
                tokens[start:stop],
                features,
                prediction,
                available,
                clocks,
            ),
        )
    if not torch.equal(rng, torch.get_rng_state()) or not all(
        torch.equal(original[key], value) for key, value in model.state_dict().items()
    ):
        raise ValueError("Native history extraction changed weights or RNG")
    return sink.complete()


def run() -> dict:
    """One exclusive resumable extraction, charged only while CUDA is in use."""
    import msvcrt

    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "WRITER.lock").open("a+b") as lock:
        if lock.tell() == 0:
            lock.write(b"0")
            lock.flush()
        lock.seek(0)
        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        try:
            no_live_training_driver()
            require_authorization()
            require_resources(starting=True)
            setup(7)
            record = register()
            rgb_receipt = rgb(record)
            dataset = SensorDataset("V")
            try:
                with gpu_charge("fusion_V_event_feature_extraction", reserve=900) as deadline:
                    model = load_model().to("cuda:0").eval().requires_grad_(False)
                    result = {"status": "PAUSED"}
                    while time.perf_counter() < deadline:
                        result = extract_shards(
                            OUT / "event",
                            model,
                            dataset,
                            FeatureIdentity(**record["identities"]["event"]),
                            manifest_sha256=record["manifest_sha256"],
                            available=np.ones(dataset.population, np.bool_),
                            input_binding={"registration_sha256": sha(OUT / "REGISTRATION.json")},
                            batch_size=record["event_batch"],
                            chunk_size=record["chunk_size"],
                            max_new_chunks=1,
                            device=torch.device("cuda:0"),
                        )
                        if result["status"] == "COMPLETE":
                            return {"status": "COMPLETE", "rgb": rgb_receipt, "event": result}
                    return result
            finally:
                dataset.close()
        finally:
            lock.seek(0)
            msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)


if __name__ == "__main__":
    print(run())

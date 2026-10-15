"""Materialize the historical RGB anchor on H, without labels or new training.

Only H is implemented here. V inference needs a separate admission after both
fusion endpoints freeze. Existing historical H input features are valid for
this unchanged RGB producer; new event producers must build their own caches.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from e_jepa_ttc.simplex_t.model import TemporalConfig, TemporalRefiner
from operational.range_transfer.common import digest, freeze, read, sha, utc
from operational.rgb_port.predict_heads import _forward

from .common import RUN, owner, require_resources
from .fusion_cache import FusionCacheWriter
from .fusion_features import FeatureBlock, FeatureIdentity
from .fusion_history_extract import FEATURE_NAMES, SCHEMA_SHA256, extract_history
from .fusion_history_preflight import run as preflight
from .fusion_pair_extract import clock_pairs
from .train import committed_sources

OUT = RUN / "fusion/inputs/H/R_CTX"


def extract() -> dict:
    """Publish resumable H shards and verify every TTC against native inference."""
    proof = preflight()
    if proof["status"] != "PASS":
        raise ValueError("Native RGB history preflight is required")
    selection = read(RUN / "rgb/SELECTION_FINAL.json")
    if selection["ranked"]:
        raise ValueError("Revisit anchor admission when a new RGB endpoint is admitted")
    root = owner()
    admission = read(RUN / "diagnostics/fusion_history_H_v1/REGISTRATION.json")
    sources = {
        **admission["sources"],
        "operational/joint_measurement/fusion_history_cache.py": sha(Path(__file__)),
    }
    commit = committed_sources(sources)
    registration = {
        "role": "H",
        "anchor": "R_CTX",
        "sources": sources,
        "parent_sha256": admission["checkpoint_sha256"],
        "historical_cache_sha256": admission["cache_sha256"],
        "historical_normalizer_sha256": admission["normalizer_sha256"],
        "manifest_sha256": admission["manifest_sha256"],
        "schema_sha256": SCHEMA_SHA256,
        "feature_names": FEATURE_NAMES,
        "rgb_selection_sha256": sha(RUN / "rgb/SELECTION_FINAL.json"),
        "preflight_sha256": sha(RUN / "diagnostics/fusion_history_H_v1/PASS.json"),
        "reason": "Both new RGB arms failed safeguards; preserve the established R_CTX anchor",
        "batch_size": 128,
        "chunk_size": 256,
        "device": "cpu",
        "precision": "FP32",
        "H_supervised_ancestry": True,
        "independent_H_evaluation": False,
        "event_parent_selected": False,
        "fusion_training_recipe_selected": False,
        "target_columns_loaded": False,
        "optimizer_updates": 0,
        "V_opened": False,
    }
    freeze(OUT / "REGISTRATION.json", registration)
    manifest = read(root / "H_MANIFEST.json")
    rows = pd.read_parquet(
        manifest["rows_path"], columns=["sample_token", "rgb_available", "producer_sensor_times_us"]
    )
    names = ("features", "timing", "valid", "expert_phase", "sample_token")
    with np.load(root / "features/H/R_HISTORY.npz", allow_pickle=False) as archive:
        if not np.all(archive["role"] == "H"):
            raise ValueError("History source is not H")
        values = {key: archive[key].copy() for key in names}
    if (
        not np.array_equal(values["sample_token"], rows.sample_token.to_numpy())
        or not rows.rgb_available.all()
    ):
        raise ValueError("Historical H rows must match complete real RGB coverage")
    identity = FeatureIdentity(
        "rgb", admission["checkpoint_sha256"], SCHEMA_SHA256, digest(sources)
    )
    sink = FusionCacheWriter(
        OUT,
        identity,
        role="H",
        manifest_sha256=admission["manifest_sha256"],
        sample_tokens=values["sample_token"],
        feature_count=len(FEATURE_NAMES),
        chunk_size=256,
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
    start_time = time.perf_counter()
    new_rows = 0
    for index in range(sink.chunks):
        if sink.read_chunk(index) is not None:
            continue
        require_resources()
        start, stop = sink.bounds(index)
        parts, predictions = [], []
        for offset in range(start, stop, 128):
            end = min(offset + 128, stop)
            batch = {key: value[offset:end] for key, value in values.items()}
            inputs = [
                torch.from_numpy(batch[key])
                for key in ("features", "timing", "valid", "expert_phase")
            ]
            features, ttc = extract_history(model, *inputs)
            reference = _forward(model, "R_CTX", batch)["prediction_ttc"]
            np.testing.assert_array_equal(ttc.astype(np.float64), reference)
            parts.append(features)
            predictions.append(ttc)
        times = np.stack([value[-2:] for value in rows.iloc[start:stop].producer_sensor_times_us])
        available = np.ones(stop - start, np.bool_)
        clocks = clock_pairs(
            rows.iloc[start:stop],
            modality="rgb",
            delta_t_s=((times[:, 1] - times[:, 0]) / 1_000_000).astype(np.float32),
            available=available,
        )
        sink.write_chunk(
            index,
            FeatureBlock(
                identity,
                "H",
                admission["manifest_sha256"],
                values["sample_token"][start:stop],
                np.concatenate(parts),
                np.concatenate(predictions),
                available,
                clocks,
            ),
        )
        new_rows += stop - start
    if not torch.equal(rng, torch.get_rng_state()) or not all(
        torch.equal(original[key], value) for key, value in model.state_dict().items()
    ):
        raise ValueError("RGB extraction changed RNG or weights")
    receipt = sink.complete()
    result = {
        "status": "PASS",
        "cache": receipt,
        "utc": utc(),
        "source_commit": commit,
        "registration_sha256": sha(OUT / "REGISTRATION.json"),
        "native_output_parity": "exact every published row",
        "target_columns_loaded": False,
        "optimizer_updates": 0,
        "H_supervised_ancestry": True,
        "independent_H_evaluation": False,
        "V_opened": False,
    }
    if not (OUT / "EXTRACTION_PROOF.json").exists():
        freeze(OUT / "EXTRACTION_PROOF.json", result)
    return {
        **result,
        "new_rows": new_rows,
        "cpu_wall_seconds_this_invocation": time.perf_counter() - start_time,
    }


def run() -> dict:
    """Own one cache writer independently of the live C1 GPU training lock."""
    import msvcrt

    if torch.cuda.is_initialized():
        raise ValueError("Historical H extraction is CPU-only")
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "WRITER.lock").open("a+b") as lock:
        if lock.tell() == 0:
            lock.write(b"0")
            lock.flush()
        lock.seek(0)
        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        try:
            return extract()
        finally:
            lock.seek(0)
            msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)


if __name__ == "__main__":
    print(run())

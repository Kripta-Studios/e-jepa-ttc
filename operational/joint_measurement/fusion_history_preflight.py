"""CPU-only four-query RGB history adapter proof; no scientific fusion fit."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from e_jepa_ttc.simplex_t.model import TemporalConfig, TemporalRefiner
from operational.range_transfer.common import digest, freeze, read, sha, utc
from operational.rgb_port.predict_heads import _forward

from .common import ROOT, RUN, owner, require_resources
from .fusion_cache import FIELDS, FusionCacheWriter
from .fusion_features import FeatureBlock, FeatureIdentity
from .fusion_history_extract import FEATURE_NAMES, SCHEMA_SHA256, extract_history
from .fusion_pair_extract import clock_pairs
from .train import committed_sources


def run() -> dict:
    """Verify original endpoint ancestry, native head output and label-free cache IO."""
    if torch.cuda.is_initialized():
        raise ValueError("RGB history preflight must run separately on CPU")
    require_resources()
    torch.set_num_threads(1)
    root = owner()
    out = RUN / "diagnostics/fusion_history_H_v1"
    old = read(RUN / "verification/HISTORICAL_MULTIMODAL_ADMISSION.json")
    parent = old["heads"]["R_CTX"]
    checkpoint = root / "fits/R_CTX/checkpoint_last.pt"
    cache = root / "features/H/R_HISTORY.npz"
    manifest = read(root / "H_MANIFEST.json")
    expected = {
        checkpoint: parent["checkpoint_sha256"],
        cache: parent["H_cache_sha256"],
        root / "features/H/R_NORMALIZER.json": parent["normalizer_sha256"],
        root / "rounds/PRODUCERS_PAIR_ENDPOINT_FREEZE.json": parent["parent_sha256"],
        root / "H_MANIFEST.json": old["manifest_hashes"]["H"],
        Path(manifest["rows_path"]): manifest["rows_sha256"],
    }
    for path, checksum in expected.items():
        if sha(path) != checksum:
            raise ValueError(f"Historical RGB identity changed: {path.name}")
    paths = [
        "operational/joint_measurement/" + name + ".py"
        for name in (
            "fusion_history_extract",
            "fusion_history_preflight",
            "fusion_cache",
            "fusion_features",
            "fusion_pair_extract",
        )
    ] + [
        "src/e_jepa_ttc/simplex_t/model.py",
        "src/e_jepa_ttc/simplex_t/phase.py",
        "operational/rgb_port/predict_heads.py",
    ]
    sources = {path: sha(ROOT / path) for path in paths}
    commit = committed_sources(sources)
    registration = {
        "indices": [0, 17, 83, 159],
        "role": "H_input_only_diagnostic",
        "checkpoint_sha256": sha(checkpoint),
        "cache_sha256": sha(cache),
        "normalizer_sha256": parent["normalizer_sha256"],
        "manifest_sha256": sha(root / "H_MANIFEST.json"),
        "feature_names": FEATURE_NAMES,
        "schema_sha256": SCHEMA_SHA256,
        "sources": sources,
        "H_supervised_ancestry": True,
        "independent_H_evaluation": False,
        "target_columns_loaded": False,
        "optimizer_updates": 0,
        "cuda": False,
        "selects_fusion_parents": False,
    }
    freeze(out / "REGISTRATION.json", registration)
    if (out / "PASS.json").exists():
        proof = read(out / "PASS.json")
        if proof["registration_sha256"] != sha(out / "REGISTRATION.json"):
            raise ValueError("History preflight identity changed")
        return proof
    rows = pd.read_parquet(
        manifest["rows_path"],
        columns=["sample_token", "rgb_available", "producer_sensor_times_us"],
    )
    ids = registration["indices"]
    names = ("features", "timing", "valid", "expert_phase", "sample_token", "role")
    with np.load(cache, allow_pickle=False) as archive:
        tokens = archive["sample_token"].astype(str)
        if not np.array_equal(tokens, rows.sample_token.to_numpy()) or not np.all(
            archive["role"] == "H"
        ):
            raise ValueError("Historical cache order or population changed")
        values = {name: archive[name][ids].copy() for name in names}
    if not rows.iloc[ids].rgb_available.all():
        raise ValueError("Preflight requires declared available RGB")
    model = TemporalRefiner(TemporalConfig(hidden=160))
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.eval().requires_grad_(False)
    weights = {key: value.clone() for key, value in model.state_dict().items()}
    rng = torch.get_rng_state().clone()
    inputs = [
        torch.from_numpy(values[key]) for key in ("features", "timing", "valid", "expert_phase")
    ]
    features, predictions = extract_history(model, *inputs)
    reference = _forward(model, "R_CTX", values)["prediction_ttc"]
    np.testing.assert_array_equal(predictions.astype(np.float64), reference)
    times = np.stack([value[-2:] for value in rows.iloc[ids].producer_sensor_times_us])
    clocks = clock_pairs(
        rows.iloc[ids],
        modality="rgb",
        delta_t_s=((times[:, 1] - times[:, 0]) / 1_000_000).astype(np.float32),
        available=np.ones(len(ids), np.bool_),
    )
    identity = FeatureIdentity("rgb", sha(checkpoint), SCHEMA_SHA256, digest(sources))
    block = FeatureBlock(
        identity,
        "H",
        registration["manifest_sha256"],
        values["sample_token"],
        features,
        predictions,
        np.ones(len(ids), np.bool_),
        clocks,
    )
    sink = FusionCacheWriter(
        out / "rgb",
        identity,
        role="H",
        manifest_sha256=block.manifest_sha256,
        sample_tokens=block.sample_tokens,
        feature_count=len(FEATURE_NAMES),
        chunk_size=2,
        input_binding={"diagnostic_registration_sha256": sha(out / "REGISTRATION.json")},
    )
    for index in range(sink.chunks):
        start, stop = sink.bounds(index)
        if sink.read_chunk(index) is None:
            sink.write_chunk(
                index, replace(block, **{key: getattr(block, key)[start:stop] for key in FIELDS})
            )
    receipt = sink.complete()
    loaded = sink.load(out / "rgb/features.npz", receipt["sha256"], block.sample_tokens)
    checks = {key: np.array_equal(getattr(block, key), getattr(loaded, key)) for key in FIELDS}
    checks["native_predictions_exact"] = np.array_equal(predictions.astype(np.float64), reference)
    checks["weights_unchanged"] = all(
        torch.equal(value, weights[key]) for key, value in model.state_dict().items()
    )
    checks["rng_unchanged"] = torch.equal(rng, torch.get_rng_state())
    proof = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "utc": utc(),
        "registration_sha256": sha(out / "REGISTRATION.json"),
        "source_commit": commit,
        "queries": len(ids),
        "optimizer_updates": 0,
        "target_columns_loaded": False,
        "H_supervised_ancestry": True,
        "independent_H_evaluation": False,
    }
    if proof["status"] != "PASS":
        raise ValueError(proof)
    freeze(out / "PASS.json", proof)
    return proof


if __name__ == "__main__":
    print(run())

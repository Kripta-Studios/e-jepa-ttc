"""Four-query CPU preflight of new pair features; no parent selection or accuracy claim."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from e_jepa_ttc.simplex_t.phase import phase_to_ttc
from operational.range_transfer.common import digest, freeze, read, sha, utc

from .common import ROOT, RUN, owner, require_resources
from .data import SensorDataset, cache_root
from .evaluate import freeze_primary
from .fusion_cache import FIELDS, FusionCacheWriter
from .fusion_features import FeatureBlock, FeatureIdentity
from .fusion_pair_extract import FEATURE_NAMES, SCHEMA_SHA256, clock_pairs, extract_pair
from .train import committed_sources, initialize, setup


def run() -> dict:
    """Verify the trained event producer, cache round-trip and unchanged weights on real P."""
    if torch.cuda.is_initialized():
        raise ValueError("This preflight is CPU-only and must run in its own process")
    require_resources(starting=True)
    root = RUN / "diagnostics/fusion_features_pair_P_v1"
    fit = "PAIR-CORR-S7"
    endpoints = freeze_primary()["endpoints"]
    paths = [
        f"operational/joint_measurement/{name}.py"
        for name in (
            "fusion_extract_preflight",
            "fusion_pair_extract",
            "fusion_cache",
            "fusion_features",
        )
    ] + ["src/e_jepa_ttc/joint_measurement/model.py", "src/e_jepa_ttc/simplex_t/phase.py"]
    binding_sources = {path: sha(ROOT / path) for path in paths}
    commit = committed_sources(binding_sources)
    manifest_path = owner() / "P_MANIFEST.json"
    manifest = read(manifest_path)
    if sha(Path(manifest["rows_path"])) != manifest["rows_sha256"]:
        raise ValueError("P query metadata changed")
    if sha(manifest_path) != read(cache_root("P") / "BINDING.json")["manifest_sha256"]:
        raise ValueError("P sensory population changed")
    registration = {
        "role": "P_diagnostic_subset",
        "indices": [0, 17, 83, 159],
        "fit_id": fit,
        "endpoint_sha256": endpoints[fit],
        "sources": binding_sources,
        "schema_sha256": SCHEMA_SHA256,
        "feature_names": list(FEATURE_NAMES),
        "manifest_sha256": sha(manifest_path),
        "sensor_complete_sha256": sha(cache_root("P") / "COMPLETE.json"),
        "optimizer_updates": 0,
        "cuda": False,
        "selects_fusion_parents": False,
    }
    freeze(root / "REGISTRATION.json", registration)
    if (root / "PASS.json").exists():
        proof = read(root / "PASS.json")
        if proof["registration_sha256"] != sha(root / "REGISTRATION.json"):
            raise ValueError("Pair feature preflight binding changed")
        return proof
    setup(7)
    model = initialize(fit)
    payload = torch.load(
        RUN / "fits" / fit / "checkpoint_last.pt", map_location="cpu", weights_only=False
    )
    model.load_state_dict(payload["model_state_dict"], strict=True)
    del payload
    model.eval().requires_grad_(False)
    initial = {name: value.clone() for name, value in model.state_dict().items()}
    dataset = SensorDataset("P")
    try:
        indices = registration["indices"]
        batch = dataset.batch(indices)
        features, prediction = extract_pair(model, batch.events, batch.delta_t_s)
        with torch.inference_mode():
            reference = phase_to_ttc(model(batch.events, batch.delta_t_s)["raw_phase"]).numpy()
        np.testing.assert_array_equal(prediction, reference)
        clocks = clock_pairs(
            dataset.rows.iloc[indices],
            modality="event",
            delta_t_s=batch.delta_t_s.numpy(),
            available=np.ones(len(indices), np.bool_),
        )
        identity = FeatureIdentity("event", endpoints[fit], SCHEMA_SHA256, digest(binding_sources))
        block = FeatureBlock(
            identity,
            "P",
            registration["manifest_sha256"],
            np.array(batch.sample_tokens),
            features,
            prediction,
            np.ones(len(indices), np.bool_),
            clocks,
        )
        sink = FusionCacheWriter(
            root / "event",
            identity,
            role="P",
            manifest_sha256=block.manifest_sha256,
            sample_tokens=block.sample_tokens,
            feature_count=196,
            chunk_size=2,
            input_binding={"diagnostic_registration_sha256": sha(root / "REGISTRATION.json")},
        )
        for index in range(sink.chunks):
            start, stop = sink.bounds(index)
            if sink.read_chunk(index) is None:
                sink.write_chunk(
                    index,
                    replace(block, **{name: getattr(block, name)[start:stop] for name in FIELDS}),
                )
        receipt = sink.complete()
        reloaded = sink.load(root / "event/features.npz", receipt["sha256"], block.sample_tokens)
        for name in FIELDS:
            np.testing.assert_array_equal(getattr(block, name), getattr(reloaded, name))
        assert all(torch.equal(initial[name], value) for name, value in model.state_dict().items())
        assert not torch.cuda.is_initialized()
        result = {
            "status": "PASS",
            "utc": utc(),
            "git_commit": commit,
            "registration_sha256": sha(root / "REGISTRATION.json"),
            "feature_cache_sha256": receipt["sha256"],
            "rows": len(indices),
            "features": 196,
            "signed_predictions_bit_exact": True,
            "lossless_reload_all_arrays_exact": True,
            "model_state_unchanged": True,
            "cuda_initialized": False,
            "optimizer_updates": 0,
            "scientific_fusion_recipe_registered": False,
            "accuracy_evaluation": False,
            "scope": "Four diagnostic P queries on the closed event PAIR-CORR endpoint",
        }
        freeze(root / "PASS.json", result)
        return result
    finally:
        dataset.close()


if __name__ == "__main__":
    print(run())

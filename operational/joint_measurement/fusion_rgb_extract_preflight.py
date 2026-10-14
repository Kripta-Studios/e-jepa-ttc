"""CPU-only real-P RGB descriptor preflight; no early H or parent selection."""

from __future__ import annotations

import numpy as np
import torch

from e_jepa_ttc.simplex_t.phase import phase_to_ttc
from operational.range_transfer.common import digest, freeze, read, sha, utc

from .common import ROOT, RUN, owner, require_resources
from .fusion_cache import FIELDS, FusionCacheWriter
from .fusion_features import FeatureBlock, FeatureIdentity
from .fusion_pair_extract import FEATURE_NAMES, SCHEMA_SHA256, clock_pairs, extract_pair
from .rgb_data import RGBPairDataset
from .rgb_evaluate import load_model
from .rgb_train import REGISTRATION, registered
from .train import committed_sources, setup

OUT = RUN / "diagnostics/fusion_features_rgb_P_v1"
FIT = "RGB-PAIR-CORR-FT-S7"


def run() -> dict:
    """Check a closed RGB endpoint's descriptors, native clock and lossless feature cache."""
    if torch.cuda.is_initialized():
        raise ValueError("Run this CPU preflight in a separate process")
    require_resources(starting=True)
    recipe = registered()
    directory = RUN / "rgb/fits" / FIT
    receipt = read(directory / "CHECKPOINT_RECEIPT.json")
    journal = read(directory / "UPDATE_JOURNAL.json")
    endpoint = sha(directory / "checkpoint_last.pt")
    if (
        receipt["status"] != "COMPLETE"
        or receipt["completed_updates"] != 12000
        or journal["completed_updates"] != journal["durable_updates"]
        or journal["durable_updates"] != 12000
        or journal["pending_update_upper"] != 0
        or receipt["checkpoint_sha256"] != endpoint
    ):
        raise ValueError("RGB P diagnostic requires the independently completed FT endpoint")
    sources = {
        **recipe["sources"],
        **{
            f"operational/joint_measurement/{name}.py": sha(
                ROOT / f"operational/joint_measurement/{name}.py"
            )
            for name in (
                "fusion_rgb_extract_preflight",
                "fusion_pair_extract",
                "fusion_cache",
                "fusion_features",
                "rgb_evaluate",
            )
        },
    }
    commit = committed_sources(sources)
    registration = {
        "role": "P_diagnostic_subset",
        "indices": [0, 17, 83, 159],
        "fit_id": FIT,
        "endpoint_sha256": endpoint,
        "RGB_registration_sha256": sha(REGISTRATION),
        "sources": sources,
        "input_identity": recipe["input_identity"],
        "schema_sha256": SCHEMA_SHA256,
        "feature_names": list(FEATURE_NAMES),
        "optimizer_updates": 0,
        "cuda": False,
        "accuracy_evaluation": False,
        "selects_fusion_parents": False,
    }
    freeze(OUT / "REGISTRATION.json", registration)
    if (OUT / "PASS.json").exists():
        proof = read(OUT / "PASS.json")
        if proof["registration_sha256"] != sha(OUT / "REGISTRATION.json"):
            raise ValueError("RGB preflight registration changed")
        return proof
    setup(7)
    model = load_model(FIT).eval().requires_grad_(False)
    initial = {name: value.clone() for name, value in model.state_dict().items()}
    dataset = RGBPairDataset("P")
    try:
        cache = dataset.source.prepared_cache
        actual_input = {
            "P_manifest_sha256": sha(owner() / "P_MANIFEST.json"),
            "P_rows_sha256": sha(dataset.source.rows_path),
            "cache_binding_sha256": cache.binding_sha256,
            "cache_manifest_sha256": cache.manifest_sha256,
            "population": dataset.population,
        }
        if actual_input != recipe["input_identity"]:
            raise ValueError("Real RGB inputs differ from the training admission")
        indices = registration["indices"]
        batch = dataset.batch(indices)
        features, predictions = extract_pair(model, batch.events, batch.delta_t_s)
        with torch.inference_mode():
            reference = phase_to_ttc(model(batch.events, batch.delta_t_s)["raw_phase"]).numpy()
        np.testing.assert_array_equal(predictions, reference)
        available = np.ones(len(indices), np.bool_)
        clocks = clock_pairs(
            dataset.rows.iloc[indices],
            modality="rgb",
            delta_t_s=batch.delta_t_s.numpy(),
            available=available,
        )
        identity = FeatureIdentity("rgb", endpoint, SCHEMA_SHA256, digest(sources))
        block = FeatureBlock(
            identity,
            "P",
            actual_input["P_manifest_sha256"],
            np.array(batch.sample_tokens),
            features,
            predictions,
            available,
            clocks,
        )
        sink = FusionCacheWriter(
            OUT / "rgb",
            identity,
            role="P",
            manifest_sha256=block.manifest_sha256,
            sample_tokens=block.sample_tokens,
            feature_count=196,
            chunk_size=len(indices),
            input_binding={"diagnostic_registration_sha256": sha(OUT / "REGISTRATION.json")},
        )
        if sink.read_chunk(0) is None:
            sink.write_chunk(0, block)
        cached = sink.complete()
        reloaded = sink.load(OUT / "rgb/features.npz", cached["sha256"], block.sample_tokens)
        for name in FIELDS:
            np.testing.assert_array_equal(getattr(block, name), getattr(reloaded, name))
        if not all(torch.equal(initial[key], value) for key, value in model.state_dict().items()):
            raise ValueError("RGB producer weights changed during extraction")
        if torch.cuda.is_initialized():
            raise ValueError("Unexpected CUDA initialization")
        result = {
            "status": "PASS",
            "utc": utc(),
            "git_commit": commit,
            "registration_sha256": sha(OUT / "REGISTRATION.json"),
            "feature_cache_sha256": cached["sha256"],
            "rows": len(indices),
            "features": 196,
            "signed_predictions_bit_exact": True,
            "lossless_reload_all_arrays_exact": True,
            "model_state_unchanged": True,
            "own_sensor_clock_verified": True,
            "sensor_pairs_us": clocks.tolist(),
            "native_deltas_s": batch.delta_t_s.tolist(),
            "cuda_initialized": False,
            "optimizer_updates": 0,
            "scientific_fusion_recipe_registered": False,
            "accuracy_evaluation": False,
        }
        freeze(OUT / "PASS.json", result)
        return result
    finally:
        dataset.close()


if __name__ == "__main__":
    print(run())

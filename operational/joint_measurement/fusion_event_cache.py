"""Extract H features from an explicitly selected frozen V18 event endpoint.

Selection is external and must follow completed C1 interpretation. This runner
does not rank models, read V, fit a normalizer or train a fusion head.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from operational.range_transfer.common import digest, freeze, read, sha

from .common import ROOT, RUN, gpu_charge, owner, require_resources
from .data import SensorDataset, cache_root
from .evaluate import freeze_primary
from .fusion_features import FeatureIdentity
from .fusion_pair_cache import extract_shards
from .fusion_pair_extract import FEATURE_NAMES, SCHEMA_SHA256
from .geometry_evaluate import freeze_endpoint, load_model
from .geometry_graph_probe import no_live_training_driver
from .geometry_train import DIRECTORY, FIT
from .rgb_run import require_authorization
from .train import committed_sources, initialize, setup

OUT = RUN / "fusion/inputs/H/event"


def register(fit_id: str, rationale: str) -> dict:
    """Freeze a deliberate event-parent choice after C1's complete evidence exists."""
    if fit_id not in {FIT, "PAIR-CORR-S7"} or not rationale.strip():
        raise ValueError("Declare a supported spatial parent and an evidence-based rationale")
    c1 = read(DIRECTORY / "evidence/COMPLETE.json")
    if c1["status"] != "C1_READY_FOR_INTERPRETATION":
        raise ValueError("Complete and interpret C1 before selecting fusion event features")
    endpoints = freeze_endpoint()["endpoints"] if fit_id == FIT else freeze_primary()["endpoints"]
    h_manifest = owner() / "H_MANIFEST.json"
    manifest = read(h_manifest)
    sensory = cache_root("H")
    if (
        sha(h_manifest) != read(sensory / "BINDING.json")["manifest_sha256"]
        or sha(Path(manifest["rows_path"])) != manifest["rows_sha256"]
    ):
        raise ValueError("Event H sensor population changed")
    if read(sensory / "COMPLETE.json")["rows"] != manifest["population_size"]:
        raise ValueError("Event H sensory cache is incomplete")
    names = (
        "fusion_event_cache",
        "fusion_pair_cache",
        "fusion_pair_extract",
        "fusion_cache",
        "fusion_features",
        "data",
        "train",
        "geometry_model",
        "geometry_evaluate",
    )
    paths = [f"operational/joint_measurement/{name}.py" for name in names]
    paths += ["src/e_jepa_ttc/joint_measurement/model.py", "src/e_jepa_ttc/simplex_t/phase.py"]
    sources = {path: sha(ROOT / path) for path in paths}
    committed_sources(sources)
    record = {
        "fit_id": fit_id,
        "rationale": rationale,
        "endpoint_sha256": endpoints[fit_id],
        "role": "H",
        "manifest_sha256": sha(h_manifest),
        "rows_sha256": manifest["rows_sha256"],
        "population": manifest["population_size"],
        "sources": sources,
        "sensor_binding_sha256": sha(sensory / "BINDING.json"),
        "sensor_complete_sha256": sha(sensory / "COMPLETE.json"),
        "C1_complete_evidence_sha256": sha(DIRECTORY / "evidence/COMPLETE.json"),
        "C1_selection_sha256": sha(DIRECTORY / "SELECTION_FINAL.json"),
        "schema_sha256": SCHEMA_SHA256,
        "feature_names": list(FEATURE_NAMES),
        "batch_size": 8,
        "chunk_size": 256,
        "device": "cuda:0",
        "precision": "FP32_no_TF32",
        "parent_training_role": "P",
        "old_neural_feature_cache_reused": False,
        "model_inputs": "native event P2 pixels and measured interval only",
        "labels_in_loader": (
            "native loader includes targets, never passed to extractor/model or output cache"
        ),
        "optimizer_updates": 0,
        "V_opened": False,
    }
    freeze(OUT / "REGISTRATION.json", record)
    return record


def extract() -> dict:
    """Advance one charged GPU scope with complete source/endpoint admission."""
    previous = read(OUT / "REGISTRATION.json")
    record = register(previous["fit_id"], previous["rationale"])
    no_live_training_driver()
    require_authorization()
    require_resources(starting=True)
    setup(7)
    dataset = SensorDataset("H")
    try:
        with gpu_charge("fusion_event_H_feature_extraction", reserve=900) as deadline:
            if record["fit_id"] == FIT:
                model = load_model()
            else:
                model = initialize(record["fit_id"])
                payload = torch.load(
                    RUN / "fits" / record["fit_id"] / "checkpoint_last.pt",
                    map_location="cpu",
                    weights_only=False,
                )
                model.load_state_dict(payload["model_state_dict"], strict=True)
                del payload
            device = torch.device(record["device"])
            model.to(device).eval().requires_grad_(False)
            identity = FeatureIdentity(
                "event", record["endpoint_sha256"], SCHEMA_SHA256, digest(record["sources"])
            )
            result = {"status": "PAUSED"}
            while time.perf_counter() < deadline:
                result = extract_shards(
                    OUT,
                    model,
                    dataset,
                    identity,
                    manifest_sha256=record["manifest_sha256"],
                    available=np.ones(dataset.population, np.bool_),
                    input_binding={"registration_sha256": sha(OUT / "REGISTRATION.json")},
                    batch_size=record["batch_size"],
                    chunk_size=record["chunk_size"],
                    max_new_chunks=1,
                    device=device,
                )
                if result["status"] == "COMPLETE":
                    return result
            return result
    finally:
        dataset.close()


def run() -> dict:
    """Hold one event feature writer; the GPU lock remains separately enforced."""
    import msvcrt

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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--register-fit", choices=[FIT, "PAIR-CORR-S7"])
    parser.add_argument("--rationale", default="")
    args = parser.parse_args()
    print(register(args.register_fit, args.rationale) if args.register_fit else run())

"""Build each replica's own features/normalizer and all four preregistered H heads."""

from __future__ import annotations

import time
from dataclasses import asdict
from pathlib import Path
from typing import cast

import numpy as np
import torch

from e_jepa_ttc.models.causal_scale_ttc import CausalScaleTTC
from e_jepa_ttc.rgb_port.normalization import FrozenNormalizer
from operational.range_transfer.common import digest, freeze, read, sha, write

from .common import ROOT, gpu_charge, owner, require_resources
from .data import SensorDataset, cache_root, role_rows
from .fusion_data import FusionTensorBank
from .fusion_engine import FusionRecipe, segment
from .fusion_features import FeatureBlock, FeatureIdentity, load_feature_cache
from .fusion_pair_cache import extract_shards
from .fusion_pair_extract import SCHEMA_SHA256
from .fusion_replication_train import DIRECTORY, REGISTRATION, SEEDS, registered
from .fusion_train import DIRECTORY as ORIGINAL
from .fusion_train import inputs, supervision
from .geometry_model import GeometricPairMeasurement
from .rgb_run import require_authorization
from .train import committed_sources, initialize, setup
from .verify_rgb_endpoint import inspect_payload

PIPELINE = DIRECTORY / "PIPELINE_REGISTRATION.json"


def producer(seed: int) -> tuple[GeometricPairMeasurement, dict]:
    """Admit a complete replica producer and independently inspect its whole trajectory."""
    record = registered()
    if seed not in SEEDS:
        raise ValueError("Unregistered replica seed")
    folder = DIRECTORY / "fits" / f"C1-FUSION-REPLICA-S{seed}"
    receipt = read(folder / "CHECKPOINT_RECEIPT.json")
    journal, pointer = (
        read(folder / "UPDATE_JOURNAL.json"),
        read(folder / "CHECKPOINT_POINTER.json"),
    )
    identity = read(folder / "IDENTITY.json")
    if not (
        receipt["status"] == "COMPLETE"
        and receipt["scientific_endpoint"]
        and receipt["completed_updates"]
        == journal["completed_updates"]
        == journal["durable_updates"]
        == pointer["completed_updates"]
        == 12000
        and journal["pending_update_upper"] == 0
        and identity["seed"] == seed
        and identity["registration_sha256"] == sha(REGISTRATION)
        and sha(folder / "checkpoint_last.pt")
        == receipt["checkpoint_sha256"]
        == pointer["checkpoint_sha256"]
        == sha(folder / "checkpoint_versions" / pointer["version"])
    ):
        raise ValueError("Replica producer is incomplete or its identity changed")
    payload = torch.load(folder / "checkpoint_last.pt", map_location="cpu", weights_only=False)
    if payload["identity"] != identity or payload["identity_sha256"] != receipt["identity_sha256"]:
        raise ValueError("Replica payload genealogy changed")
    setup(seed)
    model = GeometricPairMeasurement(cast(CausalScaleTTC, initialize("A5-P2-FROZEN")))
    proof = inspect_payload(
        model,
        payload,
        updates=12000,
        population=record["population_size"],
        batch_size=32,
        seed=seed,
        frozen=False,
    )
    proof["checkpoint_sha256"] = receipt["checkpoint_sha256"]
    freeze(folder / "ENDPOINT_VERIFICATION.json", proof)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    return model.eval().requires_grad_(False), proof


def register() -> dict:
    """Fix extraction and head implementation after both producers, before head updates."""
    registered()
    endpoints = {}
    for seed in SEEDS:
        model, proof = producer(seed)
        endpoints[str(seed)] = proof["checkpoint_sha256"]
        del model
    blocks, _, bindings = inputs()
    sources = {
        f"operational/joint_measurement/{name}.py": sha(
            ROOT / f"operational/joint_measurement/{name}.py"
        )
        for name in (
            "fusion_replication_pipeline",
            "fusion_pair_cache",
            "fusion_pair_extract",
            "fusion_cache",
            "fusion_features",
            "fusion_engine",
            "fusion_data",
            "fusion_model",
            "geometry_model",
            "verify_rgb_endpoint",
        )
    }
    committed_sources(sources)
    admission = {
        "replication_registration_sha256": sha(REGISTRATION),
        "endpoints": endpoints,
        "rgb": bindings["rgb"],
        "rgb_identity": asdict(blocks["rgb"].identity),
        "sources": sources,
        "event_schema_sha256": SCHEMA_SHA256,
        "event_extractor_sha256": digest(sources),
        "batch_size": 8,
        "chunk_size": 256,
        "precision": "FP32_no_TF32",
        "H_manifest_sha256": sha(owner() / "H_MANIFEST.json"),
        "head_recipe_rule": "Exact registered seed7 recipe with corresponding seed13/23 only",
        "V_gate": "Both producers and all four heads complete and frozen before V features",
        "H_labels": "Native global mass, sequential H supervision, never independent validation",
    }
    freeze(PIPELINE, admission)
    return admission


def feature_identity(seed: int, record: dict) -> FeatureIdentity:
    """Each trained producer gets its own genealogy despite an identical feature width."""
    return FeatureIdentity(
        "event",
        record["endpoints"][str(seed)],
        record["event_schema_sha256"],
        record["event_extractor_sha256"],
    )


def cache(seed: int, role: str) -> dict:
    """Extract complete lossless shards with pixel/time-only forward and native clocks."""
    record = read(PIPELINE)
    for path, expected in record["sources"].items():
        if sha(ROOT / path) != expected:
            raise ValueError("Replica feature/head implementation changed")
    if role == "V":
        freeze_all()
    elif role != "H":
        raise ValueError("Only declared replica H/V populations are admitted")
    manifest_path = owner() / f"{role}_MANIFEST.json"
    manifest = read(manifest_path)
    if (
        sha(Path(manifest["rows_path"])) != manifest["rows_sha256"]
        or read(cache_root(role) / "BINDING.json")["manifest_sha256"] != sha(manifest_path)
        or read(cache_root(role) / "COMPLETE.json")["rows"] != manifest["population_size"]
    ):
        raise ValueError("Replica sensor cohort changed")
    directory = DIRECTORY / "inputs" / role / f"C1-S{seed}"
    extraction = {
        "pipeline_sha256": sha(PIPELINE),
        "role": role,
        "seed": seed,
        "manifest_sha256": sha(manifest_path),
        "sensor_binding_sha256": sha(cache_root(role) / "BINDING.json"),
        "sensor_complete_sha256": sha(cache_root(role) / "COMPLETE.json"),
    }
    freeze(directory / "REGISTRATION.json", extraction)
    if (directory / "COMPLETE.json").exists():
        event_block(seed, role)
        return read(directory / "COMPLETE.json")
    dataset = SensorDataset(role)
    try:
        with gpu_charge(f"replica{seed}_{role}_event_features", reserve=900) as deadline:
            model, _ = producer(seed)
            model.to("cuda:0")
            while time.perf_counter() < deadline:
                result = extract_shards(
                    directory,
                    model,
                    dataset,
                    feature_identity(seed, record),
                    manifest_sha256=sha(manifest_path),
                    available=np.ones(dataset.population, np.bool_),
                    input_binding={"registration_sha256": sha(directory / "REGISTRATION.json")},
                    batch_size=record["batch_size"],
                    chunk_size=record["chunk_size"],
                    max_new_chunks=1,
                    device=torch.device("cuda:0"),
                )
                if result["status"] == "COMPLETE":
                    return result
            return {"status": "PAUSED"}
    finally:
        dataset.close()


def event_block(seed: int, role: str) -> FeatureBlock:
    """Load independently bound feature bytes, never fit statistics on V."""
    record = read(PIPELINE)
    directory = DIRECTORY / "inputs" / role / f"C1-S{seed}"
    receipt, binding = read(directory / "COMPLETE.json"), read(directory / "BINDING.json")
    if (
        receipt["status"] != "COMPLETE"
        or receipt["binding"] != digest(binding)
        or binding["input_binding"]["caller"]
        != {"registration_sha256": sha(directory / "REGISTRATION.json")}
    ):
        raise ValueError("Replica feature cache caller changed")
    return load_feature_cache(
        directory / "features.npz",
        expected_sha256=receipt["sha256"],
        expected_identity=feature_identity(seed, record),
        role=role,
        manifest_sha256=sha(owner() / f"{role}_MANIFEST.json"),
        sample_tokens=role_rows(role).sample_token.to_numpy(dtype=str),
        feature_count=196,
    )


def fit_heads(seed: int) -> dict:
    """New encoder-specific H normalization; two unchanged paired head recipes."""
    record = registered()
    blocks, rows, _ = inputs()
    event = event_block(seed, "H")
    normalizer = event.fit_h_normalizer()
    normalizer_path = DIRECTORY / "normalizers" / f"event_S{seed}.json"
    freeze(
        normalizer_path,
        {
            "schema": "rgb_port_normalizer_v1",
            **asdict(normalizer),
            "mean": normalizer.mean.tolist(),
            "scale": normalizer.scale.tolist(),
        },
    )
    rgb_normalizer = FrozenNormalizer.load(ORIGINAL / "normalizers/rgb.json")
    target, mass = supervision(rows)
    pending = []
    for mode in record["head_modes"]:
        folder = DIRECTORY / "head_fits" / f"FUSION-{mode}-S{seed}"
        receipt_path = folder / "CHECKPOINT_RECEIPT.json"
        if receipt_path.exists() and read(receipt_path)["status"] == "COMPLETE":
            if (
                read(receipt_path)["completed_updates"] != 2500
                or sha(folder / "checkpoint_last.pt") != read(receipt_path)["checkpoint_sha256"]
            ):
                raise ValueError("Previously complete replica head changed")
        else:
            pending.append(mode)
    if not pending:
        return {"status": "COMPLETE"}
    with gpu_charge(f"replica{seed}_fusion_heads", reserve=900) as deadline:
        bank = FusionTensorBank(
            blocks["rgb"],
            event,
            rgb_normalizer,
            normalizer,
            role="H",
            manifest_sha256=sha(owner() / "H_MANIFEST.json"),
            sample_tokens=rows.sample_token.to_numpy(dtype=str),
            device="cuda:0",
        )
        for mode in pending:
            recipe = FusionRecipe(**{**record["head_recipes"][mode], "seed": seed})
            result = segment(
                DIRECTORY / "head_fits" / f"FUSION-{mode}-S{seed}",
                bank,
                target,
                mass,
                recipe,
                binding={
                    "pipeline_sha256": sha(PIPELINE),
                    "seed": seed,
                    "event_cache_sha256": read(
                        DIRECTORY / "inputs/H" / f"C1-S{seed}" / "COMPLETE.json"
                    )["sha256"],
                    "event_normalizer_sha256": sha(normalizer_path),
                },
                limit=2500,
                deadline=deadline,
            )
            if result["status"] != "COMPLETE":
                return result
        return {"status": "COMPLETE"}


def freeze_all() -> dict:
    """Require the entire six-fit replica graph before any V feature forward."""
    record = registered()
    pipeline = read(PIPELINE)
    producers, heads = {}, {}
    for seed in SEEDS:
        folder = DIRECTORY / "fits" / f"C1-FUSION-REPLICA-S{seed}"
        proof = read(folder / "ENDPOINT_VERIFICATION.json")
        if (
            proof["status"] != "PASS"
            or proof["checkpoint_sha256"] != sha(folder / "checkpoint_last.pt")
            or proof["checkpoint_sha256"] != pipeline["endpoints"][str(seed)]
        ):
            raise ValueError("Replica producer proof changed")
        producers[str(seed)] = proof["checkpoint_sha256"]
        for mode in record["head_modes"]:
            fit = f"FUSION-{mode}-S{seed}"
            directory = DIRECTORY / "head_fits" / fit
            receipt, journal = (
                read(directory / "CHECKPOINT_RECEIPT.json"),
                read(directory / "UPDATE_JOURNAL.json"),
            )
            pointer, identity = (
                read(directory / "CHECKPOINT_POINTER.json"),
                read(directory / "IDENTITY.json"),
            )
            if not (
                receipt["status"] == "COMPLETE"
                and receipt["scientific_endpoint"]
                and receipt["completed_updates"]
                == journal["completed_updates"]
                == journal["durable_updates"]
                == pointer["completed_updates"]
                == 2500
                and journal["pending_update_upper"] == 0
                and identity["recipe"] == {**record["head_recipes"][mode], "seed": seed}
                and identity["binding"]["pipeline_sha256"] == sha(PIPELINE)
                and identity["binding"]["event_normalizer_sha256"]
                == sha(DIRECTORY / "normalizers" / f"event_S{seed}.json")
                and identity["binding"]["event_cache_sha256"]
                == read(DIRECTORY / "inputs/H" / f"C1-S{seed}" / "COMPLETE.json")["sha256"]
                and sha(directory / "checkpoint_last.pt")
                == receipt["checkpoint_sha256"]
                == pointer["checkpoint_sha256"]
                == sha(directory / "checkpoint_versions" / pointer["version"])
            ):
                raise ValueError("All four replica heads must be complete and durable before V")
            heads[fit] = receipt["checkpoint_sha256"]
    result = {"pipeline_sha256": sha(PIPELINE), "producers": producers, "heads": heads}
    freeze(DIRECTORY / "ENDPOINTS.json", result)
    return result


def run() -> dict:
    """Resume stages under one writer lock; emit no replica scores before all fits freeze."""
    import msvcrt

    require_authorization()
    require_resources(starting=True)
    register()
    with (DIRECTORY / "PIPELINE.lock").open("a+b") as lock:
        if lock.tell() == 0:
            lock.write(b"0")
            lock.flush()
        lock.seek(0)
        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        try:
            for seed in SEEDS:
                while cache(seed, "H")["status"] != "COMPLETE":
                    pass
                while fit_heads(seed)["status"] != "COMPLETE":
                    pass
            endpoints = freeze_all()
            for seed in SEEDS:
                while cache(seed, "V")["status"] != "COMPLETE":
                    pass
            result = {
                "status": "REPLICA_INPUTS_AND_HEADS_COMPLETE",
                "endpoints": endpoints,
                "full_objective_complete": False,
                "next": "Complete paired three-seed evaluation, exports and full cost",
            }
            write(DIRECTORY / "PIPELINE_STATE.json", result)
            return result
        finally:
            lock.seek(0)
            msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)


if __name__ == "__main__":
    print(run())

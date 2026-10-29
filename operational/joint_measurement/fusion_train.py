"""Fixed TRUE/ZERO RGB-anchor experiment over independently admitted H inputs."""

from __future__ import annotations

import argparse
import os
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from operational.range_transfer.common import digest, freeze, read, sha, utc, write

from .common import ROOT, RUN, accounting, gpu_charge, owner, require_resources
from .fusion_cuda_qa import OUT as CUDA_QA
from .fusion_data import FusionTensorBank
from .fusion_engine import FusionRecipe, array_digest, segment, sources
from .fusion_features import FeatureBlock, FeatureIdentity, load_feature_cache
from .geometry_graph_probe import no_live_training_driver
from .rgb_run import require_authorization
from .train import committed_sources

DIRECTORY = RUN / "fusion"
REGISTRATION = DIRECTORY / "REGISTRATION.json"
MODES = ("TRUE", "ZERO")


def inputs() -> tuple[dict[str, FeatureBlock], pd.DataFrame, dict]:
    """Bind label-free feature bytes to external receipts and the original H cohort."""
    manifest_path = owner() / "H_MANIFEST.json"
    manifest = read(manifest_path)
    if sha(Path(manifest["rows_path"])) != manifest["rows_sha256"]:
        raise ValueError("H query metadata changed")
    rows = pd.read_parquet(manifest["rows_path"])
    if set(rows.role) != {"H"} or len(rows) != manifest["population_size"]:
        raise ValueError("Fusion fitting requires the complete original H population")
    tokens = rows.sample_token.to_numpy(dtype=str)
    blocks, bindings = {}, {}
    for modality, name, width in (("rgb", "R_CTX", 28), ("event", "event", 196)):
        directory = DIRECTORY / "inputs/H" / name
        extraction = read(directory / "REGISTRATION.json")
        binding = read(directory / "BINDING.json")
        receipt = read(directory / "COMPLETE.json")
        identity = FeatureIdentity(**binding["identity"])
        endpoint = extraction["parent_sha256" if modality == "rgb" else "endpoint_sha256"]
        if (
            receipt["status"] != "COMPLETE"
            or receipt["rows"] != len(rows)
            or receipt["feature_count"] != width
            or receipt["contains_targets"]
            or receipt["binding"] != digest(binding)
            or binding["input_binding"]["caller" if modality == "event" else "registration_sha256"]
            != (
                {"registration_sha256": sha(directory / "REGISTRATION.json")}
                if modality == "event"
                else sha(directory / "REGISTRATION.json")
            )
            or identity.modality != modality
            or identity.endpoint_sha256 != endpoint
            or identity.schema_sha256 != extraction["schema_sha256"]
            or identity.extractor_sha256 != digest(extraction["sources"])
            or extraction["manifest_sha256"] != sha(manifest_path)
        ):
            raise ValueError(f"Fusion {modality} extraction admission changed")
        for path, expected in extraction["sources"].items():
            if sha(ROOT / path) != expected:
                raise ValueError(f"Fusion feature source changed: {path}")
        blocks[modality] = load_feature_cache(
            directory / "features.npz",
            expected_sha256=receipt["sha256"],
            expected_identity=identity,
            role="H",
            manifest_sha256=sha(manifest_path),
            sample_tokens=tokens,
            feature_count=width,
        )
        bindings[modality] = {
            "cache_sha256": receipt["sha256"],
            "identity": asdict(identity),
            "extraction_registration_sha256": sha(directory / "REGISTRATION.json"),
        }
    return blocks, rows, bindings


def supervision(rows: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Preserve native H targets and equal-group query masses without band retuning."""
    target = rows.target_ttc.to_numpy(dtype=np.float32)
    mass = rows.mass.to_numpy(dtype=np.float64)
    if (
        not np.isfinite(target).all()
        or np.any(target == 0)
        or not np.isfinite(mass).all()
        or np.any(mass < 0)
        or not np.isclose(mass.sum(), 1, rtol=0, atol=1e-10)
    ):
        raise ValueError("Native H supervision or global mass is invalid")
    return target, mass


def register() -> dict:
    """Freeze the numerical paired experiment before either scientific update."""
    proof = read(CUDA_QA / "PASS.json")
    qa_record = read(CUDA_QA / "REGISTRATION.json")
    if proof["status"] != "PASS" or proof["registration_sha256"] != sha(
        CUDA_QA / "REGISTRATION.json"
    ):
        raise ValueError("Exact CUDA fusion resume proof required")
    for path, expected in qa_record["sources"].items():
        if sha(ROOT / path) != expected:
            raise ValueError("CUDA QA source changed")
    blocks, rows, bindings = inputs()
    target, mass = supervision(rows)
    normalizers = {}
    for modality, block in blocks.items():
        normalizer = block.fit_h_normalizer()
        payload = {
            "schema": "rgb_port_normalizer_v1",
            "modality": modality,
            "fit_role": "H",
            "schema_sha256": normalizer.schema_sha256,
            "producer_sha256": normalizer.producer_sha256,
            "unique_ids_sha256": normalizer.unique_ids_sha256,
            "mean": normalizer.mean.tolist(),
            "scale": normalizer.scale.tolist(),
        }
        path = DIRECTORY / "normalizers" / f"{modality}.json"
        freeze(path, payload)
        normalizers[modality] = sha(path)
    binding_sources = {
        **sources(),
        "operational/joint_measurement/fusion_train.py": sha(
            ROOT / "operational/joint_measurement/fusion_train.py"
        ),
    }
    committed_sources(binding_sources)
    recipes = {
        mode: asdict(FusionRecipe(7, mode, 2500, 128, 128, 0.03, 3e-4, 1e-3, 1.0)) for mode in MODES
    }
    record = {
        "schema": "v18_rgb_anchor_fusion_v1",
        "hypothesis": (
            "New C1 spatial motion features improve a fixed RGB anchor compared with "
            "an equal-capacity ZERO control; no standalone C1 promotion is assumed"
        ),
        "recipes": recipes,
        "inputs": bindings,
        "normalizers": normalizers,
        "sources": binding_sources,
        "manifest_sha256": sha(owner() / "H_MANIFEST.json"),
        "population": len(rows),
        "target_sha256": array_digest(target),
        "mass_sha256": array_digest(mass),
        "cuda_qa_sha256": sha(CUDA_QA / "PASS.json"),
        "parent_training_roles": {"C1": "P", "R_CTX": "H head on P producers"},
        "objective": "native H global mass; emitted phase L1 + 0.1 raw phase L1 divided by 0.03",
        "optimizer": "AdamW foreach=False; cosine decay over 2500 updates; no early stopping",
        "initialization": "same seed7 modules; zero final correction preserves RGB anchor",
        "only_control_difference": "ZERO sets all event features including timing to zero",
        "fallback": "exact C1 event TTC when RGB unavailable; no average of opposing TTC signs",
        "supervision": "H sequential supervised fitting on an H-trained RGB anchor; NOT OOF",
        "endpoint_gate": "Both fixed endpoints freeze before any V inference or H fit diagnostics",
        "evaluation": (
            "Both models on complete exposed V; fixed R_CTX/F_TRUE and equal-capacity ZERO "
            "controls; unchanged system gates and paired group bootstrap"
        ),
        "selection_limit": (
            "No hyperparameter or seed rescue after V; "
            "V is internal exposed development, not official test"
        ),
        "promotion": (
            "Requires original system gates and comparable full-pipeline cost; "
            "head-only export/cost insufficient"
        ),
        "physical_scientific_updates": 5000,
        "official_submission": False,
    }
    freeze(REGISTRATION, record)
    return record


def freeze_endpoints() -> dict:
    """Open no evaluation until both complete, durable head states match the recipe."""
    record = read(REGISTRATION)
    endpoints = {}
    for mode in MODES:
        directory = DIRECTORY / "fits" / f"FUSION-{mode}-S7"
        receipt = read(directory / "CHECKPOINT_RECEIPT.json")
        journal = read(directory / "UPDATE_JOURNAL.json")
        pointer = read(directory / "CHECKPOINT_POINTER.json")
        identity = read(directory / "IDENTITY.json")
        if not (
            receipt["status"] == "COMPLETE"
            and receipt["scientific_endpoint"]
            and receipt["completed_updates"]
            == journal["completed_updates"]
            == journal["durable_updates"]
            == pointer["completed_updates"]
            == 2500
            and journal["pending_update_upper"] == 0
            and identity["recipe"] == record["recipes"][mode]
            and identity["binding"] == {"registration_sha256": sha(REGISTRATION)}
            and identity["identity_sha256"]
            == receipt["identity_sha256"]
            == journal["identity_sha256"]
            == pointer["identity_sha256"]
            and sha(directory / "checkpoint_last.pt")
            == pointer["checkpoint_sha256"]
            == receipt["checkpoint_sha256"]
            and sha(directory / "checkpoint_versions" / pointer["version"])
            == receipt["checkpoint_sha256"]
        ):
            raise ValueError(
                "Both fusion endpoints must be complete and hash-bound before evaluation"
            )
        endpoints[f"FUSION-{mode}-S7"] = receipt["checkpoint_sha256"]
    result = {"registration_sha256": sha(REGISTRATION), "endpoints": endpoints}
    freeze(DIRECTORY / "ENDPOINTS.json", result)
    return result


def advance() -> dict:
    """Run one bounded scope on immutable resident inputs, preserving full resume state."""
    from e_jepa_ttc.rgb_port.normalization import FrozenNormalizer

    require_authorization()
    no_live_training_driver()
    require_resources(starting=True)
    record = register()
    blocks, rows, _ = inputs()
    target, mass = supervision(rows)
    with gpu_charge("fusion_scientific_H_training", reserve=900) as deadline:
        normalizers = [
            FrozenNormalizer.load(DIRECTORY / "normalizers" / f"{name}.json")
            for name in ("rgb", "event")
        ]
        bank = FusionTensorBank(
            blocks["rgb"],
            blocks["event"],
            *normalizers,
            role="H",
            manifest_sha256=record["manifest_sha256"],
            sample_tokens=rows.sample_token.to_numpy(dtype=str),
            device="cuda:0",
        )
        for mode in MODES:
            fit = f"FUSION-{mode}-S7"
            receipt = segment(
                DIRECTORY / "fits" / fit,
                bank,
                target,
                mass,
                FusionRecipe(**record["recipes"][mode]),
                binding={"registration_sha256": sha(REGISTRATION)},
                limit=2500,
                deadline=deadline,
            )
            if receipt["status"] != "COMPLETE":
                return {"status": "TRAINING", "fit": fit, "updates": receipt["completed_updates"]}
    return {
        "status": "FUSION_ENDPOINTS_FROZEN",
        **freeze_endpoints(),
        "full_objective_complete": False,
    }


def run() -> dict:
    """Own the sole fusion trainer; finish the fixed pair without opening V."""
    import msvcrt

    DIRECTORY.mkdir(parents=True, exist_ok=True)
    with (DIRECTORY / "DRIVER.lock").open("a+b") as lock:
        if lock.tell() == 0:
            lock.write(b"0")
            lock.flush()
        lock.seek(0)
        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        try:
            while True:
                try:
                    state = advance()
                except Exception as error:
                    write(
                        DIRECTORY / "RUNNER_STATE.json",
                        {
                            "status": "ACTION_REQUIRED",
                            "error": str(error),
                            "utc": utc(),
                            "pid": os.getpid(),
                        },
                    )
                    raise
                state.update(utc=utc(), pid=os.getpid(), accounting=accounting())
                write(DIRECTORY / "RUNNER_STATE.json", state)
                print(state, flush=True)
                if state["status"] == "FUSION_ENDPOINTS_FROZEN":
                    return state
        finally:
            lock.seek(0)
            msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--register-only", action="store_true")
    print(register() if parser.parse_args().register_only else run())

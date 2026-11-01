"""Diagnose the original native model's numerical policy after a sealed CUDA parity failure."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import tarfile
import time
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from e_jepa_ttc.rgb_port.data import TarFrameReader
from operational.final_refit.test12 import (
    TRUSTED_INPUTS_SHA256,
    FinalV16Train40Predictor,
    _contains_private_key,
)
from operational.phase_observation_cache.replay import capture_context
from operational.phase_robust.bundle import verified_upstream_paths
from operational.release_research.protocol import ResearchRun, sha256, write_json
from operational.rgb_port import transfer as engine
from operational.rgb_port_streaming.contracts import NativeObservation
from scripts.audit_phase_native_source_closure import DEPENDENCIES
from scripts.final_train40_transfer import load_models
from scripts.release_native_pair_replay import compare_arrays
from scripts.release_research import read, sources, verify_complete

PLAN = Path("docs/release_research_implementation_20261031/NATIVE_BACKEND_DIAGNOSIS.md")
POLICIES = ((False, False), (False, True), (True, False), (True, True))


def admit(decision: dict, profile: dict, replay_result: dict) -> None:
    """Keep a platform diagnostic from bypassing the population or original-parity gates."""
    if (
        decision.get("outcome") != "native_cuda_profile_parity_failed_candidate_not_run"
        or profile.get("device") != "cuda"
        or profile.get("profile_queries") != 2
        or profile.get("targets_read") is not False
        or profile.get("candidate_evaluated") is not False
        or replay_result.get("historical_parity") != "FAIL"
        or replay_result.get("candidate_executed") is not False
    ):
        raise ValueError("Require the terminal original-only CUDA profile parity failure")


def archived_contexts(archives: list[dict], queries: dict, parent: dict) -> dict:
    """Read only the preselected label-free diagnostic tensors from authenticated archives."""
    result = {}
    for entry in archives:
        path = Path(entry["path"])
        if sha256(path) != entry["sha256"]:
            raise ValueError("Historical archive changed")
        with tarfile.open(path) as archive:

            def member(name: str) -> bytes:
                stream = archive.extractfile(name)
                if stream is None:
                    raise ValueError("Missing archived member")
                return stream.read()

            names = archive.getnames()
            admission = json.loads(member(next(n for n in names if n.endswith(".admission.json"))))
            if (
                admission["model_seal_sha256"] != parent["model_seal_sha256"]
                or admission["inputs_sha256"] != TRUSTED_INPUTS_SHA256
            ):
                raise ValueError("Historical admission differs")
            for name in names:
                if "/diagnostics/" not in name or not name.endswith(".json"):
                    continue
                metadata = json.loads(member(name))
                token = metadata["sample_token"]
                if token not in queries:
                    continue
                if (
                    token in result
                    or metadata["private_targets_read"]
                    or metadata["run_binding_sha256"] != admission["run_binding_sha256"]
                ):
                    raise ValueError("Duplicate or invalid archived query")
                if any(
                    metadata[k] != queries[token][k]
                    for k in (
                        "event_history",
                        "rgb_history",
                        "query_time_us",
                        "sequence_id",
                        "track_id",
                    )
                ):
                    raise ValueError("Historical query identity differs")
                payload = member(name.removesuffix(".json") + ".npz")
                if hashlib.sha256(payload).hexdigest() != metadata["npz_sha256"]:
                    raise ValueError("Historical tensor checksum differs")
                with np.load(io.BytesIO(payload), allow_pickle=False) as arrays:
                    result[token] = {
                        k: arrays[k]
                        for k in arrays.files
                        if k.startswith(
                            ("E_CTX_MATCHED.", "R_CTX.", "PHASE_ONLY_EVENT_S7_TRAIN40.")
                        )
                    }
    if set(result) != set(queries):
        raise ValueError("Missing historical profile queries")
    return result


def diagnose(continuation: Path, baseline: Path, upstream: Path, output: Path) -> None:
    """Compare a fixed numerical-policy matrix without running canonical PAIR or reading labels."""
    verify_complete(continuation)
    capture, parity = continuation / "native_profile_cuda", continuation / "native_profile_parity"
    for folder in (capture, parity):
        verify_complete(folder)
    freeze = read(capture / "FREEZE.json")
    admit(read(continuation / "DECISION.json"), freeze["config"], read(parity / "RESULTS.json"))
    public = Path(freeze["input_paths"]["public_inputs"])
    if sha256(public) != TRUSTED_INPUTS_SHA256:
        raise ValueError("Public inputs changed")
    data = read(public)
    if _contains_private_key(data):
        raise ValueError("Private fields forbidden")
    queries = {q["sample_token"]: q for q in read(capture / "QUERIES.json")["queries"]}
    physical = set(data["physical_aliases"].values())
    expected = [q for q in data["queries"] if q["sample_token"] in physical][:2]
    if list(queries.values()) != expected:
        raise ValueError("Profile query selection changed")
    parent = read(upstream / "BUNDLE.json")["parent"]
    reference = read(baseline / "COMPLETE.json")
    if (
        reference["status"] != "COMPLETE"
        or reference["parent"] != parent
        or reference["inputs_sha256"] != TRUSTED_INPUTS_SHA256
    ):
        raise ValueError("Historical parent differs")
    archives = [
        a
        for a in reference["archives"].values()
        if any(q["sequence_id"] in Path(a["path"]).name for q in queries.values())
    ]
    paths = verified_upstream_paths(upstream)
    files = {
        **sources(),
        "preregistration": PLAN,
        "continuation": continuation / "COMPLETE.json",
        "capture": capture / "COMPLETE.json",
        "parity": parity / "COMPLETE.json",
        "public_inputs": public,
        "baseline": baseline / "COMPLETE.json",
        "upstream_bundle": upstream / "BUNDLE.json",
        **{"model:" + k: v for k, v in paths.items()},
        **{"archive:" + str(i): Path(a["path"]) for i, a in enumerate(archives)},
    }
    for name in (
        *DEPENDENCIES,
        "scripts/release_native_backend_diagnosis.py",
        "scripts/release_native_pair_replay.py",
        "scripts/final_train40_transfer.py",
        "operational/phase_observation_cache/replay.py",
        "operational/phase_robust/bundle.py",
    ):
        files["source:" + name] = Path(name)
    inventory_path = Path(freeze["input_paths"]["media_files"])
    if sha256(inventory_path) != freeze["inputs"]["media_files"]:
        raise ValueError("Media inventory changed")
    files["media_inventory"] = inventory_path
    inventory = {str(Path(r["path"]).resolve()): r for r in read(inventory_path)["verified"]}
    for query in queries.values():
        for identity in query["event_history"]:
            item = data["event_observations"][identity]
            path = Path(item["current_tensor_path"])
            if sha256(path) != item["current_tensor_sha256"]:
                raise ValueError("Event tensor changed")
            files["event:" + identity] = path
        for identity in query["rgb_history"]:
            for shard in data["rgb_observations"][identity]["producer_shards"]:
                path = (Path(data["media_root"]) / shard).resolve()
                if "rgb:" + shard not in files:
                    if sha256(path) != inventory[str(path)]["sha256"]:
                        raise ValueError("RGB shard changed")
                    files["rgb:" + shard] = path
        identity = hashlib.sha256(query["sample_token"].encode()).hexdigest()[:24]
        files["local_context:" + identity] = parity / "queries" / identity / "ORIGINAL_CONTEXT.npz"
    run = ResearchRun(
        output,
        files,
        {
            "experiment_id": "native-original-backend-diagnosis-v1",
            "targets_read": False,
            "candidate_evaluated": False,
            "optimizer_updates": 0,
            "policies_matmul_cudnn": POLICIES,
            "device": "cuda",
            "profile_queries": 2,
            "rtol": 1e-5,
            "atol": 1e-6,
            "ttc_rtol": 1e-4,
            "ttc_atol": 1e-4,
            "benchmark": False,
            "deterministic_algorithms": True,
            "torch_version": torch.__version__,
        },
    )
    started = time.perf_counter()
    saved = (
        torch.backends.cuda.matmul.allow_tf32,
        torch.backends.cudnn.allow_tf32,
        torch.backends.cudnn.benchmark,
        torch.are_deterministic_algorithms_enabled(),
    )
    try:
        historical = archived_contexts(archives, queries, parent)
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.set_num_threads(1)
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
        models, normalizers = load_models(paths, "cuda")
        results = {}
        for matmul, cudnn in POLICIES:
            torch.backends.cuda.matmul.allow_tf32 = matmul
            torch.backends.cudnn.allow_tf32 = cudnn
            name = f"matmul_{int(matmul)}_cudnn_{int(cudnn)}"
            native = FinalV16Train40Predictor.__new__(FinalV16Train40Predictor)
            native.models, native.normalizers, native.inputs = models, normalizers, data
            native.device = torch.device("cuda")
            native.event_values, native.rgb_values = {}, {}
            native.reader = TarFrameReader(
                Path(data["media_root"]), max_open=2, max_frame_cache_bytes=128 * 1024**2
            )
            results[name] = {"allow_tf32_matmul": matmul, "allow_tf32_cudnn": cudnn, "queries": {}}
            try:
                for token, query in queries.items():
                    with (
                        patch.object(engine, "ProducerObservation", NativeObservation),
                        torch.inference_mode(),
                        capture_context(models) as context,
                    ):
                        prediction = native.predict(query, data)
                    identity = hashlib.sha256(token.encode()).hexdigest()[:24]
                    if not matmul and not cudnn:
                        with np.load(
                            files["local_context:" + identity], allow_pickle=False
                        ) as prior:
                            if set(prior.files) != set(context):
                                raise ValueError("Local profile fields differ")
                            for key in prior.files:
                                np.testing.assert_array_equal(context[key], prior[key])
                    fields = compare_arrays(context, historical[token])
                    folder = output / name / identity
                    folder.mkdir(parents=True)
                    np.savez_compressed(folder / "CONTEXT.npz", allow_pickle=False, **context)
                    results[name]["queries"][token] = {
                        "prediction": prediction,
                        "fields": fields,
                        "historical_parity": all(v["passed"] for v in fields.values()),
                    }
                observations = {}
                for modality in ("event", "rgb"):
                    values = native.event_values if modality == "event" else native.rgb_values
                    for i, (identity, (raw, experts)) in enumerate(values.items()):
                        observations[f"{modality}{i}.id"] = np.array(identity)
                        observations[f"{modality}{i}.raw17"] = raw
                        observations[f"{modality}{i}.expert3"] = experts
                np.savez_compressed(
                    output / name / "OBSERVATIONS.npz", allow_pickle=False, **observations
                )
            finally:
                native.close()
            print(
                name
                + " "
                + str([v["historical_parity"] for v in results[name]["queries"].values()]),
                flush=True,
            )
        write_json(
            output / "RESULTS.json",
            {
                "policies": results,
                "candidate_evaluated": False,
                "targets_read": False,
                "local_original_repeatability": "EXACT",
                "elapsed_seconds": time.perf_counter() - started,
                "gpu": torch.cuda.get_device_name(),
                "torch": torch.__version__,
            },
        )
        run.finish()
    except Exception as error:
        run.fail(error)
        raise
    finally:
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = saved[:2]
        torch.backends.cudnn.benchmark = saved[2]
        torch.use_deterministic_algorithms(saved[3])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--continuation",
        type=Path,
        default=Path("output/release_research_20261031/continuation_v1"),
    )
    parser.add_argument(
        "--baseline", type=Path, default=Path("output/phase_safe_fusion/test12_audit")
    )
    parser.add_argument(
        "--upstream", type=Path, default=Path("output/phase_submission/FINAL_MODEL/upstream")
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    diagnose(args.continuation, args.baseline, args.upstream, args.output)


if __name__ == "__main__":
    main()

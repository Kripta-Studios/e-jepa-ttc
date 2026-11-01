"""Capture original deployment PAIR133 once per physical eAP observation, without labels."""

from __future__ import annotations

import argparse
import hashlib
import os
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
from operational.phase_robust.bundle import verified_upstream_paths
from operational.phase_timewarp.compact_capture import capture_compact
from operational.release_research.protocol import ResearchRun, sha256, write_json
from operational.rgb_port import transfer as engine
from operational.rgb_port_streaming.contracts import NativeObservation
from scripts.audit_phase_native_source_closure import DEPENDENCIES
from scripts.final_train40_transfer import load_models
from scripts.release_research import read, sources
from scripts.report_phase_canonical_population import verified_inputs


def capture(
    inputs_path: Path,
    media_receipt: Path,
    upstream: Path,
    output: Path,
    device: str,
    profile_queries: int | None,
    population_evaluation: Path | None,
    population_screen: Path = Path("output/phase_timewarp/population2608_v1"),
) -> None:
    """Preserve deployment observation identities and record actual PAIR batch inputs."""
    if sha256(inputs_path) != TRUSTED_INPUTS_SHA256:
        raise ValueError("Unexpected native public inputs")
    data = read(inputs_path)
    if _contains_private_key(data):
        raise ValueError("Private input fields forbidden")
    complete = read(media_receipt / "COMPLETE.json")
    if (
        complete["status"] != "COMPLETE"
        or complete["inputs_sha256"] != TRUSTED_INPUTS_SHA256
        or complete["files_sha256"] != sha256(media_receipt / "FILES.json")
        or complete["freeze_sha256"] != sha256(media_receipt / "FREEZE.json")
    ):
        raise ValueError("Media verification receipt changed")
    physical = set(data["physical_aliases"].values())
    queries = [q for q in data["queries"] if q["sample_token"] in physical]
    if len(queries) != len(physical) or len(physical) != 5805:
        raise ValueError("Native physical query population differs")
    if profile_queries is not None:
        if not 1 <= profile_queries <= 2:
            raise ValueError("Profile is bounded to at most two physical queries")
        queries = queries[:profile_queries]
    if profile_queries is None or device == "cuda":
        if population_evaluation is None:
            raise ValueError("Full capture requires completed population2608 F0 admission")
        _, population_report = verified_inputs(population_screen, population_evaluation)
        if not population_report["routes"]["F0"]["eligible_for_eap_validation"]:
            raise ValueError("Population2608 did not admit F0 native validation")
    paths = verified_upstream_paths(upstream)
    inputs = {
        **sources(),
        **{"model:" + k: v for k, v in paths.items()},
        "public_inputs": inputs_path,
        "media_complete": media_receipt / "COMPLETE.json",
        "media_files": media_receipt / "FILES.json",
        "media_freeze": media_receipt / "FREEZE.json",
        "upstream_bundle": upstream / "BUNDLE.json",
        "preregistration": Path(
            "docs/release_research_implementation_20261031/NATIVE_PAIR_PREREGISTRATION.md"
        ),
    }
    if (profile_queries is None or device == "cuda") and population_evaluation is not None:
        inputs.update(
            population_complete=population_screen / "COMPLETE.json",
            population_report=population_evaluation / "REPORT.json",
            population_scored=population_evaluation / "SCORED.npz",
        )
    if device == "cuda":
        inputs["platform_followup"] = Path(
            "docs/release_research_implementation_20261031/NATIVE_PARITY_FOLLOWUP.md"
        )
    for name in (
        *DEPENDENCIES,
        "scripts/release_native_pair_capture.py",
        "scripts/final_train40_transfer.py",
        "operational/phase_timewarp/compact_capture.py",
        "operational/phase_timewarp/capture.py",
        "operational/phase_robust/bundle.py",
        "scripts/report_phase_canonical_population.py",
        "scripts/runpod_test12_staged.py",
    ):
        inputs["source:" + name] = Path(name)
    run = ResearchRun(
        output,
        inputs,
        {
            "experiment_id": "native-original-PAIR-capture-v1",
            "targets_read": False,
            "optimizer_updates": 0,
            "candidate_evaluated": False,
            "device": device,
            "torch_threads": 1,
            "torch_version": torch.__version__,
            "profile_queries": profile_queries,
            "queries": len(queries),
            "batch_size": 1,
            "tf32": False,
            "cudnn_benchmark": False,
            "deterministic_algorithms": True,
        },
    )
    began = time.perf_counter()
    native = None
    try:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.set_num_threads(1)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
        models, normalizers = load_models(paths, device)
        native = FinalV16Train40Predictor.__new__(FinalV16Train40Predictor)
        native.models, native.normalizers, native.inputs = models, normalizers, data
        native.device = torch.device(device)
        native.event_values, native.rgb_values = {}, {}
        native.reader = TarFrameReader(
            Path(data["media_root"]), max_open=2, max_frame_cache_bytes=128 * 1024**2
        )
        files = {
            str(Path(r["path"]).resolve()): r
            for r in read(media_receipt / "FILES.json")["verified"]
        }
        verified_shards = {}
        manifest = {"event": {}, "rgb": {}}
        for modality in ("event", "rgb"):
            identities = list(
                dict.fromkeys(identity for q in queries for identity in q[modality + "_history"])
            )
            for number, identity in enumerate(identities):
                started = time.perf_counter()
                item = data[modality + "_observations"][identity]
                if modality == "rgb":
                    for shard in item["producer_shards"]:
                        path = (Path(data["media_root"]) / shard).resolve()
                        if str(path) not in verified_shards:
                            expected = files[str(path)]
                            if (
                                path.stat().st_size != expected["bytes"]
                                or sha256(path) != expected["sha256"]
                            ):
                                raise ValueError("Original RGB shard changed")
                            verified_shards[str(path)] = expected["sha256"]
                with (
                    patch.object(engine, "ProducerObservation", NativeObservation),
                    capture_compact(models) as (calls, arrays),
                ):
                    if modality == "event":
                        native._event_observation(identity)
                    else:
                        native._rgb_observation(identity)
                if (
                    len(calls) != 1
                    or calls[0]["status"] != "accepted"
                    or calls[0]["sensor_shape"][0] != 1
                ):
                    raise ValueError("Original single-observation producer call differs")
                call = calls[0]
                call["observations"] = [
                    {
                        "observation_id": identity,
                        "anchor_us": item["anchor_us"],
                        "available_us": item["available_us"],
                        "batch_index": 0,
                        "source_accepted": True,
                    }
                ]
                call["dt_construction"] = "original native predictor; event nominal clock preserved"
                key = hashlib.sha256((modality + ":" + identity).encode()).hexdigest()[:24]
                folder = output / "observations" / key
                folder.mkdir(parents=True)
                np.savez_compressed(folder / "PAIR_INPUTS.npz", **arrays)
                write_json(folder / "CALLS.json", {"calls": calls, "source": item})
                manifest[modality][identity] = {
                    "folder": str(folder.relative_to(output)),
                    "elapsed_seconds": time.perf_counter() - started,
                    "pair_sha256": sha256(folder / "PAIR_INPUTS.npz"),
                    "calls_sha256": sha256(folder / "CALLS.json"),
                }
                print(
                    f"{modality} {number + 1}/{len(identities)} "
                    f"{manifest[modality][identity]['elapsed_seconds']:.3f}s",
                    flush=True,
                )
        write_json(output / "OBSERVATIONS.json", manifest)
        write_json(output / "QUERIES.json", {"queries": queries})
        write_json(
            output / "RUNTIME.json",
            {
                "elapsed_seconds": time.perf_counter() - began,
                "observations": {m: len(v) for m, v in manifest.items()},
                "verified_rgb_shards": verified_shards,
                "scope": "original producer capture, including hash IO; not full-chain latency",
            },
        )
        run.finish()
    except Exception as error:
        run.fail(error)
        raise
    finally:
        if native is not None:
            native.close()


def main() -> None:
    """Profile or capture native observations with no candidate execution or target access."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument(
        "--media-receipt", type=Path, default=Path("output/phase_timewarp/eap_media_verified_v1")
    )
    parser.add_argument(
        "--upstream", type=Path, default=Path("output/phase_submission/FINAL_MODEL/upstream")
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--profile-queries", type=int)
    parser.add_argument("--population-evaluation", type=Path)
    parser.add_argument(
        "--population-screen", type=Path, default=Path("output/phase_timewarp/population2608_v1")
    )
    args = parser.parse_args()
    capture(
        args.inputs,
        args.media_receipt,
        args.upstream,
        args.output,
        args.device,
        args.profile_queries,
        args.population_evaluation,
        args.population_screen,
    )


if __name__ == "__main__":
    main()

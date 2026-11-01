"""Reconstruct native deployment from PAIR captures and require historical parity first."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import tarfile
import time
from pathlib import Path

import numpy as np
import torch

from operational.final_refit.test12 import TRUSTED_INPUTS_SHA256
from operational.phase_observation_cache.replay import replay
from operational.phase_robust.bundle import verified_upstream_paths
from operational.phase_timewarp.canonical_pair import CanonicalPair
from operational.phase_timewarp.screen import replay_pair_batches
from operational.release_research.protocol import ResearchRun, sha256, write_json
from scripts.audit_phase_native_source_closure import DEPENDENCIES
from scripts.final_train40_transfer import load_models
from scripts.release_research import read, sources, verify_complete


def compare_arrays(actual: dict, expected: dict) -> dict:
    """Fixed historical CPU/platform tolerances, never fitted to a candidate result."""
    if not actual or set(actual) != set(expected):
        raise ValueError("Historical context/head field inventory differs")
    fields = {}
    for name, value in actual.items():
        reference = expected[name]
        if value.shape != reference.shape or value.dtype != reference.dtype:
            raise ValueError("Historical shape/dtype differs: " + name)
        if value.dtype.kind == "f":
            if not np.isfinite(value).all() or not np.isfinite(reference).all():
                raise ValueError("Nonfinite historical comparison: " + name)
            rtol, atol = (1e-4, 1e-4) if name.endswith("output_ttc") else (1e-5, 1e-6)
            passed = bool(np.isclose(value, reference, rtol=rtol, atol=atol).all())
        else:
            passed = bool(np.array_equal(value, reference))
        fields[name] = {
            "passed": passed,
            "max_abs": float(np.max(np.abs(value.astype(float) - reference.astype(float)))),
        }
    return fields


def observations(
    capture: Path, manifest: dict, models: dict, device: str, *, original: bool
) -> tuple[dict, dict]:
    """Rebuild every PHASE17 dependent column once, keeping exact observation identities."""
    cache, audit = {"event": {}, "rgb": {}}, {"event": {}, "rgb": {}}
    for modality, items in manifest.items():
        for identity, item in items.items():
            folder = capture / item["folder"]
            if (
                sha256(folder / "CALLS.json") != item["calls_sha256"]
                or sha256(folder / "PAIR_INPUTS.npz") != item["pair_sha256"]
            ):
                raise ValueError("Captured observation changed")
            calls = read(folder / "CALLS.json")["calls"]
            with np.load(folder / "PAIR_INPUTS.npz", allow_pickle=False) as archive:
                arrays = {k: archive[k] for k in archive.files}
            result = replay_pair_batches(models, calls, arrays, device)
            if original:
                for key in ("raw17", "expert3"):
                    np.testing.assert_array_equal(
                        result["arrays"]["call0." + key], arrays["call0." + key]
                    )
                pair = "PAIR_E_MATCHED" if modality == "event" else "PAIR_R"
                np.testing.assert_array_equal(
                    result["arrays"]["call0.raw_phase"], arrays[f"call0.{pair}.raw_phase"]
                )
            admitted = result["observations"][modality]
            cache[modality][identity] = admitted[0] if admitted else None
            audit[modality][identity] = {"calls": result["calls"], "accepted": bool(admitted)}
    return cache, audit


def query_observations(query: dict, cached: dict) -> dict:
    """Retain the complete declared histories; preserve explicit intervention rejections."""
    return {
        m: [
            cached[m][identity]
            for identity in query[m + "_history"]
            if cached[m][identity] is not None
        ]
        for m in ("event", "rgb")
    }


def evaluate(capture: Path, baseline: Path, upstream: Path, output: Path, candidate: bool) -> None:
    """Verify original producer/context/head reconstruction before a paired canonical replay."""
    verify_complete(capture)
    config = read(capture / "FREEZE.json")["config"]
    if config["experiment_id"] != "native-original-PAIR-capture-v1" or config["targets_read"]:
        raise ValueError("Unexpected capture")
    if candidate and config["profile_queries"] is not None:
        raise ValueError("Profiling cannot admit a native candidate")
    parent = read(upstream / "BUNDLE.json")["parent"]
    reference = read(baseline / "COMPLETE.json")
    if (
        reference["status"] != "COMPLETE"
        or reference["parent"] != parent
        or reference["inputs_sha256"] != TRUSTED_INPUTS_SHA256
    ):
        raise ValueError("Historical ancestry differs")
    paths = verified_upstream_paths(upstream)
    inputs = {
        **sources(),
        **{"model:" + k: v for k, v in paths.items()},
        "capture_complete": capture / "COMPLETE.json",
        "capture_freeze": capture / "FREEZE.json",
        "capture_observations": capture / "OBSERVATIONS.json",
        "capture_queries": capture / "QUERIES.json",
        "baseline_complete": baseline / "COMPLETE.json",
        "upstream_bundle": upstream / "BUNDLE.json",
    }
    for name in (
        *DEPENDENCIES,
        "scripts/release_native_pair_replay.py",
        "scripts/final_train40_transfer.py",
        "operational/phase_observation_cache/replay.py",
        "operational/phase_timewarp/canonical_pair.py",
        "operational/phase_timewarp/screen.py",
        "operational/phase_robust/bundle.py",
    ):
        inputs["source:" + name] = Path(name)
    run = ResearchRun(
        output,
        inputs,
        {
            "experiment_id": "native-PAIR-replay-v1",
            "targets_read": False,
            "optimizer_updates": 0,
            "candidate_requested": candidate,
            "device": config["device"],
            "profile_queries": config["profile_queries"],
            "rtol": 1e-5,
            "atol": 1e-6,
            "ttc_rtol": 1e-4,
            "ttc_atol": 1e-4,
            "tf32": False,
            "cudnn_benchmark": False,
            "deterministic_algorithms": True,
        },
    )
    started = time.perf_counter()
    try:
        device = config["device"]
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.set_num_threads(1)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
        models, normalizers = load_models(paths, device)
        manifest = read(capture / "OBSERVATIONS.json")
        queries = {q["sample_token"]: q for q in read(capture / "QUERIES.json")["queries"]}
        original_cache, original_audit = observations(
            capture, manifest, models, device, original=True
        )
        records, verified_archives = {}, {}
        for entry in reference["archives"].values():
            path = Path(entry["path"])
            # Admission binds sequence; archive filenames identify their sequence in this receipt.
            if not any(q["sequence_id"] in path.name for q in queries.values()):
                continue
            if sha256(path) != entry["sha256"]:
                raise ValueError("Historical archive changed")
            verified_archives[str(path)] = entry["sha256"]
            with tarfile.open(path) as archive:
                names = archive.getnames()

                def member(name: str) -> bytes:
                    stream = archive.extractfile(name)
                    if stream is None:
                        raise ValueError("Missing archived diagnostic")
                    return stream.read()

                admission = json.loads(
                    member(next(n for n in names if n.endswith(".admission.json")))
                )
                if (
                    admission["model_seal_sha256"] != parent["model_seal_sha256"]
                    or admission["inputs_sha256"] != TRUSTED_INPUTS_SHA256
                ):
                    raise ValueError("Archived sequence identity differs")
                for name in names:
                    if "/diagnostics/" not in name or not name.endswith(".json"):
                        continue
                    receipt = json.loads(member(name))
                    token = receipt["sample_token"]
                    if token not in queries:
                        continue
                    if (
                        token in records
                        or receipt["private_targets_read"]
                        or receipt["run_binding_sha256"] != admission["run_binding_sha256"]
                    ):
                        raise ValueError("Invalid archived query")
                    query = queries[token]
                    for key in (
                        "event_history",
                        "rgb_history",
                        "query_time_us",
                        "sequence_id",
                        "track_id",
                    ):
                        if receipt[key] != query[key]:
                            raise ValueError("Archived history or identity differs")
                    payload = member(name.removesuffix(".json") + ".npz")
                    if hashlib.sha256(payload).hexdigest() != receipt["npz_sha256"]:
                        raise ValueError("Historical arrays changed")
                    prediction, context = replay(
                        models,
                        normalizers,
                        query_observations(query, original_cache),
                        query["query_time_us"],
                        device,
                    )
                    with np.load(io.BytesIO(payload), allow_pickle=False) as stored:
                        expected = {
                            k: stored[k]
                            for k in stored.files
                            if k.startswith(
                                ("E_CTX_MATCHED.", "R_CTX.", "PHASE_ONLY_EVENT_S7_TRAIN40.")
                            )
                        }
                        fields = compare_arrays(context, expected)
                    identity = hashlib.sha256(token.encode()).hexdigest()[:24]
                    folder = output / "queries" / identity
                    folder.mkdir(parents=True)
                    np.savez_compressed(
                        folder / "ORIGINAL_CONTEXT.npz", allow_pickle=False, **context
                    )
                    records[token] = {
                        "original": prediction,
                        "historical_parity": all(f["passed"] for f in fields.values()),
                        "fields": fields,
                    }
            print(f"{len(records)}/{len(queries)} native original queries compared", flush=True)
        if set(records) != set(queries):
            raise ValueError("Missing historical queries")
        write_json(output / "ORIGINAL_PARITY.json", records)
        write_json(output / "ORIGINAL_OBSERVATIONS.json", original_audit)
        failures = sum(not r["historical_parity"] for r in records.values())
        candidate_executed = False
        event_exact = True
        if candidate and failures == 0:
            variants = {
                **models,
                **{k: CanonicalPair(models[k]) for k in ("PAIR_E_MATCHED", "PAIR_R")},
            }
            new_cache, new_audit = observations(capture, manifest, variants, device, original=False)
            for identity, old in original_cache["event"].items():
                new = new_cache["event"][identity]
                if (
                    old is None
                    or new is None
                    or any(not np.array_equal(old[k], new[k]) for k in ("raw17", "expert3"))
                ):
                    event_exact = False
            if not event_exact:
                raise ValueError("Canonical nominal event identity was not preserved")
            for token, query in queries.items():
                prediction, context = replay(
                    variants,
                    normalizers,
                    query_observations(query, new_cache),
                    query["query_time_us"],
                    device,
                )
                identity = hashlib.sha256(token.encode()).hexdigest()[:24]
                folder = output / "queries" / identity
                with np.load(folder / "ORIGINAL_CONTEXT.npz", allow_pickle=False) as original:
                    for key in original.files:
                        if key.startswith("E_CTX_MATCHED."):
                            np.testing.assert_array_equal(context[key], original[key])
                np.savez_compressed(folder / "CANONICAL_CONTEXT.npz", allow_pickle=False, **context)
                records[token]["canonical"] = prediction
                records[token]["delta_seconds"] = (
                    None if prediction is None else prediction - records[token]["original"]
                )
            write_json(output / "CANONICAL_OBSERVATIONS.json", new_audit)
            candidate_executed = True
        write_json(output / "PREDICTIONS.json", {"records": records})
        changes = [
            r["delta_seconds"] for r in records.values() if r.get("delta_seconds") is not None
        ]
        write_json(
            output / "RESULTS.json",
            {
                "queries": len(queries),
                "historical_failed_queries": failures,
                "historical_parity": "PASS" if failures == 0 else "FAIL",
                "candidate_executed": candidate_executed,
                "event_context_exact": event_exact if candidate_executed else None,
                "candidate_missing": sum(r.get("canonical") is None for r in records.values())
                if candidate_executed
                else None,
                "maximum_prediction_change_seconds": max(map(abs, changes)) if changes else None,
                "elapsed_seconds": time.perf_counter() - started,
                "archives": verified_archives,
                "native_accuracy_evaluated": False,
                "targets_read": False,
                "scope": "prediction conservation only; no accuracy or official promotion claim",
            },
        )
        run.finish()
    except Exception as error:
        run.fail(error)
        raise


def main() -> None:
    """Require original parity and full capture before canonical replay."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument(
        "--baseline", type=Path, default=Path("output/phase_safe_fusion/test12_audit")
    )
    parser.add_argument(
        "--upstream", type=Path, default=Path("output/phase_submission/FINAL_MODEL/upstream")
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--candidate", action="store_true")
    args = parser.parse_args()
    evaluate(args.capture, args.baseline, args.upstream, args.output, args.candidate)


if __name__ == "__main__":
    main()

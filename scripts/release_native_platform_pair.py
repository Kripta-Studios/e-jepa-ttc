"""Measure local canonical PAIR sensitivity, retaining the historical parity failure."""

from __future__ import annotations

import argparse
import hashlib
import os
import time
from pathlib import Path

import numpy as np
import torch

from operational.phase_observation_cache.replay import replay
from operational.phase_robust.bundle import verified_upstream_paths
from operational.phase_timewarp.canonical_pair import CanonicalPair
from operational.release_research.protocol import ResearchRun, sha256, write_json
from scripts.audit_phase_native_source_closure import DEPENDENCIES
from scripts.final_train40_transfer import load_models
from scripts.release_native_pair_replay import observations, query_observations
from scripts.release_research import read, sources, verify_complete

PLAN = Path("docs/release_research_implementation_20261031/NATIVE_PLATFORM_PAIR_PROTOCOL.md")


def sensitivity(original: list[float | None], candidate: list[float | None]) -> dict:
    """Describe output changes without targets; no output-distance threshold grants promotion."""
    old, new = np.asarray(original, float), np.asarray(candidate, float)
    if old.ndim != 1 or old.shape != new.shape or not len(old):
        raise ValueError("Expected aligned nonempty prediction vectors")
    paired = np.isfinite(old) & np.isfinite(new)
    difference = np.abs(new[paired] - old[paired])
    return {
        "queries": len(old),
        "paired": int(paired.sum()),
        "original_available": int(np.isfinite(old).sum()),
        "candidate_available": int(np.isfinite(new).sum()),
        "newly_missing": int((np.isfinite(old) & ~np.isfinite(new)).sum()),
        "newly_available": int((~np.isfinite(old) & np.isfinite(new)).sum()),
        "exact_predictions": int((old[paired] == new[paired]).sum()),
        "sign_changes": int((np.sign(old[paired]) != np.sign(new[paired])).sum()),
        "absolute_change_p50_p95_p99_seconds": np.quantile(difference, [0.5, 0.95, 0.99]).tolist()
        if len(difference)
        else None,
        "maximum_absolute_change_seconds": float(difference.max()) if len(difference) else None,
        "native_accuracy_evaluated": False,
        "historical_parity_granted": False,
        "promotion_admitted": False,
    }


def run(capture: Path, original_replay: Path, upstream: Path, output: Path) -> None:
    """Verify the complete local original exactly before the fixed same-platform intervention."""
    for folder in (capture, original_replay):
        verify_complete(folder)
    config = read(capture / "FREEZE.json")["config"]
    original_freeze = read(original_replay / "FREEZE.json")
    history = read(original_replay / "RESULTS.json")
    if (
        config["profile_queries"] is not None
        or config["device"] != "cuda"
        or config["targets_read"]
        or config["candidate_evaluated"]
        or original_freeze["inputs"]["capture_complete"] != sha256(capture / "COMPLETE.json")
        or original_freeze["config"]["candidate_requested"]
        or history["candidate_executed"]
        or history["queries"] != 5805
    ):
        raise ValueError("Require the complete original-only local native chain")
    queries = read(capture / "QUERIES.json")["queries"]
    if len(queries) != 5805 or len({q["sample_token"] for q in queries}) != 5805:
        raise ValueError("Require every physical native query")
    manifest = read(capture / "OBSERVATIONS.json")
    prior = read(original_replay / "PREDICTIONS.json")["records"]
    if set(prior) != {q["sample_token"] for q in queries}:
        raise ValueError("Original query inventory differs")
    paths = verified_upstream_paths(upstream)
    files = {
        **sources(),
        "preregistration": PLAN,
        "capture": capture / "COMPLETE.json",
        "original_replay": original_replay / "COMPLETE.json",
        "capture_manifest": capture / "OBSERVATIONS.json",
        "queries": capture / "QUERIES.json",
        "original_predictions": original_replay / "PREDICTIONS.json",
        "original_results": original_replay / "RESULTS.json",
        "upstream_bundle": upstream / "BUNDLE.json",
        **{"model:" + k: v for k, v in paths.items()},
    }
    for name in (
        *DEPENDENCIES,
        "scripts/release_native_platform_pair.py",
        "scripts/release_native_pair_replay.py",
        "scripts/final_train40_transfer.py",
        "operational/phase_observation_cache/replay.py",
        "operational/phase_timewarp/canonical_pair.py",
        "operational/phase_timewarp/screen.py",
        "operational/phase_robust/bundle.py",
    ):
        files["source:" + name] = Path(name)
    experiment = ResearchRun(
        output,
        files,
        {
            "experiment_id": "native-local-platform-pair-v1",
            "targets_read": False,
            "optimizer_updates": 0,
            "seed": 7,
            "historical_original_parity": history["historical_parity"],
            "scope": "same-platform sensitivity only, not historical conservation or accuracy",
            "device": "cuda",
            "tf32": False,
            "benchmark": False,
            "deterministic_algorithms": True,
            "torch_version": torch.__version__,
        },
    )
    started = time.perf_counter()
    try:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.set_num_threads(1)
        torch.manual_seed(7)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
        models, normalizers = load_models(paths, "cuda")
        old_cache, old_audit = observations(capture, manifest, models, "cuda", original=True)
        for number, query in enumerate(queries):
            token = query["sample_token"]
            identity = hashlib.sha256(token.encode()).hexdigest()[:24]
            baseline, original_context = replay(
                models,
                normalizers,
                query_observations(query, old_cache),
                query["query_time_us"],
                "cuda",
            )
            if baseline != prior[token]["original"]:
                raise ValueError("Local original prediction is not exactly repeatable")
            context_path = original_replay / "queries" / identity / "ORIGINAL_CONTEXT.npz"
            with np.load(context_path, allow_pickle=False) as stored:
                if set(stored.files) != set(original_context):
                    raise ValueError("Local original fields differ")
                for key in stored.files:
                    np.testing.assert_array_equal(stored[key], original_context[key])
            if (number + 1) % 100 == 0:
                print(f"{number + 1}/5805 local original exact", flush=True)
        variants = {**models, **{k: CanonicalPair(models[k]) for k in ("PAIR_E_MATCHED", "PAIR_R")}}
        new_cache, new_audit = observations(capture, manifest, variants, "cuda", original=False)
        for identity, old in old_cache["event"].items():
            new = new_cache["event"][identity]
            if old is None or new is None:
                raise ValueError("Nominal event observation rejected")
            for key in ("raw17", "expert3"):
                np.testing.assert_array_equal(new[key], old[key])
        records = []
        for number, query in enumerate(queries):
            token = query["sample_token"]
            identity = hashlib.sha256(token.encode()).hexdigest()[:24]
            baseline = prior[token]["original"]
            candidate, context = replay(
                variants,
                normalizers,
                query_observations(query, new_cache),
                query["query_time_us"],
                "cuda",
            )
            context_path = original_replay / "queries" / identity / "ORIGINAL_CONTEXT.npz"
            with np.load(context_path, allow_pickle=False) as stored:
                for key in stored.files:
                    if key.startswith("E_CTX_MATCHED."):
                        np.testing.assert_array_equal(context[key], stored[key])
            folder = output / "queries" / identity
            folder.mkdir(parents=True)
            np.savez_compressed(folder / "CANDIDATE_CONTEXT.npz", allow_pickle=False, **context)
            records.append(
                {
                    "token": token,
                    "original": baseline,
                    "canonical": candidate,
                    "local_original_exact": True,
                    "event_context_exact": True,
                    "historical_original_parity": prior[token]["historical_parity"],
                }
            )
            if (number + 1) % 100 == 0:
                print(f"{number + 1}/5805 paired native queries", flush=True)
        write_json(output / "PREDICTIONS.json", {"records": records})
        write_json(output / "PAIR_AUDIT.json", {"original": old_audit, "canonical": new_audit})
        result = sensitivity([r["original"] for r in records], [r["canonical"] for r in records])
        result.update(
            historical_original_failed_queries=history["historical_failed_queries"],
            historical_original_ttc_failed_queries=sum(
                any(not v["passed"] for k, v in r["fields"].items() if k.endswith("output_ttc"))
                for r in prior.values()
            ),
            local_original_exact=True,
            event_context_exact=True,
            elapsed_seconds=time.perf_counter() - started,
            runtime_scope="compact replay with verification/output IO; not full-chain latency",
        )
        write_json(output / "RESULTS.json", result)
        experiment.finish()
    except Exception as error:
        experiment.fail(error)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--original-replay", type=Path, required=True)
    parser.add_argument(
        "--upstream", type=Path, default=Path("output/phase_submission/FINAL_MODEL/upstream")
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.capture, args.original_replay, args.upstream, args.output)


if __name__ == "__main__":
    main()

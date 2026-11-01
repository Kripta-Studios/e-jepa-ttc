"""Describe the frozen delivered router after sealed local native PAIR sensitivity."""

import argparse
import hashlib
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch

from operational.evttc_transfer_audit.common import file_sha
from operational.final_refit.test12 import TRUSTED_INPUTS_SHA256, _contains_private_key
from operational.phase_adapt_cache import atomic_receipt
from operational.phase_campaign.data import read
from operational.phase_startup.geometry import startup_geometry
from operational.phase_submission.runtime import load_submission_model
from scripts.build_phase_native_context_augmentations import native_vector
from scripts.build_phase_safe_test12_audit import query_history

PLAN = Path("docs/research_log_canonical_population2608/EAP_ROUTER_SENSITIVITY_PROTOCOL.md")


def verified_parent(root: Path, required: set[str]) -> dict:
    """Reject unfinished or failed parents, and authenticate every consumed artifact."""
    receipt = read(root / "COMPLETE.json")
    names = {name.replace("\\", "/") for name in receipt["artifacts"]}
    if receipt["status"] != "COMPLETE" or (root / "FAILED.json").exists() or not required <= names:
        raise ValueError("Require a completed parent with all required artifacts")
    resolved_root = root.resolve(strict=True)

    def verify_one(item: tuple[str, str]) -> None:
        relative, digest = item
        path = (resolved_root / relative).resolve(strict=True)
        if not path.is_relative_to(resolved_root) or file_sha(path) != digest:
            raise ValueError(f"Changed or escaping artifact: {relative}")

    with ThreadPoolExecutor(max_workers=8) as executor:
        for _ in executor.map(verify_one, receipt["artifacts"].items()):
            pass
    return receipt


def changes(original: list, candidate: list) -> dict:
    """Keep missing predictions and signs explicit; output distance is not accuracy."""
    old, new = np.asarray(original, float), np.asarray(candidate, float)
    if old.ndim != 1 or old.shape != new.shape or not len(old):
        raise ValueError("Expected nonempty aligned predictions")
    valid_old, valid_new = np.isfinite(old), np.isfinite(new)
    paired = valid_old & valid_new
    delta = np.abs(new[paired] - old[paired])
    return {
        "queries": len(old),
        "paired": int(paired.sum()),
        "original_available": int(valid_old.sum()),
        "canonical_available": int(valid_new.sum()),
        "newly_missing": int((valid_old & ~valid_new).sum()),
        "newly_available": int((~valid_old & valid_new).sum()),
        "exact": int((old[paired] == new[paired]).sum()),
        "sign_changes": int((np.sign(old[paired]) != np.sign(new[paired])).sum()),
        "mean_abs_seconds": float(delta.mean()) if len(delta) else None,
        "max_abs_seconds": float(delta.max()) if len(delta) else None,
        "p50_p95_p99_abs_seconds": np.quantile(delta, [0.5, 0.95, 0.99]).tolist()
        if len(delta)
        else None,
    }


def assert_f0(vector: np.ndarray, expected: float | None) -> None:
    """Require the consumed context to reproduce its sealed native prediction exactly."""
    if (expected is None and np.isfinite(vector[0])) or (
        expected is not None and float(vector[0]) != expected
    ):
        raise ValueError("Context and native F0 prediction differ")


def main() -> None:
    """Reuse completed native states with causal geometry and unchanged three-seed router."""
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("capture", "original", "paired", "inputs", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--bundle", type=Path, default=Path("output/phase_submission/FINAL_MODEL"))
    args = parser.parse_args()
    if args.output.exists() or file_sha(args.inputs) != TRUSTED_INPUTS_SHA256:
        raise ValueError("Require fresh output and trusted label-free inventory")
    public = read(args.inputs)
    if _contains_private_key(public):
        raise ValueError("Private target field in input inventory")
    capture = verified_parent(args.capture, {"FREEZE.json", "QUERIES.json", "OBSERVATIONS.json"})
    original = verified_parent(args.original, {"FREEZE.json", "PREDICTIONS.json", "RESULTS.json"})
    paired = verified_parent(args.paired, {"FREEZE.json", "PREDICTIONS.json", "RESULTS.json"})
    context_inventories = {
        "original": {p.replace("\\", "/") for p in original["artifacts"]},
        "canonical": {p.replace("\\", "/") for p in paired["artifacts"]},
    }
    parent_freeze = read(args.paired / "FREEZE.json")
    result = read(args.paired / "RESULTS.json")
    if (
        parent_freeze["inputs"]["capture"] != file_sha(args.capture / "COMPLETE.json")
        or parent_freeze["inputs"]["original_replay"] != file_sha(args.original / "COMPLETE.json")
        or not result["local_original_exact"]
        or not result["event_context_exact"]
        or result["native_accuracy_evaluated"]
        or result["promotion_admitted"]
        or result["queries"] != 5805
    ):
        raise ValueError("Local sensitivity ancestry, identity or scope differs")
    del capture
    queries = read(args.capture / "QUERIES.json")["queries"]
    records = read(args.paired / "PREDICTIONS.json")["records"]
    by_token = {row["token"]: row for row in records}
    owners = {q["sample_token"]: q for q in public["queries"]}
    aliases = public["physical_aliases"]
    if (
        len(records) != len(by_token)
        or len(by_token) != 5805
        or set(by_token) != set(aliases.values())
        or {q["sample_token"] for q in queries} != set(by_token)
        or len(queries) != 5805
        or len(aliases) != 6762
    ):
        raise ValueError("Physical/logical native population differs")
    for query in queries:
        if query != owners[query["sample_token"]]:
            raise ValueError("Captured history differs from trusted physical query")
    router = load_submission_model(args.bundle)
    sources = [
        Path(__file__),
        PLAN,
        *[
            Path(p)
            for p in (
                "scripts/build_phase_safe_test12_audit.py",
                "scripts/build_phase_safe_source_audit.py",
                "scripts/build_phase_native_context_augmentations.py",
                "operational/phase_submission/runtime.py",
                "operational/phase_startup/runtime.py",
                "operational/phase_startup/model.py",
                "operational/phase_startup/geometry.py",
                "src/e_jepa_ttc/simplex_t/phase.py",
            )
        ],
    ]
    pins = {str(p): file_sha(p) for p in sources}
    pins[str(args.bundle / "MODEL.json")] = file_sha(args.bundle / "MODEL.json")
    freeze = {
        "schema": "phase_eap_local_router_sensitivity_v1",
        "sources": pins,
        "parents": {
            str(p): file_sha(p)
            for p in (
                args.capture / "COMPLETE.json",
                args.original / "COMPLETE.json",
                args.paired / "COMPLETE.json",
                args.inputs,
            )
        },
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "router_device": "cpu",
        "native_phase_conversion_device": "cuda",
        "batch_size": 1,
        "parent_verification_workers": 8,
        "targets_read": False,
        "optimizer_updates": 0,
        "promotion_admitted": False,
        "historical_gate_reclassified": False,
        "router_prior_screen": "failed; diagnostic only",
        "physical_queries": 5805,
        "logical_queries": 6762,
    }
    atomic_receipt(args.output / "FREEZE.json", freeze)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    started = time.perf_counter()
    predictions = {}
    try:
        for query in queries:
            token = query["sample_token"]
            key = hashlib.sha256(token.encode()).hexdigest()[:24]
            boxes, clocks = query_history(query, public["rgb_observations"], owners)
            geometry = startup_geometry(boxes, clocks, query["query_time_us"])
            geo = np.float32(np.clip(geometry["ttc"], -60, 60)) if geometry["valid"] else np.nan
            quality = np.asarray(geometry["quality"], np.float32)
            row: dict = {
                "group": query["sequence_id"],
                "history_active": bool(
                    np.isfinite(quality[9])
                    and quality[9] > 0
                    and np.isfinite(quality[11])
                    and quality[11] <= 0.15
                ),
            }
            for name, root, suffix in (
                ("original", args.original, "ORIGINAL_CONTEXT"),
                ("canonical", args.paired, "CANDIDATE_CONTEXT"),
            ):
                relative = f"queries/{key}/{suffix}.npz"
                if relative not in context_inventories[name]:
                    raise ValueError("Consumed context is absent from parent artifact inventory")
                with np.load(root / relative, allow_pickle=False) as arrays:
                    native = native_vector({k: arrays[k] for k in arrays.files}, "cuda")
                assert_f0(native, by_token[token][name])
                with torch.inference_mode():
                    value = float(
                        router(
                            torch.from_numpy(native)[None],
                            torch.tensor([geo], dtype=torch.float32),
                            torch.from_numpy(quality)[None],
                        )[0]
                    )
                if not row["history_active"]:
                    np.testing.assert_array_equal(np.float32(value), native[0])
                row[name] = value if np.isfinite(value) else None
            predictions[token] = row
            if len(predictions) % 500 == 0:
                elapsed = time.perf_counter() - started
                print(
                    f"ROUTER {len(predictions)}/5805 elapsed={elapsed:.1f}s "
                    f"eta={elapsed / len(predictions) * (5805 - len(predictions)):.1f}s",
                    flush=True,
                )
        for path, digest in {**pins, **freeze["parents"]}.items():
            if file_sha(Path(path)) != digest:
                raise ValueError("Router audit source/input changed during execution")

        def summary(tokens: list[str]) -> dict:
            return changes(
                [predictions[t]["original"] for t in tokens],
                [predictions[t]["canonical"] for t in tokens],
            )

        physical = sorted(predictions)
        report = {
            "physical": summary(physical),
            "logical_aliases": summary(list(aliases.values())),
            "by_sequence": {
                group: summary([t for t in physical if predictions[t]["group"] == group])
                for group in sorted({r["group"] for r in predictions.values()})
            },
            "active_history_physical_queries": sum(
                r["history_active"] for r in predictions.values()
            ),
            "aliases_are_independent_samples": False,
            "targets_read": False,
            "accuracy_evaluated": False,
            "historical_parity_granted": False,
            "promotion_admitted": False,
            "router_prior_screen": "failed; diagnostic only",
        }
        atomic_receipt(args.output / "PREDICTIONS.json", predictions)
        atomic_receipt(args.output / "REPORT.json", report)
        atomic_receipt(
            args.output / "COMPLETE.json",
            {
                "status": "COMPLETE",
                "elapsed_seconds": time.perf_counter() - started,
                "artifacts": {
                    name: file_sha(args.output / name)
                    for name in ("FREEZE.json", "PREDICTIONS.json", "REPORT.json")
                },
                "targets_read": False,
                "promotion_admitted": False,
            },
        )
        print(report["physical"])
    except Exception as exc:
        atomic_receipt(
            args.output / "FAILED.json", {"error": repr(exc), "queries": len(predictions)}
        )
        raise


if __name__ == "__main__":
    main()

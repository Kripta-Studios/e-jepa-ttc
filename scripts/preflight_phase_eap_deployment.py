"""Inventory local Test12 deployment inputs without inference, targets, or downloads."""

import argparse
import os
from collections import Counter
from pathlib import Path

import numpy as np

from e_jepa_ttc.rgb_port.history import deltas_seconds
from operational.evttc_transfer_audit.common import file_sha
from operational.final_refit.test12 import TRUSTED_INPUTS_SHA256, _contains_private_key
from operational.phase_adapt_cache import atomic_receipt
from operational.phase_campaign.data import read


def main() -> None:
    """Report availability separately from byte verification and model preservation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--baseline-audit", type=Path, required=True)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--media-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Use a fresh output; preserve prior preflight")
    if file_sha(args.inputs) != TRUSTED_INPUTS_SHA256:
        raise ValueError("Test12 inventory differs from frozen public inputs")
    inputs = read(args.inputs)
    if _contains_private_key(inputs):
        raise ValueError("Unexpected private field in input inventory")
    baseline_path = args.baseline_audit / "COMPLETE.json"
    baseline = read(baseline_path)
    bundle = read(args.upstream / "BUNDLE.json")
    if (
        baseline["status"] != "COMPLETE"
        or baseline["inputs_sha256"] != TRUSTED_INPUTS_SHA256
        or baseline["parent"] != bundle["parent"]
        or baseline["private_targets_read"]
        or baseline["feature_domain"] != "SAVED_TEST12_DEPLOYMENT_PRODUCER_STATES_NOT_OOF"
    ):
        raise ValueError("Saved diagnostic ancestry differs from deployment bundle")
    aliases = inputs["physical_aliases"]
    queries = {q["sample_token"]: q for q in inputs["queries"]}
    canonical = set(aliases.values())
    if len(queries) != 6762 or len(canonical) != 5805 or set(aliases) != set(queries):
        raise ValueError("Physical/logical population differs")
    if not canonical <= queries.keys():
        raise ValueError("Alias refers to unknown query")
    clocks = {}
    for modality in ("event", "rgb"):
        observations = inputs[f"{modality}_observations"]
        references = {identity for q in queries.values() for identity in q[f"{modality}_history"]}
        if references != set(observations):
            raise ValueError("History coverage differs from observation inventory")
        delta = np.asarray(
            [
                item["delta_t_s"]
                if modality == "event"
                else deltas_seconds(item["producer_sensor_times_us"])[-1]
                for item in observations.values()
            ],
            dtype=np.float32,
        )
        if not np.isfinite(delta).all() or not (delta > 0).all():
            raise ValueError("Invalid physical clock")
        clocks[modality] = {
            "observations": len(observations),
            "float32_100ms": int((delta == np.float32(0.1)).sum()),
            "nonidentity": int((delta != np.float32(0.1)).sum()),
            "delta_seconds_quantiles": np.quantile(delta, [0, 0.01, 0.5, 0.99, 1]).tolist(),
            "history_lengths_physical": dict(
                sorted(Counter(len(queries[t][f"{modality}_history"]) for t in canonical).items())
            ),
        }
    # Directory inventories avoid thousands of independent external-disk stat calls.
    tensor_paths = [Path(v["current_tensor_path"]) for v in inputs["event_observations"].values()]
    directories = {p.parent for p in tensor_paths}
    present = {}
    for folder in directories:
        if not folder.is_dir():
            present[folder] = set()
            continue
        with os.scandir(folder) as entries:
            present[folder] = {entry.name for entry in entries if entry.is_file()}
    missing_tensors = [str(p) for p in tensor_paths if p.name not in present[p.parent]]
    rgb_files = []
    for relative, pin in inputs["rgb_sources"].items():
        path = (args.media_root / relative).resolve()
        if not path.is_relative_to(args.media_root.resolve()):
            raise ValueError("RGB source escapes supplied media root")
        size = path.stat().st_size if path.is_file() else None
        rgb_files.append(
            {
                "relative": relative,
                "exists": size is not None,
                "expected_bytes": pin["size_bytes"],
                "observed_bytes": size,
                "size_matches": size == pin["size_bytes"],
                "expected_sha256": pin["sha256"],
            }
        )
    archives = {}
    for name, entry in baseline["archives"].items():
        actual = file_sha(Path(entry["path"]))
        if actual != entry["sha256"]:
            raise ValueError(f"Saved deployment diagnostic archive changed: {name}")
        archives[name] = actual
    source_paths = [
        Path(__file__),
        Path("operational/final_refit/test12.py"),
        Path("src/e_jepa_ttc/rgb_port/data.py"),
        Path("src/e_jepa_ttc/rgb_port/history.py"),
        Path("operational/phase_timewarp/canonical_pair.py"),
    ]
    atomic_receipt(
        args.output,
        {
            "status": "PREFLIGHT_ONLY",
            "targets_read": False,
            "npz_arrays_read": [],
            "producer_calls": 0,
            "optimizer_updates": 0,
            "candidate_evaluated": False,
            "logical_queries": len(queries),
            "physical_queries": len(canonical),
            "clock_inventory": clocks,
            "event_tensor_paths": len(tensor_paths),
            "event_tensor_missing": missing_tensors,
            "event_tensor_content_hashes_verified": False,
            "rgb_shards": rgb_files,
            "rgb_content_hashes_verified": False,
            "diagnostic_archive_hashes_verified": archives,
            "inputs_sha256": TRUSTED_INPUTS_SHA256,
            "baseline_audit_sha256": file_sha(baseline_path),
            "upstream_bundle_sha256": file_sha(args.upstream / "BUNDLE.json"),
            "source_hashes": {str(p): file_sha(p) for p in source_paths},
            "remaining": [
                "Hash each tensor and RGB shard against frozen inventory before model use",
                "Reproduce original deployment batches, raw17, contexts and head",
                "Capture missing historical PAIR133 before applying the fixed correction",
                "Retain all physical queries and frozen logical aliases without tuning",
                "Drift without targets cannot establish accuracy preservation",
            ],
        },
    )
    print(
        {
            "event_missing": len(missing_tensors),
            "rgb_missing_or_wrong_size": sum(not r["size_matches"] for r in rgb_files),
            "clocks": clocks,
        },
        flush=True,
    )


if __name__ == "__main__":
    main()

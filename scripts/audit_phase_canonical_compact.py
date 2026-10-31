"""Verify compact canonical replay against the two previously sealed full replays."""

import argparse
import hashlib
from pathlib import Path

from operational.evttc_transfer_audit.common import file_sha
from operational.phase_adapt_cache import atomic_receipt
from operational.phase_campaign.data import read
from scripts.build_phase_native_context_augmentations import verify_artifacts
from scripts.pilot_phase_producer_timewarp import exact_arrays, read_arrays


def main() -> None:
    """Compare existing evidence only; no producers, targets or new temporal pilots."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--screen", type=Path, required=True)
    parser.add_argument("--pilot", type=Path, default=Path("output/phase_timewarp/canonical_v1"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    for root in (args.screen, args.pilot):
        complete = read(root / "COMPLETE.json")
        if complete["status"] != "COMPLETE" or complete["targets_read"]:
            raise ValueError("Require sealed target-free runs")
        verify_artifacts(root, complete["artifacts"])
    results = {}
    for name, token in (
        ("road", "evttc:EXTENDED:CCRm-low-0-overlap-0:17000441"),
        ("slider500", "Slider-500:38"),
    ):
        identity = hashlib.sha256(token.encode()).hexdigest()[:24]
        source = args.screen / "queries" / identity / "canonical"
        old = args.pilot / name / "a1"
        context, old_context = read_arrays(source / "CONTEXT.npz"), read_arrays(old / "CONTEXT.npz")
        exact_arrays(context, old_context, token + " canonical context/head")
        compact, full = read_arrays(source / "PAIR_REPLAY.npz"), read_arrays(old / "PRODUCERS.npz")
        keys = [k for k in full if k.endswith((".raw17", ".expert3"))]
        exact_arrays(
            {k: compact[k] for k in keys}, {k: full[k] for k in keys}, token + " raw17/expert3"
        )
        results[token] = {"context_arrays_exact": len(context), "producer_arrays_exact": len(keys)}
    atomic_receipt(
        args.output,
        {
            "status": "PASS",
            "results": results,
            "targets_read": False,
            "producer_calls": 0,
            "screen_complete_sha256": file_sha(args.screen / "COMPLETE.json"),
            "pilot_complete_sha256": file_sha(args.pilot / "COMPLETE.json"),
            "audit_source_sha256": file_sha(Path(__file__)),
        },
    )
    print(results)


if __name__ == "__main__":
    main()

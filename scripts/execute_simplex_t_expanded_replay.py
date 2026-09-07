"""Inspect or run a pinned D1/DENSE cache launch; never train scientific heads."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from e_jepa_ttc.artifacts.simplex_t_preflight import write_new_json
from e_jepa_ttc.simplex_t.configured_expanded_replay import run_configured_expanded_replay


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-paths", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--config-sha256", required=True)
    parser.add_argument("--other-reserved-bytes", type=int, required=True)
    parser.add_argument("--max-new-queries", type=int, required=True)
    parser.add_argument("--inspect-only", action="store_true")
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    work = Path(json.loads(args.local_paths.read_text(encoding="utf-8"))["worktree"]).resolve()
    if args.report.exists() or not args.report.resolve().is_relative_to(work):
        raise ValueError("a new companion-local report path is required")
    try:
        result = run_configured_expanded_replay(
            args.local_paths,
            args.config,
            args.config_sha256,
            max_new_queries=args.max_new_queries,
            other_reserved_bytes=args.other_reserved_bytes,
            inspect_only=args.inspect_only,
        )
    except (InterruptedError, RuntimeError) as error:
        # Only known resource boundaries are pauses. A model/data/runtime failure
        # must remain a failure, not acquire a misleading resumable-resource label.
        if not str(error).startswith(("PAUSED_RESOURCE", "RESOURCE_PAUSE")):
            raise
        result = {
            "status": "PAUSED_RESOURCE",
            "reason": str(error),
            "new_blocks": None,
            "progress_count": "UNKNOWN_FOR_INTERRUPTED_INVOCATION_NOT_ASSUMED_ZERO",
            "optimizer_updates": 0,
            "resume": (
                "Same pinned configuration and existing queue receipts; use a new report path. "
                "Do not remove a lease or overwrite an unreceipted partial block."
            ),
        }
    result["launch_configuration_sha256"] = args.config_sha256
    result["requested_max_new_queries"] = args.max_new_queries
    result["inspect_only"] = args.inspect_only
    write_new_json(args.report, result)
    print(json.dumps(result))
    # Match the campaign launch contract: 3 is a resumable resource boundary.
    # argparse retains exit 2 for malformed arguments, which must not be retried.
    return 3 if result["status"] == "PAUSED_RESOURCE" else 0


if __name__ == "__main__":
    raise SystemExit(main())

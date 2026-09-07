"""Verify pinned real lineage/replay/resume components; never authorize fits."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from e_jepa_ttc.artifacts.simplex_t_preflight import write_new_json
from e_jepa_ttc.simplex_t.component_verification import verify_component_profile
from e_jepa_ttc.simplex_t.coordination import shared_write_admission
from e_jepa_ttc.simplex_t.lifecycle import admitted


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-paths", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--profile-sha256", required=True)
    parser.add_argument("--other-reserved-bytes", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--require-h16", action="store_true")
    parser.add_argument("--require-unit-qa", action="store_true")
    parser.add_argument("--require-static-qa", action="store_true")
    parser.add_argument("--require-types", action="store_true")
    parser.add_argument("--require-powershell", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("component evidence report already exists")
    if args.other_reserved_bytes < 0:
        raise ValueError("nonnegative pending output reservation required")
    paths = json.loads(args.local_paths.read_text(encoding="utf-8"))
    work = Path(paths["worktree"]).resolve(strict=True)
    if not args.output.resolve().is_relative_to(work):
        raise ValueError("report must remain in the companion worktree")

    def resources() -> bool:
        snapshot = admitted([work])
        return snapshot["has_headroom"] and shared_write_admission(
            snapshot["written_volume_free_bytes"][0], args.other_reserved_bytes + 1_048_576
        )

    torch.set_num_threads(4)
    torch.set_num_interop_threads(2)
    started = time.monotonic()
    result = verify_component_profile(
        args.local_paths,
        args.profile,
        args.profile_sha256,
        resource_ok=resources,
        require_h16=args.require_h16,
        require_unit_qa=args.require_unit_qa,
        require_static_qa=args.require_static_qa,
        require_types=args.require_types,
        require_powershell=args.require_powershell,
    )
    result["observed_seconds_excluding_imports"] = time.monotonic() - started
    write_new_json(args.output, result)
    print(json.dumps({"status": result["status"], "optimizer_updates_executed": 0}))


if __name__ == "__main__":
    main()

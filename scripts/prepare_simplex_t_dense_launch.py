"""Bind DENSE replay to three real D0 catalogs; never launch inference or fits."""

from __future__ import annotations

import argparse
import json
import uuid
from pathlib import Path

from e_jepa_ttc.artifacts.simplex_t_preflight import sha256, write_new_json
from e_jepa_ttc.simplex_t.configured_expanded_replay import (
    POOL_INDEX_SHA256,
    run_configured_expanded_replay,
)
from e_jepa_ttc.simplex_t.coordination import shared_write_admission
from e_jepa_ttc.simplex_t.lifecycle import admitted


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-paths", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--other-reserved-bytes", type=int, required=True)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if args.other_reserved_bytes < 0:
        parser.error("explicit nonnegative other pending output bytes required")
    work = Path(__file__).resolve().parents[1]
    output = args.output.resolve()
    if not output.is_relative_to(work / "artifacts"):
        raise ValueError("launch output must remain in companion artifacts")
    if Path(json.loads(args.local_paths.read_text(encoding="utf-8"))["worktree"]).resolve() != work:
        raise ValueError("local paths belong to another worktree")
    temporal = work / "artifacts/simplex_t/T1"
    manifests = [temporal / f"compiled_context/outer{outer}/COMPILED.json" for outer in range(3)]
    missing = [str(path) for path in manifests if not path.is_file()]
    if missing:
        print(
            json.dumps(
                {
                    "status": "DENSE_LAUNCH_PREREQUISITES_INCOMPLETE",
                    "missing": missing,
                    "optimizer_updates": 0,
                    "models_loaded": False,
                }
            )
        )
        return 10
    index = temporal / "dense_query_context_index/INDEX_MANIFEST.json"
    if index.stat().st_size > 1_048_576 or sha256(index) != POOL_INDEX_SHA256["DENSE_OLD"]:
        raise ValueError("DENSE input population changed")
    queries = json.loads(index.read_text(encoding="utf-8"))["queries"]
    bound = queries * 3 * 131_072 + 1_048_576
    snapshot = admitted([work])
    if not snapshot["has_headroom"] or not shared_write_admission(
        snapshot["written_volume_free_bytes"][0], args.other_reserved_bytes + bound + 2_097_152
    ):
        print(json.dumps({"status": "PAUSED_RESOURCE", "models_loaded": False}))
        return 3
    config = {
        "schema": "simplex_t_expanded_replay_launch_v1",
        "pool": "DENSE_OLD",
        "index": "artifacts/simplex_t/T1/dense_query_context_index",
        "dedup": "artifacts/simplex_t/T1/dense_query_context_dedup",
        "dedup_sha256": sha256(temporal / "dense_query_context_dedup/DEDUP_MANIFEST.json"),
        "original_index": "artifacts/simplex_t/T1/query_context_index",
        "preprocessing": (
            "e-jepa-ttc/artifacts/cache/garl_object_event_common_roi_train8192_v1/manifest.json"
        ),
        "output": "artifacts/simplex_t/T1/dense_context_features_fp32",
        "reserved_output_bytes": bound,
        "d0_reuse": {
            str(outer): {
                "compiled": path.parent.relative_to(work).as_posix(),
                "compiled_sha256": sha256(path),
                "cache": "artifacts/simplex_t/T1/context_features_fp32",
                "dedup": f"artifacts/simplex_t/T1/query_context_dedup/outer{outer}.npz",
            }
            for outer, path in enumerate(manifests)
        },
    }
    if args.verify_only:
        if not output.is_file():
            return 10
        if (
            output.stat().st_size > 1_048_576
            or json.loads(output.read_text(encoding="utf-8")) != config
        ):
            raise ValueError("DENSE launch differs from current acknowledged catalog pins")
        candidate = output
    else:
        if output.exists():
            raise FileExistsError("preserve existing launch; use --verify-only")
        output.parent.mkdir(parents=True, exist_ok=True)
        candidate = output.with_name(f"{output.stem}.candidate_{uuid.uuid4().hex}.json")
        write_new_json(candidate, config)
    digest = sha256(candidate)
    # Mandatory real inspection verifies role/time ACK, historical preprocessing
    # and all three catalog identities. The retained candidate is not authority.
    result = run_configured_expanded_replay(
        args.local_paths,
        candidate,
        digest,
        max_new_queries=1,
        other_reserved_bytes=args.other_reserved_bytes + 2_097_152,
        inspect_only=True,
    )
    if result["status"] == "PAUSED_RESOURCE":
        print(json.dumps(result))
        return 3
    if result["status"] != "EXPANDED_LAUNCH_INSPECTED_NOT_REPLAY_OR_SCIENTIFIC_ADMISSION":
        raise ValueError("DENSE launch inspection did not complete")
    if sha256(candidate) != digest or any(
        sha256(path) != config["d0_reuse"][str(outer)]["compiled_sha256"]
        for outer, path in enumerate(manifests)
    ):
        raise ValueError("DENSE launch or source catalog changed during inspection")
    if not args.verify_only:
        write_new_json(output, config)
    print(
        json.dumps(
            {
                "status": "DENSE_LAUNCH_BOUND_NOT_EXECUTED",
                "output": str(output),
                "sha256": sha256(output),
                "inspection": result,
                "optimizer_updates": 0,
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

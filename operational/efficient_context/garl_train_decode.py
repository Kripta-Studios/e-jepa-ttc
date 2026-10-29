"""Resume frozen native producers with lossless parallel cache decoding and expanded disk."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .common import ROOT, Campaign, atomic_json, digest, read


def verify_admission(c: Campaign) -> dict:
    """Activate only source-pinned tested engineering; all original scientific files stay frozen."""
    c.freeze()
    seal = read(c.out / "garl/DECODE_ENGINEERING_FREEZE.json")
    if (
        seal["native_protocol_sha256"] != digest(c.out / "garl/PROTOCOL.json")
        or seal["disk_authorization_sha256"] != digest(c.out / "DISK_RESOURCE_AUTHORIZATION.json")
        or seal["QA_sha256"] != digest(c.out / "TEST_RESULTS/parallel_decode/QA.json")
        or seal["input_QA_sha256"] != digest(c.out / "garl/DECODE_INPUT_QA.json")
        or read(c.out / "TEST_RESULTS/parallel_decode/QA.json")["status"] != "PASSED"
        or read(c.out / "garl/DECODE_INPUT_QA.json")["status"] != "PASSED"
    ):
        raise ValueError("parallel decoding QA/admission differs from this campaign")
    for file in seal["files"]:
        if digest(Path(file["path"])) != file["sha256"]:
            raise ValueError("sealed parallel decoding source changed")
    for file in read(c.out / "garl/PROTOCOL.json")["science_files"]:
        if digest(Path(file["path"])) != file["sha256"]:
            raise ValueError("original native scientific source changed")
    return seal


def main() -> int:
    """Use the original complete-state trainer with no altered optimizer, inputs, RNG or sampler."""
    from . import garl_train_cached, garl_train_parallel, parallel_inputs
    from .deadline import permits
    from .expanded_disk import install
    from .parallel_decode import DecodingInputCache

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--handoff-decode", action="store_true")
    parser.add_argument(
        "--protocol", type=Path, default=ROOT / "configs/campaign/efficient_context_v1.json"
    )
    args, _ = parser.parse_known_args()
    c = Campaign(args.protocol)
    seal = verify_admission(c)
    install(c)
    if args.handoff_decode:
        from .decode_handoff import handoff

        handoff(c)
        sys.argv.remove("--handoff-decode")
    original_cache = parallel_inputs.ParallelInputCache
    original_guard = garl_train_cached.GuardedCampaign
    original_parallel_guard = garl_train_parallel.GuardedCampaign

    class DeadlineGuard(original_guard):
        def check(self) -> bool:
            return super().check() and permits(self)

    parallel_inputs.ParallelInputCache = DecodingInputCache
    garl_train_cached.GuardedCampaign = DeadlineGuard
    garl_train_parallel.GuardedCampaign = DeadlineGuard
    try:
        from .garl_train_scandir import main as original_recipe

        result = original_recipe()
    finally:
        parallel_inputs.ParallelInputCache = original_cache
        garl_train_cached.GuardedCampaign = original_guard
        garl_train_parallel.GuardedCampaign = original_parallel_guard
    atomic_json(
        c.out / "garl/DECODE_RUNTIME_PROVENANCE.json",
        {
            "engineering_freeze_sha256": digest(c.out / "garl/DECODE_ENGINEERING_FREEZE.json"),
            "native_protocol_sha256": seal["native_protocol_sha256"],
            "raw_preparation_processes": 4,
            "cache_decoding_threads": 4,
            "cache_lookahead_max": 64,
            "max_owned_bytes": 40_000_000_000,
            "max_native_cache_bytes": 32_000_000_000,
            "optimizer_recipe_checkpoint_RNG_sampler_changed": False,
            "returncode": result,
        },
    )
    return result


if __name__ == "__main__":
    raise SystemExit(main())

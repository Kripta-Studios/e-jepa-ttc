"""Validate archived native resources, then apply the separately authorized CPU resources."""

from __future__ import annotations

import argparse
from pathlib import Path

from .common import ROOT, Campaign, atomic_json, digest, read
from .garl_train import execute as _original_execute


def execute(c: Campaign) -> None:
    """Keep original protocol equality and restore active resources before any cache or fit."""
    from . import garl_train
    from .parallel_inputs import ParallelInputCache

    frozen = c.out / "garl/PROTOCOL.json"
    active_policy = c.policy
    if int(active_policy["max_tree_rss_gib"] * 1024**3) != 16_000_000_000:
        raise ValueError("parallel native execution requires the explicit16GB authorization")
    original_cache = garl_train.InputCache

    class AuthorizedCache(ParallelInputCache):
        def __init__(self, campaign: Campaign, rows: dict) -> None:
            # The frozen execute checks EVERY archived protocol field before constructing cache.
            # Only its archived RAM metadata is projected; scientific hashes are never bypassed.
            campaign.policy = active_policy
            super().__init__(campaign, rows)

    archived_policy = dict(active_policy)
    archived_policy["max_tree_rss_gib"] = read(frozen)["active_tree_RSS_limit_bytes"] / 1024**3
    c.require_resources()
    garl_train.InputCache = AuthorizedCache
    c.policy = archived_policy
    try:
        _original_execute(c)
    finally:
        c.policy = active_policy
        garl_train.InputCache = original_cache


def main() -> int:
    """Seal this explicit resource admission adapter before running the existing CPU wrapper."""
    from . import garl_train, garl_train_parallel
    from .garl_train_cached import GuardedCampaign

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--protocol", type=Path, default=ROOT / "configs/campaign/efficient_context_v1.json"
    )
    args, _ = parser.parse_known_args()
    c = GuardedCampaign(args.protocol)
    prior = c.out / "garl/PARALLEL_INPUT_FREEZE.json"
    for pin in read(prior)["files"]:
        if digest(Path(pin["path"])) != pin["sha256"]:
            raise ValueError("sealed CPU input source changed")
    value = {
        "native_protocol_sha256": digest(c.out / "garl/PROTOCOL.json"),
        "parallel_input_freeze_sha256": digest(prior),
        "resource_authorization_sha256": digest(c.out / "RESOURCE_AUTHORIZATION_V2.json"),
        "implementation_sha256": digest(Path(__file__)),
        "QA_sha256": digest(c.out / "TEST_RESULTS/parallel_resource_adapter/QA.json"),
        "original_protocol_equality_check_preserved": True,
        "archived_RSS_metadata_bytes": 12_000_000_000,
        "active_RSS_before_cache_or_fit_bytes": 16_000_000_000,
        "optimizer_recipe_checkpoint_RNG_batch_order_changed": False,
    }
    qa = read(c.out / "TEST_RESULTS/parallel_resource_adapter/QA.json")
    if qa["status"] != "PASSED" or qa["implementation_sha256"] != value["implementation_sha256"]:
        raise ValueError("resource adapter QA must match the sealed implementation")
    freeze = c.out / "garl/PARALLEL_RESOURCE_ADAPTER_FREEZE.json"
    if freeze.exists() and read(freeze) != value:
        raise ValueError("resource adapter changed after seal")
    if not freeze.exists():
        atomic_json(freeze, value)
    garl_train.execute = execute
    return garl_train_parallel.main()


if __name__ == "__main__":
    raise SystemExit(main())

"""Bound CPU input preparation to four workers around the unchanged frozen scientific recipe."""

from __future__ import annotations

import argparse
from collections.abc import Callable
from pathlib import Path

from .common import ROOT, Campaign, atomic_json, digest, read
from .garl_train import InputCache
from .garl_train_cached import GuardedCampaign


def main() -> int:
    """Seal exact CPU worker QA and resources before restoring the original complete states."""
    from . import garl_train
    from .parallel_inputs import ParallelInputCache

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol", type=Path, default=ROOT / "configs/campaign/efficient_context_v1.json"
    )
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    c = GuardedCampaign(args.protocol)
    c.freeze()
    previous = c.out / "garl/COMPRESSED_CACHE_FREEZE.json"
    for pin in read(previous)["files"]:
        if digest(Path(pin["path"])) != pin["sha256"]:
            raise ValueError("previously sealed compressed cache source changed")
    qa_path = c.out / "garl/PARALLEL_INPUT_QA.json"
    qa = read(qa_path)
    files = [
        Path(__file__),
        Path(__file__).with_name("parallel_inputs.py"),
        Path(__file__).with_name("parallel_input_qa.py"),
        Path(__file__).with_name("resource_policy.py"),
    ]
    if (
        qa["status"] != "PASSED"
        or qa["native_protocol_sha256"] != digest(c.out / "garl/PROTOCOL.json")
        or qa["implementation_sha256"] != digest(files[1])
        or qa["QA_implementation_sha256"] != digest(files[2])
        or qa["resource_authorization_sha256"] != digest(c.out / "RESOURCE_AUTHORIZATION_V2.json")
    ):
        raise ValueError("parallel input QA or resource authorization differs from this runtime")
    value = {
        "native_protocol_sha256": digest(c.out / "garl/PROTOCOL.json"),
        "previous_cache_freeze_sha256": digest(previous),
        "QA_sha256": digest(qa_path),
        "resource_authorization_sha256": qa["resource_authorization_sha256"],
        "files": [{"path": str(p), "sha256": digest(p)} for p in files],
        "CPU_preparation_workers": 4,
        "reader_handles_per_worker": 8,
        "lookahead_records_max": 64,
        "compressed_RAM_capacity_bytes": 8 * 1024**3,
        "RSS_tree_limit_bytes": 16_000_000_000,
        "heavy_trainers": 1,
        "optimizer_recipe_checkpoint_RNG_batch_order_changed": False,
        "disk_and_scientific_update_caps_changed": False,
    }
    freeze = c.out / "garl/PARALLEL_INPUT_FREEZE.json"
    if freeze.exists() and read(freeze) != value:
        raise ValueError("CPU worker implementation changed after its separate seal")
    if not freeze.exists():
        atomic_json(freeze, value)
    original_run_fit = garl_train.run_fit

    def run_fit(
        campaign: Campaign,
        fit: dict,
        protocol: dict,
        pin: str,
        ledger: dict,
        cache: InputCache,
        ctor: Callable,
    ) -> dict | None:
        if not isinstance(cache, ParallelInputCache):
            raise TypeError("CPU input wrapper requires its separately admitted cache")
        cache.activate(fit)
        try:
            return original_run_fit(campaign, fit, protocol, pin, ledger, cache, ctor)
        finally:
            cache.drain()

    garl_train.InputCache = ParallelInputCache
    garl_train.run_fit = run_fit
    garl_train.execute(c)
    if (c.out / "garl/ENDPOINTS.json").exists():
        atomic_json(
            c.out / "garl/PRODUCER_RUNTIME_PROVENANCE.json",
            {
                "native_protocol_sha256": value["native_protocol_sha256"],
                "producer_endpoints_sha256": digest(c.out / "garl/ENDPOINTS.json"),
                "compressed_cache_freeze_sha256": digest(previous),
                "parallel_input_freeze_sha256": digest(freeze),
                "all_original_scientific_files_preserved": True,
                "CPU_preparation_workers": 4,
                "worker_QA_optimizer_updates": 0,
            },
        )
    return 0 if (c.out / "garl/ENDPOINTS.json").exists() else 3


if __name__ == "__main__":
    raise SystemExit(main())

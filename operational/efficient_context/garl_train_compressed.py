"""Separately sealed compressed input storage around the unchanged native scientific trainer."""

from __future__ import annotations

import argparse
from pathlib import Path

from .common import ROOT, atomic_json, digest, read
from .garl_train_cached import GuardedCampaign


def main() -> int:
    """Admit bit-exact storage only after QA; retain the original producer protocol and states."""
    from . import garl_train
    from .compressed_cache import CompressedInputCache

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol", type=Path, default=ROOT / "configs/campaign/efficient_context_v1.json"
    )
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    c = GuardedCampaign(args.protocol)
    c.freeze()
    original = c.out / "garl/CACHE_ENGINEERING_FREEZE.json"
    for pin in read(original)["files"]:
        if digest(Path(pin["path"])) != pin["sha256"]:
            raise ValueError("previously sealed cache engineering source changed")
    qa_path = c.out / "garl/COMPRESSED_CACHE_QA.json"
    qa = read(qa_path)
    files = [
        Path(__file__),
        Path(__file__).with_name("compressed_cache.py"),
        Path(__file__).with_name("compressed_cache_qa.py"),
    ]
    if (
        qa["status"] != "PASSED"
        or qa["native_protocol_sha256"] != digest(c.out / "garl/PROTOCOL.json")
        or qa["cache_implementation_sha256"] != digest(files[1])
        or qa["QA_implementation_sha256"] != digest(files[2])
    ):
        raise ValueError("compressed cache QA is not bound to these exact source files")
    value = {
        "native_protocol_sha256": digest(c.out / "garl/PROTOCOL.json"),
        "previous_cache_engineering_freeze_sha256": digest(original),
        "QA_sha256": digest(qa_path),
        "files": [{"path": str(p), "sha256": digest(p)} for p in files],
        "reason": "original lossless NPZ bytes also in RAM to cover TRAIN working sets",
        "scientific_recipe_checkpoint_RNG_sampler_changed": False,
        "memory_disk_capacity_increase": False,
        "optimizer_updates_for_QA": 0,
    }
    pin = c.out / "garl/COMPRESSED_CACHE_FREEZE.json"
    if pin.exists() and read(pin) != value:
        raise ValueError("compressed cache implementation changed after its separate seal")
    if not pin.exists():
        atomic_json(pin, value)
    garl_train.InputCache = CompressedInputCache
    garl_train.execute(c)
    if (c.out / "garl/ENDPOINTS.json").exists():
        atomic_json(
            c.out / "garl/PRODUCER_RUNTIME_PROVENANCE.json",
            {
                "native_protocol_sha256": value["native_protocol_sha256"],
                "producer_endpoints_sha256": digest(c.out / "garl/ENDPOINTS.json"),
                "cache_engineering_freeze_sha256": digest(original),
                "compressed_cache_freeze_sha256": digest(pin),
                "all_original_scientific_files_preserved": True,
                "cache_QA_raw_encodings_and_optimizer_updates": 0,
            },
        )
    return 0 if (c.out / "garl/ENDPOINTS.json").exists() else 3


if __name__ == "__main__":
    raise SystemExit(main())

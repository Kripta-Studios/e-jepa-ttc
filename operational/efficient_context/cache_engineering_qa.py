"""Verify a lossless cache-placement amendment against64 already encoded TRAIN inputs."""

from __future__ import annotations

import shutil
from copy import copy
from pathlib import Path

from .common import ROOT, Campaign, atomic_json, digest, read
from .exclusive_cache import ExclusiveInputCache
from .garl_train import records


def run(c: Campaign) -> dict:
    """Exercise RAM/disk transfers in an isolated own QA root, with zero raw encodings."""
    import torch

    torch.set_num_threads(4)
    torch.set_num_interop_threads(2)
    c.require_resources()
    c.freeze()
    proto = c.out / "garl/PROTOCOL.json"
    for pin in read(proto)["science_files"]:
        if digest(Path(pin["path"])) != pin["sha256"]:
            raise ValueError("native scientific freeze changed before cache QA")
    receipt = c.out / "garl/CACHE_ENGINEERING_QA.json"
    if receipt.exists():
        saved = read(receipt)
        if (
            saved["status"] == "PASSED"
            and saved["native_protocol_sha256"] == digest(proto)
            and saved["cache_implementation_sha256"]
            == digest(Path(__file__).with_name("exclusive_cache.py"))
            and saved["QA_implementation_sha256"] == digest(Path(__file__))
        ):
            return saved
        raise ValueError("existing cache QA changed; preserve it for inspection")
    destination = c.out / "verification/cache_engineering"
    isolated = copy(c)
    isolated.out = destination
    (destination / "garl/native_cache").mkdir(parents=True, exist_ok=True)
    shutil.copy2(proto, destination / "garl/PROTOCOL.json")
    rows = records(c)
    # Compute the unchanged keys without opening or mutating the live trainer's cache.
    cache = ExclusiveInputCache(isolated, rows)
    paths = sorted((c.out / "garl/native_cache").glob("*.npz"))[:64]
    if len(paths) != 64:
        raise ValueError("64 existing encoded TRAIN inputs required")
    chosen = []
    references = {}
    for source in paths:
        if source.name not in cache.key_tokens:
            raise ValueError("native cache key is outside the permitted TRAIN pool")
        token = cache.key_tokens[source.name]
        references[token] = cache.decode(source)
        target = cache.root / source.name
        shutil.copy2(source, target)
        size = target.stat().st_size
        cache.disk += size - cache.files.get(source.name, 0)
        cache.files[source.name] = size
        chosen.append({"token": token, "source_sha256": digest(source)})
    cache.limit = 2_000_000
    cache.evict(0)
    cases = []
    for repeat in range(3):
        for token, expected in references.items():
            value = cache.get(token)
            exact = (
                torch.equal(value[0], expected[0])
                and torch.equal(value[1], expected[1])
                and value[2] == expected[2]
            )
            if not exact:
                raise ValueError("cache placement changed a native TRAIN input")
            if cache.bytes > cache.memory_limit or cache.disk > cache.limit:
                raise ValueError("cache placement exceeded its fixed capacity")
            cases.append({"token": token, "repeat": repeat, "all_returned_fields_exact": exact})
    if cache.reads != 0 or cache.preserved_disk_evictions == 0:
        raise ValueError("cache QA must preserve existing disk victims without new raw extraction")
    result = {
        "status": "PASSED",
        "cases": cases,
        "selected_by": "first64 encoded TRAIN filenames sorted by SHA key; no GT selection",
        "source_inputs": chosen,
        "native_protocol_sha256": digest(proto),
        "cache_implementation_sha256": digest(Path(__file__).with_name("exclusive_cache.py")),
        "QA_implementation_sha256": digest(Path(__file__)),
        "raw_encodings": cache.reads,
        "optimizer_updates": 0,
        "preserved_disk_evictions": cache.preserved_disk_evictions,
        "production_memory_limit_bytes": cache.memory_limit,
        "isolated_QA_disk_limit_bytes": cache.limit,
        "live_trainer_inputs_modified": False,
        "scientific_recipe_modified": False,
    }
    cache.close()
    atomic_json(c.out / "garl/CACHE_ENGINEERING_QA.json", result)
    return result


def main() -> int:
    """Native cache parity QA; no optimizer, model evaluation or holdout access."""
    print(run(Campaign(ROOT / "configs/campaign/efficient_context_v1.json"))["status"], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

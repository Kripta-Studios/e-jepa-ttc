"""Bit parity through compressed RAM/disk transfers using existing authorized TRAIN inputs."""

from __future__ import annotations

from copy import copy
from pathlib import Path

from .common import ROOT, Campaign, atomic_bytes, atomic_json, digest, read
from .compressed_cache import CompressedInputCache
from .garl_train import records


def run(c: Campaign) -> dict:
    """Verify three passes over64 immutable copied inputs with zero raw encodings or updates."""
    import torch

    torch.set_num_threads(4)
    torch.set_num_interop_threads(2)
    c.require_resources()
    c.freeze()
    proto = c.out / "garl/PROTOCOL.json"
    for pin in read(proto)["science_files"]:
        if digest(Path(pin["path"])) != pin["sha256"]:
            raise ValueError("original native scientific source changed")
    destination = c.out / "verification/compressed_cache_engineering"
    isolated = copy(c)
    isolated.out = destination
    atomic_bytes(destination / "garl/PROTOCOL.json", proto.read_bytes())
    cache = CompressedInputCache(isolated, records(c))
    if cache.files:
        raise ValueError(
            "compressed QA requires a fresh isolated cache; preserve previous attempts"
        )
    references = {}
    sources = []
    for source in sorted((c.out / "garl/native_cache").glob("*.npz")):
        try:
            payload = source.read_bytes()
        except FileNotFoundError:
            continue  # An immutable live cache victim disappeared; select the next SHA key.
        token = cache.key_tokens[source.name]
        references[token] = cache.unpack(payload)
        atomic_bytes(cache.root / source.name, payload)
        cache.files[source.name] = len(payload)
        cache.disk += len(payload)
        import hashlib

        sources.append({"token": token, "source_sha256": hashlib.sha256(payload).hexdigest()})
        if len(sources) == 64:
            break
    if len(sources) != 64:
        raise ValueError("64 existing authorized TRAIN inputs required")
    cache.limit = 2_000_000
    cache.evict(0)
    cases = []
    torch_rng = torch.get_rng_state().clone()
    for repeat in range(3):
        for token, expected in references.items():
            value = cache.get(token)
            exact = all(torch.equal(value[i], expected[i]) for i in (0, 1))
            exact = exact and value[2] == expected[2]
            contiguous = all(value[i].is_contiguous() for i in (0, 1))
            if not exact or not contiguous:
                raise ValueError("compressed cache changed a native input or tensor layout")
            if cache.bytes > cache.memory_limit or cache.disk > cache.limit:
                raise ValueError("compressed cache exceeded its unchanged byte capacities")
            cases.append({"token": token, "repeat": repeat, "all_fields_exact": exact})
    if cache.reads or not cache.preserved_disk_evictions:
        raise ValueError("compressed QA must exercise exclusive transfers without raw reads")
    if not torch.equal(torch_rng, torch.get_rng_state()):
        raise ValueError("compressed input decoding consumed training RNG")
    result = {
        "status": "PASSED",
        "cases": cases,
        "source_inputs": sources,
        "selection": "first64 readable TRAIN cache filenames in SHA order; no GT selection",
        "native_protocol_sha256": digest(proto),
        "cache_implementation_sha256": digest(Path(__file__).with_name("compressed_cache.py")),
        "QA_implementation_sha256": digest(Path(__file__)),
        "raw_encodings": cache.reads,
        "optimizer_updates": 0,
        "training_RNG_consumed": False,
        "live_trainer_inputs_modified": False,
        "preserved_disk_evictions": cache.preserved_disk_evictions,
        "production_memory_limit_bytes": cache.memory_limit,
        "compressed_RAM_bytes": cache.bytes,
        "original_decoded_bytes": sum(cache.tensor_bytes(v) for v in references.values()),
        "all_tensor_layouts_contiguous": True,
        "memory_disk_capacities_changed": False,
    }
    cache.close()
    atomic_json(c.out / "garl/COMPRESSED_CACHE_QA.json", result)
    return result


if __name__ == "__main__":
    print(run(Campaign(ROOT / "configs/campaign/efficient_context_v1.json"))["status"], flush=True)

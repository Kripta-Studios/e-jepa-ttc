"""Own lossless H/V RGB input caches; reuse native decoding without changing V13."""

from __future__ import annotations

import argparse
import msvcrt
from pathlib import Path

from operational.range_transfer.common import freeze, read, sha, utc, write
from operational.rgb_port import cache_prepare as native

from .common import ROOT, RUN, config, owner, require_resources
from .data import role_rows


def local_binding(role: str) -> Path:
    """Return only V18-owned binding paths, leaving historical manifests untouched."""
    if role not in {"H", "V"}:
        raise ValueError("Reuse the existing P RGB cache")
    return RUN / "rgb/cache_bindings" / f"{role}_RGB_CACHE_BINDING.json"


def prepare(role: str) -> dict:
    """Materialize one complete held-out sensor population, without model inference."""
    binding_path = local_binding(role)
    manifest_path = owner() / f"{role}_MANIFEST.json"
    role_sha = sha(manifest_path)
    if role_sha != read(RUN / "ADMISSION.json")["files"][str(manifest_path)]:
        raise ValueError("RGB role differs from V18 admission")
    manifest = read(manifest_path)
    if sha(Path(manifest["rows_path"])) != manifest["rows_sha256"]:
        raise ValueError("RGB role rows changed")
    root = Path(config()["cache_root"]) / "rgb" / role
    root.mkdir(parents=True, exist_ok=True)
    with (root / "WRITER.lock").open("a+b") as lock:
        lock.seek(0)
        if not lock.read(1):
            lock.write(b"0")
            lock.flush()
        lock.seek(0)
        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        try:
            return _prepare_locked(role, root, manifest_path, manifest, role_sha, binding_path)
        finally:
            lock.seek(0)
            msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)


def _prepare_locked(
    role: str, root: Path, manifest_path: Path, manifest: dict, role_sha: str, binding_path: Path
) -> dict:
    rows = role_rows(role)
    if set(rows.producer_frame_count) != {2, 3}:
        raise ValueError("Review native RGB parity fixture for an unexpected frame-count cohort")
    admission = {
        "role": role,
        "population": len(rows),
        "role_manifest_sha256": role_sha,
        "rows_sha256": manifest["rows_sha256"],
        "source_hashes": {
            name: sha(ROOT / name)
            for name in (
                "operational/joint_measurement/rgb_cache.py",
                "operational/rgb_port/cache_prepare.py",
                "src/e_jepa_ttc/rgb_port/data.py",
                "src/e_jepa_ttc/rgb_port/history.py",
                "src/e_jepa_ttc/rgb_port/contracts.py",
            )
        },
        "scope": "sensor preparation only; no TTC scalar targets stored; no inference/optimizer",
    }
    freeze(root / "ADMISSION.json", admission)
    output = root / "CACHE_MANIFEST.json"
    if binding_path.exists():
        binding = read(binding_path)
        if binding["cache_manifest_sha256"] != sha(output):
            raise ValueError("Completed V18 RGB cache manifest changed")
        return read(output)
    require_resources(starting=True, disk_reserve=8 * 1024**3)
    partitions = native._partitions(rows)
    complete = []
    native._init_worker(str(Path(manifest["eap_root"])))
    try:
        for indices, records, relative in partitions:
            require_resources(disk_reserve=500 * 1024**2)
            path = root / relative
            existing = native._existing(
                path, indices, [str(r["sample_token"]) for r in records], role_sha
            )
            if existing is None:
                if path.with_suffix(".json").exists():
                    raise ValueError(f"Retain and investigate corrupt RGB shard: {path}")
                existing = native._task(records, indices, str(path), role_sha)
            complete.append(existing)
            progress = {
                "status": "PREPARING",
                "role": role,
                "completed_shards": len(complete),
                "total_shards": len(partitions),
                "completed_rows": sum(len(s["indices"]) for s in complete),
                "utc": utc(),
            }
            write(root / "PROGRESS.json", progress)
            write(RUN / f"RGB_PREPARE_{role}_PROGRESS.json", progress)
            if len(complete) % 20 == 0:
                print(f"RGB {role} {len(complete)}/{len(partitions)} shards", flush=True)
    finally:
        if native._READER is not None:
            native._READER.close()
            native._READER = None
    covered = sorted(index for shard in complete for index in shard["indices"])
    if covered != list(range(len(rows))):
        raise ValueError("RGB cache lost or duplicated population indices")
    parity = native._parity(rows, complete, Path(manifest["eap_root"]), root)
    result = {
        "schema": "rgb_port_prepared_input_cache_v1",
        "status": "COMPLETE",
        "role": role,
        "role_manifest_path": str(manifest_path),
        "role_manifest_sha256": role_sha,
        "population_size": len(rows),
        "shard_size": native.SHARD_SIZE,
        "compression": "numpy_savez_compressed_lossless",
        "stored_pixels": "uint8_RGB",
        "float_conversion": "derived_on_read_exact_uint8_float32_div255",
        "no_frame_padding_or_duplication": True,
        "decode_parity_verified": parity["all_fields_bit_exact"],
        "cache_fields": list(native.CACHE_FIELDS),
        "shards": complete,
        "compressed_bytes": sum(shard["bytes"] for shard in complete),
    }
    freeze(output, result)
    freeze(
        binding_path,
        {
            "schema": "rgb_port_cache_binding_v1",
            "status": "COMPLETE",
            "role_manifest_sha256": role_sha,
            "cache_manifest_path": str(output.resolve()),
            "cache_manifest_sha256": sha(output),
            "equivalence": "bit_exact_decoded_inputs_no_recipe_change",
        },
    )
    progress = {
        "status": "COMPLETE",
        "role": role,
        "rows": len(rows),
        "shards": len(complete),
        "bytes": result["compressed_bytes"],
        "utc": utc(),
    }
    write(root / "PROGRESS.json", progress)
    write(RUN / f"RGB_PREPARE_{role}_PROGRESS.json", progress)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("role", choices=["H", "V", "both"])
    selected = parser.parse_args().role
    for role in ("H", "V") if selected == "both" else (selected,):
        result = prepare(role)
        print({k: result[k] for k in ("status", "role", "population_size", "compressed_bytes")})

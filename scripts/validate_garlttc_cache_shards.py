#!/usr/bin/env python
"""Validate every cache shard against metadata and torch deserialization."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from e_jepa_ttc.data.garlttc_lhr_cache import _load_torch_records  # noqa: E402
from e_jepa_ttc.utils.io import read_structured  # noqa: E402


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()

    manifest_path = args.manifest.resolve()
    manifest = read_structured(manifest_path)
    root = manifest_path.parent
    failures: list[dict[str, object]] = []
    total_records = 0

    for index, shard in enumerate(manifest.get("shards", [])):
        path = root / str(shard["path"])
        print(f"[{index + 1}] {path}", flush=True)
        try:
            if not path.is_file():
                raise FileNotFoundError(path)
            actual_sha = sha256_file(path)
            if actual_sha != shard.get("sha256"):
                raise RuntimeError(
                    f"sha256 mismatch: manifest={shard.get('sha256')} actual={actual_sha}"
                )
            records = _load_torch_records(path)
            if len(records) != int(shard["count"]):
                raise RuntimeError(
                    f"count mismatch: manifest={shard['count']} loaded={len(records)}"
                )
            total_records += len(records)
        except Exception as exc:
            failures.append({"path": str(path), "error": f"{type(exc).__name__}: {exc}"})

    result = {
        "artifact_type": "garlttc_cache_integrity_audit_v1",
        "status": "failed" if failures else "passed",
        "manifest": str(manifest_path),
        "shard_count": len(manifest.get("shards", [])),
        "record_count": total_records,
        "failures": failures,
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Create a write-once byte-for-byte snapshot of integrated E/R producer glue."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections.abc import Sequence
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FILES = (
    "docs/detection_sota_20261029/INDEPENDENT_PRODUCER_FOUNDATION.md",
    "operational/independent_refit/producer_contract.py",
    "operational/independent_refit/producer_inputs.py",
    "operational/independent_refit/producer_rgb_inputs.py",
    "operational/independent_refit/producer_train.py",
    "scripts/prepare_independent_phase_e_a5_source_admission.py",
    "scripts/smoke_independent_phase_e_a5_cpu.py",
    "scripts/snapshot_independent_phase_producer_sources.py",
    "scripts/train_independent_phase_e_a5.py",
    "tests/test_independent_phase_producer_contract.py",
    "tests/test_independent_phase_producer_inputs.py",
    "tests/test_independent_phase_producer_rgb_inputs.py",
    "tests/test_independent_phase_producer_train.py",
)


def _sha256(path: Path) -> str:
    with path.resolve(strict=True).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite source snapshot: {output}")
    before = {name: _sha256(ROOT / name) for name in FILES}
    records: list[dict[str, object]] = []
    for name in FILES:
        source = (ROOT / name).resolve(strict=True)
        destination = output / "files" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        copied = _sha256(destination)
        if copied != before[name]:
            raise RuntimeError(f"Snapshot copy differs: {name}")
        records.append(
            {
                "path": name,
                "sha256": copied,
                "size_bytes": destination.stat().st_size,
            }
        )
    after = {name: _sha256(ROOT / name) for name in FILES}
    if after != before:
        raise RuntimeError("Producer source changed while snapshotting")
    manifest = {
        "schema": "independent_phase_er_producer_source_snapshot_v1",
        "status": "FROZEN_PRE_EXECUTION",
        "files": records,
        "file_count": len(records),
        "complete_bytes_copied": True,
        "source_hashes_equal_before_after": True,
        "historical_pre_r_a5_glue_recovered": False,
    }
    payload = json.dumps(manifest, sort_keys=True, indent=2, allow_nan=False) + "\n"
    manifest_path = output / "MANIFEST.json"
    manifest_path.write_text(payload, encoding="utf-8", newline="\n")
    verification = json.loads(manifest_path.read_text(encoding="utf-8"))
    if verification != manifest:
        raise RuntimeError("Source snapshot manifest round trip differs")
    print(json.dumps({"manifest": str(manifest_path), "sha256": _sha256(manifest_path)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

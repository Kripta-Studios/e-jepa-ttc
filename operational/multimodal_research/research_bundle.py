"""Build an immutable, byte-verified research snapshot without replacing historical bundles."""

from __future__ import annotations

import hashlib
import json
import subprocess
import zipfile
from datetime import UTC, datetime

from operational.multimodal_validation.common import OUT as PREVIOUS
from operational.multimodal_validation.common import ROOT, read, sha, write

from .common import OUT, accounting


def main() -> None:
    ledger = accounting()
    assert ledger["pending_upper"] == 0
    for row in ledger["fits"]:
        assert row["completed_updates"] == 2500, "Snapshot only at fixed complete fit endpoints"
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    destination = ROOT / "artifacts" / f"E_JEPA_TTC_RESEARCH_PROGRESS_{stamp}.zip"
    assert not destination.exists()
    paths = set()
    for directory in (
        ROOT / "src",
        ROOT / "operational",
        ROOT / "configs",
        ROOT / "tests",
        ROOT / "docs/multimodal_next_handoff_20261012",
        ROOT / "docs/dev32_transfer_20261012",
        OUT,
        PREVIOUS,
    ):
        for path in directory.rglob("*"):
            if not path.is_file() or "__pycache__" in path.parts or path.suffix == ".pyc":
                continue
            if ".pending" in path.name or path.suffix in {".lock", ".tmp"}:
                continue
            paths.add(path)
    for name in ("README.md", "AGENTS.md", "pyproject.toml", "uv.lock", ".gitattributes"):
        path = ROOT / name
        if path.exists():
            paths.add(path)
    metadata = dict(
        utc=stamp,
        git_commit=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        git_status=subprocess.check_output(["git", "status", "--short"], cwd=ROOT, text=True),
        status="IN_PROGRESS",
        official_SOTA=False,
        accounting=ledger,
        next_decision=read(OUT / "NEXT_DECISION.json"),
        dependencies=(
            "New readout checkpoints, H/V state caches, predictions and sources included. "
            "Original sensor HDF5/RGB archives, pretrained V13 producers/PAIR/heads and public "
            "Garl checkpoints remain external; source bindings identify them. Partial sensor "
            "replay directories are recoverable per query and are complete only with SEALED.json."
        ),
        scientific_scope=(
            "Adaptive development over exposed V/Dev32/FCWD. No private test labels, "
            "submissions or claim of official SOTA. Old V16 result archive preserved."
        ),
    )
    verifier = """from pathlib import Path
import hashlib,json
root=Path(__file__).resolve().parent
manifest=json.loads((root/'MANIFEST.json').read_text(encoding='utf-8'))
for entry in manifest['files']:
    path=(root/entry['path']).resolve()
    assert path.is_relative_to(root)
    data=path.read_bytes()
    assert len(data)==entry['bytes'] and hashlib.sha256(data).hexdigest()==entry['sha256'], path
print('Verified',len(manifest['files']),'files')
"""
    entries = []
    with zipfile.ZipFile(
        destination, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=6
    ) as archive:
        for path in sorted(paths):
            data = path.read_bytes()
            name = path.relative_to(ROOT).as_posix()
            archive.writestr(name, data)
            entries.append(
                dict(path=name, bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
            )
        extras = {
            "SNAPSHOT.json": json.dumps(metadata, ensure_ascii=False, indent=2).encode("utf-8"),
            "VERIFY.py": verifier.encode("utf-8"),
        }
        for name, data in extras.items():
            archive.writestr(name, data)
            entries.append(
                dict(path=name, bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
            )
        archive.writestr(
            "MANIFEST.json", json.dumps(dict(files=entries), ensure_ascii=False, indent=2)
        )
    with zipfile.ZipFile(destination) as archive:
        assert archive.testzip() is None
        manifest = json.loads(archive.read("MANIFEST.json"))
        assert len(archive.namelist()) == len(entries) + 1
        for entry in manifest["files"]:
            data = archive.read(entry["path"])
            assert (
                len(data) == entry["bytes"] and hashlib.sha256(data).hexdigest() == entry["sha256"]
            )
    receipt = dict(
        path=str(destination),
        bytes=destination.stat().st_size,
        sha256=sha(destination),
        files=len(entries),
        crc_pass=True,
        every_payload_sha256_verified=True,
        status="IN_PROGRESS",
        source_sha256=sha(__file__),
        official_SOTA=False,
    )
    write(destination.with_suffix(".verification.json"), receipt)
    print(receipt, flush=True)


if __name__ == "__main__":
    main()

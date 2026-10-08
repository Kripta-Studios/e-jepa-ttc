"""Verified evidence delivery for the measured1GiB resident R1 configuration."""

from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

from .common import ROOT, atomic_bytes, atomic_json, digest, read
from .r1_chain import stamp


def bundle(output: Path) -> None:
    """Generate engineering-only report and independently verify every ZIP member."""
    result = read(output / "measurement/RESULT.json")
    if not result["gpu_measured"] or result["pairs"] != 768:
        raise ValueError("three complete paired blocks required for R1 delivery")
    lines = [
        "# R1 chronological sensor replay",
        "",
        "Scientific optimizer updates: **0**.",
        "",
        "Fixed original 64 TRAIN queries, H1/H8/H16/WIDE, three paired blocks.",
        "Each route ingests all events between successive queries, including gaps.",
        "Both routes use a bounded sensor cache. Reference uses canonical per-window ROI",
        "projection; mapped uses the previously validated map-once union.",
        "",
        "| Block | Arm | Route | p50 ms | p95 ms | Ingest seconds | Fallback reads |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for row in result["summaries"]:
        lines.append(
            f"| {row['block']} | {row['label']} | {row['route']} | "
            f"{row['p50_ms']:.2f} | {row['p95_ms']:.2f} | "
            f"{row['ingestion_seconds']:.2f} | {row['fallback_reads']} |"
        )
    lines += [
        "",
        "This measures sparse chronological replay, not an online system.",
        "Do not compare it directly with independent-request R0 or training speed.",
        "All preprocessed tensors are bit exact; outputs retain the original tolerances.",
        "The resident bound is 1024 MiB per route plus source chunks and model tensors.",
        "Time starts at the first requested history in each recording.",
        "Model initialization is recorded separately and excluded from per-query latency.",
        "No protected holdouts were opened.",
    ]
    atomic_bytes(output / "REPORT.md", ("\n".join(lines) + "\n").encode())
    files = [
        p
        for p in output.rglob("*")
        if p.is_file()
        and p.suffix in (".json", ".csv", ".md")
        and p.name not in ("WRITER.lock", "BUNDLE_VERIFICATION.json", "CHAIN_STATE.json")
        and "cpu_parity" not in p.parts
    ]
    sources = list((ROOT / "operational/efficient_context").glob("r1*.py"))
    sources += [
        ROOT / "tests/unit/test_r1_reader.py",
        ROOT / "tests/unit/test_r1_supervisor.py",
        ROOT / "tests/unit/test_r1_gib.py",
    ]
    entries = {str(p.relative_to(output)).replace("\\", "/"): p for p in files}
    entries.update({"source/" + str(p.relative_to(ROOT)).replace("\\", "/"): p for p in sources})
    manifest = {name: digest(path) for name, path in entries.items()}
    destination = output / "R1_ESSENTIAL.zip"
    temporary = output / "R1_ESSENTIAL.zip.tmp"
    with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, path in entries.items():
            archive.write(path, name)
        archive.writestr("SHA256_MANIFEST.json", json.dumps(manifest, indent=2))
    with zipfile.ZipFile(temporary) as archive:
        for name, expected in manifest.items():
            if hashlib.sha256(archive.read(name)).hexdigest() != expected:
                raise ValueError(f"R1 bundle member mismatch: {name}")
    temporary.replace(destination)
    sha = digest(destination)
    atomic_bytes(output / "R1_ESSENTIAL.zip.sha256", f"{sha}  R1_ESSENTIAL.zip\n".encode())
    atomic_json(
        output / "BUNDLE_VERIFICATION.json",
        dict(status="PASSED", sha256=sha, members=len(manifest), checked_utc=stamp()),
    )

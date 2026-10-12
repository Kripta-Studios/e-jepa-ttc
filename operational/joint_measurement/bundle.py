"""Stream a quiescent V18 result bundle and verify every member independently."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import zipfile
from pathlib import Path, PurePosixPath

MANIFEST = "BUNDLE_MANIFEST.json"
FIXED_TIME = (1980, 1, 1, 0, 0, 0)


def safe_name(name: str) -> str:
    """Reject names that could escape or alias another member on Windows."""
    path = PurePosixPath(name)
    if (
        not name
        or "\\" in name
        or ":" in name
        or path.is_absolute()
        or any(part in {"", ".", ".."} for part in name.split("/"))
        or any(part.rstrip(" .") != part for part in path.parts)
    ):
        raise ValueError(f"Unsafe bundle member: {name}")
    return name


def write_archive(output: Path, files: dict[str, Path], context: dict) -> dict:
    """Hash the exact bytes streamed into the ZIP, rejecting changing input files."""
    names = [safe_name(name) for name in files]
    if len({name.casefold() for name in names}) != len(names) or any(
        name.casefold() == MANIFEST.casefold() for name in names
    ):
        raise ValueError("Duplicate or reserved bundle member")
    if output.exists() or output.with_suffix(output.suffix + ".pending").exists():
        raise FileExistsError("A bundle destination must be new")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".pending")
    payload = {}
    with zipfile.ZipFile(temporary, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=1) as arc:
        for name, source in sorted(files.items()):
            before = source.stat()
            hasher = hashlib.sha256()
            size = 0
            info = zipfile.ZipInfo(name, FIXED_TIME)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            with source.open("rb") as reader, arc.open(info, "w", force_zip64=True) as writer:
                for block in iter(lambda: reader.read(1024 * 1024), b""):
                    writer.write(block)
                    hasher.update(block)
                    size += len(block)
            after = source.stat()
            if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                raise RuntimeError(f"Bundle input changed while reading: {name}")
            if size != before.st_size:
                raise RuntimeError(f"Bundle input size changed: {name}")
            payload[name] = {"bytes": size, "sha256": hasher.hexdigest()}
        manifest = {"schema": "v18_essential_bundle_v1", "context": context, "files": payload}
        arc.writestr(MANIFEST, json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    # Do not publish an archive merely because writing it succeeded.
    verified = verify_archive(temporary)
    os.replace(temporary, output)
    return verified


def verify_archive(path: Path) -> dict:
    """Check exact membership, sizes, SHA-256 and ZIP CRC by fully reading payloads."""
    with zipfile.ZipFile(path) as arc:
        names = arc.namelist()
        for name in names:
            safe_name(name)
        if len(names) != len({name.casefold() for name in names}):
            raise ValueError("Duplicate archive member")
        manifest = json.loads(arc.read(MANIFEST))
        if manifest["schema"] != "v18_essential_bundle_v1":
            raise ValueError("Unknown bundle schema")
        expected = manifest["files"]
        if set(names) != set(expected) | {MANIFEST}:
            raise ValueError("Bundle membership differs from manifest")
        total = 0
        for name, reference in expected.items():
            digest = hashlib.sha256()
            size = 0
            with arc.open(name) as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(block)
                    size += len(block)
            if {"bytes": size, "sha256": digest.hexdigest()} != reference:
                raise ValueError(f"Bundle member differs: {name}")
            total += size
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return {
        "status": "PASS",
        "zip_sha256": digest.hexdigest(),
        "zip_bytes": path.stat().st_size,
        "payload_members": len(expected),
        "payload_bytes": total,
        "context": manifest["context"],
        "scope": "Exact member set, payload sizes, SHA-256 and ZIP CRC; no scientific replay",
    }


def require_quiescent(root: Path, run: Path) -> None:
    """Require stopped own writers and fully durable checkpoints before packaging."""
    import psutil

    from operational.range_transfer.common import read

    own_chain = {os.getpid(), *[p.pid for p in psutil.Process().parents()]}
    for process in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            if process.pid in own_chain or "python" not in (process.info["name"] or "").lower():
                continue
            command = " ".join(process.info["cmdline"] or [])
            if "operational.joint_measurement" in command and Path(process.cwd()) == root:
                raise RuntimeError(f"V18 writer still live: PID {process.pid}")
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    for path in run.glob("**/UPDATE_JOURNAL.json"):
        journal = read(path)
        if journal["pending_update_upper"] or (
            journal["completed_updates"] != journal["durable_updates"]
        ):
            raise ValueError(f"Non-durable optimization work: {path}")


def build(output: Path) -> dict:
    """Collect committed source, all V18 evidence and cache provenance, not raw sensors."""
    from operational.range_transfer.common import read, utc, write

    from .common import ROOT, RUN, config

    require_quiescent(ROOT, RUN)
    conclusion = read(RUN / "CAMPAIGN_CONCLUSION.json")
    if conclusion.get("research_status") not in {
        "RESEARCH_ONGOING",
        "CANDIDATE_FOR_CONFIRMATION",
        "EXTERNALLY_BLOCKED",
        "CONFIRMED_BENCHMARK_RESULT",
    } or not isinstance(conclusion.get("remaining_work"), list):
        raise ValueError("Explicit campaign conclusion and remaining work required")
    if subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT):
        raise ValueError("Commit source changes before building the result bundle")
    tracked = subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT).decode().split("\0")
    files = {
        "source/" + name: ROOT / name
        for name in tracked
        if name and not name.startswith(("artifacts/", "data/cache/"))
    }
    excluded = []
    for path in RUN.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(RUN)
        if (
            "__pycache__" in relative.parts
            or path.suffix in {".pyc", ".lock"}
            or ".pending" in path.name
        ):
            excluded.append(relative.as_posix())
            continue
        files["campaign/" + relative.as_posix()] = path
    for folder, namespace in (("sensor", "sensor_provenance"), ("rgb", "rgb_sensor_provenance")):
        sensor = Path(config()["cache_root"]) / folder
        for path in sensor.rglob("*.json"):
            files[namespace + "/" + path.relative_to(sensor).as_posix()] = path
    context = {
        "created_utc": utc(),
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT).decode().strip(),
        "conclusion": conclusion,
        "omitted_runtime_files": sorted(excluded),
        "raw_sensor_cache_included": False,
        "external_historical_parents_embedded": False,
        "reproduction_scope": (
            "Included weights, portable export inputs and predictions can be replayed. "
            "Full training also requires admitted public sensor sources and historical parents."
        ),
    }
    required = sum(path.stat().st_size for path in files.values())
    output.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(output.parent).free < required + 10_000_000_000:
        raise RuntimeError("Bundle would violate the 10 GB free-disk reserve")
    result = write_archive(output, files, context)
    output.with_suffix(output.suffix + ".sha256").write_text(
        f"{result['zip_sha256']}  {output.name}\n", encoding="utf-8"
    )
    write(output.with_suffix(output.suffix + ".verification.json"), result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["build", "verify"])
    parser.add_argument("path", type=Path)
    arguments = parser.parse_args()
    print(build(arguments.path) if arguments.action == "build" else verify_archive(arguments.path))

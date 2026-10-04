"""Essential bundle with full checkpoints, streaming hashes and independent numerical replay."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import re
import subprocess
import sys
import zipfile
from pathlib import Path

from .common import ROOT, Campaign, atomic_bytes, atomic_json, digest


def release_cache_for_bundle(c: Campaign) -> None:
    """Release disposable numeric inputs while retaining every scientific checkpoint."""
    if (c.out / "WRITER.lock").exists():
        raise InterruptedError("essential bundle requires no active campaign writer")
    root = (c.out / "garl/native_cache").resolve()
    if c.out.resolve() not in root.parents:
        raise ValueError("numeric cache root escaped the owned campaign")
    paths = list(root.glob("*.npz")) if root.exists() else []
    for path in paths:
        if path.resolve().parent != root or not re.fullmatch(r"[0-9a-f]{64}\.npz", path.name):
            raise ValueError("unrecognized path in the disposable numeric cache")
    size = sum(path.stat().st_size for path in paths)
    for path in paths:
        path.unlink()
    if paths:
        atomic_json(
            c.out / "garl/CACHE_RELEASE_FOR_BUNDLE.json",
            {
                "root": str(root),
                "released_files": len(paths),
                "released_bytes": size,
                "scientific_checkpoints_removed": 0,
                "raw_removed": 0,
                "cache_reconstructible_from_pinned_TRAIN": True,
            },
        )


def sources(c: Campaign) -> None:
    """Snapshot our changes and only the base kernels needed for included head replay."""
    directories = (
        "operational/efficient_context",
        "src/e_jepa_ttc/efficient_context",
        "configs/campaign",
        "docs/efficient_context_handoff_20261004",
    )
    files = [
        ROOT / "tests/unit/test_efficient_context_contracts.py",
        ROOT / "pyproject.toml",
        ROOT / "uv.lock",
        ROOT / "src/e_jepa_ttc/__init__.py",
        ROOT / "src/e_jepa_ttc/simplex_t/__init__.py",
        ROOT / "src/e_jepa_ttc/simplex_t/model.py",
        ROOT / "src/e_jepa_ttc/simplex_t/phase.py",
    ]
    for directory in directories:
        files.extend(
            p for p in (ROOT / directory).rglob("*") if p.is_file() and "__pycache__" not in p.parts
        )
    records = []
    for file in sorted(set(files)):
        relative = file.relative_to(ROOT).as_posix()
        atomic_bytes(c.out / "source" / relative, file.read_bytes())
        records.append({"path": relative, "sha256": digest(file), "bytes": file.stat().st_size})
    external = Path(c.local["garl_code_candidate"])
    native_files = [
        external / "garl_ttc/__init__.py",
        external / "configs/ablation/event_lhr.yaml",
        *sorted((external / "garl_ttc/models").glob("*.py")),
    ]
    for file in native_files:
        if not file.is_file():
            continue
        relative = "external/Garl-TTC/" + file.relative_to(external).as_posix()
        atomic_bytes(c.out / "source" / relative, file.read_bytes())
        records.append({"path": relative, "sha256": digest(file), "bytes": file.stat().st_size})
    diff = subprocess.check_output(
        [
            "git",
            "diff",
            "--binary",
            c.config["base_commit"],
            "--",
            *directories[:3],
            "tests/unit/test_efficient_context_contracts.py",
        ],
        cwd=ROOT,
    )
    atomic_bytes(c.out / "source/OWN_CHANGES.diff", diff)
    atomic_json(
        c.out / "SOURCE_CODE_MANIFEST.json",
        {
            "files": records,
            "base_commit": c.config["base_commit"],
            "raw_training_requires_base_checkout": True,
        },
    )
    atomic_json(
        c.out / "ENVIRONMENT.json",
        {
            "python": sys.version,
            "interpreter": sys.executable,
            "packages": {
                name: importlib.metadata.version(name)
                for name in (
                    "torch",
                    "numpy",
                    "pandas",
                    "pyarrow",
                    "psutil",
                    "pytest",
                    "ruff",
                    "pyright",
                )
            },
            "global_environment_modified": False,
            "numeric_threads": 4,
            "interop_threads": 2,
        },
    )


def package(c: Campaign) -> None:
    """Deliver measured and blocked branches without converting dependencies into negatives."""
    from .budget import require
    from .delivery import report
    from .export_prepared import export

    release_cache_for_bundle(c)
    require(c)
    report(c)
    export(c)
    sources(c)
    atomic_bytes(
        c.out / "REPLAY_ANALYSIS.txt",
        (
            b"From the extracted bundle root, using the recorded compatible environment:\n"
            b"python -P source/operational/efficient_context/replay_bundle.py --root .\n"
            b"-P prevents the adjacent queue.py from shadowing the Python standard library.\n"
            b"Replays included analysis/head outputs; excludes raw training and latency.\n"
        ),
    )
    excluded = {
        "BUNDLE_MANIFEST.json",
        "BUNDLE.sha256",
        "BUNDLE_VERIFICATION.json",
        "RESOURCES.json",
    }
    entries = []
    for path in sorted(c.out.rglob("*")):
        relative = path.relative_to(c.out)
        if (
            not path.is_file()
            or path.suffix in (".zip", ".pending", ".lock", ".pyc")
            or path.name in excluded
            or relative.as_posix() == "data_recovery/PROGRESS.json"
            or "__pycache__" in relative.parts
            or relative.parts[0] == "verification"
            or (relative.parts[0] == "garl" and "native_cache" in relative.parts)
        ):
            continue
        entries.append(
            {"path": relative.as_posix(), "bytes": path.stat().st_size, "sha256": digest(path)}
        )
    payload_size = sum(row["bytes"] for row in entries)
    owned_size = sum(p.stat().st_size for p in c.out.rglob("*") if p.is_file())
    if owned_size + payload_size > 10_000_000_000:
        raise InterruptedError("bundle reservation would exceed the owned10GB quota")
    atomic_json(c.out / "BUNDLE_MANIFEST.json", {"files": entries})
    target = c.out / "E_JEPA_TTC_EFFICIENT_CONTEXT_ESSENTIAL.zip"
    pending = target.with_suffix(".pending")
    with zipfile.ZipFile(pending, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for row in entries:
            z.write(c.out / row["path"], row["path"])
        z.write(c.out / "BUNDLE_MANIFEST.json", "BUNDLE_MANIFEST.json")
    extraction = c.out / "verification/extracted"
    extracted = []
    with zipfile.ZipFile(pending) as z:
        if z.testzip() is not None:
            raise ValueError("bundle CRC failed")
        for row in entries:
            check = hashlib.sha256()
            with z.open(row["path"]) as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    check.update(chunk)
            if check.hexdigest() != row["sha256"]:
                raise ValueError("bundle manifest mismatch: " + row["path"])
            relative = Path(row["path"])
            needed = (
                relative.parts[0] in ("analysis", "analytical", "prepared_heads", "source")
                or relative.parts[0] == "garl_heads"
                and "analysis" in relative.parts
                or row["path"]
                in (
                    "EWMA_TRANSPORT_CV_RESULTS.json",
                    "H8_WIDE_RESULTS.json",
                    "H8_WIDE_RESULTS_seed13.json",
                    "H8_WIDE_RESULTS_seed23.json",
                    "WIDE_REPLICATION_RESULTS.json",
                    "GARL_CONTEXT_RESULTS.json",
                )
            )
            if needed:
                path = (extraction / relative).resolve()
                if extraction.resolve() not in path.parents:
                    raise ValueError("bundle extraction path escapes the owned verification root")
                path.parent.mkdir(parents=True, exist_ok=True)
                with z.open(row["path"]) as stream, path.open("wb") as output:
                    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                        output.write(chunk)
                extracted.append(row["path"])
    script = extraction / "source/operational/efficient_context/replay_bundle.py"
    command = [sys.executable, "-P", str(script), "--root", str(extraction)]
    env = dict(os.environ, PYTHONUTF8="1")
    for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        env[name] = "4"
    env.pop("PYTHONPATH", None)
    result = subprocess.run(command, cwd=extraction, env=env, capture_output=True)
    atomic_bytes(c.out / "verification/REPLAY_STDOUT.json", result.stdout)
    atomic_bytes(c.out / "verification/REPLAY_STDERR.txt", result.stderr)
    if result.returncode:
        raise ValueError("independent extracted replay failed; preserved verification logs")
    replay = json.loads(result.stdout)
    if replay["status"] != "PASS":
        raise ValueError("independent numerical regeneration did not pass")
    pending.replace(target)
    sha = digest(target)
    atomic_bytes(c.out / "BUNDLE.sha256", (sha + "  " + target.name + "\n").encode())
    atomic_json(
        c.out / "BUNDLE_VERIFICATION.json",
        {
            "status": "PASS",
            "CRC": "PASS",
            "manifest_files_verified": len(entries),
            "bundle_sha256": sha,
            "bundle_bytes": target.stat().st_size,
            "independently_extracted_files": len(extracted),
            "replay": replay,
            "replay_command": command,
            "extra_optimizer_updates": 0,
            "raw_training_replay": False,
            "full_checkpoint_archives_sha256_verified": True,
        },
    )
    require(c)
    print("BUNDLE_VERIFIED", sha, flush=True)

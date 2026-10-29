"""Build or materialize a compact SHA-addressed delta over immutable council_v1.

The materialize/verify commands use only stdlib and also work from a copied
materialize.py inside a delta. Build runs no sensors, models or GPU.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
ROUND = "train40_runtime_council_20261028"


def digest(raw: bytes) -> str:
    """Hash exact stored bytes."""
    return hashlib.sha256(raw).hexdigest()


def safe_child(root: Path, relative: str) -> Path:
    """Reject absolute, traversal and symlink escapes in untrusted manifests."""
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"Unsafe manifest path: {relative}")
    target = (root / path).resolve()
    if not target.is_relative_to(root.resolve()):
        raise ValueError(f"Manifest path escapes root: {relative}")
    return target


def verified_bytes(root: Path, row: dict[str, Any], path_key: str = "path") -> bytes:
    """Read one manifest entry only if hash and size match."""
    raw = safe_child(root, row[path_key]).read_bytes()
    if digest(raw) != row["sha256"] or len(raw) != row["bytes"]:
        raise ValueError(f"Evidence mismatch: {row[path_key]}")
    return raw


def read_base(base: Path) -> tuple[dict[str, Any], str]:
    """Validate every immutable base entry without editing the base tree."""
    raw = (base / "MANIFEST.json").read_bytes()
    manifest = json.loads(raw)
    for row in manifest["files"]:
        verified_bytes(base, row)
    return manifest, digest(raw)


def materialize(delta: Path, base: Path, output: Path | None) -> dict[str, Any]:
    """Verify base and delta, optionally reconstruct the declared full tree in a new path."""
    manifest = json.loads((delta / "MANIFEST.json").read_bytes())
    _, base_sha = read_base(base)
    if base_sha != manifest["base_manifest_sha256"]:
        raise ValueError("The base manifest does not match this delta")
    if output is not None:
        output = output.resolve()
        if (
            output.exists()
            or output.is_relative_to(base.resolve())
            or output.is_relative_to(delta.resolve())
        ):
            raise ValueError("Materialization requires a new path outside base and delta")
    seen: set[str] = set()
    for row in manifest["files"]:
        if row["path"] in seen:
            raise ValueError("Duplicate output path in delta manifest")
        seen.add(row["path"])
        origin = base if row["storage"] == "base" else delta
        if row["storage"] not in ("base", "delta"):
            raise ValueError("Unknown evidence storage")
        raw = verified_bytes(origin, row, "stored_path")
        if output is not None:
            target = safe_child(output, row["path"])
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(raw)
    return {"verified": True, "files": len(seen), "materialized": str(output) if output else None}


def build(base: Path, output: Path) -> dict[str, Any]:
    """Snapshot current compact evidence, referencing base blobs instead of duplicating them."""
    sys.path.insert(0, str(ROOT))
    from docs.train40_runtime_council_20261028.archive_council import checked_copy, recover_sources
    from scripts.report_train40_resume import copy_evidence, generate

    base, output = base.resolve(), output.resolve()
    if output.exists() or not output.is_relative_to(ROOT / "docs" / ROUND / "evidence"):
        raise ValueError("Build output must be new under the council evidence directory")
    base_manifest, base_sha = read_base(base)
    base_by_sha = {row["sha256"]: row for row in base_manifest["files"]}
    source, docs = ROOT / "artifacts" / ROUND, ROOT / "docs" / ROUND
    generated = generate(source, docs / "RESULTS_GENERATED.md")
    scratch_root = (ROOT / "tmp").resolve()
    scratch_root.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="council_delta_", dir=scratch_root) as temporary:
        staging = Path(temporary).resolve()
        if not staging.is_relative_to(scratch_root):
            raise ValueError("Temporary archive escaped the dedicated scratch directory")
        assembled = staging / "assembled"
        artifacts = assembled / "artifacts" / ROUND
        artifact_manifest = copy_evidence(
            source,
            artifacts,
            exclude_top_level=("hash_probe_dependencies", "crop_cache_integration_patch_v1"),
        )
        # Root QA uses .log; retain these small text logs beside their hash receipts.
        for path in sorted(source.rglob("*.log")):
            relative = path.relative_to(source)
            if not any(part == "qa" or part.endswith("_qa_v1") for part in relative.parts):
                continue
            checked_copy(path, artifacts / relative)
            raw = path.read_bytes()
            artifact_manifest["files"].append(
                {
                    "source": relative.as_posix(),
                    "origin": "artifact",
                    "published": relative.as_posix(),
                    "sha256": digest(raw),
                    "bytes": len(raw),
                }
            )
        recover_sources(artifacts, artifact_manifest)
        if artifact_manifest["unavailable_executed_source_snapshots"]:
            raise ValueError("Historical executed source snapshots remain unavailable")
        for path in sorted(docs.iterdir()):
            if path.is_file() and path.suffix in {".md", ".json", ".py"}:
                checked_copy(path, assembled / "docs" / ROUND / path.name)
        code = list((ROOT / "operational/train40_runtime").glob("*.py"))
        code += list((ROOT / "tests/unit").glob("test_train40*.py"))
        code += [
            ROOT / "scripts" / name
            for name in (
                "report_train40_resume.py",
                "benchmark_train40_paced.py",
                "benchmark_train40_multiobject.py",
                "benchmark_train40_subpacket_backend.py",
            )
            if (ROOT / "scripts" / name).exists()
        ]
        for path in code:
            checked_copy(path, assembled / path.relative_to(ROOT))
        subprocess.run(
            [
                sys.executable,
                "-I",
                "-B",
                "scripts/report_train40_resume.py",
                "--source",
                f"artifacts/{ROUND}",
                "--output",
                f"docs/{ROUND}/RESULTS_GENERATED.md",
            ],
            cwd=assembled,
            check=True,
            capture_output=True,
            text=True,
        )
        if (
            digest((assembled / "docs" / ROUND / "RESULTS_GENERATED.md").read_bytes())
            != generated["report_sha256"]
        ):
            raise ValueError("Portable report differs from native receipts")
        output.mkdir(parents=True, exist_ok=False)
        entries = []
        delta_blobs: set[str] = set()
        for path in sorted(assembled.rglob("*")):
            if not path.is_file():
                continue
            raw = path.read_bytes()
            raw.decode("utf-8-sig")
            if path.suffix in {".pt", ".pth", ".npy", ".npz", ".h5", ".hdf5", ".whl", ".pyc"}:
                raise ValueError(f"Forbidden evidence payload: {path}")
            sha = digest(raw)
            if sha in base_by_sha:
                storage, stored = "base", base_by_sha[sha]["path"]
            else:
                storage, stored = "delta", f"blobs/{sha}.txt"
                if sha not in delta_blobs:
                    target = output / stored
                    target.parent.mkdir(exist_ok=True)
                    target.write_bytes(raw)
                    delta_blobs.add(sha)
            entries.append(
                {
                    "path": path.relative_to(assembled).as_posix(),
                    "sha256": sha,
                    "bytes": len(raw),
                    "storage": storage,
                    "stored_path": stored,
                }
            )
        manifest = {
            "schema": "train40_council_delta_v1",
            "created_utc": datetime.now(UTC).isoformat(),
            "base_name": base.name,
            "base_manifest_sha256": base_sha,
            "goal_admitted": False,
            "closure_requested": True,
            "pause_after_publication_owned_by_root": True,
            "files": entries,
            "logical_bytes": sum(row["bytes"] for row in entries),
            "referenced_base_files": sum(row["storage"] == "base" for row in entries),
            "stored_unique_blobs": len(delta_blobs),
            "stored_blob_bytes": sum(
                (output / f"blobs/{sha}.txt").stat().st_size for sha in delta_blobs
            ),
            "portable_report_sha256": generated["report_sha256"],
            "portable_report_regenerated_exactly": True,
            "unavailable_executed_sources": [],
            "raw_weights_dependencies_included": False,
        }
        (output / "MANIFEST.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        shutil.copyfile(Path(__file__), output / "materialize.py")
        (output / "README.md").write_text(
            "# Council final delta\n\n"
            "Requiere el directorio hermano council_v1 intacto. No duplica sus blobs; "
            "MANIFEST.json vincula cada ruta final a bytes exactos de la base o del delta.\n\n"
            "Desde este directorio, con Python stdlib:\n\n```text\n"
            "python materialize.py verify --delta . --base ../council_v1\n"
            "python materialize.py materialize --delta . --base ../council_v1 "
            "--output ../council_final_materialized\n"
            "```\n\nEn el árbol reconstruido, leer docs/" + ROUND + "/HANDOFF.md. "
            "El informe se regenera con scripts/report_train40_resume.py usando "
            "--source artifacts/" + ROUND + " --output docs/" + ROUND + "/RESULTS_GENERATED.md. "
            "No se incluyen RAW, pesos ni tensores .pt. "
            "El objetivo de 50 Hz/<20 ms no está admitido.\n",
            encoding="utf-8",
        )
        materialize(output, base, None)
        if digest((base / "MANIFEST.json").read_bytes()) != base_sha:
            raise ValueError("Immutable base manifest changed")
    return {
        key: manifest[key]
        for key in (
            "stored_unique_blobs",
            "stored_blob_bytes",
            "logical_bytes",
            "referenced_base_files",
            "portable_report_sha256",
        )
    }


def main() -> None:
    """CLI for a new delta build, full verification or safe materialization."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("build", "verify", "materialize"))
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--delta", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.action == "build":
        if args.output is None:
            parser.error("build requires --output")
        result = build(args.base, args.output)
    else:
        if args.delta is None or (args.action == "materialize" and args.output is None):
            parser.error("verify requires --delta; materialize also requires --output")
        result = materialize(
            args.delta, args.base, args.output if args.action == "materialize" else None
        )
    print(json.dumps(result))


if __name__ == "__main__":
    main()

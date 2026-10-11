"""Build or verify a text-source handoff ZIP from committed V12/V13 Git objects."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import subprocess
import tarfile
import zipfile
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[2]
DIRECTORIES = {"src", "operational", "tests", "configs", "docs", "scripts"}
ROOT_FILES = {
    "README.md",
    "AGENTS.md",
    "LICENSE",
    "CITATION.cff",
    "CONTRIBUTING.md",
    "SECURITY.md",
    "Makefile",
    "pyproject.toml",
    "uv.lock",
    ".python-version",
    ".gitattributes",
    ".gitignore",
}
SUFFIXES = {
    ".py",
    ".md",
    ".json",
    ".jsonl",
    ".csv",
    ".tsv",
    ".txt",
    ".toml",
    ".yaml",
    ".yml",
    ".lock",
    ".ps1",
    ".sh",
    ".xml",
    ".bib",
    ".cff",
    ".rst",
}


def digest(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def verify(path: Path) -> dict:
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)) or archive.testzip() is not None:
            raise ValueError("Duplicate entries or CRC failure")
        manifest = json.loads(archive.read("BUNDLE_MANIFEST.json"))
        expected = {r["path"] for r in manifest["files"]} | {"BUNDLE_MANIFEST.json"}
        if set(names) != expected:
            raise ValueError("ZIP inventory differs from manifest")
        for entry in manifest["files"]:
            p = PurePosixPath(entry["path"])
            if p.is_absolute() or ".." in p.parts:
                raise ValueError("Unsafe archive path")
            content = archive.read(entry["path"])
            if len(content) != entry["bytes"] or digest(content) != entry["sha256"]:
                raise ValueError(f"Hash mismatch: {entry['path']}")
    return {
        "status": "PASSED",
        "files_verified": len(manifest["files"]),
        "commits": manifest["commits"],
        "zip_bytes": path.stat().st_size,
        "zip_sha256": digest(path.read_bytes()),
    }


def build(v12: Path, output: Path) -> dict:
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite an existing delivery: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest: dict = {
        "schema": "ejepa_text_handoff_v1",
        "commits": {},
        "files": [],
        "excluded": [
            "datasets",
            "checkpoints",
            "private bindings",
            "untracked files",
            "binary artifacts",
            "Python environments",
            "Git internals",
        ],
    }
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:

        def add(name: str, content: bytes) -> None:
            archive.writestr(name, content)
            manifest["files"].append(
                {"path": name, "bytes": len(content), "sha256": digest(content)}
            )

        for label, repo in (("V12", v12), ("V13", ROOT)):
            head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo).decode().strip()
            manifest["commits"][label] = head
            data = subprocess.check_output(["git", "archive", "--format=tar", head], cwd=repo)
            with tarfile.open(fileobj=io.BytesIO(data)) as tree:
                for member in tree:
                    p = PurePosixPath(member.name)
                    accepted = p.as_posix() in ROOT_FILES or (
                        p.parts[0] in DIRECTORIES
                        and (p.suffix.lower() in SUFFIXES or p.name == ".gitattributes")
                    )
                    if not member.isfile() or not accepted:
                        continue
                    stream = tree.extractfile(member)
                    if stream is None:
                        raise ValueError(member.name)
                    content = stream.read()
                    if b"\0" in content:
                        raise ValueError(f"Unexpected binary content: {member.name}")
                    add(label + "/" + member.name, content)
        add(
            "START_HERE.md",
            (
                "# E-JEPA V12/V13 — handoff 2026-10-12\n\n"
                "Empieza por V13/docs/agent_handoff_20261012/README.md.\n\n"
                "Informes, código y evidencia textual de dos revisiones Git separadas. "
                "Los resultados son exploratorios; no acreditan SOTA.\n\n"
                "Desde V13: python docs/agent_handoff_20261012/replay.py --output RUTA_SALIDA\n\n"
                "Desde V13: python docs/agent_handoff_20261012/build_bundle.py "
                "--verify RUTA_ZIP\n\n"
                "Datos crudos, pesos y binarios no están incluidos. Las rutas locales de los "
                "recibos son históricas. El manifiesto contiene los commits y hashes exactos.\n"
            ).encode(),
        )
        archive.writestr(
            "BUNDLE_MANIFEST.json", json.dumps(manifest, indent=2, ensure_ascii=False) + "\n"
        )
    receipt = verify(output)
    output.with_suffix(".sha256").write_text(
        receipt["zip_sha256"] + "  " + output.name + "\n", encoding="utf-8"
    )
    output.with_suffix(".receipt.json").write_text(
        json.dumps(receipt, indent=2) + "\n", encoding="utf-8"
    )
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v12", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--verify", type=Path)
    args = parser.parse_args()
    if args.verify:
        result = verify(args.verify)
    elif args.v12 and args.output:
        result = build(args.v12.resolve(), args.output.resolve())
    else:
        parser.error("Use --verify ZIP or --v12 REPOSITORY --output ZIP")
    print(json.dumps(result, indent=2))

"""Build a new portable, hash-verified council evidence tree without sensors or weights."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.report_train40_resume import copy_evidence, generate  # noqa: E402

ROUND = "train40_runtime_council_20261028"
EXCLUDED = (
    "qa",
    "hash_probe_dependencies",
    "subpacket_proposal",
    "crop_cache_integration_patch_v1",
)


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def checked_copy(source: Path, target: Path) -> None:
    """Copy exact UTF-8 bytes, failing on concurrent changes or a noncompact file."""
    data = source.read_bytes()
    if len(data) > 8_000_000:
        raise ValueError(f"Noncompact source: {source}")
    data.decode("utf-8-sig")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data)
    if source.read_bytes() != data or target.read_bytes() != data:
        raise ValueError(f"Source changed while archiving: {source}")


def recover_sources(artifact_root: Path, manifest: dict[str, Any]) -> None:
    """Recover older executed bytes only when their receipt SHA matches a Git blob."""
    by_hash = {
        row["sha256"]: row["published"]
        for row in manifest["files"]
        if row["published"].endswith(".py.txt")
    }
    unresolved, resolved = [], []
    for item in manifest["unavailable_executed_source_snapshots"]:
        name, expected = item["source"], item["expected_sha256"]
        if expected in by_hash:
            resolved.append(
                {**item, "published": by_hash[expected], "method": "existing_exact_snapshot"}
            )
            continue
        path = Path(name)
        path = path if path.is_absolute() else ROOT / path
        try:
            relative = path.resolve().relative_to(ROOT).as_posix()
        except ValueError:
            unresolved.append(item)
            continue
        history = subprocess.run(
            ["git", "log", "-40", "--format=%H", "--", relative],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        restored = False
        for revision in history.stdout.splitlines():
            blob = subprocess.run(
                ["git", "show", f"{revision}:{relative}"],
                cwd=ROOT,
                capture_output=True,
                check=False,
            )
            if blob.returncode or digest(blob.stdout) != expected:
                continue
            blob.stdout.decode("utf-8-sig")
            destination = Path("source_snapshots") / expected / (relative + ".txt")
            target = artifact_root / destination
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(blob.stdout)
            if digest(target.read_bytes()) != expected:
                raise ValueError("Recovered source changed while writing")
            manifest["files"].append(
                {
                    "source": name,
                    "origin": "exact_git_source_snapshot",
                    "published": destination.as_posix(),
                    "sha256": expected,
                    "bytes": len(blob.stdout),
                    "git_commit": revision,
                }
            )
            resolved.append(
                {
                    **item,
                    "published": destination.as_posix(),
                    "method": "exact_git_blob",
                    "git_commit": revision,
                }
            )
            by_hash[expected] = destination.as_posix()
            restored = True
            break
        if not restored:
            unresolved.append(item)
    manifest["unavailable_executed_source_snapshots"] = unresolved
    manifest["resolved_executed_source_snapshots"] = resolved
    (artifact_root / "MANIFEST.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )


def main() -> None:
    """Keep original receipts immutable and verify report regeneration from the new tree."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    destination = args.output.resolve()
    if destination.exists() or not destination.is_relative_to(ROOT / "docs" / ROUND / "evidence"):
        raise ValueError("Output must be new under this round's docs/evidence directory")
    source = ROOT / "artifacts" / ROUND
    docs = ROOT / "docs" / ROUND
    report_manifest = generate(source, docs / "RESULTS_GENERATED.md")
    destination.mkdir(parents=True, exist_ok=False)
    artifact_root = destination / "artifacts" / ROUND
    artifact_manifest = copy_evidence(source, artifact_root, exclude_top_level=EXCLUDED)
    recover_sources(artifact_root, artifact_manifest)
    for path in sorted(docs.iterdir()):
        if path.is_file() and path.suffix in {".md", ".json", ".py"}:
            checked_copy(path, destination / "docs" / ROUND / path.name)
    code_paths = [
        "scripts/report_train40_resume.py",
        "operational/train40_runtime/paced_statistics.py",
        "operational/train40_runtime/__init__.py",
        "tests/unit/test_train40_runtime_paced.py",
        "tests/unit/test_train40_council_report.py",
    ]
    for relative in code_paths:
        checked_copy(ROOT / relative, destination / relative)
    command = [
        sys.executable,
        "-I",
        "-B",
        "scripts/report_train40_resume.py",
        "--source",
        f"artifacts/{ROUND}",
        "--output",
        f"docs/{ROUND}/RESULTS_GENERATED.md",
    ]
    result = subprocess.run(command, cwd=destination, capture_output=True, text=True, check=True)
    portable_report = destination / "docs" / ROUND / "RESULTS_GENERATED.md"
    if digest(portable_report.read_bytes()) != report_manifest["report_sha256"]:
        raise ValueError("Portable regeneration differs from the native receipt-derived report")
    generators = [
        [f"artifacts/{ROUND}/frame_sealing_cpu/write_findings.py.txt"],
        [f"artifacts/{ROUND}/rgb_exact_gpu/write_findings.py.txt"],
        [f"docs/{ROUND}/build_linspace_attribution.py"],
    ]
    for argv in generators:
        subprocess.run(
            [sys.executable, "-I", "-B", *argv],
            cwd=destination,
            capture_output=True,
            text=True,
            check=True,
        )
    for name in (
        "FRAME_SEALING_CPU_GENERATED.md",
        "FRAME_SEALING_CPU_GENERATED.json",
        "RGB_GPU_FINDINGS.md",
        "LINSPACE_ATTRIBUTION.md",
    ):
        if (destination / "docs" / ROUND / name).read_bytes() != (docs / name).read_bytes():
            raise ValueError(f"Portable scientific note differs: {name}")
    report_command_text = (
        f"python scripts/report_train40_resume.py --source artifacts/{ROUND} "
        f"--output docs/{ROUND}/RESULTS_GENERATED.md"
    )
    readme = f"""# Council v1 — snapshot de investigación activa

Resultados y alcance: [COUNCIL](docs/{ROUND}/COUNCIL.md) y
[tablas regeneradas](docs/{ROUND}/RESULTS_GENERATED.md).

Este paquete conserva recibos positivos y negativos, fuentes exactas disponibles,
documentación y generadores. No contiene RAW, pesos, dependencias, caches ni .pt.
Los diagnósticos cuyos recibos referencian tensores .pt necesitan esos artefactos
externos para volver a ejecutarse; aquí se pueden regenerar sus tablas JSON.
No es un cierre ni una admisión de 50 Hz/<20 ms. El objetivo permanece activo.

Desde la raíz del paquete, con Python (sólo stdlib para el informe):

```text
{report_command_text}
python artifacts/{ROUND}/frame_sealing_cpu/write_findings.py.txt
python artifacts/{ROUND}/rgb_exact_gpu/write_findings.py.txt
python docs/{ROUND}/build_linspace_attribution.py
```

Los archivos .py.txt son snapshots legibles que Python puede ejecutar con esa
extensión. Las rutas absolutas en recibos se conservan como procedencia histórica;
los generadores anteriores leen los recibos locales relativos. La reproducción de
GPU/datasets requiere el checkout y sus datos/pesos externos verificados.

MANIFEST.json vincula todos los bytes del paquete excepto el propio manifiesto.
El manifiesto de artifacts registra recuperaciones exactas desde Git y cualquier
fuente histórica no recuperada; nunca se sustituye por una versión distinta.
QA/propuestas en curso y dependencias de descarga quedaron fuera de esta ronda.
"""
    (destination / "README.md").write_text(readme, encoding="utf-8")
    inventory = []
    forbidden = {".pt", ".pth", ".npz", ".npy", ".h5", ".hdf5", ".whl", ".dll", ".pyd", ".pyc"}
    for path in sorted(destination.rglob("*")):
        if path.is_file():
            if path.suffix.lower() in forbidden:
                raise ValueError(f"Forbidden payload in portable evidence: {path}")
            raw = path.read_bytes()
            raw.decode("utf-8-sig")
            inventory.append(
                {
                    "path": path.relative_to(destination).as_posix(),
                    "sha256": digest(raw),
                    "bytes": len(raw),
                }
            )
    manifest = {
        "schema": "train40_council_portable_evidence_v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "goal_active": True,
        "goal_admitted": False,
        "round": "council_v1",
        "files": inventory,
        "file_count": len(inventory),
        "total_bytes": sum(row["bytes"] for row in inventory),
        "report_regenerated_byte_exact": True,
        "scientific_notes_regenerated_byte_exact": True,
        "regeneration_command": command,
        "regeneration_stdout": result.stdout.strip(),
        "unavailable_executed_sources": artifact_manifest["unavailable_executed_source_snapshots"],
        "excluded_top_level_artifacts": list(EXCLUDED),
        "raw_weights_pt_dependencies_included": False,
    }
    (destination / "MANIFEST.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "output": str(destination),
                "files": len(inventory),
                "bytes": manifest["total_bytes"],
                "unavailable_sources": len(manifest["unavailable_executed_sources"]),
            }
        )
    )


if __name__ == "__main__":
    main()

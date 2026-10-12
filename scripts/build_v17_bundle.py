"""Package a closed V17 campaign, immutable endpoints and exact dependency inventory."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import zipfile
from datetime import UTC, datetime
from pathlib import Path

from operational.range_transfer.common import read, sha, write
from operational.representation_audit.run import OLD, OWNER, ROOT, RUN


def dependencies() -> list[dict]:
    paths = {OLD / "ADMISSION.json", OLD / "PRODUCER_METRICS_P_H.json"}
    paths.add(
        ROOT.parent
        / "e-jepa-ttc-v14-range-transfer/artifacts/range_transfer_20261010"
        / "evaluation/screen/RT-M0-L0-S7/METRICS.json"
    )
    for role in ("P", "H", "V"):
        manifest = OWNER / f"{role}_MANIFEST.json"
        paths.add(manifest)
        paths.add(Path(read(manifest)["rows_path"]))
        paths.update((OLD / "vector_cache" / role).glob("*.json"))
        paths.update((OLD / "vector_cache" / role).glob("*.npz"))
    for arm in ("E_A5_MATCHED", "E_C2F_MATCHED", "PAIR_E_MATCHED"):
        paths.add(OWNER / f"fits/{arm}/checkpoint_last.pt")
        paths.add(OWNER / f"fits/{arm}/CHECKPOINT_RECEIPT.json")
        paths.add(OWNER / f"fits/{arm}/RUN_PROVENANCE.json")
    paths.add(OWNER / "SPLIT_MANIFEST.json")
    paths.add(OWNER / "features/H/E_NORMALIZER.json")
    paths.add(OWNER / "rounds/PRODUCERS_PAIR_ENDPOINT_FREEZE.json")
    for role in ("H", "V"):
        paths.add(OWNER / f"features/{role}/E_HISTORY.npz")
        paths.add(OWNER / f"features/{role}/E_HISTORY.json")
        paths.update((OWNER / f"features/{role}/E_RAW").glob("*.npz"))
        paths.update((OWNER / f"features/{role}/E_RAW").glob("*.json"))
    result = []
    for path in sorted(paths):
        if not path.is_file():
            raise FileNotFoundError(f"Exact dependency missing: {path}")
        result.append(
            {
                "path": str(path.resolve()),
                "bytes": path.stat().st_size,
                "sha256": sha(path),
                "included": False,
            }
        )
    # Large new spatial cache shards can be rebuilt from raw events; preserve exact hashes.
    for path in sorted((RUN / "spatial_cache").glob("*/part_*.npz")):
        result.append(
            {
                "path": str(path.resolve()),
                "bytes": path.stat().st_size,
                "sha256": sha(path),
                "included": False,
            }
        )
    return result


def build(*, allow_paused: bool = False) -> None:
    status = read(RUN / "CAMPAIGN_NEXT_DECISION.json")
    paused = status["status"] == "RESOURCE_PAUSED"
    if status["status"] != "NO_CANDIDATE" and not (allow_paused and paused):
        raise ValueError("This closed-negative bundle requires all registered queues finished")
    if (RUN / "PROCESS.lock").exists():
        raise ValueError("A worker lock remains; do not package a moving campaign")
    dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT).decode().strip()
    if dirty:
        raise ValueError("Commit the reviewed source/report changes locally before packaging")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT).decode().strip()
    dependency_rows = dependencies()
    write(
        RUN / "BUNDLE_DEPENDENCIES.json",
        {
            "files": dependency_rows,
            "raw_event_source_root": read(OWNER / "fits/E_A5_MATCHED/RUN_PROVENANCE.json")[
                "source_identity"
            ]["event_source_root"],
            "scope": "readout/head reproduction with frozen caches; raw data are external",
        },
    )
    label = "PROGRESS" if paused else "RESULTS"
    destination = (
        Path.home() / "Downloads" / f"E_JEPA_TTC_V17_RESEARCH_{label}_20261012_{commit[:8]}.zip"
    )
    if destination.exists():
        raise FileExistsError("Never overwrite a delivered bundle")
    tracked = (
        subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT).decode("utf-8").split("\0")
    )
    files: dict[str, Path] = {}
    for relative in tracked:
        if relative:
            path = ROOT / relative
            if path.is_file():
                if not path.resolve().is_relative_to(ROOT.resolve()):
                    raise ValueError("Tracked path escapes worktree")
                files["source/" + Path(relative).as_posix()] = path
    durable = {}
    for pointer in RUN.glob("*/**/CHECKPOINT_POINTER.json"):
        durable[pointer.parent] = read(pointer)["version"]
    for path in sorted(RUN.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(RUN)
        if any(part in ("__pycache__", ".pytest_cache", "cache") for part in rel.parts):
            continue
        if "spatial_cache" in rel.parts and path.suffix == ".npz":
            continue
        if path.suffix in (".lock", ".tmp", ".pyc") or path.name == "PAUSE":
            continue
        if "checkpoint_versions" in rel.parts:
            if path.name != durable.get(path.parent.parent):
                continue
        files["campaign/" + rel.as_posix()] = path
    scope = {
        "status": "IN_PROGRESS_RESOURCE_PAUSE" if paused else "CLOSED_NO_CANDIDATE",
        "scientifically_closed": not paused,
        "source_commit": commit,
        "created_utc": datetime.now(UTC).isoformat(),
        "includes": [
            "tracked source and reports",
            "all new training checkpoints; receipt distinguishes complete and pending fits",
            "durable checkpoint versions",
            "portable inference weights",
            "predictions including invalid rows",
            "curves and gradients",
            "group metrics and bootstrap",
            "QA and accounting",
            "exact dependency inventory",
        ],
        "excludes": [
            "raw datasets",
            "V13 parent checkpoint files",
            "V15 frozen vector caches",
            "large V17 feature caches",
            "Git object database and credentials",
        ],
        "reproducibility": (
            "metrics can be replayed from predictions; "
            "complete fits require listed external caches; "
            "portable exports load without parent files but receive prepared inputs"
        ),
        "SOTA": False,
    }
    write(RUN / "BUNDLE_SCOPE.json", scope)
    files["campaign/BUNDLE_SCOPE.json"] = RUN / "BUNDLE_SCOPE.json"
    readme = RUN / "BUNDLE_README.md"
    readme.write_text(
        "# V17: bundle verificable\n\n"
        + (
            "**Campaña pendiente por recursos; no es un cierre científico.**\n\n"
            if paused
            else "**Campaña cerrada sin candidato admisible.**\n\n"
        )
        + f"Código local: `{commit}`. No hay afirmación SOTA.\n\n"
        "- [Informes y protocolos](source/docs/representation_audit_20261012/README.md).\n"
        "- [Estado y presupuesto](campaign/CAMPAIGN_NEXT_DECISION.json).\n"
        "- [Scope y exclusiones](campaign/BUNDLE_SCOPE.json).\n"
        "- [Dependencias exactas de cachés](campaign/BUNDLE_DEPENDENCIES.json).\n"
        "- [Portabilidad y comandos](source/docs/representation_audit_20261012/PORTABILITY.md).\n\n"
        "`source/` contiene código y documentación; `campaign/` contiene evidencia, pesos, "
        "curvas, predicciones y estados durables. El manifiesto verifica cada miembro. "
        "Las tablas de controles reutilizados no son observaciones independientes.\n\n"
        "Los pesos portables reproducen ejemplos preparados incluidos. Los fits completos "
        "requieren las cachés externas listadas; no se incluyen datasets crudos ni todos "
        "los activos de los productores. Los receipts identifican los fits pendientes.\n",
        encoding="utf-8",
    )
    files["README.md"] = readme
    files["campaign/BUNDLE_README.md"] = readme
    manifest = {"scope": scope, "files": []}
    temporary = destination.with_suffix(".building.zip")
    with zipfile.ZipFile(
        temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6
    ) as archive:
        for index, (name, path) in enumerate(sorted(files.items())):
            payload = path.read_bytes()
            digest = hashlib.sha256(payload).hexdigest()
            archive.writestr(name, payload)
            manifest["files"].append({"path": name, "bytes": len(payload), "sha256": digest})
            if index % 500 == 0:
                print("packed", index, len(files), flush=True)
        archive.writestr(
            "MANIFEST.json", json.dumps(manifest, sort_keys=True, ensure_ascii=False, indent=2)
        )
    temporary.replace(destination)
    digest = sha(destination)
    destination.with_suffix(destination.suffix + ".sha256").write_text(
        f"{digest}  {destination.name}\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "path": str(destination),
                "sha256": digest,
                "bytes": destination.stat().st_size,
                "files": len(files),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-paused", action="store_true")
    build(allow_paused=parser.parse_args().allow_paused)

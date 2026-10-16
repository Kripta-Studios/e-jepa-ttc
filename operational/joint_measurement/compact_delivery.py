"""Deliver complete result tables and portable examples under a strict 100 MB cap."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
from pathlib import Path

from operational.range_transfer.common import read, utc, write

from .bundle import require_quiescent, verify_archive, write_archive
from .common import ROOT, RUN

LIMIT = 100_000_000
TEXT = {".json", ".csv", ".md", ".txt", ".log", ".yaml", ".yml"}


def collect() -> tuple[dict[str, Path], list[dict]]:
    """Retain positive and negative evidence; omit raw caches and optimizer payloads."""
    tracked = (
        subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT).decode("utf-8").split("\0")
    )
    files = {
        "source/" + name: ROOT / name
        for name in tracked
        if name and not name.startswith(("artifacts/", "data/cache/"))
    }
    omitted = []
    for path in sorted(RUN.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(RUN)
        ephemeral = (
            "__pycache__" in relative.parts
            or path.suffix in {".lock", ".pyc", ".pending"}
            or ".pending" in path.name
        )
        portable = "exports" in relative.parts and path.suffix in {".pt", ".npz", ".npy"}
        historical_source = "admission_revisions" in relative.parts and path.suffix == ".py"
        table = path.name == "predictions.parquet" or (
            relative.parent == Path("diagnostics/query_panel")
            and path.name in {"P_QUERY_PANEL.parquet", "H_QUERY_PANEL.parquet"}
        )
        include = not ephemeral and (
            path.suffix in TEXT | {".png", ".pdf"} or portable or table or historical_source
        )
        if include:
            files["campaign/" + relative.as_posix()] = path
        else:
            omitted.append(
                {
                    "path": relative.as_posix(),
                    "bytes": path.stat().st_size,
                    "reason": "runtime file" if ephemeral else "nonportable tensor/cache/fragment",
                }
            )
    return files, omitted


def build(output: Path) -> dict:
    """Verify a private candidate before publishing a ZIP of at most 100,000,000 bytes."""
    output = output.resolve()
    if output.suffix.lower() != ".zip":
        raise ValueError("A new .zip destination is required")
    candidate = output.with_suffix(".candidate")
    if any(
        path.exists()
        for path in (output, candidate, candidate.with_suffix(candidate.suffix + ".pending"))
    ):
        raise FileExistsError("Never overwrite a previous delivery or interrupted candidate")
    require_quiescent(ROOT, RUN)
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT):
        raise ValueError("Commit all source and report changes before delivery")
    conclusion = read(RUN / "CAMPAIGN_CONCLUSION.json")
    if not isinstance(conclusion.get("remaining_work"), list):
        raise ValueError("An explicit current conclusion with remaining work is required")
    files, omitted = collect()
    context = {
        "created_utc": utc(),
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT).decode().strip(),
        "conclusion": conclusion,
        "max_zip_bytes": LIMIT,
        "selection": "All metadata, curves, consolidated predictions and portable exports",
        "omitted_files": omitted,
        "optimizer_checkpoint_payloads_included": False,
        "raw_sensor_or_feature_caches_included": False,
        "external_historical_parents_embedded": False,
        "reproduction_scope": (
            "Recompute reported metrics from consolidated predictions and replay included "
            "portable examples. Training and raw-sensor inference require external data, "
            "historical parents and local full checkpoints. Metadata can reference omitted "
            "tensor payloads. Fusion head exports alone do not include their producers."
        ),
    }
    readme = RUN / "delivery/COMPACT_README.md"
    readme.parent.mkdir(parents=True, exist_ok=True)
    reports = [
        "CURRENT_CONCLUSION.md",
        "RESEARCH_STATUS.md",
        "RESULTS_TABLES.md",
        "FUSION_REPLICATION_RESULTS.md",
        "FUSION_RESULTS.md",
        "FUSION_COST_DIAGNOSTIC.md",
        "C1_RESULTS.md",
        "RGB_RESULTS.md",
        "D1_RESULTS.md",
        "B1_RESULTS.md",
        "RESUME_20261016.md",
        "D1_EXECUTION.md",
        "B1_EXECUTION.md",
        "LEARNING_RECIPE_AUDIT.md",
        "QUERY_COMPLEMENTARITY.md",
        "HISTORICAL_BASELINES.md",
    ]
    links = [
        f"- [{name}](source/docs/joint_measurement_20261012/{name})"
        for name in reports
        if (ROOT / "docs/joint_measurement_20261012" / name).exists()
    ]
    readme.write_text(
        "\n".join(
            [
                "# V18: documentación, resultados y exports portables",
                "",
                f"Commit: `{context['commit']}`.",
                "",
                "La conclusión y el trabajo pendiente están en "
                "[CAMPAIGN_CONCLUSION.json](campaign/CAMPAIGN_CONCLUSION.json). "
                "Este paquete no concede una afirmación SOTA ni convierte "
                "desarrollo interno en test.",
                "",
                "## Informes",
                "",
                *links,
                "",
                "## Contenido y límites",
                "",
                "`source/` contiene el código y documentación versionados. `campaign/` conserva "
                "metadatos, curvas, predicciones consolidadas, verificaciones y exports, incluidos "
                "los resultados negativos. Los pesos portables y ejemplos están en los directorios "
                "`exports`; los modelos de cabeza necesitan las features de sus productores.",
                "",
                "No se incluyen sensores crudos, cachés de features, estados de optimizador ni "
                "padres históricos externos. Permanecen locales. El manifiesto enumera las "
                "omisiones; los recibos pueden referenciar esos archivos no incluidos. La "
                "reproducción de entrenamiento requiere esos datos y padres admitidos.",
                "",
                "El ZIP publicado tiene como máximo 100.000.000 bytes. "
                "[BUNDLE_MANIFEST.json](BUNDLE_MANIFEST.json) contiene tamaños y SHA-256 "
                "de cada miembro. El `.sha256` externo verifica el ZIP completo.",
                "",
                "Para verificar la estructura desde la extracción, con las dependencias "
                "del proyecto disponibles: `python source/operational/joint_measurement/bundle.py "
                "verify RUTA_DEL_ZIP`. Es una comprobación de integridad, no un reentrenamiento.",
                "",
                "Para recalcular todas las tablas y ejecutar todos los exports, desde `source/` "
                "y con `PYTHONPATH=src;.` en PowerShell, ejecutar "
                "`python -B -m operational.joint_measurement.replay .. "
                "--output RUTA_EXTERNA.json`. "
                "Guardar el recibo fuera de la extracción para mantener intacto el manifiesto. "
                "Las pruebas que necesitan checkpoints omitidos se enumeran como no reproducidas; "
                "no se confunden con verificaciones aprobadas.",
                "",
                "Ejemplo de replay independiente de una cabeza: "
                "`python source/operational/joint_measurement/replay_fusion_export.py "
                "campaign/fusion/exports/FUSION-TRUE-S7`. Requiere NumPy/PyTorch y usa "
                "los ejemplos empaquetados; no importa el proyecto ni accede a sensores.",
                "",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    files["README.md"] = readme
    output.parent.mkdir(parents=True, exist_ok=True)
    payload_bytes = sum(path.stat().st_size for path in files.values())
    if shutil.disk_usage(output.parent).free < payload_bytes + 10_000_000_000:
        raise RuntimeError("Delivery would violate the 10 GB free disk reserve")
    write_archive(candidate, files, context)
    if candidate.stat().st_size > LIMIT:
        measured = candidate.stat().st_size
        candidate.unlink()
        raise RuntimeError(f"Unpublished candidate removed: {measured} bytes exceeds {LIMIT}")
    verified = verify_archive(candidate)
    os.replace(candidate, output)
    output.with_suffix(".zip.sha256").write_text(
        f"{verified['zip_sha256']}  {output.name}\n", encoding="utf-8"
    )
    write(output.with_suffix(".zip.verification.json"), verified)
    return {key: verified[key] for key in ("status", "zip_bytes", "zip_sha256", "payload_members")}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    arguments = parser.parse_args()
    print(build(arguments.output))

"""Regenerate V15 Markdown, plots and a hash-verified local bundle from evidence."""
# ruff: noqa: E501 -- Markdown paragraphs preserve readable prose in generated reports.

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import zipfile
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "artifacts/range_readout_20261012"
OUT = ROOT / "docs/range_readout_results_20261012"


def read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def table(frame: pd.DataFrame) -> str:
    def cell(value: object) -> str:
        if value is None:
            return "ausente"
        if isinstance(value, float):
            return f"{value:.6g}"
        return str(value).replace("|", "\\|")

    values = [
        list(frame.columns),
        ["---"] * len(frame.columns),
        *frame.itertuples(index=False, name=None),
    ]
    return "\n" + "\n".join("| " + " | ".join(map(cell, row)) + " |" for row in values) + "\n\n"


def plots() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    destination = RUN / "figures"
    destination.mkdir(exist_ok=True)
    historical = RUN / "historical_V14"
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), constrained_layout=True)
    for mass in (0, 1):
        for directory, label in (
            (historical / "fits" / f"RT-M{mass}-L0-S7", "Uniforme V14"),
            (RUN / "fits" / f"S-M{mass}-L0-S7", "Importancia V15"),
        ):
            frame = pd.read_csv(directory / "CURVE.csv")
            axes[0, mass].plot(
                frame["update"], frame.loss.rolling(100, min_periods=1).mean(), label=label
            )
            axes[1, mass].plot(
                frame["update"],
                (frame.gradient_l2_before_clip > 1).rolling(100, min_periods=1).mean(),
                label=label,
            )
        axes[0, mass].set_title(f"M{mass}: mismo riesgo esperado, media móvil 100")
        axes[0, mass].set_ylabel("Estimador de loss")
        axes[1, mass].set_ylabel("Fracción con clipping")
        axes[1, mass].set_xlabel("Updates")
        axes[0, mass].legend()
    fig.savefig(destination / "S_SAMPLING_CURVES.png", dpi=150)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    found = False
    for name in ("R-C05-S7", "R-C20-S7"):
        path = RUN / "fits" / name / "CURVE.csv"
        if not path.exists():
            continue
        found = True
        frame = pd.read_csv(path)
        axes[0].plot(frame["update"], frame.loss.rolling(100, min_periods=1).mean(), label=name)
        axes[1].plot(
            frame["update"],
            (frame.gradient_l2_before_clip > 1).rolling(100, min_periods=1).mean(),
            label=name,
        )
    if found:
        axes[0].set_title("Readout: objetivo nativo P")
        axes[1].set_title("Readout: clipping")
        axes[0].set_ylabel("Loss nativa, media móvil 100")
        axes[1].set_ylabel("Fracción con clipping, media móvil 100")
        axes[1].set_ylim(0, 1.05)
        for ax in axes:
            ax.legend()
            ax.set_xlabel("Updates")
        fig.savefig(destination / "R_READOUT_CURVES.png", dpi=150)
    plt.close(fig)


def generate() -> None:
    snapshot_historical()
    OUT.mkdir(parents=True, exist_ok=True)
    decision = read(RUN / "NEXT_DECISION.json")
    ledger = read(RUN / "ACCOUNTING.json")
    rows = []
    checks = []
    costs = []
    for path in sorted((RUN / "evaluation").glob("*/*/METRICS.json")):
        value = read(path)
        name = path.parent.name
        rows.append(
            {
                "fit": name,
                "stage": path.parent.parent.name,
                **{
                    key: value["bins"][band]["group_macro_mid"]
                    for key, band in (
                        ("MiDc", "crucial"),
                        ("MiDs", "small"),
                        ("MiDl", "large"),
                        ("MiDn", "negative"),
                    )
                },
                "overall_MiD": value["overall_MiD"],
                "mean_MiD": value["mean_MiD"],
                "MAE_s": value["diagnostics"]["mae_s"],
                "late_warning": value["late_warning"]["rate"],
            }
        )
        receipt = read(path.parent / "RECEIPT.json")
        if sha(path.parent / "PREDICTIONS.csv") != receipt["predictions_sha256"]:
            raise ValueError("Prediction hash mismatch")
    for path in sorted((RUN / "fits").glob("*/CHECKPOINT_RECEIPT.json")):
        value = read(path)
        if sha(Path(value["checkpoint_path"])) != value["checkpoint_sha256"]:
            raise ValueError("Checkpoint hash mismatch")
        checks.append(
            {
                "fit": path.parent.name,
                "status": value["status"],
                "updates": value["completed_updates"],
                "sha256": value["checkpoint_sha256"],
            }
        )
        provenance = read(path.parent / "RUN_PROVENANCE.json")
        costs.append(
            {
                "fit": path.parent.name,
                "seconds": provenance.get("elapsed_fit_s", provenance.get("elapsed_s")),
                "scope": provenance["scope"],
            }
        )
    result_frame = pd.DataFrame(rows)
    result_frame.to_csv(RUN / "RESULTS_TABLE.csv", index=False)
    result_table = table(result_frame) if rows else "\nNo hay evaluación V nueva todavía.\n"
    header = f"# V15 RANGE-READOUT\n\nEstado registrado: **{decision['status']}**.\n\n"
    links = """1. [Método y ejecución](01_METODO.md).
2. [Resultados y decisión](02_RESULTADOS.md).
3. [Muestreo, clipping y alcance del productor](03_DIAGNOSTICOS.md).
4. [Pruebas, fallos, recursos y coste](04_PRUEBAS_Y_COSTE.md).
5. [Reproducción y contenido del bundle](05_REPRODUCCION.md).
"""
    (OUT / "README.md").write_text(
        header
        + links
        + "\n"
        + result_table
        + f"Contabilidad: {ledger['scientific']} científicos + {ledger['technical']} técnicos; "
        f"{ledger['recovery_upper']} recuperación, {ledger['pending_upper']} inciertos. "
        "No se reasigna saldo. V14 permanece NO_CANDIDATE y V13 es de sólo lectura.\n",
        encoding="utf-8",
    )
    method = (ROOT / "docs/range_readout_integration_20261012.md").read_text(encoding="utf-8")
    (OUT / "01_METODO.md").write_text(method, encoding="utf-8")
    selection = (
        read(RUN / "SELECTION_FREEZE.json")
        if (RUN / "SELECTION_FREEZE.json").exists()
        else decision
    )
    gate_rows = []
    for family, entry in selection.get("gates", {}).items():
        for contrast in ("paired", "historical_V14_M0"):
            item = entry[contrast]
            gate_rows.append(
                {
                    "family": family,
                    "contrast": contrast,
                    "eligible": item["eligible"],
                    "failed": ", ".join(k for k, v in item["checks"].items() if not v) or "ninguno",
                }
            )
    gate_table = table(pd.DataFrame(gate_rows)) if gate_rows else "Selección pendiente.\n"
    (OUT / "02_RESULTADOS.md").write_text(
        "# Resultados y selección\n\n"
        + result_table
        + "## Gates registrados\n\n"
        + gate_table
        + "Los candidatos prospectivos son S-M1 y R-C20-CTX. Se aplican los gates V14 "
        "frente a su control pareado y al M0/L0 histórico. S-M0 y R-C05 no se promocionan "
        "a posteriori. Un intervalo favorable no es un gate adicional.\n\n"
        "MiD por banda promedia primero dentro de cada grupo y después entre grupos. "
        "Overall = 0,5 MiDc + 0,3 MiDs + 0,1 MiDl + 0,1 MiDn; permanece ausente si falta "
        "una banda. mean_MiD y MAE son medias por consulta. El bootstrap agrupa por escenario: "
        "1.000 draws, seed12345, descriptivo ante múltiples contrastes.\n\n"
        "Las tablas completas por consulta y grupo están en evaluation/. Los siete "
        "contrastes iniciales se conservan en bootstrap/. Si hay réplica, se promedian "
        "pérdidas por consulta sobre seeds7/13/23; no se construye un ensemble TTC. "
        "El p95 agregado usa el error positivo medio por consulta antes de su cuantil.\n\n"
        "V es desarrollo interno expuesto; no es test oficial ni confirmación externa. "
        "La comparación histórica tiene seed7 y no se inventan sus seeds13/23.\n",
        encoding="utf-8",
    )
    diag = "# Diagnósticos\n\n"
    for filename, title in (
        ("H_GRADIENT_VARIANCE.csv", "Gradientes iniciales H"),
        ("CLIPPING.csv", "Clipping durante S"),
        ("CLIPPING_ALL_FITS.csv", "Clipping de todos los fits V15"),
        ("PRODUCER_REACH_SUMMARY.csv", "Alcance condicional P/H"),
    ):
        path = RUN / filename
        if path.exists():
            diag += "## " + title + "\n" + table(pd.read_csv(path))
    diag += """La varianza se estima con 32 batches de diagnóstico y se contrasta con el gradiente exacto de todo H en la inicialización seed7. No garantiza menor varianza para todos los parámetros o durante todo el entrenamiento. La igualdad de riesgo esperado no implica igualdad tras clipping y AdamW.

La masa no se normaliza dentro del batch. El sampler usa probabilidades globales float64 y la corrección w/p. El cambio no crea escenas independientes. SAMPLER_EXPOSURE.csv conserva los draws por grupo y rango.

El alcance del productor mantiene la geometría fija y permite residuales máximos independientes en ambas parejas. Se marca como no disponible cuando se cruza un polo del transporte. No es un límite absoluto de PAIR/H8 ni de un encoder distinto. Se conservan predicciones, known, soporte, edades y correcciones por consulta en PRODUCER_REACH_P_H.csv.

EVENT_CROP_ROWS_P_H.csv y EVENT_CROP_SUMMARY.csv usan las coordenadas reales del índice de eventos congelado. La fuente identifica secuencias y tracks, pero no episodios independientes; estos últimos permanecen sin inventar.

PRODUCER_METRICS_P_H.json compara A5 original y ambos readouts sobre P/H. P es entrenamiento de readout; H entrena las cabezas. Estas cifras describen el mecanismo y no son selección ni validación independiente. Las normas de gradiente R proceden de un objetivo distinto del de S; sus valores absolutos no son comparables como una única pérdida.
"""
    (OUT / "03_DIAGNOSTICOS.md").write_text(diag, encoding="utf-8")
    qa = (
        "# Pruebas, contabilidad y coste\n\n"
        + table(pd.DataFrame(checks))
        + table(pd.DataFrame(costs))
    )
    qa += f"Updates físicos superiores: {ledger['physical_upper']}. Máximos: 56.000 científicos, 500 técnicos, 3.500 recuperación y 60.000 físicos.\n\n"
    qa += """Los tests contractuales y logs se conservan en qa/. Las pruebas de continuidad comparan 10 updates sintéticos continuos frente a 5+restore+5. S consume 20 técnicos; R conserva una primera comprobación de 20 y otra de 20 tras añadir los guards explícitos. No se utilizan datos reales para consumir el presupuesto técnico.

Las paridades raw/caché y GPU/CPU verifican outputs, pérdidas y gradientes. La tolerancia 2e-5 se fijó antes de los fits R. Los módulos congelados se comparan exactamente tras el entrenamiento sintético y cada fit R. Los contratos T2/T3 y negativos usan fixtures; no son evidencia de precisión.

Los tiempos de fit proceden de RUN_PROVENANCE e incluyen trabajo vectorial/cabeza y persistencia durante el bucle; excluyen entrenar productores y la preparación de cachés. Los COST.json de V miden sólo forward de cabeza desde caché, batch128. No son latencias E2E ni batch1. PROGRESS.json de extracción registra el tiempo de esa invocación; cuando existe resume, no equivale al total de todos los intentos.

Se preservan el piloto CPU, la pausa controlada de extracción para validar la caché por filas, los fallos de colección iniciales y el ajuste del radio del fixture sintético. No se cambiaron los radios, datos ni parámetros científicos para resolver esos tests. Los artefactos históricos siguen identificados con sus versiones de código.

Ruff/Pyright se ejecutan sobre V15. No se afirma que la suite o CI completa del histórico esté verde. V13/R1 no se modifican; los snapshots de propietarios y recursos conservan el estado observado. No se cambian drivers, paging, energía ni afinidad.

Durante la ejecución cambió la admisión Git de V13, observada y conservada en OWNER_GIT_CHANGE_OBSERVATION.json. V15 no escribió ese archivo. Los padres event y la caché H vinculados por hash se volvieron a validar; la auditoría distingue estos activos científicos de la metadata viva de otra campaña.
"""
    (OUT / "04_PRUEBAS_Y_COSTE.md").write_text(qa, encoding="utf-8")
    reproduction = """# Reproducción y bundle

La raíz del worktree es `e-jepa-ttc-v15-range-readout`, junto a V13 y V14. Los resultados viven en `artifacts/range_readout_20261012/`.

```powershell
pwsh -NoProfile -File scripts/Resume-RangeReadout.ps1 -Help
pwsh -NoProfile -File scripts/Resume-RangeReadout.ps1
& '../e-jepa-ttc/.venv/Scripts/python.exe' scripts/build_range_readout_reports.py --bundle
```

No lance otra continuación mientras exista el coordinador activo. Los endpoints completos se restauran sin añadir updates. Un fallo requiere diagnosticar el log indicado antes de reanudar; no use reset, clean ni borre checkpoints.

El bundle contiene informes, código nuevo, protocolo recibido, pesos y estados nuevos, curvas, predicciones, diagnósticos, fragmentos vectoriales, features de las familias, pruebas y logs. Los archivos están relacionados mediante BUNDLE_MANIFEST.json y el SHA-256 externo. El verificador comprueba también duplicados y CRC.

```powershell
& '../e-jepa-ttc/.venv/Scripts/python.exe' scripts/build_range_readout_reports.py --verify-zip 'artifacts/V15_RANGE_READOUT_REPORTS_RESULTS_20261012.zip'
Get-FileHash -Algorithm SHA256 -LiteralPath 'artifacts/V15_RANGE_READOUT_REPORTS_RESULTS_20261012.zip'
```

El bundle no sustituye los datos sensoriales originales ni los tres padres matched V13. ADMISSION.json registra sus rutas y hashes; los manifiestos conservan los paths del host original. Repetir desde sensores requiere esas dependencias. La reproducción de tablas desde predicciones y la verificación de hashes no equivale a repetir inferencia ni entrenamiento.

Un checkpoint R requiere `load_endpoint(..., ResidualSpec(bound))`; su atributo `.model` conserva el forward raw del productor con el operador explícito. C20 no puede interpretarse como `0.20*tanh(s)` nativo. Los encoders son los mismos padres compartidos, no nuevas representaciones entrenadas con tres semillas.

No se hace push en esta campaña. Un resultado FCWD/Dev32, si la rama de transferencia llega a habilitarse, seguirá siendo desarrollo externo expuesto.
"""
    (OUT / "05_REPRODUCCION.md").write_text(reproduction, encoding="utf-8")
    (RUN / "FINAL_REPORT.md").write_text(
        header
        + result_table
        + gate_table
        + f"Informes detallados: ../../docs/range_readout_results_20261012/README.md\n\n"
        f"Contabilidad: {ledger['scientific']} científicos, {ledger['technical']} técnicos, "
        f"{ledger['recovery_upper']} recuperación.\n",
        encoding="utf-8",
    )
    plots()


def snapshot_historical() -> None:
    """Carry the two referenced baselines so report reproduction needs no sibling V14."""
    source = ROOT.parent / "e-jepa-ttc-v14-range-transfer/artifacts/range_transfer_20261010"
    destination = RUN / "historical_V14"
    records = {}
    for mass in (0, 1):
        name = f"RT-M{mass}-L0-S7"
        relative_paths = [
            f"fits/{name}/CURVE.csv",
            f"fits/{name}/CHECKPOINT_RECEIPT.json",
            f"evaluation/screen/{name}/PREDICTIONS.csv",
            f"evaluation/screen/{name}/PREDICTIONS_RECEIPT.json",
            f"evaluation/screen/{name}/METRICS.json",
        ]
        for relative in relative_paths:
            target = destination / relative
            original = source / relative
            if target.exists():
                if original.exists() and sha(target) != sha(original):
                    raise ValueError("Historical snapshot changed: " + relative)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(original, target)
            records[relative] = sha(target)
        csv = destination / f"evaluation/screen/{name}/PREDICTIONS.csv"
        if sha(csv) != read(csv.parent / "PREDICTIONS_RECEIPT.json")["sha256"]:
            raise ValueError("Historical prediction receipt mismatch")
    (destination / "SNAPSHOT_MANIFEST.json").write_text(
        json.dumps(records, indent=2) + "\n", encoding="utf-8"
    )


def verify(path: Path) -> dict[str, Any]:
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)) or archive.testzip() is not None:
            raise ValueError("Invalid ZIP members/CRC")
        manifest = json.loads(archive.read("BUNDLE_MANIFEST.json"))
        if set(names) != set(manifest) | {"BUNDLE_MANIFEST.json"}:
            raise ValueError("Manifest membership mismatch")
        for name, digest in manifest.items():
            if hashlib.sha256(archive.read(name)).hexdigest() != digest:
                raise ValueError("Hash mismatch: " + name)
    return {
        "status": "PASS",
        "files_verified": len(manifest),
        "bytes": path.stat().st_size,
        "sha256": sha(path),
    }


def bundle() -> None:
    if read(RUN / "NEXT_DECISION.json")["status"] not in (
        "NO_CANDIDATE",
        "NO_REPLICATED_CANDIDATE",
        "TRANSFER_COMPLETE",
    ):
        raise ValueError("Campaign is not closed; do not publish an incomplete final bundle")
    destination = ROOT / "artifacts/V15_RANGE_READOUT_REPORTS_RESULTS_20261012.zip"
    files = {}
    for folder in (
        RUN,
        OUT,
        ROOT / "operational/range_readout",
        ROOT / "src/e_jepa_ttc/range_readout",
        ROOT / "docs/range_readout_handoff_20261012",
    ):
        for path in folder.rglob("*"):
            if (
                path.is_file()
                and path.suffix not in (".zip", ".lock", ".pyc")
                and "__pycache__" not in path.parts
                and ".tmp" not in path.name
            ):
                files[path.relative_to(ROOT).as_posix()] = path
    for path in [
        *ROOT.glob("tests/test_range_readout*.py"),
        Path(__file__),
        ROOT / "scripts/Resume-RangeReadout.ps1",
        ROOT / "docs/range_readout_integration_20261012.md",
        ROOT / "pyproject.toml",
        ROOT / "uv.lock",
    ]:
        files[path.relative_to(ROOT).as_posix()] = path
    # Preserve the native implementation imported by the new integration as well.
    for folder in (ROOT / "src", ROOT / "operational"):
        for path in folder.rglob("*.py"):
            files[path.relative_to(ROOT).as_posix()] = path
    for path in (ROOT / "configs").rglob("*"):
        if path.is_file() and path.suffix in (".yaml", ".yml", ".json"):
            files[path.relative_to(ROOT).as_posix()] = path
    manifest = {name: sha(path) for name, path in sorted(files.items())}
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    metadata = json.dumps(
        {
            "git_head": commit,
            "base_commit": "1c251781ac845a5d661523e607e0f7c4a8d3af8a",
            "state": read(RUN / "NEXT_DECISION.json"),
            "optimizer_updates_during_packaging": 0,
        },
        indent=2,
    ).encode()
    manifest["BUNDLE_PROVENANCE.json"] = hashlib.sha256(metadata).hexdigest()
    with zipfile.ZipFile(destination, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, path in sorted(files.items()):
            archive.write(path, name)
            if sha(path) != manifest[name]:
                raise RuntimeError("Source changed while packaging: " + name)
        archive.writestr("BUNDLE_PROVENANCE.json", metadata)
        archive.writestr("BUNDLE_MANIFEST.json", json.dumps(manifest, indent=2))
    result = verify(destination)
    destination.with_suffix(".zip.sha256").write_text(
        result["sha256"] + "  " + destination.name + "\n", encoding="ascii"
    )
    destination.with_suffix(".zip.verification.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"bundle": str(destination), **result}))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", action="store_true")
    parser.add_argument("--verify-zip", type=Path)
    args = parser.parse_args()
    if args.verify_zip:
        print(json.dumps(verify(args.verify_zip)))
        return
    generate()
    if args.bundle:
        bundle()


if __name__ == "__main__":
    main()

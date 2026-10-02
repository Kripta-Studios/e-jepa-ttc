"""Assemble the extended local essential delivery after canonical verification."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import zipfile
from pathlib import Path
from typing import Any

import pandas as pd
from extras import ANALYSIS, CAMPAIGN, OUT, ROOT, record
from runtime import atomic_bytes, atomic_json, digest

ART = ROOT / "artifacts/simplex_t/closure_20261002"
STAGING = ART / "essential_staging"
DELIVERY = ART / "essential_delivery"


def put(source: Path, name: str) -> None:
    path = STAGING / name
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if digest(path) != digest(source):
            raise ValueError(f"staging input changed: {name}")
        return
    # All sources are closed immutable publications; use bounded copies.
    shutil.copyfile(source, path)


def git_state() -> dict[str, Any]:
    return {
        "worktree": str(ROOT),
        "branch": subprocess.check_output(
            ["git", "branch", "--show-current"], cwd=ROOT, text=True
        ).strip(),
        "execution_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "status": subprocess.check_output(["git", "status", "--short"], cwd=ROOT, text=True),
        "scientific_code": "0c1a7285b6b5af869b0bf5a13a9629f08997e7a2",
    }


def write_documents(verified: bool) -> None:
    state = git_state()
    metrics = pd.read_csv(OUT / "METRICS.csv")
    scores = metrics[metrics.scope == "all"].set_index("arm_seed").MiD.to_dict()
    review = record(ROOT / "artifacts/simplex_t/review_20261002/evidence.json")
    seed = record(ANALYSIS / "three_seed/TPR/THREE_SEED_ANALYSIS.json")["summary"]
    seed_uncertainty = record(ANALYSIS / "three_seed/TPR/uncertainty/PAIRED_UNCERTAINTY.json")
    accounting = record(ANALYSIS / "CAMPAIGN_ACCOUNTING.json")
    ewma = record(OUT / "ARITHMETIC_VERIFICATION.json")
    authority = record(ART / "CANONICAL_CALL_RESULT.json")
    freeze = record(CAMPAIGN / "launches/T6.json")["freeze_sha256"]
    table = "| Brazo | MiD OLD_DEV |\n|---|---:|\n" + "".join(
        f"| {name} | {value:.17g} |\n" for name, value in scores.items()
    )
    contrasts = "| Contraste | Delta | CI95 jerárquico | CI95 secuencias |\n|---|---:|---|---|\n"
    for name, row in review["contrasts"].items():
        contrasts += (
            f"| {name} | {row['delta']:.17g} | {row['hierarchical_ci95']} | "
            f"{row['sequence_only_ci95']} |\n"
        )
    residual = pd.read_csv(OUT / "RESIDUAL_DECOMPOSITION.csv")
    distributions = pd.read_csv(OUT / "LATENT_DISTRIBUTIONS.csv")
    similarities = pd.read_csv(OUT / "LATENT_SIMILARITY.csv")
    availability_table = pd.read_parquet(ANALYSIS / "analyses/T2/DIAGNOSTICS.parquet")
    availability = (
        pd.DataFrame(
            availability_table[
                (availability_table.arm == "TPR-D1-H8-C160")
                & (availability_table.axis == "availability")
            ]
        )
        .set_index("stratum")
        .queries.to_dict()
    )
    temporal_table = pd.read_parquet(ANALYSIS / "analyses/T2/TEMPORAL_STRATA.parquet")
    temporal = (
        pd.DataFrame(temporal_table[temporal_table.arm == "TPR-D1-H8-C160"])
        .set_index("stratum")
        .queries.to_dict()
    )
    primary_summary = next(row for row in review["scores"] if row["arm_seed"] == "TPR-D1-H8-C160@7")
    regeneration_status = "VERIFICADA" if verified else "PENDIENTE, archivo de preentrega"
    note = (
        "# Diagnóstico LATENT posterior, sin entrenamiento\n\n"
        "Los nueve endpoints LATENT/ZERO, sus predicciones y el informe histórico se conservan. "
        "Se verificaron las identidades TRAIN/OLD, normalizadores y productores en los tres folds. "
        "Las descomposiciones son descriptivas post hoc; no son gates de aceptación.\n\n"
        "## Distribuciones y productores\n\n"
        f"Se incluyen {len(distributions)} filas por coordenada/fold/rol en "
        "[LATENT_DISTRIBUTIONS.csv](supplement/LATENT_DISTRIBUTIONS.csv): momentos, extremos, "
        "cuantiles 1/99, escala TRAIN y conteos |z|>10. El universo es el de observaciones "
        "únicas consumidas por la historia H16 de normalización, sin duplicar queries. "
        f"En las coordenadas latentes, el máximo |z| observado es "
        f"{distributions[distributions.is_latent].normalized_abs_max.max():.17g}; "
        f"hay {int(distributions[distributions.is_latent].raw_std_below_1e_minus8.sum())} "
        "filas coordenada/rol/fold con std raw <1e-8. Las escalas casi nulas no se corrigen.\n\n"
        "[LATENT_PRODUCERS.json](supplement/LATENT_PRODUCERS.json) documenta las familias A5 "
        "inner/outer independientes. La dimensión 128 no garantiza coordenadas semánticamente "
        "alineadas; una cabeza comparte posiciones numéricas entre esos productores. "
        "Se compararon únicamente observaciones actuales ya cacheadas de igual query/ROI "
        "entre folds, con selección determinista por identidad y agrupación por par de familias. "
        f"[LATENT_SIMILARITY.csv](supplement/LATENT_SIMILARITY.csv) contiene {len(similarities)} "
        "comparaciones de estructuras de similitud coseno. "
        f"Rango de MAE de Gram: [{similarities.gram_cosine_MAE.min():.17g}, "
        f"{similarities.gram_cosine_MAE.max():.17g}]. "
        "No se ajustó un alineador, normalizador ni transformación con OLD_DEV. "
        "La heterogeneidad observada no demuestra la causa del fallo LATENT.\n\n"
        "## Residuales, todas las secuencias\n\n"
        "Se separa sobreestimación/infraestimación de TTC sólo cuando target y predicción "
        "son positivos. Los errores de signo se cuentan por separado. Un escape perjudicial "
        "es un punto fuera del hull de fases de expertos cuya pérdida supera la mediana actual. "
        "El 76,3% histórico fuera del hull no describía exclusivamente sobreestimación.\n\n"
        "| Brazo | Secuencia | Queries cruciales | Sobreestimación | Infraestimación | "
        "Signo erróneo | Escape perjudicial |\n"
        "|---|---|---:|---:|---:|---:|---:|\n"
    )
    selected = pd.DataFrame(
        residual[
            (residual.scope == "sequence")
            & (residual.stratum == "positive_crucial_0_to_3s")
            & residual.arm_seed.str.startswith("LATENT")
        ]
    )
    for row in selected.to_dict("records"):
        note += (
            f"| {row['arm_seed']} | {row['value']} | {row['queries']} | "
            f"{row['positive_TTC_overestimate_count']} | "
            f"{row['positive_TTC_underestimate_count']} | {row['wrong_sign_count']} | "
            f"{row['harmful_phase_escape_count']} |\n"
        )
    note += (
        "\n`mHGFBekt7X` permanece incluida; véanse también sus MiD y contribuciones ponderadas "
        "en METRICS.csv y RESIDUAL_DECOMPOSITION.csv, sin eliminarla del global. "
        "La cobertura q10–q90 del H8 registrado es "
        f"{primary_summary['phase_interval_covers']:.17g}, "
        "inferior al nominal 80%. "
        f"Se conservan {availability['cold_start']} cold starts, "
        f"{availability['partial_history']} historias parciales adicionales, "
        f"{temporal['rapid_change']} cambios rápidos bajo el umbral phase/s=1 y "
        f"{temporal['sign_transition']} transición de signo muestreada. "
        "No se cambió ese umbral ni se infirió latencia hasta detección.\n"
    )
    atomic_bytes(STAGING / "LATENT_DIAGNOSTIC_NOTE.md", note.encode("utf-8"))
    report = (
        "# Cierre local SIMPLEX-T\n\n"
        "Se completaron los 13 checkpoints T6 y la baseline EWMA registrada en tres folds. "
        "Esta sesión ejecutó cero actualizaciones de optimizador y no abrió confirmación.\n\n"
        "## Identidades y ejecución\n\n"
        f"Scientific code: `{state['scientific_code']}`. Freeze SHA-256: `{freeze}`. "
        f"Worktree: `{state['worktree']}`. Rama: `{state['branch']}`. "
        f"HEAD/commit de entrega: `{state['execution_head']}`. "
        "El entrenamiento conserva su identidad científica histórica; el commit operativo "
        "no lo modifica retroactivamente. La entrega canónica previa fue verificada en su HEAD "
        "histórico y queda preservada en canonical_delivery/.\n\n"
        f"Verificación canónica: `{authority['status']}`. Los 72 endpoints2500 y sus hashes "
        "están reconciliados; total científico guardado 180.000 updates. La contabilidad física "
        f"histórica conserva el intervalo [{accounting['recorded_total_updates_lower']}, "
        f"{accounting['recorded_total_updates_upper']}] incluyendo técnica/incertidumbre de "
        "terminaciones previas, sin consumo adicional aquí. El techo no autorizaba más trabajo.\n\n"
        "La política local fue enmendada por el usuario a 2 GiB de RAM disponible y 10 GB "
        "decimales de emergencia tras reservas; permanece techo RSS del árbol 4 GiB. "
        "No se alteró Stage70–76. Se conservan los cambios locales ajenos registrados en "
        "[GIT_STATE.json](GIT_STATE.json).\n\n"
        "## Reparación y paridad\n\n"
        "Se conservaron los seis checkpoints iniciales y los fragmentos posteriores válidos. "
        "Los cuatro grupos pendientes usan 33 fragmentos cada uno: 8.195 intentos publicados "
        "T2, 8.192 aceptados, tres rechazados. No hubo nueva selección de draws ni cambio "
        "de población, seeds, pesos, operaciones numéricas o intervalos. Se repararon la "
        "publicación atómica Parquet en Windows, la comparación JSON tupla/lista y la declaración "
        "incompleta de recibos T2/T4. El recibo T2 de una invocación fallida no acredita picos "
        "o tiempos exhaustivos de entrenamiento.\n\n"
        "[PARITY.json](verification/PARITY.json) y los XML guardan la equivalencia bit a bit "
        "entre ejecución continua e interrumpida, frente a la rutina congelada; para T2 "
        "publicado se conserva tolerancia preregistrada absoluta 1e-10. También se probaron "
        "hash cambiado, raíz ausente, Windows1455, journal cortado y reemplazo interrumpido. "
        "Los tests no entrenaron modelos.\n\n"
        "## Números reconciliados\n\n"
        "Cohorte OLD_DEV: 8.192 queries, nueve secuencias, tres folds. Los CSV físicos y "
        "evidence.json se verificaron a precisión completa, con diferencia absoluta máxima "
        "permitida 1e-10; el markdown redondeado no se usó como oráculo. La tabla siguiente "
        "se regenera desde [METRICS.csv](supplement/METRICS.csv).\n\n" + table + "\n"
        "La comparación principal conserva H8 registrado. Los controles FREE, REVERSED, "
        "LATENT_ZERO y el H16 exploratorio no se promovieron.\n\n"
        f"Tres seeds de cabeza: H8 media {seed['scores']['TPR-D1-H8-C160']['mean']:.17g}, "
        f"std muestral {seed['scores']['TPR-D1-H8-C160']['sample_std']:.17g}; H1 media "
        f"{seed['scores']['TPR-D1-H1-C160']['mean']:.17g}. Delta medio de pérdidas emparejadas "
        f"{seed['paired_delta']['mean']:.17g}. CI95 jerárquico condicionado a esas seeds: "
        f"{seed_uncertainty['comparisons']['TPR-D1-H8-C160']['hierarchical_ci95']}. "
        "No es un ensemble de TTC ni replicación de todos los expertos.\n\n"
        "EWMA fija: mediana de fases expertas por observación, pesos exp(-lag/0,3) sobre "
        "el sufijo H8 válido, normalización y emisión registrada FP32 con conversión TTC "
        "float64. No se promediaron TTC firmados ni se barrió alpha. Cada fold conserva "
        "binding de queries, historias, ACK y productor. Sus contrastes post hoc están en "
        "[ARITHMETIC_VERIFICATION.json](supplement/ARITHMETIC_VERIFICATION.json):\n\n"
        + "```json\n"
        + json.dumps(ewma["ewma_comparisons"], indent=2, allow_nan=False)
        + "\n```\n\n"
        "## Factorial, interacciones y controles\n\n" + contrasts + "\n"
        "Las tablas por query, secuencia, fold y estratos, incluidos los vacíos, están "
        "incluidas junto con la receta histórica y todos los draws jerárquicos. El intervalo "
        "por secuencia no crea adquisiciones independientes nuevas.\n\n"
        "## Conclusiones separadas\n\n"
        "- Ejecución: T6 y transporte local verificados; cero updates en esta sesión.\n"
        "- Validez: fuentes, identidades de query, normalizadores y números sellados.\n"
        "- Desarrollo: H8 mejora sustancialmente la estimación en OLD_DEV reutilizado.\n"
        "- Replicación: tres seeds de cabezas, mismos expertos congelados.\n"
        "- Contexto: el pasado aporta información frente a H1 y REPEAT_CURRENT.\n"
        "- Mecanismo: la cronología no está demostrada; REVERSED fue entrenado con "
        "inversión sistemática y puede reaprenderla.\n"
        "- Seguridad/disponibilidad: no hay evidencia de AEB más rápida, tracking online "
        "persistente o incertidumbre calibrada. La GRU reinicia estado por query y el "
        "ROI depende de la query actual.\n"
        "- Confirmación: Stage76, public validation, private test, EvTTC test y CodaBench "
        "permanecen cerrados.\n\n"
        "## LATENT, regeneración y siguiente decisión\n\n"
        "[LATENT_DIAGNOSTIC_NOTE.md](LATENT_DIAGNOSTIC_NOTE.md) separa distribuciones, "
        "productores, similitud de observaciones compartidas y errores TTC; no corrige "
        "los resultados LATENT. Contiene todas las secuencias, incluida mHGFBekt7X.\n\n"
        "El bundle contiene los 72 pesos compactos, normalizadores, entradas normalizadas "
        "OLD_DEV para sus cabezas, predicciones y 180.000 puntos de curvas TRAIN. "
        "Los 36 checkpoints de expertos no se incluyen: se registra ruta, tamaño, hash y "
        "rol en [EXPERT_CHECKPOINT_INVENTORY.json](supplement/EXPERT_CHECKPOINT_INVENTORY.json). "
        "El alcance es reproducción de agregados y salidas de cabezas desde contextos "
        "cacheados incluidos; no reconstrucción raw-autónoma ni inferencia nueva de expertos.\n\n"
        f"Regeneración desde extracción independiente: {regeneration_status}. "
        "La verificación incluye hashes internos/CRC y el script `operational/regenerate.py`. "
        "El recibo final externo vincula el SHA-256 del ZIP entregado.\n\n"
        "[NEXT_EXPERIMENT_PROPOSAL.md](NEXT_EXPERIMENT_PROPOSAL.md) propone una única "
        "replicación acotada H16 con seis fits/15.000 updates futuros. No está ejecutada "
        "ni autorizada aquí. La literatura recuperada prepara hipótesis posteriores, "
        "sin cambiar ni retrasar la campaña.\n"
    )
    decision = {
        "schema": "simplex_t_local_closure_decision_v1",
        "status": "CLOSED_LOCAL_DEVELOPMENT" if verified else "PREDELIVERY_REGENERATION_PENDING",
        "scientific_code": state["scientific_code"],
        "scientific_freeze_sha256": freeze,
        "analysis_commit": state["execution_head"],
        "candidate": "TPR-D1-H8-C160",
        "T6_steps_complete": 13,
        "fits": 72,
        "saved_scientific_updates": 180000,
        "optimizer_updates_in_this_session": 0,
        "ewma_folds_complete": [0, 1, 2],
        "canonical_verification": authority,
        "replication_scope": "SUPERVISED_HEAD_SEEDS_ONLY",
        "past_context_useful": True,
        "chronological_mechanism_demonstrated": False,
        "faster_AEB_demonstrated": False,
        "persistent_online_tracking_demonstrated": False,
        "calibrated_uncertainty_demonstrated": False,
        "holdouts_opened": False,
        "holdout_authorized": False,
        "future_optimizer_updates_authorized": 0,
        "next_proposal": "NEXT_EXPERIMENT_PROPOSAL.md",
        "proposal_executed": False,
        "raw_autonomous_regeneration": False,
    }
    atomic_bytes(STAGING / "CODEX_SIMPLEX_T_FINAL_REPORT.md", report.encode("utf-8"))
    atomic_json(STAGING / "NEXT_DECISION_SIMPLEX_T.json", decision)
    atomic_json(STAGING / "GIT_STATE.json", state)


def create_archive(path: Path) -> str:
    members = {
        p.relative_to(STAGING).as_posix(): {"sha256": digest(p), "bytes": p.stat().st_size}
        for p in sorted(STAGING.rglob("*"))
        if p.is_file() and p.relative_to(STAGING).as_posix() != "CONTENT_MANIFEST.json"
    }
    atomic_json(
        STAGING / "CONTENT_MANIFEST.json",
        {
            "schema": "simplex_t_extended_delivery_v1",
            "analysis_commit": git_state()["execution_head"],
            "members": members,
        },
    )
    partial = path.with_suffix(".zip.pending")
    if path.exists() or partial.exists():
        raise FileExistsError("preserve previous delivery archive")
    with zipfile.ZipFile(
        partial, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6
    ) as archive:
        for name in [*members, "CONTENT_MANIFEST.json"]:
            archive.write(STAGING / name, name)
    with zipfile.ZipFile(partial) as archive:
        if archive.testzip() is not None or set(archive.namelist()) != {
            *members,
            "CONTENT_MANIFEST.json",
        }:
            raise ValueError("ZIP CRC/member inventory mismatch")
        import hashlib

        for name, pin in members.items():
            if hashlib.sha256(archive.read(name)).hexdigest() != pin["sha256"]:
                raise ValueError("ZIP member hash mismatch")
    os.replace(partial, path)
    value = digest(path)
    atomic_bytes(path.with_suffix(".zip.sha256"), f"{value}  {path.name}\n".encode("ascii"))
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["prepare", "final"], required=True)
    args = parser.parse_args()
    import psutil

    if psutil.disk_usage(str(ROOT)).free - 3 * 1024**3 < 10_000_000_000:
        raise InterruptedError("10GB emergency plus3GiB conservative package reservation required")
    if (
        record(ART / "CANONICAL_CALL_RESULT.json")["status"]
        != "DELIVERY_REVERIFIED_AGAINST_BOUND_SCIENTIFIC_EVIDENCE"
    ):
        raise ValueError("canonical verify-only required before extended delivery")
    final_pin = CAMPAIGN / "T6/CHECKPOINTED_WORK/control/CANONICAL_FINAL_VERIFICATION.json"
    if record(final_pin)["status"] != "CANONICAL_ADMISSION_AND_FREEZE_REVERIFIED_AFTER_DELIVERY":
        raise ValueError("final canonical admission/freeze required")
    if len(record(OUT / "HEAD_EXPORT.json")["heads"]) != 72:
        raise ValueError("all72 compact input exports required")
    DELIVERY.mkdir(parents=True, exist_ok=True)
    STAGING.mkdir(parents=True, exist_ok=True)
    if args.stage == "prepare":
        canonical = ART / "canonical_attempt/delivery"
        bundle = next(canonical.glob("*.zip"))
        if digest(bundle) != (bundle.with_suffix(".zip.sha256").read_text().split()[0]):
            raise ValueError("canonical ZIP SHA mismatch")
        with zipfile.ZipFile(bundle) as archive:
            if archive.testzip() is not None:
                raise ValueError("canonical CRC mismatch")
            for name in archive.namelist():
                target_name = (
                    "canonical_delivery/" + name
                    if name
                    in {
                        "CODEX_SIMPLEX_T_FINAL_REPORT.md",
                        "NEXT_DECISION_SIMPLEX_T.json",
                        "CONTENT_MANIFEST.json",
                    }
                    else name
                )
                target = STAGING / target_name
                if not target.resolve().is_relative_to(STAGING.resolve()):
                    raise ValueError("ZIP traversal")
                if not target.exists():
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with archive.open(name) as source, target.open("xb") as destination:
                        shutil.copyfileobj(source, destination)
        for path in sorted(OUT.rglob("*")):
            if path.is_file():
                put(path, "supplement/" + path.relative_to(OUT).as_posix())
        for path in sorted(Path(__file__).parent.iterdir()):
            if path.is_file():
                put(path, "operational/" + path.name)
        for name in (
            "PARITY.json",
            "PUBLICATION_PARITY.xml",
            "ALL_PARITY.xml",
            "CANONICAL_CALL_RESULT.json",
            "RESOURCE_RECEIPTS_RECONCILIATION.json",
            "GRAPH_SERIALIZATION_PARITY.json",
            "HANDOFF_VERIFICATION.json",
        ):
            put(ART / name, "verification/" + name)
        for path in sorted((CAMPAIGN / "T6/CHECKPOINTED_WORK/control").iterdir()):
            if path.is_file():
                put(path, "journal/" + path.name)
        for name in (
            "evidence.json",
            "scores.csv",
            "per_sequence.csv",
            "training_curves.csv",
            "training_fits.csv",
        ):
            put(ROOT / "artifacts/simplex_t/review_20261002" / name, "historical_review/" + name)
        for path in (canonical / "DELIVERY.json", bundle.with_suffix(".zip.sha256")):
            put(path, "canonical_delivery/" + path.name)
        put(Path(__file__).with_name("NEXT_EXPERIMENT_PROPOSAL.md"), "NEXT_EXPERIMENT_PROPOSAL.md")
        for fold in range(3):
            folder = ROOT / f"artifacts/simplex_t/T1/fixed_baselines_outer{fold}_fp64_emission"
            for name in ("BASELINES.json", "outer_dev.npz"):
                put(folder / name, f"fixed_baselines/fold{fold}/" + name)
        for name in ("pyproject.toml", "uv.lock", ".python-version"):
            put(ROOT / name, "environment/" + name)
        put(final_pin, "verification/CANONICAL_FINAL_VERIFICATION.json")
        put(
            ART / "saved_analysis/SCORE_RECONCILIATION.csv",
            "verification/SCORE_RECONCILIATION.csv",
        )
        write_documents(False)
        archive = DELIVERY / "PREDELIVERY_REGENERATION_INPUT.zip"
    else:
        proof = record(ART / "PREDELIVERY_REGENERATION.json")
        if proof["status"] != "EXTRACTED_BUNDLE_REGENERATION_VERIFIED" or len(proof["heads"]) != 72:
            raise ValueError("extracted regeneration required")
        put(ART / "PREDELIVERY_REGENERATION.json", "verification/PREDELIVERY_REGENERATION.json")
        write_documents(True)
        archive = DELIVERY / (
            "E_JEPA_TTC_SIMPLEX_T_ESSENTIAL_RESULTS_" + git_state()["execution_head"][:12] + ".zip"
        )
        for name in (
            "CODEX_SIMPLEX_T_FINAL_REPORT.md",
            "NEXT_DECISION_SIMPLEX_T.json",
            "LATENT_DIAGNOSTIC_NOTE.md",
            "NEXT_EXPERIMENT_PROPOSAL.md",
        ):
            shutil.copyfile(STAGING / name, DELIVERY / name)
    archive_sha = create_archive(archive)
    if args.stage == "final":
        # Make the standalone report's relative links usable locally, without
        # duplicating the payload bytes already in immutable staging.
        for source in sorted(STAGING.rglob("*")):
            if not source.is_file():
                continue
            destination = DELIVERY / source.relative_to(STAGING)
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists():
                if digest(destination) != digest(source):
                    raise ValueError("preserve conflicting final browsable payload")
            else:
                os.link(source, destination)
    print(
        json.dumps({"archive": str(archive), "sha256": archive_sha}, ensure_ascii=False),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

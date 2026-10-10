"""Audit frozen predictions against paper metrics without fitting or selecting weights."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import platform
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from operational.sota_evidence.metrics import observations, paired_macro, summarize

METHODS = [
    "H8_median3",
    "Direct_median3",
    "H8_seed7",
    "H8_seed13",
    "H8_seed23",
    "Direct_seed7",
    "Direct_seed13",
    "Direct_seed23",
    "public_Garl_event_lhr",
    "public_Garl_rgb_event_full",
]


def read(path: Path) -> dict:
    """Read JSON receipts including PowerShell UTF8 BOMs."""
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def digest(path: Path) -> str:
    """Hash artifacts in bounded memory."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def json_safe(value: object) -> object:
    """Use JSON null, never nonstandard NaN/Infinity, for unavailable metrics."""
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    return value


def inventory(sources: Path, media_root: Path) -> dict:
    """Check the official test population without trying to reconstruct private labels."""
    frame = pd.read_parquet(sources / "test_inputs.parquet")
    if frame.sample_token.duplicated().any() or frame.sample_token.isna().any():
        raise ValueError("official test tokens are not unique")
    template = read(sources / "sample_submission.json")
    if set(template["results"]) != set(frame.sample_token):
        raise ValueError("official template and input population differ")
    train = set((sources / "official_train.txt").read_text().split())
    test = set((sources / "official_test.txt").read_text().split())
    if train & test or set(frame.sequence_id) != test:
        raise ValueError("official split mismatch")
    tree_items = json.loads((sources / "eap_test_tree.json").read_text(encoding="utf-8"))
    tree = {item["path"]: item for item in tree_items}
    events = []
    resolved_root = media_root.resolve()
    for name in sorted(set(frame.events_path)):
        path = (media_root / name).resolve()
        if not path.is_relative_to(resolved_root):
            raise ValueError("media reference escapes dataset root")
        events.append(
            {
                "relative_path": name,
                "exists": path.is_file(),
                "remote_bytes": tree.get(name, {}).get("size"),
            }
        )
    known_sizes = all(row["remote_bytes"] is not None for row in events)
    return {
        "official_test_samples": len(frame),
        "test_sequences": len(test),
        "train_sequences": len(train),
        "split_intersection": sorted(train & test),
        "public_test_columns": frame.columns.tolist(),
        "public_test_has_ttc": "ttc" in frame.columns,
        "template_token_set_exact": True,
        "event_files": events,
        "missing_event_bytes": sum(row["remote_bytes"] for row in events if not row["exists"])
        if known_sizes
        else None,
        "official_scoring": "Private labels; JSON evaluation requires CodaBench",
    }


def training_split_audit(train_index: Path, sources: Path) -> dict:
    """Verify actual V12 training sequence IDs against the official held-out split."""
    actual = set(pd.read_parquet(train_index, columns=["sequence_id"]).sequence_id.astype(str))
    official = set((sources / "official_train.txt").read_text().split())
    test = set((sources / "official_test.txt").read_text().split())
    return {
        "v12_train_sequences": len(actual),
        "official_train_sequences": len(official),
        "same_train_ids": actual == official,
        "train_test_overlap": sorted(actual & test),
        "status": "PASSED" if actual == official and not actual & test else "FAILED",
        "train_index_sha256": digest(train_index),
    }


def render_report(output: Path, report: Path, status: dict) -> None:
    """Generate human-readable findings from computed tables and archived sources."""
    metrics = pd.read_csv(output / "METRICS.csv")
    selected = pd.DataFrame(
        metrics[
            (metrics.policy == "native")
            & (metrics.cohort == "all")
            & metrics.method.isin(
                [
                    "H8_median3",
                    "Direct_median3",
                    "public_Garl_event_lhr",
                    "public_Garl_rgb_event_full",
                ]
            )
        ]
    )
    lines = [
        "# Evidencia frente a Garl-TTC y REACT — 10 de octubre de 2026",
        "",
        "**Estado: superioridad SOTA no establecida.** Esta ejecución recalcula métricas",
        "sobre predicciones congeladas; no entrena, selecciona checkpoints ni crea un test ciego.",
        "V13 conserva su ejecución. GPU adicional consumida: 0 segundos.",
        "",
        "## Resultados recalculados",
        "",
        "RTE micro pondera consultas; RTE macro pondera secuencias por igual.",
        "Los resultados son diagnósticos locales; no son las poblaciones de los artículos.",
        "",
        "| Cohorte | Método | n | RTE micro (%) | RTE macro (%) |",
        "|---|---|---:|---:|---:|",
    ]
    for row in selected.to_dict("records"):
        micro = f"{row['micro_rte_percent']:.3f}" if pd.notna(row["micro_rte_percent"]) else "N/D"
        macro = f"{row['macro_rte_percent']:.3f}" if pd.notna(row["macro_rte_percent"]) else "N/D"
        lines.append(f"| {row['dataset']} | {row['method']} | {row['n']} | {micro} | {macro} |")
    lines.extend(
        [
            "",
            "Las tablas también contienen cada semilla, cada secuencia, TTC positivos, bandas",
            "de ambos artículos y una vista secundaria con límite ±60 s igual para todos.",
            "El bootstrap pareado remuestrea secuencias enteras (10.000 réplicas); no elimina",
            "la correlación entre familias ni la exposición previa.",
            "FCWD solo aporta tres secuencias.",
            "No se selecciona el modelo con mejor resultado después de observar estas tablas.",
            "",
            "## Contratos de los artículos",
            "",
            "- Garl: MiD = 10^4 |log(1−0,1/pred) − log(1−0,1/GT)|; bandas",
            "  (0,3], (3,6], (6,10], [−10,0), pesos 0,5/0,3/0,1/0,1.",
            "  El código público excluye −10 exactamente, a diferencia del texto. Se registran",
            "  ambas convenciones. La tabla VI publica RTE promedio 10,60 %; es referencia",
            "  bibliográfica, no una reproducción nuestra. [Artículo](https://arxiv.org/abs/2603.16303).",
            "- REACT: su MiDw usa |log(pred/GT)| para TTC positivo, con bandas",
            "  [0,3), [3,6), [6,10). Es otra magnitud. Sus pesos suman 0,9; registramos",
            "  suma literal y normalizada como sensibilidad, sin elegir la más favorable.",
            "  La tabla II declara 9,59±0,74 % sobre validación en dominio con cinco semillas",
            "  y 4,6 ms; la tabla III declara 25,7 % en FCWD. Sin población/ejecutable",
            "  idénticos no se restan esas cifras a las nuestras para proclamar victoria.",
            "  [Artículo](https://arxiv.org/abs/2609.19204).",
            "",
            "El artículo de REACT anuncia código tras publicación. La web del autor y la lista",
            "de repositorios públicos revisadas no proporcionaron una reproducción verificable.",
            "Esto describe las fuentes examinadas; no demuestra ausencia absoluta de código.",
            "El 9,44 % de Garl citado en REACT tampoco equivale al promedio 10,60 % de la",
            "tabla VI de Garl: hace falta reconstruir la población antes de comparar rankings.",
            "",
            "## Integridad de las métricas",
            "",
            "`REFERENCE_PARITY.json` contrasta cada fila local con el evaluador Garl archivado.",
            "Su media MiD omite NaN, y una predicción infinita puede generar MiD finito pese",
            "a marcarse como fallo. Conservamos esa salida solo como referencia de compatibilidad.",
            "La salida estricta declara N/D si alguna fila requerida es inválida; registra",
            "fallos y tamaños de banda. No cambia umbrales para mejorar resultados.",
            "Una banda vacía impide el agregado ponderado: no se redistribuyen sus pesos.",
            "",
            "## Barreras de la evaluación oficial",
            "",
            "Las 40 secuencias del índice real de entrenamiento coinciden con train40 oficial",
            "y no intersectan test12 (`TRAIN_TEST_DISJOINT.json`). Esto cierra la comprobación",
            "de IDs de secuencia; no sustituye auditoría de escenarios ni puntuación de test.",
            f"- Test oficial: {status['inventory']['official_test_samples']} consultas, "
            f"{status['inventory']['test_sequences']} secuencias, sin etiquetas TTC públicas.",
            f"- Eventos ausentes: {status['inventory']['missing_event_bytes']} bytes",
            "  según el inventario remoto archivado. Descarga completa pendiente.",
            "- Preparar inferencia causal para esos IDs, comprobar relojes y ROI y obtener",
            "  predicciones completas antes de exportar. `submission.py` exige todos los IDs",
            "  y valores finitos; la plantilla descargada no se usa como predicción.",
            "- Obtener puntuación externa de CodaBench con el candidato y protocolo fijados",
            "  previamente. Las etiquetas privadas no se pueden reconstruir del conjunto público.",
            "- Para REACT: disponer de pesos/código, split y timestamps exactos, configuración",
            "  de entrenamiento y medición, y reevaluar ambos métodos en la misma población.",
            "- Las tres cabezas comparten productores: faltan repeticiones independientes del",
            "  sistema completo para estimar variabilidad de entrenamiento.",
            "",
            "## Alcance de una afirmación",
            "",
            "H8 usa eventos con ROI oráculo e historia 650 ms; Garl usa sus entradas nativas",
            "200 ms, y la rama full añade RGB. REACT no recibe ROI. El entrenamiento previo",
            "de nuestros productores incluyó teacher RGB: event-only describe inferencia, no",
            "toda la supervisión. Los protocolos no son equivalentes.",
            "La latencia propia incluye lectura HDF5; los costes publicados tienen otros",
            "límites y hardware. La historia de 650 ms no es automáticamente 650 ms de",
            "espera adicional en cada consulta de un sistema causal ya inicializado.",
            "Se necesitan warm-start, cadencia, antigüedad de observaciones y cola separados.",
            "",
            "Podemos afirmar mejoras medidas en los protocolos locales descritos. No podemos",
            "afirmar mejor rendimiento global, tiempo real, autonomía sin ROI ni SOTA.",
            "",
            "## Reproducir",
            "",
            "```powershell",
            "python -m operational.sota_evidence.run `",
            "  --source-root artifacts/ttc_revision_20261009 `",
            "  --train-index artifacts/train40_system_20261005/TRAIN40_ROWS.parquet `",
            "  --output artifacts/sota_evidence_20261010 --media-root $env:EAP_ROOT `",
            "  --garl-evaluator $env:GARL_EVALUATOR --report docs/sota_evidence_20261010/README.md",
            "python -m pytest tests/unit/test_sota_evidence.py `",
            "  tests/unit/test_ttc_revision_score.py",
            "```",
            "",
            "Fuentes y entradas: `SOURCES.json`, `INPUTS.json`, `PROTOCOL.json`.",
            "Resultados: `METRICS.csv`, `PAIRED_MACRO_RTE.csv`, `REFERENCE_PARITY.json`,",
            "`INVENTORY.json`, `RESULT.json`. Las fuentes bibliográficas nunca se mezclan",
            "con resultados reproducidos en una clasificación conjunta.",
        ]
    )
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    """Recompute frozen evidence, reference parity and official-test feasibility."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--media-root", type=Path, required=True)
    parser.add_argument("--garl-evaluator", type=Path, required=True)
    parser.add_argument("--train-index", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    sources = output / "sources"
    source_manifest = read(output / "SOURCES.json")
    for source in source_manifest["sources"]:
        if source["status"] == "DOWNLOADED":
            if digest(sources / source["name"]) != source["sha256"]:
                raise ValueError(f"downloaded source changed: {source['name']}")
    reference_copy = sources / "evaluate_garlttc.py"
    if reference_copy.exists() and digest(reference_copy) != digest(args.garl_evaluator):
        raise ValueError("reference evaluator changed; use a new campaign")
    shutil.copy2(args.garl_evaluator, reference_copy)
    spec = importlib.util.spec_from_file_location("garl_reference", reference_copy)
    if spec is None or spec.loader is None:
        raise ValueError("cannot import inspected reference evaluator")
    reference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference)
    protocol = {
        "schema": "sota_evidence_v1",
        "selection": "none; frozen pre-existing endpoints",
        "methods": METHODS,
        "policies": ["native", "common_bound_60"],
        "eligibility": "finite nonzero GT only; all missing predictions retained",
        "primary_reporting": "micro and equal-sequence macro RTE; no SOTA ranking",
        "bootstrap_draws": 10000,
        "seed": 20261010,
        "garl_dt_seconds": 0.1,
        "react_weight_normalization": "ambiguous; publish raw and normalized",
        "training_updates": 0,
        "additional_gpu_seconds": 0,
        "cohort_exposure": "Dev32 and FCWD already exposed; not blind",
    }
    protocol_path = output / "PROTOCOL.json"
    if protocol_path.exists() and read(protocol_path) != protocol:
        raise ValueError("protocol changed after freeze; use a new campaign")
    protocol_path.write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    results, paired, parity, invalid_rows = [], [], [], []
    inputs = {str(reference_copy): digest(reference_copy)}
    auxiliary_inputs = {
        str(args.train_index): digest(args.train_index),
        str(output / "SOURCES.json"): digest(output / "SOURCES.json"),
        str(sources / "eap_test_tree.json"): digest(sources / "eap_test_tree.json"),
    }
    for dataset in ("DEV32", "FCWD"):
        path = args.source_root / f"{dataset}_PREDICTIONS.csv"
        inputs[str(path)] = digest(path)
        frame = pd.read_csv(path)
        if frame.query_id.duplicated().any() or frame.sequence_id.isna().any():
            raise ValueError("duplicate IDs or missing sequence metadata")
        truth_all = np.asarray(frame.truth_ttc_seconds, dtype=float)
        eligible = np.isfinite(truth_all) & (truth_all != 0)
        frame = frame.loc[eligible]
        truth = np.asarray(frame.truth_ttc_seconds, dtype=float)
        sequences = np.asarray(frame.sequence_id)
        cohorts = {"all": np.ones(len(truth), bool), "positive": truth > 0}
        cohorts.update({f"sequence:{s}": sequences == s for s in np.unique(sequences)})
        ref_frame = frame[["query_id", "sequence_id", "truth_ttc_seconds"]].rename(
            columns={"query_id": "sample_token", "truth_ttc_seconds": "ttc"}
        )
        for method in METHODS:
            raw = np.asarray(frame[method], dtype=float)
            ref_rows = reference._rows_from_prediction(
                ref_frame,
                dict(zip(frame.query_id, raw, strict=True)),
                strict_tokens=True,
                dT=0.1,
            )
            row_metrics = observations(truth, raw)
            ours = row_metrics["garl_reference_mid"]
            bad_mid = ~np.isfinite(row_metrics["garl_mid"])
            bad_react = (truth > 0) & ~np.isfinite(row_metrics["react_logratio"])
            for index in np.flatnonzero(bad_mid | bad_react):
                invalid_rows.append(
                    {
                        "dataset": dataset,
                        "method": method,
                        "query_id": frame.iloc[index].query_id,
                        "truth": truth[index],
                        "prediction": raw[index],
                        "garl_mid_invalid": bool(bad_mid[index]),
                        "react_positive_logratio_invalid": bool(bad_react[index]),
                        "prediction_nonfinite": bool(not np.isfinite(raw[index])),
                        "prediction_zero": bool(raw[index] == 0),
                        "garl_gt_ratio_nonpositive": bool(0 < truth[index] <= 0.1),
                        "garl_prediction_ratio_nonpositive": bool(0 < raw[index] <= 0.1),
                        "react_prediction_nonpositive": bool(raw[index] <= 0),
                    }
                )
            ref_values = np.asarray(ref_rows.MiD, dtype=float)
            if not np.allclose(ours, ref_values, rtol=1e-10, atol=1e-9, equal_nan=True):
                raise ValueError(f"reference MiD parity failed: {dataset}/{method}")
            finite = np.isfinite(ours) & np.isfinite(ref_values)
            parity.append(
                {
                    "dataset": dataset,
                    "method": method,
                    "status": "PASSED",
                    "rows": len(raw),
                    "max_abs_mid_difference": float(
                        np.max(np.abs(ours[finite] - ref_values[finite]))
                    )
                    if finite.any()
                    else None,
                    "reference_summary": reference.summarize_results(ref_rows),
                }
            )
        for policy in ("native", "common_bound_60"):
            predictions = {}
            for method in METHODS:
                raw = np.asarray(frame[method], dtype=float)
                values = (
                    np.where(np.isfinite(raw), np.clip(raw, -60, 60), raw)
                    if policy != "native"
                    else raw
                )
                predictions[method] = values
                for cohort, mask in cohorts.items():
                    results.append(
                        {
                            "dataset": dataset,
                            "policy": policy,
                            "method": method,
                            "cohort": cohort,
                            "population_n": len(truth_all),
                            "gt_ineligible_n": int((~eligible).sum()),
                            **summarize(truth[mask], values[mask], sequences[mask]),
                        }
                    )
            for left in ("H8_median3", "Direct_median3"):
                rights = ["public_Garl_event_lhr", "public_Garl_rgb_event_full"]
                if left == "Direct_median3":
                    rights.append("H8_median3")
                for right in rights:
                    for cohort in ("all", "positive"):
                        mask = cohorts[cohort]
                        paired.append(
                            {
                                "dataset": dataset,
                                "policy": policy,
                                "cohort": cohort,
                                "left": left,
                                "right": right,
                                **paired_macro(
                                    truth[mask],
                                    predictions[left][mask],
                                    predictions[right][mask],
                                    sequences[mask],
                                ),
                            }
                        )
    previous = output / "INPUTS.json"
    if previous.exists() and read(previous) != inputs:
        raise ValueError("frozen inputs changed; use a new campaign")
    auxiliary_path = output / "AUXILIARY_INPUTS.json"
    if auxiliary_path.exists() and read(auxiliary_path) != auxiliary_inputs:
        raise ValueError("auxiliary source inputs changed; use a new campaign")
    pd.DataFrame(results).to_csv(output / "METRICS.csv", index=False)
    pd.DataFrame(paired).to_csv(output / "PAIRED_MACRO_RTE.csv", index=False)
    pd.DataFrame(invalid_rows).to_csv(output / "INVALID_METRICS.csv", index=False)
    inv = inventory(sources, args.media_root)
    split = training_split_audit(args.train_index, sources)
    status = {
        "status": "AUDIT_COMPLETE_SOTA_NOT_ESTABLISHED",
        "utc": datetime.now(UTC).isoformat(),
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "python": platform.python_version(),
        "host": platform.node(),
        "gpu_seconds": 0,
        "optimizer_updates": 0,
        "inventory": inv,
        "reference_parity_runs": len(parity),
        "metric_rows": len(results),
        "gates": {
            "frozen_prediction_rescoring": "PASSED",
            "garl_formula_parity": "PASSED",
            "v12_official_sequence_disjointness": split["status"],
            "official_test_score": "MISSING_PRIVATE_EVALUATION",
            "official_test_media": "INCOMPLETE"
            if any(not r["exists"] for r in inv["event_files"])
            else "AVAILABLE",
            "react_reproduction": "MISSING_VERIFIED_CODE_WEIGHTS_AND_QUERY_SPLIT",
            "matched_runtime_measurement": "NOT_PERFORMED_FOR_PAPER_EXECUTABLES",
            "independent_full_model_seeds": "NOT_ESTABLISHED",
            "blind_comparison": "NOT_ESTABLISHED",
        },
    }
    for name, value in (
        ("INPUTS.json", inputs),
        ("AUXILIARY_INPUTS.json", auxiliary_inputs),
        ("REFERENCE_PARITY.json", parity),
        ("INVENTORY.json", inv),
        ("TRAIN_TEST_DISJOINT.json", split),
        ("RESULT.json", status),
    ):
        (output / name).write_text(
            json.dumps(json_safe(value), indent=2, allow_nan=False), encoding="utf-8"
        )
    render_report(output, args.report, status)
    print(json.dumps({k: v for k, v in status.items() if k != "inventory"}, indent=2))


if __name__ == "__main__":
    main()

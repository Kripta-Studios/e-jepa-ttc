"""Score sealed post-V18 predictions; never train, infer or choose a checkpoint."""

from __future__ import annotations

import argparse
import platform
import shutil
import socket
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

from operational.range_transfer.analyze import metrics
from scripts.post_v18_transfer_compare import DOC, OUT, ROOT, RUN18, verified
from scripts.post_v18_transfer_inputs import read, sha, write

MODELS = ["Garl_full", "PHASE_ONLY_V16_S7", "FUSION_C1_V18_S7", "F_TRUE", "R_CTX", "C1_event"]


def sealed(directory: Path, registration: Path) -> list[dict[str, Any]]:
    """Reject incomplete, reordered or modified prediction fragments."""
    seal = read(directory / "SEALED.json")
    binding = sha(registration)
    if seal["registration_sha256"] != binding or len(seal["files"]) != seal["queries"]:
        raise ValueError("Invalid prediction seal")
    result = []
    for name, expected in sorted(seal["files"].items()):
        path = directory / name
        if sha(path) != expected:
            raise ValueError(f"Prediction changed: {path}")
        row = read(path)
        if row["registration_sha256"] != binding or row["targets_read"]:
            raise ValueError("Invalid prediction ancestry")
        result.append(row)
    return result


def match_target(table: np.ndarray, time_s: float) -> dict[str, Any]:
    """Nearest public TTC within 50 ms; preserve ties and exclusions explicitly."""
    if table.ndim != 2 or table.shape[1] != 5 or not len(table):
        raise ValueError("Expected five-column public TTC table")
    clock = table[:, 1]
    if not np.isfinite(clock).all() or (np.diff(clock) <= 0).any():
        raise ValueError("TTC timestamps must be strictly increasing and finite")
    insertion = int(np.searchsorted(clock, time_s))
    candidates = sorted({max(0, insertion - 1), min(len(clock) - 1, insertion)})
    index = min(candidates, key=lambda i: (abs(float(clock[i]) - time_s), i))
    delta = abs(float(clock[index]) - time_s)
    value = float(table[index, 4])
    reason = None
    if delta > 0.05 + 1e-12:
        reason = "TTC_TIMESTAMP_MISMATCH"
    elif not np.isfinite(value) or value == 0:
        reason = "TTC_INVALID_OR_ZERO"
    return {
        "target_ttc": value if reason is None else None,
        "target_timestamp_s": float(clock[index]),
        "target_delta_s": delta,
        "target_exclusion": reason,
    }


def summary(frame: pd.DataFrame, model: str) -> dict[str, Any]:
    """Report complete-cohort errors, with explicit conditional finite diagnostics."""
    target = frame.target_ttc.to_numpy(dtype=float)
    prediction = frame[model].to_numpy(dtype=float)
    if not np.isfinite(target).all() or (target == 0).any():
        raise ValueError("Invalid scoring targets")
    valid = np.isfinite(prediction)
    ae = np.abs(prediction[valid] - target[valid])
    rte = 100 * ae / np.abs(target[valid])
    conditional = {
        "RTE_pct": float(rte.mean()) if len(rte) else None,
        "MAE_s": float(ae.mean()) if len(ae) else None,
        "median_AE_s": float(np.median(ae)) if len(ae) else None,
        "p95_AE_s": float(np.quantile(ae, 0.95)) if len(ae) else None,
        "sign_error_pct": (
            float(100 * np.mean(np.sign(prediction[valid]) != np.sign(target[valid])))
            if valid.any()
            else None
        ),
    }
    return {
        "queries": len(frame),
        "finite_predictions": int(valid.sum()),
        "coverage": float(valid.mean()),
        "groups": int(frame.group_id.nunique()),
        "complete_cohort": conditional if valid.all() else None,
        "finite_only_conditional": conditional,
    }


def contrasts(frame: pd.DataFrame, pairs: list[tuple[str, str]]) -> list[dict[str, Any]]:
    """Paired equal-sequence macro contrasts, bootstrapping whole groups only."""
    rows = []
    for candidate, reference in pairs:
        if not np.isfinite(frame[[candidate, reference]].to_numpy(dtype=float)).all():
            rows.append({"candidate": candidate, "reference": reference, "status": "INCOMPLETE"})
            continue
        by_group = []
        for group, part in frame.groupby("group_id", sort=True):
            y = part.target_ttc.to_numpy(dtype=float)
            delta = np.abs(part[candidate].to_numpy() - y) - np.abs(part[reference].to_numpy() - y)
            by_group.append(
                {
                    "group": str(group),
                    "MAE_s": float(delta.mean()),
                    "RTE_pct": float(np.mean(100 * delta / np.abs(y))),
                }
            )
        rng = np.random.default_rng(7)
        indices = rng.integers(0, len(by_group), size=(2000, len(by_group)))
        result: dict[str, Any] = {
            "candidate": candidate,
            "reference": reference,
            "per_group": by_group,
            "estimand": "Equal-group macro candidate minus reference; negative favors candidate",
            "bootstrap": "2000 whole-group draws, seed7; exploratory, not a confirmation test",
        }
        for metric in ("MAE_s", "RTE_pct"):
            values = np.array([group[metric] for group in by_group])
            result[metric] = {
                "point": float(values.mean()),
                "CI95": np.quantile(values[indices].mean(1), [0.025, 0.975]).tolist(),
            }
        rows.append(result)
    return rows


def tables(frame: pd.DataFrame, models: list[str]) -> dict[str, Any]:
    """Produce pooled and sequence-specific tables without changing coverage."""
    return {
        "pooled": {model: summary(frame, model) for model in models},
        "per_group": {
            str(group): {model: summary(part, model) for model in models}
            for group, part in frame.groupby("group_id", sort=True)
        },
    }


def evttc() -> None:
    """Join public EvTTC targets only after both complete prediction seals."""
    verified()
    own = sealed(OUT / "own", OUT / "REGISTRATION.json")
    garl = sealed(OUT / "garl", OUT / "REGISTRATION.json")
    manifest = read(OUT / "QUERY_MANIFEST.json")
    rows, labels = [], {}
    for query, a, b in zip(manifest["rows"], own, garl, strict=True):
        if query["query_id"] != a["query_id"] or a["query_id"] != b["query_id"]:
            raise ValueError("Query pairing changed")
        path = Path(query["raw_path"]).parent / "ttc.csv"
        labels[str(path)] = sha(path)
        target = match_target(
            np.genfromtxt(path, dtype=np.float64, ndmin=2), query["anchor_us"] / 1e6
        )
        rows.append(
            {
                "query_id": query["query_id"],
                "group_id": query["sequence_id"],
                "anchor_s": query["anchor_us"] / 1e6,
                **target,
                **a["predictions"],
                **b["predictions"],
            }
        )
    frame = pd.DataFrame(rows)
    DOC.mkdir(parents=True, exist_ok=True)
    frame.to_csv(DOC / "EVTTC_QUERIES.csv", index=False)
    valid = cast(pd.DataFrame, frame.loc[frame.target_exclusion.isna(), :]).copy()
    result = tables(valid, MODELS)
    result.update(
        {
            "registration_sha256": sha(OUT / "REGISTRATION.json"),
            "prediction_seals": {r: sha(OUT / r / "SEALED.json") for r in ("own", "garl")},
            "label_sha256": labels,
            "sensor_queries": len(frame),
            "scored_queries": len(valid),
            "target_exclusions": cast(
                pd.DataFrame, frame.loc[frame.target_exclusion.notna(), :]
            ).to_dict(orient="records"),
            "target_range_s": [float(valid.target_ttc.min()), float(valid.target_ttc.max())],
            "negative_targets": int((valid.target_ttc < 0).sum()),
            "contrasts": contrasts(
                valid,
                [
                    ("FUSION_C1_V18_S7", "PHASE_ONLY_V16_S7"),
                    ("PHASE_ONLY_V16_S7", "Garl_full"),
                    ("FUSION_C1_V18_S7", "Garl_full"),
                ],
            ),
            "timing": {
                route: {
                    key: {
                        "median_s": float(np.median([row["seconds"][key] for row in data])),
                        "p95_s": float(np.quantile([row["seconds"][key] for row in data], 0.95)),
                    }
                    for key in data[0]["seconds"]
                }
                for route, data in (("own_shared_five_models", own), ("garl_standalone", garl))
            },
            "optimizer_updates": 0,
            "official_evaluation": False,
        }
    )
    write(DOC / "EVTTC_RESULTS.json", result)
    # This static scientific figure is regenerated solely from the scored table.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    for ax, (group, part) in zip(axes, valid.groupby("group_id", sort=True), strict=True):
        ax.plot(part.anchor_s, part.target_ttc, "k-", label="GT", linewidth=2)
        for model in MODELS[:4]:
            ax.plot(part.anchor_s, part[model], marker=".", label=model)
        ax.set(title=str(group), xlabel="Recording time (s)", ylabel="Signed TTC (s)")
        ax.grid(alpha=0.25)
    axes[-1].legend(fontsize=7)
    fig.savefig(DOC / "EVTTC_TRAJECTORIES.png", dpi=160)
    fig.savefig(DOC / "EVTTC_TRAJECTORIES.svg")
    plt.close(fig)
    print({model: result["pooled"][model]["complete_cohort"] for model in MODELS})


def internal_v() -> tuple[pd.DataFrame, dict[str, Any]]:
    """Re-score existing, sealed seed7 V predictions with exact one-to-one joins."""
    paths = {
        "PHASE_ONLY_V16_S7": RUN18 / "fusion/v16_compare/predictions/V/V16-PHASE-ONLY-S7",
        "FUSION_C1_V18_S7": RUN18 / "fusion/predictions/V/FUSION-TRUE-S7",
        "F_TRUE": RUN18 / "fusion/predictions/V/F_TRUE",
        "R_CTX": RUN18 / "fusion/predictions/V/R_CTX",
    }
    merged = None
    sources = {}
    for model, directory in paths.items():
        path = directory / "predictions.parquet"
        receipt = read(directory / "COMPLETE.json")
        if receipt["sha256"] != sha(path) or receipt["status"] != "COMPLETE":
            raise ValueError("Historical V predictions changed")
        sources[model] = {"path": str(path), "sha256": sha(path)}
        part = pd.read_parquet(path).rename(columns={"prediction_ttc": model})
        if len(part) != receipt["rows"] or part.sample_token.duplicated().any():
            raise ValueError("Invalid V query set")
        if merged is None:
            merged = part
        else:
            merged = merged.merge(
                part,
                on=["sample_token", "group_id"],
                suffixes=("", "_other"),
                how="outer",
                validate="one_to_one",
                indicator=True,
            )
            if not (merged["_merge"] == "both").all() or not np.allclose(
                merged.target_ttc, merged.target_ttc_other, atol=1e-5, rtol=0
            ):
                raise ValueError("V pairing/targets differ")
            merged = merged.drop(columns=["target_ttc_other", "_merge"])
    assert merged is not None
    result = {
        model: metrics(
            merged.target_ttc.to_numpy(), merged[model].to_numpy(), merged.group_id.to_numpy()
        )
        for model in paths
    }
    return merged, {
        "queries": len(merged),
        "groups": int(merged.group_id.nunique()),
        "seed": 7,
        "new_inference": False,
        "sources": sources,
        "metrics": result,
    }


def eap() -> None:
    """Score native Garl after sealing, and re-use exact historical V predictions."""
    directory = OUT / "eap"
    garl = sealed(directory, directory / "REGISTRATION.json")
    query = read(directory / "QUERY_MANIFEST.json")["rows"]
    for a, b in zip(query, garl, strict=True):
        if a["sample_token"] != b["sample_token"] or a["group_id"] != b["group_id"]:
            raise ValueError("eAP query identity changed")
    frame, result = internal_v()
    frame.to_csv(DOC / "EAP_V_EXISTING_PREDICTIONS.csv", index=False)
    write(DOC / "EAP_V_RESULTS.json", result)
    selected = pd.DataFrame(
        [{k: row[k] for k in ("sample_token", "group_id", "Garl_full")} for row in garl]
    )
    subset = selected.merge(
        frame, on=["sample_token", "group_id"], validate="one_to_one", how="left", indicator=True
    )
    if not (subset["_merge"] == "both").all():
        raise ValueError("Missing historical predictions")
    subset = subset.drop(columns="_merge")
    subset.to_csv(DOC / "EAP_64_QUERIES.csv", index=False)
    result = tables(subset, MODELS[:5])
    result.update(
        {
            "registration_sha256": sha(directory / "REGISTRATION.json"),
            "seal_sha256": sha(directory / "SEALED.json"),
            "upstream_bit_exact_checks": [
                row["upstream_bit_exact"] for row in garl if row["upstream_bit_exact"] is not None
            ],
            "independent_generalization_for_public_Garl": False,
            "public_train_overlap_groups": read(directory / "REGISTRATION.json")[
                "V_groups_in_public_train"
            ],
            "optimizer_updates": 0,
        }
    )
    write(DOC / "EAP_64_RESULTS.json", result)
    print({model: result["pooled"][model]["complete_cohort"] for model in MODELS[:5]})


def markdown_table(result: dict[str, Any], models: list[str]) -> list[str]:
    """Render computed errors without manually transcribing scientific numbers."""
    lines = [
        "| Sistema | RTE (%) | MAE (s) | Mediana AE (s) | p95 AE (s) |",
        "|---|---:|---:|---:|---:|",
    ]
    for model in models:
        values = result["pooled"][model]["complete_cohort"]
        if values is None:
            lines.append(f"| {model} | Incompleto | — | — | — |")
        else:
            lines.append(
                f"| {model} | {values['RTE_pct']:.3f} | {values['MAE_s']:.3f} | "
                f"{values['median_AE_s']:.3f} | {values['p95_AE_s']:.3f} |"
            )
    return lines


def report() -> None:
    """Publish reproducible tables and a small evidence bundle, never raw sensor data."""
    ev = read(DOC / "EVTTC_RESULTS.json")
    ea = read(DOC / "EAP_64_RESULTS.json")
    internal = read(DOC / "EAP_V_RESULTS.json")
    primary = ev["contrasts"][0]
    interval = primary["RTE_pct"]["CI95"]
    lines = [
        "# Comparación congelada posterior a V18 — 17-10-2026",
        "",
        "**Ejecutado:** inferencia nueva de seis sistemas sobre 46 consultas EvTTC; "
        "Garl full sobre 64 consultas eAP; recálculo de cuatro sistemas en las 16.432 consultas V "
        "ya guardadas. Cero optimizer updates, cero refit y ningún envío oficial. "
        "V18 conserva `NO_CANDIDATE`.",
        "",
        "La autorización posterior del usuario («haz esos análisis y comparaciones») habilita "
        "esta comparación congelada. El protocolo anterior era una propuesta no ejecutada. "
        "Esta reconstrucción EvTTC no se identifica con su comparación FCWD: FCWD usa otra "
        "población/cámara y no aporta un mapeo validado para el Garl full de esta comparación.",
        "",
        "## EvTTC: consultas emparejadas en los intervalos comunicados",
        "",
        "Pesos seed7 fijados; ningún ensemble ni selección de semilla. `PHASE_ONLY_EVENT_S7` "
        "es un sistema RGB+eventos: EVENT identifica la contribución de fase, no una modalidad "
        "exclusivamente de eventos. C1_event sí es event-only; R_CTX es RGB. "
        "F_TRUE, V18 y Garl full son RGB+eventos.",
        "",
        *markdown_table(ev, MODELS),
        "",
        "Menor error es mejor. RTE = 100·|pred−GT|/|GT|, media por consulta; MAE en segundos. "
        "Cobertura de predicción y GT: 46/46 en todos los sistemas. Sin eliminar outliers, "
        "sin clipping añadido y sin calibrar los pesos o la conversión tras ver GT.",
        "",
        "**F_TRUE histórico obtiene el menor RTE y MAE de esta cohorte.** V16 y V18 mejoran "
        "a Garl full en los tres grupos, pero V18 sólo mejora a V16 en CCRm; empeora "
        "ligeramente en los dos CCRs2. No hay un reemplazo universal demostrado.",
        "",
        "| Escenario local | N | Garl RTE | V16 RTE | V18 RTE | F_TRUE RTE |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for group, models in ev["per_group"].items():
        scores = [models[m]["complete_cohort"]["RTE_pct"] for m in MODELS[:4]]
        lines.append(
            f"| {group} | {models['Garl_full']['queries']} | "
            + " | ".join(f"{value:.3f}" for value in scores)
            + " |"
        )
    lines += [
        "",
        f"El contraste macro por grupo V18−V16 es {primary['RTE_pct']['point']:.3f} puntos "
        f"RTE; IC95% bootstrap [{interval[0]:.3f}, {interval[1]:.3f}]. "
        "Se remuestrean tres secuencias completas, 2.000 veces, seed7. El intervalo incluye cero. "
        "Es exploratorio: sólo tres grupos, ventanas correlacionadas, desarrollo EvTTC "
        "previamente observado y varios contrastes; no es confirmación independiente. "
        "La media macro del contraste y la media por consulta de la tabla "
        "son estimandos distintos.",
        "",
        f"Los targets cubren {ev['target_range_s'][0]:.3f}–{ev['target_range_s'][1]:.3f} s; "
        "todos son positivos. Esta cohorte no prueba precisión negativa, TTC largo, "
        "receding ni la corrección de A1-FULL. Un error de signo nulo aquí tiene alcance limitado.",
        "",
        "![Trayectorias TTC](EVTTC_TRAJECTORIES.png)",
        "",
        "## eAP: comparación descriptiva, no test independiente de Garl",
        "",
        "Ocho IDs equiespaciados en el orden lexicográfico de sample_token de cada grupo V: "
        "64 consultas fijadas sin seleccionar por GT, banda o error. Garl se ejecutó de nuevo; "
        "los otros cuatro sistemas reutilizan sus predicciones exactas selladas sobre esos IDs. "
        "En la primera consulta de cada grupo se comprobó igualdad bit a bit entre sensores "
        "del preprocesado local y el upstream: ocho comprobaciones satisfechas.",
        "",
        *markdown_table(ea, MODELS[:5]),
        "",
        "Garl full gana en RTE y MAE en este subconjunto. Los ocho grupos V figuran en "
        "`E:/Garl-TTC/configs/splits/train.txt` (46 IDs publicados). Por ello la independencia "
        "para Garl no está establecida; ese listado tampoco certifica por sí solo toda la "
        "genealogía del checkpoint publicado. No extrapolar las 64 consultas a todo eAP. "
        "No hemos sustituido el subconjunto por otro más favorable.",
        "",
        "## V completa: predicciones existentes, misma semilla 7",
        "",
        "| Sistema | Agregado MiD | MiDc | MiDs | MiDl | MiDn | MAE (s) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for model, value in internal["metrics"].items():
        values = [
            value["overall_MiD"],
            *[
                value["bins"][band]["group_macro_mid"]
                for band in ("crucial", "small", "large", "negative")
            ],
            value["diagnostics"]["mae_s"],
        ]
        lines.append(f"| {model} | " + " | ".join(f"{v:.3f}" for v in values) + " |")
    lines += [
        "",
        "16.432 consultas y ocho grupos, sin inferencias nuevas. Estos números son "
        "seed7, no las medias de tres semillas del cierre V18. V18 mejora agregado y MAE "
        "frente a V16, pero empeora MiDs, MiDl y MiDn. V sigue siendo desarrollo observado. "
        "No comparar estos MiD con los RTE EvTTC ni con scores de H.",
        "",
        "## Alcance y decisión",
        "",
        "La comparación nueva sostiene conservar V18 como alternativa; no sostiene sustituir "
        "V16 ni descartar F_TRUE. Se mantiene el candidato oficial V16 congelado. "
        "Para conocer la posición oficial falta un recibo de CodaBench y resolver elegibilidad "
        "de modalidad e historial; esta tarea no envía nada ni demuestra SOTA. No se consultó "
        "GT privado ni se usaron estos resultados para ajustar pesos, semillas o clipping.",
        "",
        "El [diagnóstico A1 anterior](../post_v18_20261017/A1_NEGATIVE_ANALYSIS.md) conserva "
        "la descomposición de dirección, magnitud y saturación sobre P. Esta evaluación "
        "no ejecuta A1-FULL ni una ablación de gradientes y no cambia esa conclusión.",
        "",
        "La latencia registrada es CPU FP32, dos hilos: Garl tiene una ruta individual; "
        "la ruta propia calcula cinco sistemas con productores compartidos. No son latencias "
        "comparables de dos productos ni reproducen el hardware de la Tabla VI. Las cifras "
        "por fase permanecen en EVTTC_RESULTS.json; no se afirma tiempo real.",
        "",
        "## Evidencia y reproducción",
        "",
        "[Auditoría del correo y diferencias](AUTHOR_PROTOCOL_AUDIT.md), "
        "[EvTTC JSON](EVTTC_RESULTS.json), [consultas EvTTC](EVTTC_QUERIES.csv), "
        "[eAP-64 JSON](EAP_64_RESULTS.json), [consultas eAP-64](EAP_64_QUERIES.csv), "
        "[V completa JSON](EAP_V_RESULTS.json), [V completa CSV](EAP_V_EXISTING_PREDICTIONS.csv). "
        "Los registros previos, manifiestos, recibos y predicciones nuevas se conservan "
        "en `evidence/`; no contiene datos crudos ni checkpoints.",
        "",
        "Desde la raíz, con el entorno local del proyecto y `PYTHONPATH=src;.`:",
        "",
        "```powershell",
        "python -B -m scripts.post_v18_transfer_report evttc",
        "python -B -m scripts.post_v18_transfer_report eap",
        "python -B -m scripts.post_v18_transfer_report report",
        "```",
        "",
        "Esto sólo vuelve a puntuar predicciones selladas y genera documentación. Para "
        "reproducir inferencia en otro entorno, respetar las rutas locales declaradas en "
        "REGISTRATION y los SHA de padres; los runners `post_v18_transfer_compare` y "
        "`post_v18_eap_compare` rechazan registros incompatibles y reanudan fragmentos "
        "completos sin sobrescribirlos. No ejecutar `register` sobre un registro existente. "
        "El checkout necesita los artefactos de las ramas hermanas y los datasets locales; "
        "la evidencia publicada por sí sola permite inspección, no reconstruye los sensores.",
        "",
    ]
    (DOC / "REPORT.md").write_text("\n".join(lines), encoding="utf-8", newline="\n")
    for path in OUT.rglob("*.json"):
        if "runtime_deps" in path.parts:
            continue
        target = DOC / "evidence" / path.relative_to(OUT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
    import torch
    import torchvision

    write(
        DOC / "EXECUTION_RECEIPT.json",
        {
            "status": "COMPLETE_FROZEN_DESCRIPTIVE_COMPARISON",
            "optimizer_updates": 0,
            "refit": False,
            "submission": False,
            "V18": "CLOSED_NO_CANDIDATE",
            "git_base": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
            "generated_utc": datetime.now(UTC).isoformat(),
            "host": socket.gethostname(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torchvision": torchvision.__version__,
            "numpy": np.__version__,
            "device": "CPU FP32, torch threads=2",
            "GPU_used": False,
            "opencv": "4.14.0.94; isolated runtime_deps, no global/shared environment modification",
            "seal_write_times_utc": {
                route: datetime.fromtimestamp(
                    (OUT / route / "SEALED.json").stat().st_mtime, UTC
                ).isoformat()
                for route in ("own", "garl", "eap")
            },
            "sources": {
                str(path.relative_to(ROOT)): sha(path)
                for path in sorted((ROOT / "scripts").glob("post_v18_*compare.py"))
                + sorted((ROOT / "scripts").glob("post_v18_transfer_*.py"))
            },
            "checks": {
                "targeted_tests": "6 passed",
                "ruff": "passed",
                "pyright": "0 errors",
                "eap_upstream_bit_exact": "8/8",
                "sealed_outputs": "46 own + 46 Garl + 64 eAP",
            },
            "technical_corrections": [
                "Pre-inference corrections retained; no weights or cohort changes",
                "Missing OpenCV resolved with the lockfile version in an isolated local target",
                "Scorer fractional-second matching corrected and regression-tested; "
                "no new inference",
            ],
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("evttc", "eap", "report"))
    args = parser.parse_args()
    {"evttc": evttc, "eap": eap, "report": report}[args.action]()

"""Full-cohort, clock-bound query comparisons; no model selection or learned routing."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from operational.range_transfer.common import digest, read, sha, utc, write

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "artifacts/joint_measurement_20261012"
MODELS = {
    "A5": ("A5-P2-FROZEN", ""),
    "NATIVE": ("NATIVE-P2-FT-S7", ""),
    "CORR": ("PAIR-CORR-S7", ""),
    "C1": ("PAIR-CORR-GEOMETRY-S7", "adaptive/C1_geometry"),
}
CLOCK_COLUMNS = [
    "query_time_us",
    "cutoff_us",
    "event_previous_start_us",
    "event_previous_end_us",
    "event_current_start_us",
    "event_current_end_us",
    "native_delta_t_s",
    "metric_delta_t_s",
]


def validate_clock_bindings(identities: dict, role: str, sensor_sha: str) -> None:
    """Require every prediction to bind the same native sensor cohort and role."""
    if set(identities) != set(MODELS):
        raise ValueError("Four named model identities required")
    for model, identity in identities.items():
        if (
            identity.get("sensor_complete") != sensor_sha
            or identity.get("role") != role
            or identity.get("fit_id") != MODELS[model][0]
        ):
            raise ValueError(f"Incompatible clock/cohort identity: {model}")


def join_queries(metadata: pd.DataFrame, predictions: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Reject duplicates, missing/extra queries and inconsistent metadata; never intersect."""
    if set(predictions) != set(MODELS):
        raise ValueError("Four prediction populations required")
    if metadata.sample_token.isna().any() or metadata.sample_token.duplicated().any():
        raise ValueError("Duplicate/null native query in metadata")
    if pd.DataFrame(metadata[["group_id", "sequence_id", "role"]]).isna().to_numpy().any():
        raise ValueError("Null group, sequence or role in metadata")
    if not set(CLOCK_COLUMNS).issubset(metadata.columns):
        raise ValueError("Missing native clock fields")
    panel = metadata.set_index("sample_token").copy()
    for column in CLOCK_COLUMNS:
        if not np.isfinite(panel[column].to_numpy(dtype=float)).all():
            raise ValueError("Nonfinite native clock")
    if not (
        (panel.event_previous_end_us > panel.event_previous_start_us).all()
        and (panel.event_current_end_us > panel.event_current_start_us).all()
        and (panel.event_current_start_us >= panel.event_previous_end_us).all()
        and (panel.native_delta_t_s > 0).all()
        and (panel.metric_delta_t_s == 0.1).all()
        and (panel.cutoff_us <= panel.query_time_us).all()
    ):
        raise ValueError("Incompatible native clock intervals")
    truth = panel.target_ttc.to_numpy(dtype=float)
    if not (
        np.isfinite(truth) & (((truth > 0.1) & (truth <= 10)) | ((truth > -10) & (truth < 0)))
    ).all():
        raise ValueError("Invalid target population")
    panel["target_phase_recomputed"] = -np.log1p(-0.1 / truth)
    for model, frame in predictions.items():
        if frame.sample_token.isna().any() or frame.sample_token.duplicated().any():
            raise ValueError(f"Duplicate/null native query: {model}")
        incoming = frame.set_index("sample_token")
        if len(incoming) != len(panel) or set(incoming.index) != set(panel.index):
            raise ValueError(f"Different query population: {model}; no intersection permitted")
        incoming = incoming.loc[panel.index]
        for key in ["target_ttc", "group_id", "sequence_id", "role", *CLOCK_COLUMNS]:
            if key in incoming and not np.array_equal(
                incoming[key].to_numpy(), panel[key].to_numpy()
            ):
                raise ValueError(f"Incompatible {key}: {model}")
        for key in ("target_ttc", "group_id", "sequence_id", "role"):
            if key not in incoming:
                raise ValueError(f"Missing identity column: {key}")
        pred = incoming.prediction_ttc.to_numpy(dtype=float)
        valid = np.isfinite(pred) & ((pred < 0) | (pred > 0.1))
        phase = np.full(len(pred), np.nan)
        phase[valid] = -np.log1p(-0.1 / pred[valid])
        panel[f"{model}_ttc"] = pred
        panel[f"{model}_phase_saved"] = incoming.raw_phase.to_numpy(dtype=float)
        panel[f"{model}_phase_emitted"] = phase
        panel[f"{model}_valid"] = valid
        panel[f"{model}_sign_correct"] = valid & (np.signbit(pred) == np.signbit(truth))
        panel[f"{model}_MiD"] = 10000 * np.abs(phase - panel.target_phase_recomputed)
        panel[f"{model}_AE_s"] = np.abs(pred - truth)
    panel["negative"] = truth < 0
    panel["crucial"] = (truth > 0.1) & (truth <= 3)
    panel["negative_CORR_sign_error"] = panel.negative & ~panel.CORR_sign_correct
    for model in ("A5", "NATIVE", "C1"):
        panel[f"{model}_minus_CORR_MiD"] = panel[f"{model}_MiD"] - panel.CORR_MiD
        panel[f"{model}_minus_CORR_AE_s"] = panel[f"{model}_AE_s"] - panel.CORR_AE_s
        panel[f"crucial_CORR_beats_{model}"] = panel.crucial & (
            panel[f"{model}_minus_CORR_MiD"] > 0
        )
    panel["crucial_CORR_beats_A5_and_NATIVE"] = (
        panel.crucial_CORR_beats_A5 & panel.crucial_CORR_beats_NATIVE
    )
    return panel.reset_index()


def summarize(panel: pd.DataFrame, role: str, group: str = "ALL") -> list[dict]:
    """Describe paired outcomes on fixed sets; sign correctness is not TTC accuracy."""
    result = []
    subsets = {
        "negative_all": panel.negative,
        "negative_CORR_sign_error": panel.negative_CORR_sign_error,
        "crucial_all": panel.crucial,
        "crucial_CORR_beats_A5": panel.crucial_CORR_beats_A5,
        "crucial_CORR_beats_NATIVE": panel.crucial_CORR_beats_NATIVE,
        "crucial_CORR_beats_A5_and_NATIVE": panel.crucial_CORR_beats_A5_and_NATIVE,
    }
    for label, mask in subsets.items():
        part = panel.loc[mask]
        for model in MODELS:
            valid = bool(part[f"{model}_valid"].all())
            row = {
                "role": role,
                "group": group,
                "subset": label,
                "model": model,
                "rows": len(part),
                "invalid": int((~part[f"{model}_valid"]).sum()),
                "sign_correct": int(part[f"{model}_sign_correct"].sum()),
                "sign_correct_rate": float(part[f"{model}_sign_correct"].mean())
                if len(part)
                else None,
            }
            for metric in ("MiD", "AE_s"):
                values = part[f"{model}_{metric}"]
                difference = values - part[f"CORR_{metric}"]
                complete = valid and bool(part.CORR_valid.all()) and len(part) > 0
                row[f"mean_{metric}"] = float(values.mean()) if valid and len(part) else None
                row[f"mean_delta_{metric}"] = float(difference.mean()) if complete else None
                row[f"median_delta_{metric}"] = float(difference.median()) if complete else None
                row[f"p95_delta_{metric}"] = float(difference.quantile(0.95)) if complete else None
                row[f"worse_than_CORR_{metric}_count"] = int((difference > 0).sum())
                row[f"better_than_CORR_{metric}_count"] = int((difference < 0).sum())
                row[f"tie_with_CORR_{metric}_count"] = int((difference == 0).sum())
            result.append(row)
    return result


def build() -> dict:
    """Admit original metadata and clocks, then publish complete P/H query panels."""
    cfg = read(ROOT / "configs/joint_measurement/primary.json")
    owner = (ROOT / cfg["owner_relative"]).resolve()
    output = RUN / "diagnostics/query_panel"
    output.mkdir(parents=True, exist_ok=True)
    bindings, summaries, contingencies, critical_contingencies, outputs = {}, [], [], [], {}
    role_ids, role_groups = {}, {}
    primary = read(RUN / "primary_evidence/COMPLETE.json")
    c1 = read(RUN / "adaptive/C1_geometry/evidence/COMPLETE.json")
    admission = read(RUN / "ADMISSION.json")
    for role in ("P", "H"):
        manifest_path = owner / f"{role}_MANIFEST.json"
        manifest = read(manifest_path)
        rows_path = Path(manifest["rows_path"])
        sensor = Path(cfg["cache_root"]) / "sensor" / role
        sensor_binding, complete = read(sensor / "BINDING.json"), read(sensor / "COMPLETE.json")
        sensor_sha = sha(sensor / "COMPLETE.json")
        if (
            complete["status"] != "COMPLETE"
            or complete["binding"] != digest(sensor_binding)
            or sensor_binding["manifest_sha256"] != sha(manifest_path)
            or manifest["rows_sha256"] != sha(rows_path)
        ):
            raise ValueError("Native cohort/sensor metadata admission failed")
        if sha(manifest_path) not in admission["files"].values():
            raise ValueError("Role manifest not admitted by V18")
        rows = pd.read_parquet(rows_path)
        if set(rows.role) != {role}:
            raise ValueError("Role metadata contains a different population role")
        if len(rows) != complete["rows"] or len(rows) != manifest["population_size"]:
            raise ValueError("Native population size changed")
        source = sensor_binding["source_identity"]
        index_path = Path(source["event_source_root"]) / "TRAIN40_INDEX.npz"
        native_rows_path = index_path.with_name("TRAIN40_ROWS.parquet")
        if sha(native_rows_path) != source["train40_rows_sha256"]:
            raise ValueError("Native query clock metadata changed")
        native_rows = pd.read_parquet(native_rows_path, columns=["sample_token", "timestamp_us"])
        if native_rows.sample_token.duplicated().any():
            raise ValueError("Duplicate native query clock")
        native_clock = native_rows.set_index("sample_token").reindex(rows.sample_token).timestamp_us
        if not np.array_equal(native_clock.to_numpy(), rows.query_time_us.to_numpy()):
            raise ValueError("Query anchor differs from native query clock")
        if (
            sha(index_path) != source["train40_index_sha256"]
            or source["target_anchor"] != "query_time_us"
        ):
            raise ValueError("Native clock index or anchor changed")
        with np.load(index_path, allow_pickle=False) as index:
            tokens = pd.Index(index["tokens"].astype(str))
            if tokens.has_duplicates:
                raise ValueError("Duplicate query in native event index")
            positions = tokens.get_indexer(rows.sample_token)
            if (positions < 0).any():
                raise ValueError("Native event index lacks query; no dropping permitted")
            windows = index["windows_us"][positions, -2:]
            delta = index["delta_t_s"][positions]
            if windows.dtype.kind not in "iu" or windows.shape != (len(rows), 2, 2):
                raise ValueError("Invalid native sensor clock shape/type")
            if not np.array_equal(index["sequences"][positions].astype(str), rows.sequence_id):
                raise ValueError("Native event sequence differs")
            if not np.array_equal(index["ttc_s"][positions], rows.target_ttc):
                raise ValueError("Native indexed target differs")
        metadata = pd.DataFrame(
            rows[
                [
                    "sample_token",
                    "sequence_id",
                    "group_id",
                    "track_id",
                    "role",
                    "target_ttc",
                    "target_phase",
                    "query_time_us",
                    "cutoff_us",
                ]
            ]
        ).copy()
        for offset, name in enumerate(("previous", "current")):
            metadata[f"event_{name}_start_us"] = windows[:, offset, 0]
            metadata[f"event_{name}_end_us"] = windows[:, offset, 1]
        metadata["native_delta_t_s"] = delta
        metadata["metric_delta_t_s"] = 0.1
        metadata["event_end_separation_us"] = windows[:, 1, 1] - windows[:, 0, 1]
        predictions, identities, model_bindings = {}, {}, {}
        for alias, (name, relative) in MODELS.items():
            folder = RUN / relative / "predictions" / role / name
            path = folder / "predictions.parquet"
            receipt, identity = read(folder / "COMPLETE.json"), read(folder / "IDENTITY.json")
            proof = (c1 if alias == "C1" else primary)["verification"][f"{role}/{name}"]
            cohort = (c1["cohorts"] if alias == "C1" else primary["binding"]["cohorts"])[
                f"{role}/{name}"
            ]
            if (
                receipt["status"] != "COMPLETE"
                or receipt["binding"] != digest(identity)
                or receipt["sha256"] != sha(path)
                or receipt["sha256"] != cohort
                or proof["status"] != "PASS"
                or proof["metrics_sha256"] != sha(folder / "METRICS.json")
            ):
                raise ValueError(f"Verified prediction binding changed: {role}/{name}")
            predictions[alias] = pd.read_parquet(path)
            identities[alias] = identity
            model_bindings[alias] = {"predictions_sha256": sha(path), "identity": identity}
        validate_clock_bindings(identities, role, sensor_sha)
        panel = join_queries(metadata, predictions)
        role_ids[role] = set(panel.sample_token)
        role_groups[role] = set(panel.group_id)
        path = output / f"{role}_QUERY_PANEL.parquet"
        pending = path.with_suffix(".parquet.pending")
        panel.to_parquet(pending, index=False, compression="zstd", compression_level=9)
        os.replace(pending, path)
        outputs[role] = {
            "path": path.relative_to(ROOT).as_posix(),
            "sha256": sha(path),
            "rows": len(panel),
            "columns": list(panel.columns),
        }
        summaries.extend(summarize(panel, role))
        for group, part in panel.groupby("group_id", sort=True):
            summaries.extend(summarize(part, role, str(group)))
        crucial = panel.loc[panel.crucial]
        outcomes = {}
        for model in ("A5", "NATIVE"):
            difference = crucial[f"{model}_minus_CORR_MiD"].to_numpy()
            outcomes[model] = np.select(
                [difference > 0, difference < 0, difference == 0],
                ["CORR_better", "CORR_worse", "tie"],
                default="invalid",
            )
        for a5 in ("CORR_better", "CORR_worse", "tie", "invalid"):
            for native in ("CORR_better", "CORR_worse", "tie", "invalid"):
                critical_contingencies.append(
                    {
                        "role": role,
                        "versus_A5": a5,
                        "versus_NATIVE": native,
                        "rows": int(
                            ((outcomes["A5"] == a5) & (outcomes["NATIVE"] == native)).sum()
                        ),
                    }
                )
        negative = panel.loc[panel.negative]
        for a5 in (False, True):
            for native in (False, True):
                for corr in (False, True):
                    for geometry in (False, True):
                        mask = (
                            (negative.A5_sign_correct == a5)
                            & (negative.NATIVE_sign_correct == native)
                            & (negative.CORR_sign_correct == corr)
                            & (negative.C1_sign_correct == geometry)
                        )
                        contingencies.append(
                            {
                                "role": role,
                                "A5_sign_correct": a5,
                                "NATIVE_sign_correct": native,
                                "CORR_sign_correct": corr,
                                "C1_sign_correct": geometry,
                                "rows": int(mask.sum()),
                            }
                        )
        bindings[role] = {
            "manifest_sha256": sha(manifest_path),
            "rows_sha256": sha(rows_path),
            "sensor_complete_sha256": sensor_sha,
            "sensor_binding": sensor_binding,
            "native_index_sha256": sha(index_path),
            "models": model_bindings,
        }
    if role_ids["P"] & role_ids["H"] or role_groups["P"] & role_groups["H"]:
        raise ValueError("P/H query or group overlap")
    pd.DataFrame(summaries).to_csv(output / "PAIRED_SUMMARIES.csv", index=False)
    pd.DataFrame(contingencies).to_csv(output / "NEGATIVE_SIGN_CONTINGENCY.csv", index=False)
    pd.DataFrame(critical_contingencies).to_csv(output / "CRUCIAL_MID_CONTINGENCY.csv", index=False)
    result = {
        "status": "COMPLETE_DESCRIPTIVE_QUERY_PANEL",
        "utc": utc(),
        "bindings": bindings,
        "outputs": outputs,
        "rows": sum(v["rows"] for v in outputs.values()),
        "optimizer_updates": 0,
        "V_opened": False,
        "selection_changed": False,
        "source_sha256": sha(Path(__file__)),
        "clock_scope": "Same bound sensor cohort for all four models; query/annotation clock "
        "and native event sensor clock retained separately. Native model delta "
        "and fixed metric delta are not inferred by subtracting different clocks.",
        "saved_phase_scope": "A5/NATIVE saved raw_phase is phase recomputed from native TTC; "
        "CORR/C1 saved raw_phase is actual pre-emission head output.",
        "interpretation": "Outcome-conditioned descriptive sets; no inference-time selector, "
        "no evidence that the winning model is predictable without targets",
    }
    write(output / "RESULT.json", result)
    publish_report(result, summaries, contingencies, critical_contingencies)
    print(json.dumps({"status": result["status"], "rows": result["rows"]}))
    return result


def publish_report(
    result: dict, summaries: list[dict], negative: list[dict], critical: list[dict]
) -> None:
    """Publish group tables and outcome-conditioned results, including shared failures."""
    folder = ROOT / "docs/joint_measurement_20261012"
    evidence = folder / "evidence_query_panel_20261016"
    write(evidence / "RESULT.json", result)
    pd.DataFrame(summaries).to_csv(evidence / "PAIRED_SUMMARIES.csv", index=False)
    pd.DataFrame(negative).to_csv(evidence / "NEGATIVE_SIGN_CONTINGENCY.csv", index=False)
    pd.DataFrame(critical).to_csv(evidence / "CRUCIAL_MID_CONTINGENCY.csv", index=False)
    lines = [
        "# Panel de consultas V18: complementariedad y fallos compartidos",
        "",
        "Se unen los cuatro endpoints verificados por `sample_token`, sin intersección "
        "selectiva: 53.890 consultas P y 18.422 H. P es entrenamiento; H es desarrollo "
        "del productor. No se abre V, no se entrena un selector ni se cambia la selección.",
        "",
        "## Admisión y relojes",
        "",
        "El constructor rechaza identificadores duplicados o nulos, "
        "consultas ausentes/adicionales, "
        "targets/grupos/secuencias/roles distintos y bindings sensoriales incompatibles. "
        "Verifica los SHA-256 originales de las ocho predicciones, manifiestos, filas y "
        "TRAIN40_INDEX; contrasta el reloj de consulta contra TRAIN40_ROWS. "
        "Los cuatro modelos comparten exactamente el mismo sensor_complete por rol. "
        "No se releyeron todos los tensores sensoriales: la garantía se apoya en sus "
        "recibos de admisión y verificación previos, conservados en RESULT.json.",
        "",
        "Se conservan query_time_us/cutoff_us (reloj de consulta), los inicios/finales de "
        "las dos ventanas event (reloj del sensor), su separación real, el delta nativo "
        "de entrada y el delta de métrica. No se confunden relojes ni se infiere el "
        "delta por restar timestamps de dominios distintos. Las tablas originales "
        "de predicciones no contienen esos tiempos: se adjuntan desde su genealogía "
        "común verificada. Si cambian las fuentes o un binding, se rechaza el panel.",
        "",
        "Se guardan target, fase target original y recalculada, TTC y fases de cada modelo. "
        "En A5/NATIVE, la columna histórica raw_phase fue calculada desde TTC emitido; "
        "en CORR/C1 sí es la salida anterior a la conversión. El panel distingue "
        "phase_saved y phase_emitted; no atribuye al nativo una fase interna no exportada.",
        "",
        "## Negativos concretos donde CORR equivoca el signo",
        "",
        "Aquí acertar significa únicamente recuperar el signo negativo. Se informa "
        "también el error en segundos: un signo correcto no garantiza un TTC cercano.",
        "",
        "| Rol | Casos | Modelo | Signo corregido | Corrección % | Delta MAE frente a CORR, s |",
        "|---|---:|---|---:|---:|---:|",
    ]
    for row in summaries:
        if (
            row["group"] == "ALL"
            and row["subset"] == "negative_CORR_sign_error"
            and row["model"] != "CORR"
        ):
            lines.append(
                f"| {row['role']} | {row['rows']} | {row['model']} | {row['sign_correct']} | "
                f"{100 * row['sign_correct_rate']:.2f} | {row['mean_delta_AE_s']:+.3f} |"
            )
    lines += ["", "Valores negativos del delta significan menos error que CORR.", ""]
    for role in ("P", "H"):
        rows = [r for r in negative if r["role"] == role and not r["CORR_sign_correct"]]
        both = sum(r["rows"] for r in rows if r["A5_sign_correct"] and r["NATIVE_sign_correct"])
        neither = sum(
            r["rows"] for r in rows if not r["A5_sign_correct"] and not r["NATIVE_sign_correct"]
        )
        all_fail = sum(r["rows"] for r in rows if not any(r[f"{m}_sign_correct"] for m in MODELS))
        lines.append(
            f"- {role}: A5 y NATIVE corrigen simultáneamente {both} casos; "
            f"ambos siguen fallando en {neither}; los cuatro fallan en {all_fail}."
        )
    lines += [
        "",
        "## Casos críticos donde CORR mejora",
        "",
        "Se define mejora por menor MiD individual, sin umbral añadido. "
        "La banda crítica es 0,1 < TTC <= 3 s. Se conserva el contraste completo "
        "y los subsets separados contra A5, NATIVE y ambos en los CSV.",
        "",
        "| Rol | CORR mejor que ambos | Total crítico | CORR peor que ambos |",
        "|---|---:|---:|---:|",
    ]
    for role in ("P", "H"):
        rows = [r for r in critical if r["role"] == role]
        better = sum(
            r["rows"] for r in rows if r["versus_A5"] == r["versus_NATIVE"] == "CORR_better"
        )
        worse = sum(r["rows"] for r in rows if r["versus_A5"] == r["versus_NATIVE"] == "CORR_worse")
        lines.append(f"| {role} | {better} | {sum(r['rows'] for r in rows)} | {worse} |")
    lines += [
        "",
        "En los casos donde CORR gana a ambos:",
        "",
        "| Rol | Modelo | Delta MiD medio | Delta MAE, s |",
        "|---|---|---:|---:|",
    ]
    for row in summaries:
        if (
            row["group"] == "ALL"
            and row["subset"] == "crucial_CORR_beats_A5_and_NATIVE"
            and row["model"] != "CORR"
        ):
            lines.append(
                f"| {row['role']} | {row['model']} | {row['mean_delta_MiD']:+.3f} | "
                f"{row['mean_delta_AE_s']:+.3f} |"
            )
    lines += [
        "",
        "## Interpretación y reproducción",
        "",
        "Existe complementariedad observada por consulta, además de fallos compartidos. "
        "Estas particiones usan GT para describir los resultados: no son una regla "
        "disponible en inferencia ni prueban que pueda anticiparse qué modelo ganará. "
        "No se presenta un selector oráculo como sistema alcanzable. Los promedios "
        "anteriores son pooled dentro de cada subset, no el agregado oficial ni "
        "un intervalo estadístico. Los resultados por grupo están incluidos; "
        "las ventanas de una misma secuencia no se consideran réplicas independientes.",
        "",
        "Ejecutar `python -B -m operational.joint_measurement.query_panel` con "
        "`PYTHONPATH=src;.`. Requiere las predicciones y metadatos admitidos. "
        "Los paneles completos P_QUERY_PANEL.parquet y H_QUERY_PANEL.parquet "
        "están en `artifacts/joint_measurement_20261012/diagnostics/query_panel/`, "
        "con compresión Zstandard sin pérdida. RESULT.json liga sus tamaños de "
        "población, columnas y hashes. No se eliminan predicciones inválidas: "
        "se marcan y los promedios de métricas incompletas quedan indefinidos.",
        "",
        "[Bindings y paneles](evidence_query_panel_20261016/RESULT.json) · "
        "[Contrastes y grupos](evidence_query_panel_20261016/PAIRED_SUMMARIES.csv) · "
        "[Contingencia de signo](evidence_query_panel_20261016/NEGATIVE_SIGN_CONTINGENCY.csv) · "
        "[Contingencia crítica](evidence_query_panel_20261016/CRUCIAL_MID_CONTINGENCY.csv)",
        "",
    ]
    (folder / "QUERY_COMPLEMENTARITY.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    build()

"""Describe the registered PAIR objective on fixed P endpoints, without updates."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from e_jepa_ttc.joint_measurement.loss import phase_components, training_mass
from e_jepa_ttc.simplex_t.phase import phase_to_ttc
from operational.range_transfer.common import sha, utc, write

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "artifacts/joint_measurement_20261012"
SOURCES = {
    "PAIR-DIRECT-S7": RUN,
    "PAIR-CORR-S7": RUN,
    "PAIR-CORR-FROZEN-S7": RUN,
    "PAIR-CORR-GEOMETRY-S7": RUN / "adaptive/C1_geometry",
    "PAIR-CORR-WIDE-S7": RUN / "adaptive/D1_predictor_width",
    "RGB-PAIR-CORR-FT-S7": RUN / "rgb",
    "RGB-PAIR-CORR-FROZEN-S7": RUN / "rgb",
}


def build() -> dict:
    """Reconstruct endpoint loss contributions; no model backward or selection."""
    rows, bindings, checks = [], {}, {}
    aligned = None
    for name, directory in SOURCES.items():
        folder = directory / "predictions/P" / name
        path = folder / "predictions.parquet"
        receipt = json.loads((folder / "COMPLETE.json").read_text())
        fingerprint = sha(path)
        if receipt["status"] != "COMPLETE" or receipt["sha256"] != fingerprint:
            raise ValueError(f"Prediction admission failed: {name}")
        table = pd.read_parquet(path).sort_values("sample_token").reset_index(drop=True)
        identity = table[["sample_token", "group_id", "target_ttc", "role"]]
        if aligned is None:
            aligned = identity
        elif not identity.equals(aligned):
            raise ValueError("P populations or targets differ")
        if not table.role.eq("P").all() or table.sample_token.duplicated().any():
            raise ValueError("Only unique P queries admitted")
        target = table.target_ttc.to_numpy(dtype=np.float64)
        prediction = table.prediction_ttc.to_numpy(dtype=np.float64)
        raw = table.raw_phase.to_numpy(dtype=np.float64, copy=True)
        groups = table.group_id.to_numpy(dtype=str)
        masses = training_mass(target, groups, role="P")
        masks = [
            (target > 0.1) & (target <= 3),
            (target > 3) & (target <= 6),
            (target > 6) & (target <= 10),
            (target > -10) & (target < 0),
        ]
        # Independent group-band count reconstruction of the producer masses.
        band_id = np.select(masks, [0, 1, 2, 3], default=-1)
        counts = pd.crosstab(groups, band_id).reindex(columns=range(4), fill_value=0)
        nominal = np.array([0.5, 0.3, 0.1, 0.1])
        active = (counts.to_numpy() > 0) * nominal
        cells = active / active.sum(axis=1, keepdims=True) / len(counts)
        group_index = counts.index.get_indexer(groups)
        independent = cells[group_index, band_id] / counts.to_numpy()[group_index, band_id]
        np.testing.assert_allclose(masses, independent, rtol=1e-13, atol=1e-15)
        if not (np.isfinite(raw).all() and np.isfinite(prediction).all()):
            raise ValueError("Nonfinite endpoint values")
        phi = -np.log1p(-0.1 / target)
        emitted = -np.log1p(-0.1 / prediction)
        emitted_error = np.abs(emitted - phi) / 0.03
        raw_error = 0.1 * np.abs(raw - phi) / 0.03
        # Float64 reconstruction vs production functions, separate from stored FP32 outputs.
        torch_raw, torch_phi = torch.from_numpy(raw), torch.from_numpy(phi)
        terms = phase_components(torch_raw, torch_phi)
        replay = phase_to_ttc(torch_raw).numpy()
        np.testing.assert_allclose(prediction, replay, rtol=1e-5, atol=1e-4)
        np.testing.assert_allclose(raw_error, terms["raw"].numpy(), rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(emitted_error, terms["emitted"].numpy(), rtol=1e-5, atol=1e-5)
        total = float(np.dot(masses, emitted_error + raw_error))
        sign = np.signbit(target) != np.signbit(prediction)
        metrics = json.loads((folder / "METRICS.json").read_text())
        np.testing.assert_allclose(
            sign.mean(), metrics["diagnostics"]["sign_error_rate"], rtol=1e-12, atol=1e-12
        )
        if len(table) != metrics["num_rows"] or len(table) != receipt["rows"]:
            raise ValueError("Prediction row count does not match evidence")
        for band, mask in zip(("crucial", "small", "large", "negative"), masks, strict=True):
            contribution = float(np.dot(masses[mask], (emitted_error + raw_error)[mask]))
            rows.append(
                {
                    "model": name,
                    "band": band,
                    "rows": int(mask.sum()),
                    "population_fraction": float(mask.mean()),
                    "effective_mass": float(masses[mask].sum()),
                    "weighted_emitted": float(np.dot(masses[mask], emitted_error[mask])),
                    "weighted_raw": float(np.dot(masses[mask], raw_error[mask])),
                    "loss_contribution": contribution,
                    "loss_fraction": contribution / total,
                    "endpoint_objective": total,
                    "sign_error_rate": float(sign[mask].mean()),
                    "sign_errors": int(sign[mask].sum()),
                    "fraction_of_all_sign_errors": (
                        float(sign[mask].sum() / sign.sum()) if sign.any() else None
                    ),
                    "mae_s": float(np.abs(prediction[mask] - target[mask]).mean()),
                    "sign_error_loss_contribution": float(
                        np.dot(masses[mask & sign], (emitted_error + raw_error)[mask & sign])
                    ),
                }
            )
        bindings[name] = {
            "path": path.relative_to(ROOT).as_posix(),
            "sha256": fingerprint,
            "metrics_sha256": sha(folder / "METRICS.json"),
        }
        checks[name] = {
            "rows": len(table),
            "mass_sum": float(masses.sum()),
            "max_emitted_reconstruction_difference": float(
                np.max(np.abs(emitted_error - terms["emitted"].numpy()))
            ),
        }
    result = {
        "utc": utc(),
        "status": "COMPLETE_DESCRIPTIVE_P_AUDIT",
        "bindings": bindings,
        "source_sha256": sha(Path(__file__)),
        "loss_source_sha256": sha(ROOT / "src/e_jepa_ttc/joint_measurement/loss.py"),
        "checks": checks,
        "rows": rows,
        "optimizer_updates": 0,
        "cuda_used": False,
        "H_or_V_opened": False,
        "selection_changed": False,
        "scope": "Full P endpoint objective, float64 reconstruction from stored FP32 outputs; "
        "not historical minibatch loss, parameter gradients, or a causal ablation",
    }
    output = RUN / "diagnostics/loss_recipe_P"
    write(output / "RESULT.json", result)
    pd.DataFrame(rows).to_csv(output / "BAND_CONTRIBUTIONS.csv", index=False)
    published = ROOT / "docs/joint_measurement_20261012/evidence_loss_recipe_20261016"
    write(published / "RESULT.json", result)
    pd.DataFrame(rows).to_csv(published / "BAND_CONTRIBUTIONS.csv", index=False)
    document = ROOT / "docs/joint_measurement_20261012/LEARNING_RECIPE_AUDIT.md"
    lines = [
        "# V18: pérdida, ponderación y errores de signo",
        "",
        "La receta de aprendizaje es una hipótesis pendiente de contraste causal. "
        "El deterioro de signo y TTC negativo aparece también en P y en ambas modalidades; "
        "una explicación exclusivamente basada en transferencia P/H resulta insuficiente.",
        "",
        "## Alcance y reproducción",
        "",
        "Ejecutar `python -B -m operational.joint_measurement.loss_audit` con `PYTHONPATH=src;.`. "
        "Se leen siete endpoints P ya congelados (53.890 consultas cada uno); "
        "no se abre H/V ni se realiza inferencia de modelos, backward, CUDA o updates. "
        "Los hashes de predicciones se contrastan con sus recibos COMPLETE. "
        "Se verifica igualdad de poblaciones y una reconstrucción independiente de las masas. "
        "La reconstrucción NumPy de la pérdida se contrasta con las funciones de producción.",
        "",
        "Fuentes numéricas: [RESULT.json](evidence_loss_recipe_20261016/RESULT.json) "
        "y [BAND_CONTRIBUTIONS.csv](evidence_loss_recipe_20261016/BAND_CONTRIBUTIONS.csv). "
        "Se conservan también en `artifacts/joint_measurement_20261012/diagnostics/loss_recipe_P`. "
        "Son pérdidas del endpoint sobre todo P, "
        "reconstruidas en float64 desde salidas guardadas; no son curvas históricas ni "
        "mediciones de gradientes de parámetros.",
        "",
        "## Qué optimiza la receta",
        "",
        "`src/e_jepa_ttc/joint_measurement/loss.py` asigna igual masa a cada grupo y "
        "masas nominales 0,5/0,3/0,1/0,1 a las bandas (0,1;3], (3;6], (6;10] y (-10;0). "
        "Redistribuye por bandas presentes. En esta población las masas efectivas "
        "coinciden con las nominales. El muestreo uniforme con los pesos globales "
        "implementados en `operational/joint_measurement/train.py` estima ese objetivo.",
        "",
        "Por consulta: `|fase_emitida - fase_GT| / 0,03 + "
        "0,1 * |fase_raw - fase_GT| / 0,03`. No hay término separado de clasificación "
        "de signo. La fase sí conserva el signo: un cruce de cero tiene coste, pero "
        "no una penalización categórica adicional. La conversión firmada y el límite "
        "de 60 segundos también intervienen; cerca de fase cero, diferencias pequeñas "
        "pueden producir grandes diferencias en segundos. No se ha aislado su efecto causal.",
        "",
        "## Distribución observada en la banda negativa de P",
        "",
        "| Modelo | Masa % | Signo incorrecto dentro de negativos % | "
        "Fracción de todos los errores de signo % | Contribución a pérdida total % |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        if row["band"] == "negative":
            lines.append(
                f"| {row['model']} | {100 * row['effective_mass']:.2f} | "
                f"{100 * row['sign_error_rate']:.2f} | "
                f"{100 * row['fraction_of_all_sign_errors']:.2f} | "
                f"{100 * row['loss_fraction']:.2f} |"
            )
    lines += [
        "",
        "Una masa del 10% no implica que el 10% de la pérdida o del gradiente proceda "
        "de esa banda. Los errores grandes elevan su contribución. Estas cifras no "
        "demuestran que el optimizador ignore el alejamiento ni que una BCE de signo "
        "lo vaya a resolver; documentan el equilibrio alcanzado con esta receta.",
        "",
        "## Selección y límites de la inferencia",
        "",
        "CORR mejora agregado y banda crucial frente a A5-P2, pero empeora bandas "
        "3–6, 6–10, negativa y signo en H. C1 mejora todas las bandas MiD de H frente "
        "a CORR libre y reduce sus errores de signo, pero aumenta MAE y cola absoluta. "
        "No hay dominancia global. Véase [C1_RESULTS.md](C1_RESULTS.md) y "
        "[RGB_RESULTS.md](RGB_RESULTS.md). C1 sí fue utilizado y replicado como "
        "productor para fusión; su rechazo standalone no eliminó esa señal.",
        "",
        "La arquitectura nativa usa otra pérdida. Comparar sus resultados con PAIR "
        "no aísla el encoder ni demuestra que la pérdida nueva cause las regresiones. "
        "B1 contrasta aprendizaje de correspondencias frente a TTC adicional con "
        "cómputo emparejado; tampoco aísla la hipótesis de pérdida.",
        "",
        "## Contraste que falta",
        "",
        "Mantener arquitectura, inicialización, población, muestreo y cómputo, "
        "y cambiar un único factor: ponderación por bandas o término de signo. "
        "Definir coeficientes y decisiones antes de nuevos updates y comparar P/H "
        "sin rescates en V. Medir además contribuciones y conflictos de gradiente "
        "por banda, con un conjunto P fijado prospectivamente.",
        "",
        "Este diagnóstico no registra ni ejecuta esa nueva ablación y no modifica "
        "B1 en curso. El programa adaptativo original restringe las cuatro familias "
        "y no incluye un sweep de pérdidas; ampliar recursos no convierte por sí "
        "solo una búsqueda de coeficientes en una prueba registrada. Se conserva "
        "la hipótesis como trabajo científico pendiente, sin reclasificar los "
        "endpoints históricos ni alterar los gates.",
        "",
    ]
    document.write_text("\n".join(lines), encoding="utf-8")
    print(
        json.dumps(
            {
                "status": result["status"],
                "models": len(bindings),
                "rows": sum(item["rows"] for item in checks.values()),
            }
        )
    )
    return result


if __name__ == "__main__":
    build()

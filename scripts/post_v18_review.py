"""Prepare CPU-only post-V18 evidence; never run a model, optimizer or submission."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
import zipfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs/post_v18_20261017"
V16 = ROOT.parent / "e-jepa-ttc-v16-multimodal-validation"
V19 = ROOT.parent / "e-jepa-ttc-v19-decision-recovery"
RUN16 = V16 / "artifacts/multimodal_research_20261012"
RUN19 = V19 / "artifacts/decision_recovery_20261017"
ZIP_HASH = "60b5a620848659633b875772430cddd7d7532b0fe180ec59904d04b81a9d3ca1"


def read(path: Path) -> Any:
    """Read a JSON document without importing its producer."""
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path: Path) -> str:
    """Hash a local artifact."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def relative(path: Path) -> str:
    """Express sibling worktree sources relative to the shared workspace."""
    return path.resolve().relative_to(ROOT.parent).as_posix()


def write(name: str, value: Any) -> None:
    """Write exclusively to the new review directory."""
    target = (OUT / name).resolve()
    if not target.is_relative_to(OUT.resolve()):
        raise ValueError("Review output escaped its directory")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def official_status() -> dict[str, Any]:
    """Verify the existing submission without regenerating any prediction."""
    directory = RUN16 / "test12_phase_only"
    path = directory / "PHASE_ONLY_S7_GarlTTC_submission.zip"
    if sha(path) != ZIP_HASH:
        raise ValueError("Frozen V16 submission changed")
    with zipfile.ZipFile(path) as archive:
        if archive.namelist() != ["submission.json"] or archive.testzip() is not None:
            raise ValueError("Unexpected submission membership or CRC")
        payload = json.loads(archive.read("submission.json"))
    if payload["meta"]["format"] != "garlttc_prediction_v1":
        raise ValueError("Unexpected submission schema")
    inputs = read(directory / "INPUTS.json")
    tokens = [row["sample_token"] for row in inputs["queries"]]
    results = payload["results"]
    with (directory / "TEST_PREDICTIONS.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    original = {row["sample_token"]: float(row["prediction"]) for row in rows}
    if (
        len(tokens) != 6762
        or len(set(tokens)) != 6762
        or len(rows) != 6762
        or set(results) != set(tokens)
        or set(original) != set(tokens)
    ):
        raise ValueError("Frozen token population differs")
    for token, item in results.items():
        value = item["ttc"]
        if not math.isfinite(value) or not (value < 0 or value > 0.1):
            raise ValueError("Invalid signed TTC")
        if value != original[token]:
            raise ValueError("Submission changed relative to sealed CSV")
    freeze = read(directory / "MODEL_INPUT_FREEZE.json")
    if sha(directory / "INPUTS.json") != freeze["inputs_sha256"]:
        raise ValueError("Input manifest changed")
    return {
        "schema": "post_v18_official_status_v1",
        "observed_utc": datetime.now(UTC).isoformat(),
        "candidate": "PHASE_ONLY_EVENT_S7",
        "package": {"path": relative(path), "sha256": ZIP_HASH, "bytes": path.stat().st_size},
        "local_checks": {
            "status": "PASS",
            "rows": len(results),
            "crc": True,
            "exact_frozen_tokens": True,
            "exact_sealed_csv_values": True,
            "private_targets_read": False,
        },
        "submission_status": "NOT_CONFIRMED_WITH_AUTHENTICATED_ACCOUNT",
        "last_local_status": read(directory / "PACKAGE_VERIFICATION.json")["status"],
        "receipt_search_scope": [
            relative(directory),
            relative(V16 / "docs/continuidad_v16_v18_20261012/submission"),
        ],
        "official_score": None,
        "modalities": ["RGB", "event"],
        "history": {
            "causal": True,
            "same_object": True,
            "max_observations_per_modality": 8,
            "admitted_support_ms": 650,
        },
        "pending": [
            "authenticated_submission_history",
            "participation_approval",
            "current_quota",
            "organizer_admission_of_cross_sample_causal_history",
        ],
        "current_rules_status": (
            "UNVERIFIED: web reader returned unrelated content; browser inventory empty"
        ),
        "official_repository": "https://github.com/NAIL-HNU/Garl-TTC",
        "competition": "https://www.codabench.org/competitions/17289/",
        "historical_rules_evidence": relative(
            V16 / "docs/continuidad_v16_v18_20261012/EVALUACION_OFICIAL.md"
        ),
        "submission_authorized": False,
        "external_actions": [],
        "next_action": (
            "Confirm account history and history eligibility; obtain separate "
            "explicit authorization before uploading exact frozen ZIP once."
        ),
        "source_hashes": {
            relative(directory / name): sha(directory / name)
            for name in (
                "MODEL_INPUT_FREEZE.json",
                "INPUTS.json",
                "TEST_PREDICTIONS.csv",
                "PACKAGE_VERIFICATION.json",
            )
        },
    }


def diagnostics() -> dict[str, Any]:
    """Extend existing P diagnostics with magnitude/range/group/phase decomposition."""
    import numpy as np
    import pandas as pd
    import torch

    from e_jepa_ttc.simplex_t.phase import MAX_PHASE, MIN_PHASE, emitted_phase, phase_to_ttc

    receipt_path = RUN19 / "diagnostics/COMPLETE.json"
    existing_path = RUN19 / "diagnostics/P.json"
    receipt = read(receipt_path)
    if sha(existing_path) != receipt["artifacts"]["P.json"]:
        raise ValueError("Existing P diagnostics changed")
    existing = read(existing_path)
    if existing["role"] != "P" or existing["rows"] != 53890:
        raise ValueError("Unexpected diagnostic population")
    source_paths = {
        "FULL": ROOT
        / (
            "artifacts/joint_measurement_20261012/adaptive/A1_adaptation_depth/pr"
            "edictions/P/A1-FULL-S7/predictions.parquet"
        ),
        "A5": ROOT
        / "artifacts/joint_measurement_20261012/predictions/P/A5-P2-FROZEN/predictions.parquet",
    }
    tables = {}
    for name, path in source_paths.items():
        expected = receipt["sources"]["prediction_bindings"]["P"][path.relative_to(ROOT).as_posix()]
        if sha(path) != expected:
            raise ValueError(f"Existing prediction binding changed: {name}")
        tables[name] = pd.read_parquet(path).sort_values("sample_token").reset_index(drop=True)
        if sha(path) != expected:
            raise ValueError("Prediction changed during read")
    identity = ["sample_token", "group_id", "target_ttc", "role"]
    if not tables["FULL"][identity].equals(tables["A5"][identity]):
        raise ValueError("Comparators do not have identical queries/targets/groups")
    if len(tables["FULL"]) != 53890 or set(tables["FULL"].role) != {"P"}:
        raise ValueError("Only complete frozen P is allowed")
    if tables["FULL"].sample_token.duplicated().any():
        raise ValueError("Duplicate P tokens")
    cells = []
    phase_summaries = {}
    macro_contributions = {}
    bounds = [-10.0, -8.0, -4.0, -2.0, -1.0, -0.5, 0.0]
    for name, table in tables.items():
        target = table.target_ttc.to_numpy(dtype=np.float64)
        pred = table.prediction_ttc.to_numpy(dtype=np.float64)
        raw = torch.tensor(table.raw_phase.to_numpy(dtype=np.float32))
        if (
            not np.isfinite(target).all()
            or not np.isfinite(pred).all()
            or not torch.isfinite(raw).all()
        ):
            raise ValueError("Nonfinite diagnostic input")
        negative = target < 0
        if not np.all((target[negative] >= -10) & (target[negative] < 0)):
            raise ValueError("Negative range does not cover the frozen population")
        if not np.all((pred < 0) | (pred > 0.1)):
            raise ValueError("Prediction outside native TTC domain")
        correct = np.signbit(target) == np.signbit(pred)
        clipped = raw.clamp(MIN_PHASE, MAX_PHASE)
        inverse_ttc = -torch.expm1(-clipped) / 0.1
        saturation = (inverse_ttc.abs() <= 1 / 60).numpy()
        converted = phase_to_ttc(raw).numpy().astype(np.float64)
        max_difference = float(np.abs(converted - pred).max())
        if max_difference > 1e-5:
            raise ValueError("Stored raw phase does not reproduce emitted TTC within 1e-5 s")
        if (
            int((negative & saturation).sum())
            != existing["models"][name]["negative_saturated_count"]
        ):
            raise ValueError("Native saturation disagrees with saved diagnostics")
        mid = 10000 * np.abs(np.log1p(-0.1 / target) - np.log1p(-0.1 / pred))
        absolute = np.abs(target - pred)
        ratio = np.abs(pred / target)
        groups = table.group_id.to_numpy()
        negative_groups = sorted(set(groups[negative]))
        macro_weight = np.zeros(len(table), dtype=np.float64)
        for group in negative_groups:
            selected = negative & (groups == group)
            macro_weight[selected] = 1.0 / (len(negative_groups) * int(selected.sum()))
        contributions = {}
        for sign in ("correct", "wrong"):
            for sat in ("saturated", "unsaturated"):
                selected = negative & (correct if sign == "correct" else ~correct)
                selected &= saturation if sat == "saturated" else ~saturation
                contributions[f"{sign}_{sat}"] = float(
                    (mid[selected] * macro_weight[selected]).sum()
                )
        expected_macro = existing["models"][name]["metrics"]["bins"]["negative"]["group_macro_mid"]
        if not math.isclose(sum(contributions.values()), expected_macro, abs_tol=1e-9):
            raise ValueError("MiDn contribution decomposition differs from original metric")
        macro_contributions[name] = contributions
        for lower, upper in zip(bounds[:-1], bounds[1:], strict=True):
            range_mask = (target >= lower) & (target < upper)
            for sign in ("all", "correct", "wrong"):
                for sat in ("all", "saturated", "unsaturated"):
                    mask = range_mask.copy()
                    if sign != "all":
                        mask &= correct if sign == "correct" else ~correct
                    if sat != "all":
                        mask &= saturation if sat == "saturated" else ~saturation
                    for group in ["ALL", *sorted(set(groups))]:
                        selected = mask if group == "ALL" else mask & (groups == group)
                        n = int(selected.sum())
                        group_means = [
                            float(mid[selected & (groups == g)].mean())
                            for g in sorted(set(groups[selected]))
                        ]
                        cells.append(
                            {
                                "model": name,
                                "range_lower_inclusive": lower,
                                "range_upper_exclusive": upper,
                                "sign": sign,
                                "saturation": sat,
                                "group": group,
                                "count": n,
                                "mae_s": float(absolute[selected].mean()) if n else None,
                                "p95_ae_s": float(np.quantile(absolute[selected], 0.95))
                                if n
                                else None,
                                "pooled_MiD": float(mid[selected].mean()) if n else None,
                                "conditional_group_macro_MiD": float(np.mean(group_means))
                                if n
                                else None,
                                "median_magnitude_ratio": float(np.median(ratio[selected]))
                                if n
                                else None,
                                "magnitude_over_2x_fraction": float((ratio[selected] > 2).mean())
                                if n
                                else None,
                                "magnitude_under_half_fraction": float(
                                    (ratio[selected] < 0.5).mean()
                                )
                                if n
                                else None,
                            }
                        )
        negative_raw = raw.numpy()[negative]
        phase_summaries[name] = {
            "native_reconversion_max_error_s": max_difference,
            "negative_count": int(negative.sum()),
            "negative_raw_phase_zero": int((negative_raw == 0).sum()),
            "negative_raw_phase_clipped": int(
                (negative & ((raw.numpy() < MIN_PHASE) | (raw.numpy() > MAX_PHASE))).sum()
            ),
            "negative_raw_phase_quantiles": {
                str(q): float(np.quantile(negative_raw, q))
                for q in (0, 0.01, 0.1, 0.5, 0.9, 0.99, 1)
            },
            "negative_emitted_vs_raw_phase_abs_mean": float(
                (emitted_phase(raw) - raw).abs().numpy()[negative].mean()
            ),
            "inverse_ttc_regimes": {
                "saturated_le_1_over_60": int((negative & saturation).sum()),
                "unsaturated_to_1_over_30": int(
                    (negative & ~saturation & (inverse_ttc.abs().numpy() <= 1 / 30)).sum()
                ),
                "above_1_over_30": int((negative & (inverse_ttc.abs().numpy() > 1 / 30)).sum()),
            },
            "correct_sign_negative_error_share": existing["models"][name]["negative_correct_sign"][
                "absolute_error_sum_s"
            ]
            / existing["models"][name]["negative"]["absolute_error_sum_s"],
            "raw_phase_caution": existing["models"][name]["native_saturation"]["caution"]
            if name == "A5"
            else "Stored FULL raw phase; no producer rerun",
        }
    csv_path = OUT / "A1_NEGATIVE_RANGE_GROUP.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(cells[0]))
        writer.writeheader()
        writer.writerows(cells)
    result = {
        "schema": "post_v18_negative_diagnostics_v1",
        "role": "P",
        "rows": 53890,
        "optimizer_updates": 0,
        "model_inference_runs": 0,
        "gradient_evaluations": 0,
        "scope": (
            "Frozen training-population diagnostics; neither generalization nor "
            "causal loss evidence"
        ),
        "reused_models": {name: existing["models"][name] for name in source_paths},
        "paired_sign_cells": existing["negative_sign_cross_vs_FULL"]["A5"],
        "phase_magnitude": phase_summaries,
        "MiDn_additive_contributions": macro_contributions,
        "MiDn_contribution_definition": (
            "Sum per-query MiD/(all_negative_groups * negative_queries_in_its_group) "
            "inside each disjoint sign/saturation cell; denominators remain those of "
            "the complete negative band. These four contributions sum to original MiDn."
        ),
        "range_edges_s": bounds,
        "empty_cells": "CSV empty values mean undefined, never zero",
        "range_csv": {"path": csv_path.name, "sha256": sha(csv_path), "rows": len(cells)},
        "conditional_metrics": (
            "Cell/group quantities do not equal the original whole-band macro "
            "score; conditional model sign subsets differ. Use paired_sign_cells "
            "for matched-query comparisons."
        ),
        "sources": {
            relative(p): sha(p)
            for p in [
                *source_paths.values(),
                receipt_path,
                existing_path,
                ROOT / "src/e_jepa_ttc/simplex_t/phase.py",
                Path(__file__),
            ]
        },
        "environment": {
            "torch": torch.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "device": "cpu",
        },
    }
    write("A1_NEGATIVE_DIAGNOSTICS.json", result)
    lines = [
        "# A1-FULL: magnitud y saturación en P",
        "",
        (
            "Analizamos 53.890 predicciones P guardadas; 22.397 tienen TTC "
            "negativo. Reutilizamos el diagnóstico V19 y añadimos rangos, grupos "
            "y comprobación CPU de la conversión de fase. No ejecutamos el "
            "modelo ni calculamos gradientes."
        ),
        "",
        (
            "| Modelo | MiDn macro por grupo | MiD negativo pooled | MAE "
            "negativo, s | Error de signo negativo | MAE con signo correcto, s |"
        ),
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name in source_paths:
        m = existing["models"][name]
        lines.append(
            f"| {name} | {m['metrics']['bins']['negative']['group_macro_mid']:.3f} | "
            f"{m['negative']['mean_MiD']:.3f} | {m['negative']['mae_s']:.3f} | "
            f"{100 * m['negative']['sign_error_rate']:.2f}% | "
            f"{m['negative_correct_sign']['mae_s']:.3f} |"
        )
    full = existing["models"]["FULL"]
    lines += [
        "",
        (
            "El error de signo de esta tabla se condiciona a targets negativos "
            "de P. No equivale a la tasa global en H. Cada modelo acierta el "
            "signo en consultas distintas; el JSON conserva también celdas "
            "emparejadas FULL/A5."
        ),
        "",
        (
            "| A1-FULL, targets negativos | n | MAE, s | MiD pooled | Fracción "
            "del error absoluto negativo |"
        ),
        "|---|---:|---:|---:|---:|",
    ]
    for name, cell in full["negative_sign_saturation_cells"].items():
        lines.append(
            f"| {name} | {cell['count']} | {cell['mae_s']:.3f} | "
            f"{cell['mean_MiD']:.3f} | "
            f"{100 * cell['fraction_of_negative_absolute_error']:.2f}% |"
        )
    saturated_share = sum(
        c["fraction_of_negative_absolute_error"]
        for k, c in full["negative_sign_saturation_cells"].items()
        if k.endswith("_True")
    )
    lines += [
        "",
        f"FULL acumula el "
        f"{100 * phase_summaries['FULL']['correct_sign_negative_error_share']:.2f}% "
        "del error absoluto negativo en consultas con signo correcto. Hay un problema "
        "de magnitud además del error de dirección. "
        f"La saturación afecta a {full['negative_saturated_count']} consultas negativas "
        f"({100 * full['negative_saturated_count'] / full['negative']['count']:.2f}%) "
        f"y concentra el {100 * saturated_share:.2f}% del error absoluto negativo. "
        "Las cuatro celdas anteriores son disjuntas; no sumamos otra vez saturación y signo.",
        "",
        "| Componente aditivo MiDn | FULL | A5 | Delta FULL - A5 |",
        "|---|---:|---:|---:|",
    ]
    for key, value in macro_contributions["FULL"].items():
        baseline = macro_contributions["A5"][key]
        lines.append(f"| {key} | {value:.3f} | {baseline:.3f} | {value - baseline:.3f} |")
    macro_gap = sum(macro_contributions["FULL"].values()) - sum(macro_contributions["A5"].values())
    saturated_gap = sum(
        macro_contributions["FULL"][key] - macro_contributions["A5"][key]
        for key in ("correct_saturated", "wrong_saturated")
    )
    wrong_unsaturated_gap = (
        macro_contributions["FULL"]["wrong_unsaturated"]
        - macro_contributions["A5"]["wrong_unsaturated"]
    )
    lines += [
        "",
        "Estas contribuciones conservan los denominadores de la banda negativa "
        "completa y suman el MiDn original. El delta por celda combina cambios de "
        "error y de pertenencia: no constituye una atribución causal. La fracción "
        "del error absoluto de arriba mide otra cantidad.",
        "",
        f"El incremento total MiDn es {macro_gap:.3f}. Las celdas saturadas "
        f"aportan {saturated_gap:.3f} ({100 * saturated_gap / macro_gap:.2f}% de ese delta); "
        f"la celda de signo incorrecto sin saturación aporta {wrong_unsaturated_gap:.3f} "
        f"({100 * wrong_unsaturated_gap / macro_gap:.2f}%). La saturación pesa en MAE, "
        "pero no concentra la mayor parte del deterioro MiDn. Quitar el límite "
        "de 60 s no queda justificado como solución por este análisis.",
        "",
        "| Rango target P, s | n FULL | MAE FULL, s | MAE A5, s | Razón de magnitudes, mediana |",
        "|---|---:|---:|---:|---:|",
    ]
    for lower, upper in zip(bounds[:-1], bounds[1:], strict=True):
        selected = [
            c
            for c in cells
            if c["range_lower_inclusive"] == lower
            and c["range_upper_exclusive"] == upper
            and c["group"] == "ALL"
            and c["sign"] == c["saturation"] == "all"
        ]
        a, b = selected
        values = [
            "N/D" if x is None else f"{x:.3f}"
            for x in (a["mae_s"], b["mae_s"], a["median_magnitude_ratio"])
        ]
        lines.append(f"| [{lower:g}, {upper:g}) | {a['count']} | {' | '.join(values)} |")
    paired = existing["negative_sign_cross_vs_FULL"]["A5"]["cells"]
    both_correct = paired["FULL_correct_True_comparator_correct_True"]
    lines += [
        "",
        "Los rangos son descriptivos y se fijan para este desglose. El intervalo "
        "[-1,-0,5) tiene solo tres consultas y [-0,5,0) ninguna: no permiten concluir "
        "comportamiento cerca de TTC cero. Cerca de fase cero significa TTC de gran "
        "magnitud, no TTC próximo a cero.",
        "",
        f"En las mismas {both_correct['count']} consultas donde ambos modelos aciertan "
        f"el signo, el MiD pooled es {both_correct['FULL']['mean_MiD']:.3f} para FULL "
        f"y {both_correct['comparator']['mean_MiD']:.3f} para A5. La regresión de "
        "magnitud persiste al fijar las consultas, por lo que no depende solo de "
        "comparar subconjuntos de signo diferentes.",
        "",
        "| Grupo P | n negativos | MiD FULL | MiD A5 | Delta MiD |",
        "|---|---:|---:|---:|---:|",
    ]
    group_rows = []
    for group, group_full in full["negative_by_group"].items():
        group_a5 = existing["models"]["A5"]["negative_by_group"][group]
        if group_full["count"]:
            group_rows.append(
                (
                    group_full["mean_MiD"] - group_a5["mean_MiD"],
                    group,
                    group_full,
                    group_a5,
                )
            )
    for difference, group, group_full, group_a5 in sorted(group_rows, reverse=True)[:5]:
        lines.append(
            f"| {group} | {group_full['count']} | {group_full['mean_MiD']:.3f} | "
            f"{group_a5['mean_MiD']:.3f} | {difference:.3f} |"
        )
    lines += [
        "",
        "Mostramos los cinco mayores incrementos de MiD negativo por grupo; "
        "el JSON conserva todos los grupos y el CSV todos los cruces. "
        f"FULL empeora en {sum(d > 0 for d, *_ in group_rows)} de {len(group_rows)} "
        "grupos con negativos. Este ranking no selecciona consultas de entrenamiento.",
        "",
        f"La reconversión CPU reproduce el TTC guardado con error máximo "
        f"{phase_summaries['FULL']['native_reconversion_max_error_s']:.9g} s. "
        f"Hay {phase_summaries['FULL']['negative_raw_phase_clipped']} fases crudas "
        "negativas sujetas al clipping MIN_PHASE/MAX_PHASE. Por tanto, ese clipping "
        "no explica estos casos; el límite de magnitud de 60 s sí está presente.",
    ]
    lines += [
        "",
        (
            "La conversión nativa usa q = -expm1(-clamp(fase))/0,1 y TTC = "
            "sign(q)/max(|q|,1/60). Cerca de fase cero, el inverso amplifica "
            "discrepancias hasta alcanzar el límite de 60 s. Dentro del límite, "
            "la salida queda saturada; no afirmamos que su derivada siga "
            "creciendo. Fase cero emite +60 s. El clipping de fase y el límite "
            "de magnitud son operaciones distintas."
        ),
        "",
        (
            "A5 puede guardar fase reconstruida desde TTC: su etiqueta de "
            "saturación describe el umbral de conversión, sin probar el estado "
            "crudo original del productor. FULL sí guarda raw_phase. No "
            "modificamos clipping ni salidas."
        ),
        "",
        (
            "El CSV contiene todos los grupos, rangos y cruces signo/saturación, "
            "incluidos los vacíos. Los estadísticos condicionales son "
            "diagnósticos, no una nueva puntuación oficial. No derivamos "
            "conflictos de gradiente ni causalidad de estas tablas."
        ),
        "",
        (
            "Único contraste pertinente: el L0/LW ya registrado en V19, que "
            "mantiene FULL, padre, sampler y presupuesto y cambia la masa "
            "negativa. No abrimos otra copia ni proponemos BCE como solución "
            "demostrada. El resultado de esa intervención pertenece a V19 y "
            "requiere su propia evidencia."
        ),
        "",
        (
            "Reproducir: `python -B scripts/post_v18_review.py prepare` con el "
            "entorno del proyecto y `PYTHONPATH=src;.` en Windows. Fuentes y "
            "hashes: [A1_NEGATIVE_DIAGNOSTICS.json](A1_NEGATIVE_DIAGNOSTICS.json)"
            ". Desglose: [A1_NEGATIVE_RANGE_GROUP.csv](A1_NEGATIVE_RANGE_GROUP.cs"
            "v)."
        ),
        "",
    ]
    (OUT / "A1_NEGATIVE_ANALYSIS.md").write_text("\n".join(lines), encoding="utf-8")
    return {"rows": 53890, "range_cells": len(cells), "negative": full["negative"]["count"]}


def plans() -> None:
    """Bind the candidate registry and prospective specifications to existing artifacts."""
    run18 = ROOT / "artifacts/joint_measurement_20261012"
    freeze_path = RUN16 / "test12_phase_only/MODEL_INPUT_FREEZE.json"
    freeze = read(freeze_path)

    def binding(path: Path) -> dict[str, str]:
        return {"path": relative(path), "sha256": sha(path)}

    registry = {
        "schema": "post_v18_candidate_registry_v1",
        "path_base": "workspace_parent",
        "historical_v18_status": "NO_CANDIDATE",
        "new_optimizer_updates_authorized": 0,
        "new_comparison_inference_authorized": False,
        "official_SOTA": False,
        "candidates": [
            {
                "id": "PHASE_ONLY_EVENT_S7",
                "family": "V16",
                "status": "FROZEN_OFFICIAL_EVALUATION_CANDIDATE",
                "modalities": ["RGB", "event"],
                "seed": 7,
                "ensemble": False,
                "model_freeze": binding(freeze_path),
                "head_sha256": freeze["readout_sha256"]["EVENT"],
                "submission_sha256": ZIP_HASH,
                "genealogy": (
                    "V13 P producers/PAIR -> H normalizers/context refiners -> H "
                    "PHASE_ONLY head; V exposed development"
                ),
                "parent_hashes": {
                    relative(Path(p)): value for p, value in freeze["parent_sha256"].items()
                },
                "eligibility": "Cross-sample causal history and account approval unconfirmed",
            },
            {
                "id": "FUSION-C1-V18-S7",
                "family": "V18",
                "status": "PARTIAL_INTERNAL_IMPROVEMENT_NOT_PROMOTED",
                "modalities": ["RGB", "event"],
                "seed": 7,
                "ensemble": False,
                "endpoint": read(run18 / "fusion/ENDPOINTS.json")["endpoints"]["FUSION-TRUE-S7"],
                "endpoints_receipt": binding(run18 / "fusion/ENDPOINTS.json"),
                "registration": binding(run18 / "fusion/REGISTRATION.json"),
                "genealogy": "Shared P A5 -> P C1 adaptation; historical R_CTX; new H fusion head",
                "replications": [7, 13, 23],
                "replication_limit": "Shared A5 and R_CTX; scalar metric means are not an ensemble",
                "comparison_scope": "V exposed development only; interval vs V16 includes zero",
            },
            {
                "id": "A1-FULL-S7",
                "family": "V18",
                "status": "POSITIVE_BAND_GAIN_NEGATIVE_BAND_REGRESSION",
                "modalities": ["event"],
                "seed": 7,
                "ensemble": False,
                "endpoint": read(run18 / "adaptive/A1_adaptation_depth/ENDPOINTS.json")[
                    "A1-FULL-S7"
                ],
                "endpoints_receipt": binding(run18 / "adaptive/A1_adaptation_depth/ENDPOINTS.json"),
                "genealogy": "A5 -> C1 -> B1 correspondence stage1 -> FULL 12000 TTC updates on P",
                "evaluation_role": "H",
                "direct_score_comparison_to_fusion_V_allowed": False,
            },
        ],
    }
    write("CANDIDATE_REGISTRY.json", registry)
    manifest_path = ROOT.parent / (
        "e-jepa-ttc-v12-efficient-context/artifacts/sota_campaign_20261008/fc"
        "wd_population/QUERY_MANIFEST.json"
    )
    manifest = read(manifest_path)
    rows = manifest["rows"]
    if len({r["query_id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate external query identity")
    write(
        "FROZEN_COMPARISON_PROTOCOL.json",
        {
            "schema": "post_v18_frozen_comparison_v1",
            "status": "PROPOSED_NOT_AUTHORIZED_NOT_EXECUTED",
            "hypothesis": (
                "Does frozen V18 C1 fusion retain a descriptive advantage against "
                "frozen V16 outside V?"
            ),
            "population": {
                "name": "FCWD",
                **binding(manifest_path),
                "rows": len(rows),
                "selection": "Every registered query; no selection using new errors",
                "exposure": "Previously observed transfer development; not blind confirmation",
                "negative_band": (
                    "Not available; cannot establish negative-band protection or full "
                    "signed benchmark promotion"
                ),
            },
            "models": [
                "PHASE_ONLY_EVENT_S7",
                "FUSION-C1-V18-S7",
                "F_TRUE",
                "R_CTX",
                "Garl_RGB_event_official_checkpoint",
            ],
            "seed_policy": "seed7 fixed; no selection of seed13, no ensemble",
            "identity_registry": "CANDIDATE_REGISTRY.json",
            "new_endpoint_required": False,
            "before_inference": [
                "Separate explicit scope and compute-budget authorization",
                (
                    "Seal complete producer/normalizer/adapter/checkpoint hashes for "
                    "every route, including Garl"
                ),
                "Verify native timestamp, ROI, calibration and causal history per query",
                (
                    "Seal paired input coverage using sensor availability only; preserve "
                    "full manifest and disclose every excluded/missing query"
                ),
                "Keep historical-native and common-input comparability claims distinct",
            ],
            "sensor_contract": {
                "query_anchor": "Frozen manifest reference time",
                "future_sensor_access": False,
                "history": (
                    "Declare each native route; same-object causal support at most 650ms "
                    "for contextual systems"
                ),
                "Garl": (
                    "Official preprocessing and native temporal pair; cost and "
                    "historical support differ"
                ),
                "modalities": (
                    "RGB+event where both present; fallback results reported separately, "
                    "not silently mixed into paired multimodal comparison"
                ),
            },
            "metrics": [
                "native overall_MiD with actual available bands declared",
                "MiDc",
                "MiDs",
                "MiDl",
                "MiDn=undefined_if_no_negative_targets",
                "MAE_s",
                "p95_absolute_error_s",
                "sign_error",
                "coverage",
                "per_acquisition_group",
            ],
            "primary_contrast": "V18-S7 minus V16-S7 on identical query IDs",
            "statistics": {
                "paired_bootstrap_replicates": 2000,
                "seed": 12345,
                "unit": "acquisition_group",
                "interpretation": (
                    "Descriptive conditional intervals on exposed transfer data; not "
                    "independent replication"
                ),
            },
            "latency": (
                "Separate authorized paired raw pipeline measurement; no reuse of "
                "prepared-forward milliseconds as E2E"
            ),
            "decision_policy": {
                "mechanistic_signal": "Report direction and uncertainty vs control",
                "further_evaluation": (
                    "Report aggregate, every band, sign, coverage and cost separately; "
                    "no automatic promotion"
                ),
                "substantial_gain": (
                    "10% crucial remains a historical V18 goal, not a new retrospective "
                    "admission rule"
                ),
                "promotion": "Not available from this exposed positive-only cohort",
                "SOTA": "Requires eligible official comparable evaluation",
            },
            "optimizer_updates": 0,
            "inference_executed": False,
            "autostart": False,
        },
    )
    sealed = {role: read(RUN16 / f"cache/{role}/SEALED.json") for role in ("H", "V")}
    population = sum(x["population"] for x in sealed.values())
    updates = math.ceil(2500 * population / sealed["H"]["population"])
    warmup = math.ceil(100 * updates / 2500)
    plan = {
        "schema": "post_v18_refit_plan_v1",
        "family": "PHASE_ONLY_V16",
        "seed": 7,
        "status": "PROPOSED_HEAD_ONLY_REFIT_NOT_EXECUTED",
        "execution_enabled": False,
        "authorized_optimizer_updates": 0,
        "scope": (
            "One new PHASE_ONLY EVENT head on H union V; all producers, PAIR, "
            "context refiners and normalizers remain frozen. This is not a full "
            "TRAIN40 producer refit."
        ),
        "reason": (
            "Smallest coherent extension of the already prepared family, using "
            "more upper-stage TRAIN without swapping parents under an old head"
        ),
        "selection_dependency": (
            "Review current official candidate status first; do not tune this "
            "recipe from test12 scores"
        ),
        "parent_freeze": binding(freeze_path),
        "candidate_registry": "CANDIDATE_REGISTRY.json",
        "roles": {
            "P": "24 producer-training groups; frozen lower stages",
            "H": "8 historical refiner/head-training groups; becomes final head TRAIN",
            "V": (
                "8 exposed development groups; proposed conversion to final head "
                "TRAIN only after authorization"
            ),
            "FCWD_Dev32": "Exclude from fit and model selection; already exposed transfer",
            "test12": (
                "Exclude from fit; private TTC not accessed; new predictions/export "
                "need separate scope"
            ),
        },
        "stacking": {
            "strategy": "Declared sequential in-sample stacking",
            "OOF": False,
            "reason": (
                "H features derive from refiners trained on H; no claim that H or "
                "H+V residuals estimate generalization"
            ),
            "cross_fitting": (
                "Not performed and not implied; group cross-fitting would require "
                "rebuilding fold-specific dependent stages under a different plan"
            ),
            "normalizers": (
                "Retain frozen historical H normalizers; no fitting on test or "
                "transfer; bind their hashes through parent freeze"
            ),
        },
        "training_inputs": {
            role: {
                "path": relative(RUN16 / f"cache/{role}/states.npz"),
                "expected_sha256": sealed[role]["states_sha256"],
                "rows": sealed[role]["population"],
                "seal": binding(RUN16 / f"cache/{role}/SEALED.json"),
            }
            for role in sealed
        },
        "model": {
            "class": "PhaseOnlyAnchor",
            "mode": "EVENT",
            "parameters": 1955,
            "initialization": "fresh seed7; zero final output; do not continue old trained head",
            "architecture": "20->32 LayerNorm SiLU->32 LayerNorm SiLU->3",
            "source": binding(V16 / "operational/multimodal_research/phase_only_anchor.py"),
        },
        "objective": {
            "source": binding(V16 / "src/e_jepa_ttc/rgb_port/fusion.py"),
            "function": "fusion_loss",
            "phase_scale": 0.03,
            "quantile_weight": 0.1,
            "mass": (
                "Preserve original group-balanced training mass: "
                "1/(number_of_TRAIN_groups * queries_in_group). Recompute across H+V; "
                "do not concatenate separately normalized mass vectors. "
                "Evaluation band weights 0.5/0.3/0.1/0.1 are not training weights."
            ),
            "mass_source": binding(V16 / "operational/rgb_port/prepare.py"),
        },
        "optimizer": {
            "name": "AdamW",
            "lr": 0.0003,
            "betas": [0.9, 0.999],
            "eps": 1e-8,
            "weight_decay": 0.001,
            "foreach": False,
            "clip_norm": 1.0,
        },
        "sampling": {
            "name": "uniform_with_replacement",
            "batch_size": 128,
            "population": population,
            "updates": updates,
            "baseline_updates": 2500,
            "baseline_population": sealed["H"]["population"],
            "rule": "ceil(2500 * new_population / original_H_population)",
            "expected_draws_per_query": updates * 128 / population,
        },
        "scheduler": {
            "warmup_updates": warmup,
            "warmup": "lr_max * update / warmup_updates",
            "after_warmup": "lr_min+(lr_max-lr_min)*(1+cos(pi*(update-warmup)/(total-warmup)))/2",
            "lr_max": 0.0003,
            "lr_min": 0.00003,
            "step_unit": "committed_optimizer_update",
        },
        "budget_proposed_only": {
            "scientific_updates": updates,
            "technical_updates": 0,
            "recovery_update_cap": 250,
            "physical_update_cap": updates + 250,
            "GPU_seconds": 0,
            "CPU_wall_seconds": 7200,
            "RAM_bytes": 4000000000,
            "device": "cpu_FP32_one_thread",
            "source": "New authorization only; no V18 remainder",
        },
        "checkpoint": {
            "every_updates": 100,
            "endpoint": "fixed final update; no early stopping on H/V/test",
            "contents": [
                "model",
                "optimizer",
                "scheduler",
                "RNG",
                "sampler",
                "committed_and_pending_journal",
                "plan_hash",
                "input_hashes",
                "source_commit",
                "environment",
            ],
            "resume": (
                "Exact identity only; count replayed updates; refuse existing "
                "historical destinations"
            ),
        },
        "export": {
            "format": "TorchScript head plus metadata and five fixed TRAIN examples",
            "output_root": "artifacts/post_v18_refit/PHASE_ONLY_HV_S7",
            "full_sensor_pipeline": False,
            "test12_predictions": "Not authorized by this plan",
            "parity": (
                "CPU FP32 same examples, max absolute TTC error <=1e-5 s; disclose "
                "any exception before proceeding"
            ),
        },
        "stages": [
            {
                "id": "authorize",
                "depends_on": [],
                "operation": "Bind explicit new authorization, plan SHA256 and CPU/update budget",
            },
            {
                "id": "verify_parents",
                "depends_on": ["authorize"],
                "operation": (
                    "Verify full frozen genealogy and no changed producer/normalizer; "
                    "never replace a failed binding"
                ),
            },
            {
                "id": "build_HV_inputs",
                "depends_on": ["verify_parents"],
                "operation": (
                    "Verify both caches; join by frozen IDs; disjoint groups; recompute "
                    "TRAIN masses; seal union"
                ),
            },
            {
                "id": "initialize_new_head",
                "depends_on": ["build_HV_inputs"],
                "operation": "Fresh PhaseOnlyAnchor EVENT seed7 with frozen input semantics",
            },
            {
                "id": "fit_fixed_endpoint",
                "depends_on": ["initialize_new_head"],
                "operation": (
                    "Apply declared sampler, optimizer, scaled schedule and durable checkpoints"
                ),
            },
            {
                "id": "export_train_examples",
                "depends_on": ["fit_fixed_endpoint"],
                "operation": (
                    "Export head and verify fixed TRAIN examples; no benchmark inference "
                    "or submission"
                ),
            },
        ],
        "commands_available_now": [
            ["python", "-B", "scripts/post_v18_review.py", "validate"],
            ["python", "-B", "scripts/run_post_v18_refit.py", "dry-run"],
        ],
        "training_adapter": {
            "status": "IMPLEMENTED_NOT_TRAINED",
            "source": {
                "path": "scripts/run_post_v18_refit.py",
                "sha256": sha(ROOT / "scripts/run_post_v18_refit.py"),
            },
            "runtime_sources": {
                relative(V16 / name): sha(V16 / name)
                for name in (
                    "src/e_jepa_ttc/simplex_t/phase.py",
                    "operational/rgb_port/train_producers.py",
                )
            },
            "authorization": (
                "Separate post_v18_refit_authorization_v1 receipt with authorized=true, "
                "exact plan_sha256, user_instruction_reference and budget equal to "
                "budget_proposed_only. None created by this review."
            ),
            "future_command": [
                "python",
                "-B",
                "scripts/run_post_v18_refit.py",
                "train",
                "--authorization",
                "<explicit-receipt.json>",
            ],
            "resume": (
                "Same command plus --resume. Exact plan, source, environment and authorization "
                "identity. A stale RUNNING.lock requires manual inspection; no automatic restart."
            ),
            "resource_guards": (
                "CPU only, one thread, RSS checked before each batch. Wall time reserved "
                "conservatively in durable 60-second slices, including resumes. Cooperative "
                "guards, not OS-enforced hard limits; an operation can overshoot a slice."
            ),
            "validation_limit": (
                "Actual caches and hashes checked; synthetic checkpoint/resume/export tests "
                "use zero optimizer steps. End-to-end optimization has not been executed."
            ),
        },
    }
    write("REFIT_PLAN.json", plan)


def prepare() -> None:
    """Write new evidence only; preserve historical campaigns and their processes."""
    OUT.mkdir(parents=True, exist_ok=True)
    write("OFFICIAL_EVALUATION_STATUS.json", official_status())
    summary = diagnostics()
    plans()
    snapshots = {}
    for name in ("REGISTRATION.json", "PIPELINE_STATE.json", "USER_CONTINUE_RECEIPT.json"):
        path = RUN19 / name
        content = path.read_bytes()
        snapshots[name] = {
            "source": relative(path),
            "sha256": hashlib.sha256(content).hexdigest(),
            "record": json.loads(content),
        }
    write(
        "V19_RECONCILIATION.json",
        {
            "observed_utc": datetime.now(UTC).isoformat(),
            "git_commit": subprocess.check_output(
                ["git", "-C", str(V19), "rev-parse", "HEAD"], text=True
            ).strip(),
            "snapshots": snapshots,
            "process_liveness_checked": False,
            "actions_on_v19": [],
            "scope": (
                "Recorded history, not proof of current process liveness or successful training"
            ),
        },
    )
    print(json.dumps({"status": "PASS", **summary, "optimizer_updates": 0, "inference_runs": 0}))


def validate() -> None:
    """Validate deliverable references and the unexecuted refit DAG."""
    plan = read(OUT / "REFIT_PLAN.json")
    if plan["authorized_optimizer_updates"] != 0 or plan["execution_enabled"]:
        raise ValueError("This review cannot authorize training")
    known = set()
    ordered = []
    for stage in plan["stages"]:
        if not set(stage["depends_on"]) <= known or stage["id"] in known:
            raise ValueError("Refit DAG is not topologically ordered")
        known.add(stage["id"])
        ordered.append(stage["id"])
    if plan["family"] != "PHASE_ONLY_V16" or plan["seed"] != 7:
        raise ValueError("Unexpected family or seed")
    adapter = plan["training_adapter"]["source"]
    if sha(ROOT / adapter["path"]) != adapter["sha256"]:
        raise ValueError("Refit adapter changed; regenerate proposal and preflight")
    sampling = plan["sampling"]
    if sampling["population"] != sum(x["rows"] for x in plan["training_inputs"].values()):
        raise ValueError("Refit population differs from sealed role sizes")
    if sampling["updates"] != math.ceil(
        sampling["baseline_updates"] * sampling["population"] / sampling["baseline_population"]
    ):
        raise ValueError("Refit exposure scaling changed")
    for item in plan["training_inputs"].values():
        seal_path = ROOT.parent / item["seal"]["path"]
        if sha(seal_path) != item["seal"]["sha256"]:
            raise ValueError("Refit input seal changed")
        if read(seal_path)["states_sha256"] != item["expected_sha256"]:
            raise ValueError("Refit cache binding changed")
    if plan["stacking"]["OOF"]:
        raise ValueError("Historical H features are not OOF")
    status = read(OUT / "OFFICIAL_EVALUATION_STATUS.json")
    if status["package"]["sha256"] != ZIP_HASH or status["submission_authorized"]:
        raise ValueError("Submission identity or authorization changed")
    for name in (
        "CANDIDATE_REGISTRY.json",
        "FROZEN_COMPARISON_PROTOCOL.json",
        "A1_NEGATIVE_DIAGNOSTICS.json",
        "V19_RECONCILIATION.json",
    ):
        read(OUT / name)
    diag = read(OUT / "A1_NEGATIVE_DIAGNOSTICS.json")
    if sha(OUT / diag["range_csv"]["path"]) != diag["range_csv"]["sha256"]:
        raise ValueError("Diagnostic table changed")
    with (OUT / diag["range_csv"]["path"]).open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != diag["range_csv"]["rows"]:
        raise ValueError("CSV cell count changed")
    for model, saved in diag["reused_models"].items():
        base = [
            r
            for r in rows
            if r["model"] == model and r["group"] == "ALL" and r["sign"] == r["saturation"] == "all"
        ]
        count = sum(int(r["count"]) for r in base)
        if count != saved["negative"]["count"]:
            raise ValueError("Range decomposition is not exhaustive")
        mean = (
            sum(int(r["count"]) * float(r["pooled_MiD"]) for r in base if int(r["count"])) / count
        )
        if not math.isclose(mean, saved["negative"]["mean_MiD"], abs_tol=1e-9):
            raise ValueError("CSV does not reconstruct saved pooled negative MiD")
        for row in base:
            parts = [
                r
                for r in rows
                if r["model"] == model
                and r["group"] == "ALL"
                and r["range_lower_inclusive"] == row["range_lower_inclusive"]
                and r["sign"] != "all"
                and r["saturation"] != "all"
            ]
            if sum(int(r["count"]) for r in parts) != int(row["count"]):
                raise ValueError("Sign/saturation cells do not partition a range")
        macro = sum(diag["MiDn_additive_contributions"][model].values())
        if not math.isclose(
            macro, saved["metrics"]["bins"]["negative"]["group_macro_mid"], abs_tol=1e-9
        ):
            raise ValueError("Additive components do not reconstruct original MiDn")
    for path, expected in diag["sources"].items():
        if sha(ROOT.parent / path) != expected:
            raise ValueError(f"Diagnostic source changed: {path}")
    print(
        json.dumps(
            {
                "status": "PASS",
                "training_started": False,
                "execution_enabled": False,
                "ordered_stages": ordered,
            }
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "validate", "dry-run"))
    arguments = parser.parse_args()
    if arguments.action == "prepare":
        prepare()
    else:
        validate()

"""P-only, non-deployable selection ceiling of the frozen L0/LW loss screen."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from operational.range_transfer.analyze import gate

from . import common, diagnostics, loss_negative_analysis

OUT = common.RUN / "loss_pair_oracle"


def _changes(target: np.ndarray, before: np.ndarray, after: np.ndarray, mask: np.ndarray) -> dict:
    old_mid, new_mid = diagnostics._mid(target, before), diagnostics._mid(target, after)
    old_abs, new_abs = np.abs(target - before), np.abs(target - after)
    old_correct = np.signbit(target) == np.signbit(before)
    new_correct = np.signbit(target) == np.signbit(after)
    return {
        "count": int(mask.sum()),
        "MiD_improved": int((mask & (new_mid < old_mid)).sum()),
        "MiD_harmed": int((mask & (new_mid > old_mid)).sum()),
        "MiD_tied": int((mask & (new_mid == old_mid)).sum()),
        "absolute_error_improved": int((mask & (new_abs < old_abs)).sum()),
        "absolute_error_harmed": int((mask & (new_abs > old_abs)).sum()),
        "absolute_error_tied": int((mask & (new_abs == old_abs)).sum()),
        "absolute_error_sum_delta_s": float((new_abs[mask] - old_abs[mask]).sum()),
        "sign_rescued": int((mask & ~old_correct & new_correct).sum()),
        "sign_harmed": int((mask & old_correct & ~new_correct).sum()),
        "sign_both_wrong": int((mask & ~old_correct & ~new_correct).sum()),
    }


def compute(
    tables: dict[str, pd.DataFrame], rows: pd.DataFrame, baseline: pd.DataFrame
) -> tuple[dict, pd.DataFrame]:
    """Choose minimum MiD once per P query, then score that one vector everywhere."""
    loss_negative_analysis.validate_pair(tables, rows, "P")
    if (
        not baseline[common.IDENTITY]
        .reset_index(drop=True)
        .equals(rows[common.IDENTITY].reset_index(drop=True))
    ):
        raise ValueError("A5 comparator must match every P target/ID/group/order")
    if baseline.sample_token.duplicated().any():
        raise ValueError("Duplicated A5 queries")
    target = rows.target_ttc.to_numpy(float)
    predictions = {name: frame.prediction_ttc.to_numpy(float) for name, frame in tables.items()}
    predictions["A5"] = baseline.prediction_ttc.to_numpy(float)
    for name, values in predictions.items():
        if not np.isfinite(values).all() or not ((values < 0) | (values > 0.1)).all():
            raise ValueError(f"Invalid signed TTC: {name}; do not filter")
    losses = {name: diagnostics._mid(target, pred) for name, pred in predictions.items()}
    choose_lw = losses["LW"] < losses["L0"]
    selected = np.where(choose_lw, predictions["LW"], predictions["L0"])
    predictions["ORACLE"] = selected
    # Retain original raw phase from the SAME selected expert for saturation diagnostics.
    panel = rows.copy()
    panel["selected_expert"] = np.where(choose_lw, "LW", "L0")
    panel["prediction_ttc"] = selected
    panel["raw_phase"] = np.where(choose_lw, tables["LW"].raw_phase, tables["L0"].raw_phase)
    for name in ("L0", "LW", "A5"):
        panel[f"{name}_prediction_ttc"] = predictions[name]
        panel[f"{name}_MiD"] = losses[name]
    panel["ORACLE_MiD"] = diagnostics._mid(target, selected)
    metrics = {name: common.score(rows, pred) for name, pred in predictions.items()}
    if not all(m["complete_prediction_and_mid_coverage"] for m in metrics.values()):
        raise ValueError("Incomplete scorer coverage")
    masks = {
        "all": np.ones(len(rows), bool),
        "crucial": (target > 0.1) & (target <= 3),
        "small": (target > 3) & (target <= 6),
        "large": (target > 6) & (target <= 10),
        "negative": target < 0,
    }
    saturation = {}
    for name, table in {**tables, "A5": baseline, "ORACLE": panel}.items():
        saturated, proof = diagnostics._saturation(table, name)
        saturation[name] = {
            "native_proof": proof,
            "counts": {band: int((saturated & mask).sum()) for band, mask in masks.items()},
        }
    comparisons = {}
    for name in ("L0", "LW", "A5"):
        candidate, reference = metrics["ORACLE"], metrics[name]
        comparisons[name] = {
            "overall_MiD_delta": candidate["overall_MiD"] - reference["overall_MiD"],
            "overall_MiD_ratio": candidate["overall_MiD"] / reference["overall_MiD"],
            "band_MiD_deltas": {
                band: candidate["bins"][band]["group_macro_mid"]
                - reference["bins"][band]["group_macro_mid"]
                for band in ("crucial", "small", "large", "negative")
            },
            "MAE_delta_s": candidate["diagnostics"]["mae_s"] - reference["diagnostics"]["mae_s"],
            "sign_error_rate_delta": candidate["diagnostics"]["sign_error_rate"]
            - reference["diagnostics"]["sign_error_rate"],
            "accuracy_gates_diagnostic_not_admission": gate(reference, candidate),
            "rescues_and_harms": {
                band: _changes(target, predictions[name], selected, mask)
                for band, mask in masks.items()
            },
            "by_group": {
                str(group): _changes(target, predictions[name], selected, rows.group_id == group)
                for group in rows.group_id.unique()
            },
        }
    return {
        "schema": "v19_P_loss_pair_oracle_v1",
        "status": "ORACLE_NON_DEPLOYABLE_DIAGNOSTIC",
        "role": "P",
        "rows": len(rows),
        "query_ids_sha256": diagnostics._tokens_hash(rows),
        "prediction_vector_sha256_float64_le": diagnostics._array_hash(selected),
        "objective": "minimum native per-query MiD, ties L0; same vector for EVERY metric",
        "aggregation": "group-macro within bands; weights crucial/small/large/negative .5/.3/.1/.1",
        "selected_counts": {"L0": int((~choose_lw).sum()), "LW": int(choose_lw.sum())},
        "selection_by_band": {
            band: {"L0": int((~choose_lw & mask).sum()), "LW": int((choose_lw & mask).sum())}
            for band, mask in masks.items()
        },
        "metrics": metrics,
        "comparisons": comparisons,
        "pure_selection_negative_floor": {
            "oracle_MiDn": metrics["ORACLE"]["bins"]["negative"]["group_macro_mid"],
            "A5_MiDn": metrics["A5"]["bins"]["negative"]["group_macro_mid"],
            "A5_original_system_gate_limit": metrics["A5"]["bins"]["negative"]["group_macro_mid"]
            + max(2.0, 0.05 * metrics["A5"]["bins"]["negative"]["group_macro_mid"]),
            "A5_development_band_gate_limit": 1.1
            * metrics["A5"]["bins"]["negative"]["group_macro_mid"],
            "proof_scope": (
                "Per-query minimum with nonnegative registered group/band weights is a lower "
                "bound for EVERY hard selector of these same L0/LW outputs on this fixed P. "
                "This is not a bound for interpolation, a different model, or unseen data."
            ),
        },
        "saturation": saturation,
        "scientific_updates": 0,
        "optimizer_updates": 0,
        "inference_calls": 0,
        "router_fit": False,
        "H_opened": False,
        "candidate": False,
        "system_promotion": False,
        "cost": "Unmeasured; an output-based router requires both producers",
        "interpretation_limit": (
            "Uses GT on producer-training P. Complementarity is not evidence of causal routing "
            "signal or out-of-group generalization; minimum MiD can still harm MAE/sign."
        ),
    }, panel


def document(report: dict) -> str:
    floor = report["pure_selection_negative_floor"]
    negative_floor_fails = floor["oracle_MiDn"] > floor["A5_original_system_gate_limit"]
    lines = [
        "# Oráculo P de la pantalla L0/LW",
        "",
        "Diagnóstico no desplegable: se escoge el menor MiD por consulta usando GT, con empate "
        "a L0. Todas las métricas usan ese mismo vector. No se abre H ni se ajusta un router.",
        "",
        "| Modelo | MiD agregado | Crítico | Pequeño | Largo | Negativo | MAE s | Signo |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, m in report["metrics"].items():
        bands = [m["bins"][b]["group_macro_mid"] for b in ("crucial", "small", "large", "negative")]
        values = [m["overall_MiD"], *bands, m["diagnostics"]["mae_s"]]
        lines.append(
            f"| {name} | "
            + " | ".join(f"{v:.6f}" for v in values)
            + f" | {100 * m['diagnostics']['sign_error_rate']:.4f}% |"
        )
    lines += [
        "",
        f"Selección: {report['selected_counts']['L0']:,} L0 y "
        f"{report['selected_counts']['LW']:,} LW "
        f"de {report['rows']:,} consultas P. A5 es referencia, no un experto seleccionable.",
        "",
    ]
    for name, comparison in report["comparisons"].items():
        failed = [
            key
            for key, ok in comparison["accuracy_gates_diagnostic_not_admission"]["checks"].items()
            if not ok
        ]
        change = comparison["rescues_and_harms"]["all"]
        lines.append(
            f"Frente a {name}: cambio agregado {100 * (comparison['overall_MiD_ratio'] - 1):.3f}%; "
            f"rescates/daños de signo {change['sign_rescued']}/{change['sign_harmed']}; "
            f"mejoras/daños de error absoluto {change['absolute_error_improved']}/"
            f"{change['absolute_error_harmed']}. "
            f"Gates numéricos fallidos: {failed}."
        )
        lines.append("")
    lines += [
        "El mínimo MiD por consulta es también el mínimo posible de cada banda para cualquier "
        "selector duro entre estos mismos outputs L0/LW, porque todos los pesos registrados "
        "son positivos. "
        + (
            "Su MiDn supera el límite original frente a A5; ningún selector puro de este par "
            "puede cumplir esa protección en este P, aunque sí hay complementariedad. "
            if negative_floor_fails
            else "Su MiDn no descarta por sí solo un selector que respete la protección A5. "
        )
        + "No es un límite para interpolar, cambiar modelos o evaluar otras poblaciones.",
        "",
        "La complementariedad mide margen de selección ideal; no demuestra que un gate causal "
        "pueda aprenderlo. Es P de entrenamiento, una semilla y padres compartidos. No habilita "
        "promoción, nueva réplica ni una conclusión sobre generalización.",
        "",
        "Las saturaciones, rescates/daños por banda y grupo están en REPORT.json. El Parquet "
        "P_ORACLE.parquet conserva IDs, targets, relojes originales, experto escogido y vector "
        "único. RECEIPT.json vincula entradas, código y salidas. Cero updates y cero inferencias.",
        "",
    ]
    return "\n".join(lines)


def run() -> dict:
    """Execute only existing-table CPU arithmetic and isolated diagnostic writes."""
    torch.set_num_threads(1)
    tables, rows, admission, bindings = loss_negative_analysis.load("P")
    baseline = common.load_population("P", ["A5"])["A5"]
    bindings.update(
        {
            str(common.SOURCE_ROOT / path): value
            for path, value in common.source_bindings("P", ["A5"]).items()
        }
    )
    report, panel = compute(tables, rows, baseline)
    if len(rows) != 53890:
        raise ValueError("The real diagnostic must use every one of the 53,890 P queries")
    report["admission"] = admission
    OUT.mkdir(parents=True, exist_ok=True)
    panel_path = OUT / "P_ORACLE.parquet"
    panel.to_parquet(panel_path, index=False)
    common.write_json(OUT / "REPORT.json", report)
    doc = common.ROOT / "docs/decision_recovery_20261017/LOSS_PAIR_ORACLE.md"
    doc.write_text(document(report), encoding="utf-8")
    if any(common.sha(Path(path)) != value for path, value in bindings.items()):
        raise ValueError("Input bytes changed during CPU diagnostic")
    sources = [
        Path(__file__),
        Path(diagnostics.__file__),
        Path(common.__file__),
        Path(loss_negative_analysis.__file__),
        common.ROOT / "operational/range_transfer/analyze.py",
        common.ROOT / "operational/rgb_port/evaluate.py",
        common.ROOT / "src/e_jepa_ttc/simplex_t/phase.py",
    ]
    receipt = {
        "status": "COMPLETE_P_ONLY_ORACLE_DIAGNOSTIC",
        "inputs": bindings,
        "sources": {str(path.relative_to(common.ROOT)): common.sha(path) for path in sources},
        "outputs": {
            str(path.relative_to(common.ROOT)): common.sha(path)
            for path in (panel_path, OUT / "REPORT.json", doc)
        },
        "scientific_updates": 0,
        "inference_calls": 0,
        "H_opened": False,
        "historical_seven_expert_oracles_modified": False,
        "input_bytes_unchanged_after_analysis": True,
    }
    common.write_json(OUT / "RECEIPT.json", receipt)
    return receipt


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))

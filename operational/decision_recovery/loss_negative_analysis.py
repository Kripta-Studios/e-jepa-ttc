"""Paired negative-band decomposition of complete frozen L0/LW predictions."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from . import common, diagnostics, training

ARMS = ("L0", "LW")
COUNTS = {"P": 53890, "H": 18422}
OUT = common.RUN / "loss_negative_analysis"
CELLS = ("both_correct", "L0_only_correct", "LW_only_correct", "both_wrong")


def validate_pair(tables: dict[str, pd.DataFrame], rows: pd.DataFrame, role: str) -> None:
    """Require exact full-order identities and valid coverage without filtering."""
    if role not in COUNTS or set(tables) != set(ARMS) or rows.empty:
        raise ValueError("Exactly L0/LW on P or H are required")
    for arm, frame in tables.items():
        required = {*common.IDENTITY, "prediction_ttc", "raw_phase"}
        if not required.issubset(frame.columns):
            raise ValueError(f"Missing columns: {arm}")
        if (
            len(frame) != len(rows)
            or frame.sample_token.duplicated().any()
            or frame[common.IDENTITY].isna().to_numpy().any()
            or set(frame.role) != {role}
            or not frame[common.IDENTITY]
            .reset_index(drop=True)
            .equals(rows[common.IDENTITY].reset_index(drop=True))
        ):
            raise ValueError(f"Population identity/order/target mismatch: {arm}")
        values = frame[["target_ttc", "prediction_ttc", "raw_phase"]].to_numpy(float)
        if not np.isfinite(values).all():
            raise ValueError(f"Nonfinite values retained as failed admission: {arm}")
        target, prediction = values[:, 0], values[:, 1]
        if not (((target > -10) & (target < 0)) | ((target > 0.1) & (target <= 10))).all():
            raise ValueError("Target outside registered four-band population")
        if not ((prediction < 0) | (prediction > 0.1)).all():
            raise ValueError("Prediction outside native signed phase domain")


def load(role: str) -> tuple[dict[str, pd.DataFrame], pd.DataFrame, dict, dict]:
    """Read existing receipts and common clock-bearing rows; never run inference."""
    from operational.range_transfer.common import digest

    if role not in COUNTS:
        raise ValueError("Only P/H are admitted")
    manifest_path = training.manifest_path(role)
    manifest = common.read_json(manifest_path)
    rows_path = Path(manifest["rows_path"])
    sensor = training.cache_root(role)
    sensor_binding = common.read_json(sensor / "BINDING.json")
    sensor_complete = common.read_json(sensor / "COMPLETE.json")
    registration_path = common.RUN / "evaluation/REGISTRATION.json"
    registration = common.read_json(registration_path)
    paths = [manifest_path, rows_path, sensor / "BINDING.json", sensor / "COMPLETE.json"]
    paths += [registration_path, common.RUN / "evaluation/REPORT.json"]
    if (
        manifest["status"] != "COMPLETE"
        or common.sha(rows_path) != manifest["rows_sha256"]
        or sensor_binding["manifest_sha256"] != common.sha(manifest_path)
        or sensor_binding["source_identity"]["rows_sha256"] != manifest["rows_sha256"]
        or sensor_complete["status"] != "COMPLETE"
        or sensor_complete["binding"] != digest(sensor_binding)
        or sensor_complete["rows"] != COUNTS[role]
    ):
        raise ValueError("Source population or sensor clock binding changed")
    rows = pd.read_parquet(rows_path)
    if len(rows) != COUNTS[role] or len(rows) != manifest["population_size"]:
        raise ValueError("Incomplete source population")
    clocks = [name for name in rows.columns if name.endswith("_us")]
    if not {"query_time_us", "cutoff_us", "producer_times_us"}.issubset(clocks):
        raise ValueError("Missing source query/observation clocks")
    tables = {}
    identities = {}
    for arm in ARMS:
        directory = common.RUN / f"evaluation/predictions/{role}/{arm}"
        identity_path, complete_path = directory / "IDENTITY.json", directory / "COMPLETE.json"
        prediction_path, metrics_path = (
            directory / "predictions.parquet",
            directory / "METRICS.json",
        )
        identity, complete = common.read_json(identity_path), common.read_json(complete_path)
        if (
            identity["arm"] != arm
            or identity["role"] != role
            or identity["seed"] != 7
            or identity["endpoint"] != registration["endpoints"][arm]
            or identity["evaluation_registration_sha256"] != common.sha(registration_path)
            or identity["manifest_sha256"] != common.sha(manifest_path)
            or identity["rows_sha256"] != manifest["rows_sha256"]
            or identity["sensor_complete_sha256"] != common.sha(sensor / "COMPLETE.json")
            or complete["status"] != "COMPLETE"
            or complete["rows"] != COUNTS[role]
            or complete["binding"] != digest(identity)
            or complete["sha256"] != common.sha(prediction_path)
            or complete["metrics_sha256"] != common.sha(metrics_path)
        ):
            raise ValueError(f"Changed endpoint/clock/coverage receipt: {role}/{arm}")
        tables[arm] = pd.read_parquet(prediction_path)
        identities[arm] = identity
        paths.extend([identity_path, complete_path, prediction_path, metrics_path])
    validate_pair(tables, rows, role)
    for arm, table in tables.items():
        if common.score(table, table.prediction_ttc.to_numpy()) != common.read_json(
            common.RUN / f"evaluation/predictions/{role}/{arm}/METRICS.json"
        ):
            raise ValueError("Existing full-population metrics did not reproduce exactly")
    admission = {
        "status": "PASS",
        "rows": len(rows),
        "query_ids_sha256": diagnostics._tokens_hash(rows),
        "identity_columns_exact": common.IDENTITY,
        "clocks": {
            "source_columns": clocks,
            "rows_sha256": manifest["rows_sha256"],
            "sensor_binding_sha256": common.sha(sensor / "BINDING.json"),
            "source_identity": sensor_binding["source_identity"],
            "method": "Both prediction receipts bind the same exact clock-bearing rows and sensor",
            "prediction_files_contain_clocks": False,
            "raw_sensor_payloads_replayed": False,
        },
        "endpoints": {arm: identities[arm]["endpoint"] for arm in ARMS},
        "full_metrics_reproduced_exactly": True,
    }
    return tables, rows, admission, {str(path): common.sha(path) for path in paths}


def summarize(frame: pd.DataFrame, mask: np.ndarray, arm: str) -> dict:
    """Pooled cell statistics plus additive contributions to group-macro MiDn."""
    selected = frame.loc[mask]
    count = len(selected)
    error = selected[f"{arm}_error_s"].to_numpy()
    mid = selected[f"{arm}_MiD"].to_numpy()
    return {
        "count": count,
        "mae_s": float(np.abs(error).mean()) if count else None,
        "absolute_error_sum_s": float(np.abs(error).sum()),
        "bias_s": float(error.mean()) if count else None,
        "signed_error_sum_s": float(error.sum()),
        "mean_MiD": float(mid.mean()) if count else None,
        "MiD_sum": float(mid.sum()),
        "group_macro_negative_MiD_contribution": float(
            (selected[f"{arm}_MiD"] * selected.negative_group_weight).sum()
        ),
        "magnitude_mae_s": float(selected[f"{arm}_magnitude_error_s"].mean()) if count else None,
        "sign_error_count": int((~selected[f"{arm}_sign_correct"]).sum()),
        "saturated_count": int(selected[f"{arm}_saturated"].sum()),
        "saturated_positive_count": int(
            (selected[f"{arm}_saturated"] & (selected[f"{arm}_ttc"] > 0)).sum()
        ),
        "saturated_negative_count": int(
            (selected[f"{arm}_saturated"] & (selected[f"{arm}_ttc"] < 0)).sum()
        ),
    }


def compute(
    tables: dict[str, pd.DataFrame], rows: pd.DataFrame, role: str
) -> tuple[dict, pd.DataFrame]:
    """Partition every negative query; report changes without fitting or selection."""
    validate_pair(tables, rows, role)
    negative = rows.target_ttc.to_numpy() < 0
    if not negative.any():
        raise ValueError("Negative population is empty")
    frame = rows.loc[negative, common.IDENTITY].reset_index(drop=True).copy()
    truth = frame.target_ttc.to_numpy(float)
    group_sizes = frame.groupby("group_id").sample_token.transform("size")
    frame["negative_group_weight"] = 1.0 / (frame.group_id.nunique() * group_sizes)
    native = {}
    metrics = {}
    for arm in ARMS:
        prediction = tables[arm].prediction_ttc.to_numpy(float)
        saturation, native[arm] = diagnostics._saturation(tables[arm], arm)
        pred = prediction[negative]
        frame[f"{arm}_ttc"] = pred
        frame[f"{arm}_error_s"] = pred - truth
        frame[f"{arm}_magnitude_error_s"] = np.abs(np.abs(pred) - np.abs(truth))
        frame[f"{arm}_MiD"] = diagnostics._mid(truth, pred)
        frame[f"{arm}_sign_correct"] = np.signbit(pred) == np.signbit(truth)
        frame[f"{arm}_saturated"] = saturation[negative]
        metrics[arm] = common.score(tables[arm], prediction)
        if not metrics[arm]["complete_prediction_and_mid_coverage"]:
            raise ValueError("Incomplete native MiD coverage")
    left, right = frame.L0_sign_correct.to_numpy(), frame.LW_sign_correct.to_numpy()
    masks = (left & right, left & ~right, ~left & right, ~left & ~right)
    if not np.array_equal(sum(mask.astype(int) for mask in masks), np.ones(len(frame), int)):
        raise AssertionError("Cells must be disjoint and exhaustive")
    frame["cell"] = ""
    cells = {}
    for name, mask in zip(CELLS, masks, strict=True):
        frame.loc[mask, "cell"] = name
        values = {arm: summarize(frame, mask, arm) for arm in ARMS}
        cells[name] = {
            "count": int(mask.sum()),
            **values,
            "delta_LW_minus_L0": {
                key: values["LW"][key] - values["L0"][key]
                for key in (
                    "absolute_error_sum_s",
                    "signed_error_sum_s",
                    "MiD_sum",
                    "group_macro_negative_MiD_contribution",
                )
            },
        }
    all_negative = np.ones(len(frame), dtype=bool)
    totals = {arm: summarize(frame, all_negative, arm) for arm in ARMS}
    for arm in ARMS:
        expected = metrics[arm]["bins"]["negative"]["group_macro_mid"]
        if not np.isclose(
            totals[arm]["group_macro_negative_MiD_contribution"], expected, rtol=1e-12, atol=1e-9
        ):
            raise AssertionError("Cell contributions do not reconstruct native group-macro MiDn")
        for field in ("absolute_error_sum_s", "signed_error_sum_s", "MiD_sum"):
            if not np.isclose(sum(cell[arm][field] for cell in cells.values()), totals[arm][field]):
                raise AssertionError("Cell sums do not reconstruct the full negative population")
    by_group = {
        str(group): {
            name: {
                arm: summarize(frame, mask & (frame.group_id.to_numpy() == group), arm)
                for arm in ARMS
            }
            for name, mask in zip(CELLS, masks, strict=True)
        }
        for group in sorted(frame.group_id.unique())
    }
    return {
        "role": role,
        "rows": len(rows),
        "negative_count": len(frame),
        "negative_groups": int(frame.group_id.nunique()),
        "complete_population_metrics": metrics,
        "negative_totals": totals,
        "cells": cells,
        "by_group": by_group,
        "native_saturation": native,
        "candidate": False,
        "selection": False,
    }, frame


def document(report: dict) -> str:
    """Render a bounded Spanish report directly from admitted results."""
    lines = [
        "# V19: descomposición del error negativo L0/LW",
        "",
        "Análisis descriptivo de predicciones congeladas completas P/H. Cero fits, updates o "
        "inferencias nuevas. H es desarrollo expuesto y no selecciona ningún modelo aquí.",
        "",
        "Los IDs, targets, roles y orden coinciden exactamente con las filas fuente. Los relojes "
        "se admiten mediante los mismos hashes de manifest, filas y sensor en ambos recibos; "
        "no están duplicados en los parquet de predicciones ni se reproduce el sensor crudo.",
        "",
        "Las cuatro celdas particionan todos los targets negativos. MAE, sesgo y MiD de celda "
        "son promedios pooled. La contribución MiDn usa 1/(número de grupos × negativos del "
        "grupo), por lo que las cuatro contribuciones suman la métrica negativa nativa. "
        "Saturación se calcula con la conversión FP32 original, no con igualdad decimal a 60.",
        "",
    ]
    for role, result in report["populations"].items():
        totals = result["negative_totals"]
        lines += [
            f"## {role}: {result['rows']:,} consultas completas",
            "",
            f"Negativos: {result['negative_count']:,}; grupos: {result['negative_groups']}.",
            "",
            "| Celda | n | MAE L0/LW s | MiD pooled L0/LW | Sesgo L0/LW s | "
            "Saturados L0/LW | Δ suma error absoluto s | Δ contribución MiDn |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for name, cell in result["cells"].items():
            left, right, delta = cell["L0"], cell["LW"], cell["delta_LW_minus_L0"]

            def pair(key: str, left: dict = left, right: dict = right) -> str:
                return "/".join("—" if v is None else f"{v:.3f}" for v in (left[key], right[key]))

            lines.append(
                f"| {name} | {cell['count']} | {pair('mae_s')} | {pair('mean_MiD')} | "
                f"{pair('bias_s')} | {left['saturated_count']}/{right['saturated_count']} | "
                f"{delta['absolute_error_sum_s']:.3f} | "
                f"{delta['group_macro_negative_MiD_contribution']:.6f} |"
            )
        lines += ["", "| Total negativo | L0 | LW |", "|---|---:|---:|"]
        for key in (
            "mae_s",
            "bias_s",
            "absolute_error_sum_s",
            "mean_MiD",
            "group_macro_negative_MiD_contribution",
            "sign_error_count",
            "saturated_count",
        ):
            lines.append(f"| {key} | {totals['L0'][key]:.6f} | {totals['LW'][key]:.6f} |")
        lines.append("")
        cells = result["cells"]
        full_change = (
            totals["LW"]["group_macro_negative_MiD_contribution"]
            - totals["L0"]["group_macro_negative_MiD_contribution"]
        )
        preserved_change = cells["both_correct"]["delta_LW_minus_L0"][
            "group_macro_negative_MiD_contribution"
        ]
        share = 100 * preserved_change / full_change if full_change else None
        fraction = f" ({share:.1f}% del cambio neto)" if share is not None else ""
        lines += [
            f"La contribución de ambos signos correctos es {preserved_change:.3f} MiDn"
            f"{fraction}: la mejora incluye cambios de magnitud sin recuperar signo. "
            f"Hay {cells['LW_only_correct']['count']} correcciones de signo y "
            f"{cells['L0_only_correct']['count']} regresiones.",
            "",
            f"En los {cells['both_wrong']['count']} casos de signo incorrecto compartido, "
            f"el MAE pasa de {cells['both_wrong']['L0']['mae_s']:.3f} a "
            f"{cells['both_wrong']['LW']['mae_s']:.3f} s y el MiD pooled de "
            f"{cells['both_wrong']['L0']['mean_MiD']:.3f} a "
            f"{cells['both_wrong']['LW']['mean_MiD']:.3f}. "
            "No son correcciones de signo. La transformación de fase comprime diferencias "
            "cuando TTC se aleja hacia infinito; por eso MiD puede mejorar mientras el "
            "error en segundos empeora. Las saturaciones correspondientes figuran en la tabla.",
            "",
        ]
    lines += [
        "## Interpretación y límites",
        "",
        "La celda both_correct compara magnitudes conservando signo correcto. LW_only_correct "
        "recoge correcciones de signo; L0_only_correct recoge regresiones. both_wrong conserva "
        "errores de signo. Estas categorías describen dónde cambia el error: una celda que "
        "cambia de signo también puede cambiar de magnitud, por lo que no estima un efecto "
        "causal puro del signo. No se crea ningún vector contrafactual ni selector por target.",
        "",
        "La mejora negativa no revoca el empeoramiento crítico/agregado ni habilita por sí sola "
        "replicación o promoción. La partición se aplica idénticamente a P y H, sin buscar "
        "umbrales o coeficientes. La variación entre entrenamientos no se estima con una semilla.",
        "",
        "Evidencia: `artifacts/decision_recovery_20261017/loss_negative_analysis/REPORT.json`, "
        "CSV de todos los negativos P/H y `RECEIPT.json` con hashes de entradas, código y salidas.",
        "",
        "Reproducción CPU: `python -B -m operational.decision_recovery.loss_negative_analysis`. "
        "Coordinar la ventana con el propietario GPU antes de iniciarla.",
        "",
    ]
    return "\n".join(lines)


def run() -> dict:
    """Create only this diagnostic's output subtree and its dedicated document."""
    import torch

    torch.set_num_threads(1)
    report = {"schema": "v19_loss_negative_analysis_v1", "populations": {}, "admission": {}}
    bindings = {}
    outputs = {}
    OUT.mkdir(parents=True, exist_ok=True)
    for role in COUNTS:
        tables, rows, admission, inputs = load(role)
        result, frame = compute(tables, rows, role)
        report["populations"][role], report["admission"][role] = result, admission
        bindings.update(inputs)
        path = OUT / f"{role}_NEGATIVE_QUERIES.csv"
        frame.to_csv(path, index=False, float_format="%.17g")
        outputs[str(path.relative_to(common.ROOT))] = common.sha(path)
    report.update(optimizer_updates=0, inference_calls=0, fitting=False, system_promotion=False)
    common.write_json(OUT / "REPORT.json", report)
    doc = common.ROOT / "docs/decision_recovery_20261017/LOSS_NEGATIVE_ANALYSIS.md"
    doc.write_text(document(report), encoding="utf-8")
    sources = [
        Path(__file__),
        Path(diagnostics.__file__),
        Path(common.__file__),
        Path(training.__file__),
        common.ROOT / "operational/range_transfer/analyze.py",
        common.ROOT / "operational/rgb_port/evaluate.py",
        common.ROOT / "src/e_jepa_ttc/simplex_t/phase.py",
        common.ROOT / "tests/unit/test_decision_recovery_loss_negative_analysis.py",
    ]
    if any(common.sha(Path(path)) != value for path, value in bindings.items()):
        raise ValueError("Source bytes changed during diagnostic")
    for path in (OUT / "REPORT.json", doc):
        outputs[str(path.relative_to(common.ROOT))] = common.sha(path)
    receipt = {
        "status": "PASS",
        "source_git_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=common.ROOT, text=True
        ).strip(),
        "source_commit_exact": False,
        "source_identity": "File hashes are authoritative; analysis may be uncommitted",
        "working_tree_changes": subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=common.ROOT, text=True
        ).splitlines(),
        "inputs": bindings,
        "sources": {str(path.relative_to(common.ROOT)): common.sha(path) for path in sources},
        "outputs": outputs,
        "optimizer_updates": 0,
        "inference_calls": 0,
        "source_bytes_unchanged_after_analysis": True,
        "historical_panel_modified": False,
        "report_sha256": common.sha(OUT / "REPORT.json"),
        "tokens_sha256": {role: report["admission"][role]["query_ids_sha256"] for role in COUNTS},
    }
    common.write_json(OUT / "RECEIPT.json", receipt)
    return receipt


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))

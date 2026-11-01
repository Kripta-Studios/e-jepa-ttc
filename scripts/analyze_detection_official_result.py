"Audit organizer-returned aggregate eAP results without private test labels."

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


def sha(path: Path) -> str:
    """Hash bounded input or output bytes."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write(path: Path, value: Any) -> None:
    """Persist exactly the UTF-8 bytes hashed in the report inventory."""
    path.write_bytes(
        (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode()
    )


def reconstruct_ap(precision: list[float], min_recall: float, min_precision: float) -> float:
    "Reconstruct the clipped AP from the returned 101-point interpolated curve."
    if len(precision) != 101 or not 0 <= min_recall < 1 or not 0 <= min_precision < 1:
        raise ValueError("Unsupported AP curve/configuration")
    values = np.asarray(precision, dtype=np.float64)
    if not np.isfinite(values).all() or np.any(values < 0) or np.any(values > 1):
        raise ValueError("Invalid precision curve")
    return float(
        np.maximum(0, values[round(100 * min_recall) + 1 :] - min_precision).mean()
        / (1 - min_precision)
    )


def main() -> None:
    "Reconstruct summary arithmetic, class diagnostics and a source-bound report."
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scoring-zip", type=Path, required=True)
    parser.add_argument("--prediction-zip", type=Path, required=True)
    parser.add_argument("--submission-receipt", type=Path, required=True)
    parser.add_argument("--local-comparison", type=Path, required=True)
    parser.add_argument("--local-gt-shards", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output
    output.mkdir(parents=True, exist_ok=False)
    raw_names = {
        "metrics_summary.json": "detailed_results/metrics_summary.json",
        "metrics_details.json": "detailed_results/metrics_details.json",
        "scores.json": "scores.json",
        "scoring_metadata.txt": "metadata",
    }
    with zipfile.ZipFile(args.scoring_zip) as archive:
        for name, member in raw_names.items():
            if archive.getinfo(member).file_size > 2_000_000:
                raise ValueError("Unexpectedly large organizer result")
            (output / name).write_bytes(archive.read(member))
    with zipfile.ZipFile(args.prediction_zip) as archive:
        if archive.namelist() != ["metadata"]:
            raise ValueError("Prediction-result layout differs; inspect before proceeding")
        (output / "prediction_metadata.txt").write_bytes(archive.read("metadata"))
    summary = json.loads((output / "metrics_summary.json").read_text())
    detail = json.loads((output / "metrics_details.json").read_text())
    scores = json.loads((output / "scores.json").read_text())
    receipt = json.loads(args.submission_receipt.read_text())
    local = json.loads(args.local_comparison.read_text())
    cfg = summary["cfg"]
    assert summary["num_samples"] == receipt["frames"]
    assert summary["num_pred_boxes"] == receipt["boxes"]
    assert sum(summary["gt_count_by_class"].values()) == summary["num_gt_boxes"]
    assert scores["AP"] == summary["mean_ap"]
    assert "exitCode: 0" in (output / "scoring_metadata.txt").read_text()
    first = round(100 * cfg["min_recall"]) + 1
    classes = {}
    ap_errors = []
    tp_errors = []
    unclipped = []
    for name in cfg["class_names"]:
        curves = {}
        for threshold in cfg["dist_ths"]:
            key = str(float(threshold))
            curve = detail[f"{name}:{key}"]
            ap = reconstruct_ap(curve["precision"], cfg["min_recall"], cfg["min_precision"])
            ap_errors.append(abs(ap - summary["label_aps"][name][key]))
            confidence = np.asarray(curve["confidence"])
            indices = np.flatnonzero(confidence > 0)
            last = int(indices[-1]) if len(indices) else 0
            curves[key] = {
                "official_AP": summary["label_aps"][name][key],
                "reconstructed_AP": ap,
                "last_nonzero_confidence_recall_grid": curve["recall"][last],
                "precision_at_recall_10pct": curve["precision"][10],
                "precision_at_recall_20pct": curve["precision"][20],
                "mean_returned_precision_without_clipping": float(np.mean(curve["precision"])),
            }
            unclipped.append(curves[key]["mean_returned_precision_without_clipping"])
            if threshold == cfg["dist_th_tp"]:
                calculated_tp = {
                    metric: float(np.mean(curve[metric][first : last + 1]))
                    if last >= first
                    else 1.0
                    for metric in cfg["tp_metrics"]
                }
                for metric, value in calculated_tp.items():
                    tp_errors.append(abs(value - summary["label_tp_errors"][name][metric]))
        classes[name] = {
            "ground_truth_boxes": summary["gt_count_by_class"][name],
            "official_AP": summary["mean_dist_aps"][name],
            "development_selection_AP_different_protocol": local["results"]["learned"]["per_class"][
                name
            ],
            "tp_errors": summary["label_tp_errors"][name],
            "curves": curves,
        }
    reconstructed_mean = float(np.mean([r["official_AP"] for r in classes.values()]))
    weight = cfg["mean_ap_weight"]
    eap = (weight * reconstructed_mean + sum(summary["tp_scores"].values())) / (
        weight + len(cfg["tp_metrics"])
    )
    assert max(ap_errors) < 1e-12 and max(tp_errors) < 1e-12
    assert abs(eap - summary["eap_score"]) < 1e-12
    assert abs(reconstructed_mean - summary["mean_ap"]) < 1e-12
    for key, metric in (("ATE", "trans_err"), ("ASE", "scale_err"), ("AOE", "orient_err")):
        assert scores[key] == summary["tp_errors"][metric]
        assert (
            abs(np.mean([x["tp_errors"][metric] for x in classes.values()]) - scores[key]) < 1e-12
        )
    eligible: Counter[str] = Counter()
    ignored: Counter[str] = Counter()
    reasons: Counter[str] = Counter()
    frames = 0
    gt_receipts = {}
    for path in sorted(args.local_gt_shards.glob("*/gt.json")):
        gt_receipts[path.parent.name] = sha(path)
        for frame in json.loads(path.read_text())["ground_truth_by_token"].values():
            frames += 1
            eligible.update(x["class_name"] for x in frame["eligible"])
            ignored.update(x["class_name"] for x in frame["ignored"])
            reasons.update(frame["ignored_by_reason"])
    assert frames == local["frames"]
    ignored_fraction = sum(ignored.values()) / (sum(ignored.values()) + sum(eligible.values()))
    analysis = {
        "schema": "phase_eap_official_return_analysis_v1",
        "source_sha256": {
            "scoring_result.zip": sha(args.scoring_zip),
            "prediction_result.zip": sha(args.prediction_zip),
            "submission_receipt": sha(args.submission_receipt),
            "local_comparison": sha(args.local_comparison),
        },
        "submission_zip_sha256": receipt["submission_zip_sha256"],
        "count_match": {
            "samples": summary["num_samples"],
            "gt_boxes": summary["num_gt_boxes"],
            "pred_boxes": summary["num_pred_boxes"],
        },
        "official": {**scores, "eap_score": summary["eap_score"]},
        "classes": classes,
        "returned_curve_reconstruction": {
            "max_AP_abs_error": max(ap_errors),
            "max_TP_abs_error": max(tp_errors),
            "eap_score": eap,
            "scope": (
                "Exact summary arithmetic from returned curves, not reproduction of "
                "private matching or test labels."
            ),
        },
        "mean_interpolated_test_precision_without_clipping": float(np.mean(unclipped)),
        "unclipped_diagnostic_scope": (
            "Same returned test curves with clipping removed; not local AP101, not "
            "an alternative official score."
        ),
        "development_population": {
            "frames": frames,
            "eligible": dict(eligible),
            "ignored": dict(ignored),
            "eligible_total": sum(eligible.values()),
            "ignored_total": sum(ignored.values()),
            "ignored_fraction": ignored_fraction,
            "ignored_reasons": dict(reasons),
            "gt_shard_sha256": gt_receipts,
        },
        "velocity_evaluated": cfg["velocity_evaluated"],
        "private_labels_read": False,
        "test_predictions_changed": False,
        "official_rank_known": False,
    }
    write(output / "ANALYSIS.json", analysis)
    lines = [
        "# Resultado oficial eAP 3D Detection — 1 de noviembre de 2026",
        "",
        (
            "Generado desde los ZIP devueltos por el scorer. No se leen ni "
            "reconstruyen etiquetas test."
        ),
        "",
        f"**AP oficial: {scores['AP']:.8f}; eAP score: {eap:.8f}.**",
        f"ATE: {scores['ATE']:.8f}; ASE: {scores['ASE']:.8f}; AOE: {scores['AOE']:.8f}.",
        "",
        f"El scorer terminó con exitCode 0. Coinciden los {receipt['frames']:,} frames "
        f"y las {receipt['boxes']:,} predicciones del envío.",
        (
            "La entrega contiene un frame vacío por un PNG público truncado, "
            "documentado antes del scoring."
        ),
        (
            "prediction_result.zip solo contiene metadata de ingestión nula; no "
            "contiene predicciones ni un error de scoring."
        ),
        "",
        "| Clase | GT test | AP oficial (%) | Recall alcanzado a 2 m, rejilla (%) |",
        "|---|---:|---:|---:|",
    ]
    for name, row in classes.items():
        recall = row["curves"]["2.0"]["last_nonzero_confidence_recall_grid"]
        lines.append(
            f"| {name} | {row['ground_truth_boxes']:,} | "
            f"{100 * row['official_AP']:.3f} | {100 * recall:.0f} |"
        )
    lines += [
        "",
        (
            "El recall se lee en la rejilla de 1 % devuelta; no es un recuento "
            "exacto de matches ni un oracle del pool."
        ),
        (
            "AP cero no implica ausencia absoluta de aciertos: las curvas de esas "
            "clases contienen precisión positiva en algún umbral."
        ),
        "",
        "## Diferencia de métricas y población",
        "",
        (
            "La configuración oficial fija min_recall=0.1 y min_precision=0.1. "
            "Sobre cada curva de 101 puntos:"
        ),
        "`AP = mean(max(precision[11:] - 0.1, 0)) / 0.9`.",
        (
            "Esta reconstrucción coincide con los 32 AP por clase/distancia del "
            "scorer. La macro de ocho clases coincide con AP."
        ),
        (
            "La fórmula está también documentada en el [devkit público "
            "nuScenes](https://github.com/nutonomy/nuscenes-devkit/blob/master/pyth"
            "on-sdk/nuscenes/eval/detection/algo.py)."
        ),
        (
            "Esa coincidencia no acredita todos los detalles del matching o los "
            "filtros privados del servidor."
        ),
        "",
        f"La media de las curvas test devueltas, sin esos recortes, es {np.mean(unclipped):.8f}. "
        "Es un diagnóstico",
        (
            "aritmético; no reproduce nuestro AP101 con envolvente ni constituye "
            "otra puntuación oficial."
        ),
        "",
        f"La evaluación local usó {sum(eligible.values()):,} anotaciones elegibles "
        f"e ignoró {sum(ignored.values()):,}",
        f"de {sum(eligible.values()) + sum(ignored.values()):,} anotaciones "
        f"({100 * ignored_fraction:.2f} %) por profundidad fuera del protocolo.",
        (
            "El perfil local limita profundidad de cámara a 0,1–60 m y usa "
            "reglas de dominio/ignorados para GT y predicciones."
        ),
        (
            "La selección local y el test son además secuencias distintas. No se "
            "puede atribuir toda la diferencia a generalización"
        ),
        (
            "ni toda a la métrica; faltan igualar población y scoring sobre "
            "TRAIN/selección públicos."
        ),
        "",
        "## Lectura de los errores y del score compuesto",
        "",
        (
            "Las cuatro clases con recall insuficiente a 2 m reciben 1.0 en los "
            "tres errores TP; el promedio global incluye"
        ),
        (
            "esos valores de penalización. ATE no es simplemente la distancia "
            "media de todas las cajas, ni ASE un porcentaje"
        ),
        (
            "de error de longitud. La orientación es angular; convertir la macro a "
            "grados no elimina las penalizaciones."
        ),
        "",
        (
            "`eAP = (5*mAP + (1-ATE) + (1-ASE) + (1-AOE)) / 8` para estos errores, "
            "todos inferiores a uno en la macro."
        ),
        ("La regla general usa max(0,1-error). La reconstrucción coincide con el score devuelto."),
        (
            "El score compuesto 0.1398 no significa mAP 13.98 %. El JSON indica "
            "velocity_evaluated=false."
        ),
        "",
        "## Conclusión y siguiente decisión",
        "",
        (
            "El envío es válido y demuestra señal útil principalmente en "
            "coches. El resultado no acredita un detector robusto"
        ),
        (
            "para las ocho clases ni liderazgo oficial: no se ha verificado un "
            "leaderboard comparable."
        ),
        (
            "Truck tiene 37.230 anotaciones test; su debilidad no puede explicarse "
            "por falta de soporte test."
        ),
        (
            "No hay un envío oficial baseline sin ranker: este resultado no "
            "permite atribuirle una mejora o un perjuicio en test."
        ),
        (
            "Las 50 propuestas por frame explican el total enviado, pero no prueban "
            "que bajar un umbral o aplicar NMS mejore AP."
        ),
        "",
        (
            "Antes de gastar más entrenamiento, alinear en TRAIN/selección el "
            "cálculo con la configuración observada y reportar"
        ),
        (
            "por separado población raw y dominio local. Reevaluar las "
            "predicciones guardadas de baseline y ranking sin inferir"
        ),
        (
            "otra vez, manteniendo separado el perfil histórico. Después "
            "diagnosticar recall por clase, rango y confusiones."
        ),
        (
            "Elegir las siguientes intervenciones con validación pública, no "
            "ajustando umbrales, modelos o geometría contra test."
        ),
        (
            "Este análisis no lanza reentrenamiento ni nuevas participaciones y no "
            "modifica el artefacto enviado."
        ),
    ]
    (output / "ANALYSIS.md").write_bytes(("\n".join(lines) + "\n").encode())
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
    names = cfg["class_names"]
    values = [100 * classes[c]["official_AP"] for c in names]
    axes[0].barh(names[::-1], values[::-1], color="#267a91")
    axes[0].set_xlabel("AP oficial (%)")
    axes[0].set_title("AP media en 0,5 / 1 / 2 / 4 m")
    for name in names:
        v = detail[f"{name}:2.0"]
        axes[1].plot(v["recall"], v["precision"], label=name)
    axes[1].axvline(0.1, color="gray", linestyle="--", linewidth=0.8)
    axes[1].axhline(0.1, color="gray", linestyle="--", linewidth=0.8)
    axes[1].set(
        xlabel="Recall",
        ylabel="Precisión",
        title="Curvas devueltas a 2 m",
        xlim=(0, 0.65),
        ylim=(0, 1),
    )
    axes[1].legend(fontsize=7, loc="upper right")
    fig.savefig(output / "official_class_analysis.png", dpi=170)
    plt.close(fig)
    write(
        output / "MANIFEST.json", {p.name: sha(p) for p in sorted(output.iterdir()) if p.is_file()}
    )
    print(
        json.dumps(
            {
                "status": "VERIFIED",
                "mAP": scores["AP"],
                "eap_score": eap,
                "AP_reconstruction_max_error": max(ap_errors),
                "output": str(output),
            }
        )
    )


if __name__ == "__main__":
    main()

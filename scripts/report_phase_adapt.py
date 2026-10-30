"""Generate final model card, curves and verification table from completed refit receipts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    """Write only measured training-fit results and explicit deployment limitations."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    args = parser.parse_args()
    result = read(args.model / "RESULTS.json")
    data = read(args.data / "COMPLETE.json")
    bundle = args.model / "FINAL_MODEL"
    manifest = read(bundle / "BUNDLE.json")
    rows = []
    for seed, report in result["reports"].items():
        for epoch in report["history"]:
            for domain, metrics in epoch["domains"].items():
                rows.append(
                    {
                        "seed": seed,
                        "epoch": epoch["epoch"],
                        "updates": epoch["updates"],
                        "domain": domain,
                        "loss": epoch["loss"],
                        **metrics,
                    }
                )
    with (bundle / "training_metrics.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    primary = result["reports"]["7"]
    lines = [
        "# PHASE-ADAPT F1 — modelo final local",
        "",
        "Candidato entrenado con la población local compatible. Se entrega la última "
        "iteración de la semilla 7, elegida antes del ajuste. Las semillas 13 y 21 se "
        "conservan como réplicas, sin escoger la mejor por sus métricas.",
        "",
        f"- Ejemplos utilizados: **{data['rows']}**, de los cuales {data['eap_rows']} "
        f"son TRAIN40 y {data['evttc_rows']} son EvTTC.",
        f"- Grabaciones EvTTC utilizadas: {data['evttc_recordings']}.",
        f"- Consultas EvTTC con sensores/ROI compatibles: {data['evttc_sensor_eligible']}; "
        f"no utilizadas: {data['evttc_sensor_eligible'] - data['evttc_rows']}.",
        f"- Parámetros ajustados: {primary['parameters']}; ocho componentes anteriores "
        "y ambos normalizadores permanecen congelados.",
        f"- Entrenamiento por réplica: {primary['epochs']} épocas completas, "
        f"{primary['updates']} actualizaciones. Todos los ejemplos se visitan en cada época.",
        "- Optimización: AdamW, LR 1e-4, weight decay 1e-4, batch 128, clip 1.0; "
        "pérdida original de fase y cuantiles, masa igual por dominio y grabación.",
        "",
        "## Resultados medidos sobre entrenamiento",
        "",
        "Estas cifras son resustitución sobre los datos del ajuste. No son un test "
        "independiente, un score oficial ni evidencia de SOTA. Los errores usan TTC "
        "convertido desde la coordenada de fase con el soporte original de la cabeza.",
        "",
        "| Semilla | Dominio | MAE inicial (s) | MAE final (s) | Error relativo inicial (%) | "
        "Error relativo final (%) |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for seed, report in result["reports"].items():
        for domain, final in report["history"][-1]["domains"].items():
            before = report["baseline"]["domains"][domain]
            lines.append(
                f"| {seed} | {domain} | {before['mae_seconds']:.6f} | "
                f"{final['mae_seconds']:.6f} | "
                f"{before['relative_absolute_error_pct']:.6f} | "
                f"{final['relative_absolute_error_pct']:.6f} |"
            )
    lines += [
        "",
        "## Uso y verificaciones",
        "",
        "`components/head.pt` contiene el checkpoint nuevo, el optimizador, la receta, "
        "las curvas y la vinculación con datos/código/padres. `BUNDLE.json` verifica "
        "los hashes del grafo original y de la cabeza adaptada. Los componentes "
        "originales se incluyen por separado; su sello histórico no se modifica.",
        "",
        "`head.torchscript.pt` es la cabeza exportada para entradas FP32 finitas "
        "`[B,340]` y logits `[B,2]`. Las columnas son TTC en segundos, fase puntual, "
        "q10 de fase y q90 de fase. El exportado requiere entradas válidas; las "
        "comprobaciones Python de forma/finitud no forman parte del trace.",
        "En Windows, para rutas con caracteres Unicode, cargar TorchScript desde "
        "un archivo binario abierto: `with open(path, 'rb') as stream: "
        "head = torch.jit.load(stream)`. Esto evita el fopen nativo de rutas que "
        "puede fallar aunque el archivo exista.",
        "",
        "```python",
        "from pathlib import Path",
        "from operational.phase_adapt_bundle import AdaptedPredictor",
        "predictor = AdaptedPredictor(Path('FINAL_MODEL'), device='cuda')",
        "prediction = predictor.predict(sensor_row)",
        "```",
        "",
        "`sensor_row` sigue el contrato del adaptador EvTTC nativo del repositorio. "
        "No debe contener etiquetas TTC. La salida adaptada se llama "
        "`PHASE_ADAPT_FINAL_S7`. Los pesos son portables; el entorno y código del "
        "repositorio siguen siendo necesarios para preparar sensores e inferir.",
        "",
        f"Paridad de exportación exacta: `{result['export_exact_parity']}`. "
        f"Replay desde sensores crudos: `{result['raw_sensor_replay']}`. "
        "Detalles en `RAW_REPLAY.json`, `example_input.npz` y `example_output.json`.",
        "",
        "## Datos y límites",
        "",
        "TRAIN40 reutiliza sus estados físicos OOF preservados; EvTTC usa productores "
        "congelados y targets CSV locales con interpolación acotada a20ms. No se "
        "extrapolan targets ni se imputan etiquetas fuera de soporte. Las consultas "
        "sin estado de fusión quedan registradas por separado.",
        "",
        "Las grabaciones EvTTC tienen exposición histórica; su independencia de "
        "adquisición y la equivalencia del target con el scorer oficial siguen sin "
        "certificar. Slider y TEST12 no se usan en este ajuste. La adaptación cambia "
        "el estatus de transferencia del modelo, que debe declararse como entrenado "
        "con EvTTC. No hay validación de seguridad ni demostración de calibración "
        "en dominios nuevos. El runtime conserva el fallback de eventos si falta RGB.",
        "La preparación usa las cajas públicas del objeto y la proyección geométrica "
        "existente basada en rotación. No incluye un detector automático ni demuestra "
        "que omitir la traslación sea adecuado para todas las distancias.",
        "",
        f"Checkpoint final SHA-256: `{manifest['files']['head']['sha256']}`.",
        "",
    ]
    (bundle / "MODEL_CARD.md").write_text("\n".join(lines), encoding="utf-8")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(7, 4))
    for seed, report in result["reports"].items():
        axis.plot(
            [0] + [x["epoch"] for x in report["history"]],
            [report["baseline"]["loss"]] + [x["loss"] for x in report["history"]],
            marker="o",
            label=f"seed {seed}",
        )
    axis.set(xlabel="Época completa", ylabel="Pérdida ponderada de entrenamiento")
    axis.legend()
    figure.tight_layout()
    figure.savefig(bundle / "training_loss.png", dpi=160)
    plt.close(figure)
    artifact_hashes = {}
    for path in sorted(bundle.rglob("*")):
        if path.is_file() and path.name != "ARTIFACTS.json":
            with path.open("rb") as stream:
                artifact_hashes[path.relative_to(bundle).as_posix()] = hashlib.file_digest(
                    stream, "sha256"
                ).hexdigest()
    (bundle / "ARTIFACTS.json").write_text(
        json.dumps(artifact_hashes, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"Generated model card, metrics CSV, loss figure and {len(artifact_hashes)} artifact hashes"
    )


if __name__ == "__main__":
    main()

"""Export live observations and regenerate their table without importing Torch."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path


def window(rows: list[dict]) -> dict:
    """Compute observed wall-clock throughput, including pauses between samples."""
    first, last = rows[0], rows[-1]
    seconds = (
        datetime.fromisoformat(last["time"]) - datetime.fromisoformat(first["time"])
    ).total_seconds()
    updates = last["updates"] - first["updates"]
    assert seconds > 0 and updates > 0
    return {
        "start": first["time"],
        "end": last["time"],
        "seconds": seconds,
        "updates": updates,
        "updates_per_minute": updates * 60 / seconds,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path)
    parser.add_argument("--run", type=Path)
    args = parser.parse_args()
    output = Path(__file__).resolve().parent
    if args.evidence:
        assert args.run is not None

        def read(path: Path) -> dict | list:
            return json.loads(path.read_text(encoding="utf-8-sig"))

        baseline = read(args.evidence / "BASELINE_TELEMETRY.json")
        live = read(args.evidence / "LIVE_TELEMETRY.json")
        extra = args.evidence / "LIVE_TELEMETRY_EXTENSION.json"
        if extra.exists():
            live += read(extra)
        live = sorted(live, key=lambda x: x["time"])
        live = [row for row in live if row["updates"] > 1500]
        assert all(row["recovery"] == 0 for row in live)
        origin = read(args.run / "fits/R_A5/rgb_loader_checkpoints/origin_001500.json")
        expected = read(args.evidence / "cpu_abba/RESULT.json")["checkpoint_sha256"]
        assert origin["origin"]["checkpoint_sha256"] == expected
        runtime = read(args.run / "fits/R_A5/RGB_LOADER_RUNTIME.json")
        profile = read(args.evidence / "OPTIMIZED_STACK_SUMMARY.json")
        for thread in profile:
            for key in ("top_leaf", "top_inclusive"):
                for frame in thread[key]:
                    name = frame.get("file", "").replace("\\", "/")
                    frame["file"] = next(
                        (
                            name[name.index(marker) :]
                            for marker in ("operational/", "src/e_jepa_ttc/", "site-packages/")
                            if marker in name
                        ),
                        name.rsplit("/", 1)[-1],
                    )
        result = {
            "schema": "rgb_loader_live_observation_v1",
            "baseline": baseline,
            "optimized": live,
            "origin": origin,
            "last_runtime_receipt": runtime,
            "optimized_main_thread_profile": profile,
            "source_sha256": {
                name: hashlib.sha256((args.evidence / name).read_bytes()).hexdigest()
                for name in (
                    "BASELINE_TELEMETRY.json",
                    "LIVE_TELEMETRY.json",
                    "OPTIMIZED_STACKS.json",
                )
            },
            "limits": (
                "Sequential live observations, different batches and windows, "
                "not a controlled whole-training ABBA. ETA only for remaining R_A5 updates; "
                "excludes other V13 stages."
            ),
        }
        (output / "LIVE_RESULT.json").write_text(
            json.dumps(result, indent=2) + "\n", encoding="utf-8"
        )
    result = json.loads((output / "LIVE_RESULT.json").read_text(encoding="utf-8"))
    baseline, live = window(result["baseline"]), window(result["optimized"])
    last = result["optimized"][-1]
    eta_hours = (30330 - last["updates"]) / live["updates_per_minute"] / 60
    lines = [
        "# Entrenamiento reanudado: observación en vivo",
        "",
        "Tabla regenerada desde `LIVE_RESULT.json` por `publish_live.py` sin argumentos.",
        "",
        "| Ventana | Duración (s) | Updates | Updates/min |",
        "|---|---:|---:|---:|",
    ]
    for title, value in (("Anterior", baseline), ("Optimizada", live)):
        lines.append(
            f"| {title} | {value['seconds']:.2f} | {value['updates']} "
            f"| {value['updates_per_minute']:.2f} |"
        )
    lines += [
        "",
        f"Última muestra: {last['time']}; **{last['updates']}/30.330**, "
        f"duraderos {last['durable']}; recuperación acumulada {last['recovery']}.",
        "",
        f"Extrapolación para terminar sólo R_A5 a este ritmo: **{eta_hours:.1f} horas**.",
        "",
        "Estas ventanas tienen distinta duración, distintos batches y carga del equipo variable. "
        "El cociente de sus ritmos es descriptivo; no demuestra una aceleración "
        "controlada de todo el entrenamiento.",
        "",
        "Se restauró el hash exacto del checkpoint 1.500. El recibo del siguiente checkpoint "
        "está incluido en el JSON. Se mantienen BF16, objetivo, sampler y estado del optimizador.",
        "",
        "El perfil restante señala preparación de imágenes RGB y trabajo del modelo. "
        "Los porcentajes de pilas del hilo principal no son tiempos CUDA "
        "y los inclusivos se solapan.",
        "",
    ]
    (output / "LIVE.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"baseline": baseline, "optimized": live, "eta_R_A5_hours": eta_hours}))


if __name__ == "__main__":
    main()

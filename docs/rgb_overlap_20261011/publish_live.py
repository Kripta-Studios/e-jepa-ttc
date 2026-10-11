"""Publish bounded live observations without importing Torch or changing training."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ORIGIN_UPDATE = 2124
TARGET_UPDATES = 30330


def read(path: Path) -> tuple[Any, str]:
    """Tolerate the telemetry writer replacing a JSON file during observation."""
    for attempt in range(10):
        raw = path.read_bytes()
        try:
            return json.loads(raw.decode("utf-8-sig")), hashlib.sha256(raw).hexdigest()
        except json.JSONDecodeError:
            if attempt == 9:
                raise
            time.sleep(0.05)
    raise AssertionError("Unreachable")


def stamp(row: dict[str, Any]) -> float:
    return datetime.fromisoformat(row["time"]).timestamp()


def window(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Use observed endpoints, retaining zero-rate samples and pauses inside them."""
    if len(rows) < 2:
        raise ValueError("At least two observations are required")
    first, last = rows[0], rows[-1]
    seconds = stamp(last) - stamp(first)
    if seconds <= 0 or any(
        b["updates"] < a["updates"] for a, b in zip(rows, rows[1:], strict=False)
    ):
        raise ValueError("Telemetry timestamps or update counters are inconsistent")
    advances = int(last["updates"]) - int(first["updates"])
    rate = advances * 60 / seconds
    return {
        "start": first["time"],
        "end": last["time"],
        "seconds": seconds,
        "start_updates": first["updates"],
        "end_updates": last["updates"],
        "updates": advances,
        "updates_per_minute": rate,
        "eta_R_A5_hours": (TARGET_UPDATES - last["updates"]) / rate / 60 if rate > 0 else None,
    }


def public_rows(raw: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Retain only timestamps, counters and numeric resource measurements."""
    result = []
    for item in raw:
        row = {
            name: item[name]
            for name in ("time", "updates", "durable", "pending", "recovery", "available_ram")
        }
        row["processes"] = [
            {name: process.get(name) for name in ("pid", "rss", "private", "cpu_s")}
            for process in item.get("processes", [])
        ]
        if item.get("gpu_csv"):
            parts = item["gpu_csv"].split(",")
            try:
                values = [float(part.strip()) for part in parts]
            except ValueError:
                values = []
            if len(values) == 5:
                row["gpu"] = dict(
                    zip(
                        (
                            "utilization_percent",
                            "memory_used_mib",
                            "power_w",
                            "sm_clock_mhz",
                            "temperature_c",
                        ),
                        values,
                        strict=True,
                    )
                )
        result.append(row)
    return sorted({row["time"]: row for row in result}.values(), key=stamp)


def receipt(raw: dict[str, Any]) -> dict[str, Any]:
    """Project known runtime fields; no local paths or individual samples are copied."""
    fields = (
        "schema status fit_id freeze_sha256 checkpoint_sha256 completed_updates "
        "elapsed_s_since_restore objective_changed geometry_precision "
        "speculative_preparation_included_in_counters"
    ).split()
    result = {name: raw[name] for name in fields if name in raw}
    nested = {
        "origin": "checkpoint_sha256 completed_updates",
        "geometry": "enabled",
        "decoder": "workers max_pending_bytes peak_pending_bytes peak_pending_reads "
        "parallel_reads fallback_reads hits decode_worker_seconds consumer_wait_seconds "
        "largest_decoded_shard_bytes rgb_cache_bytes",
        "prefetch": "bound captured consumed_batches prefetch_depth prepared_batches "
        "queued_batches_max source_prepare_s source_wait_s submitted_batches worker_threads",
    }
    for section, names in nested.items():
        result[section] = {
            name: raw.get(section, {})[name]
            for name in names.split()
            if name in raw.get(section, {})
        }
    return result


def summarize_stacks(raw: dict[str, Any]) -> list[dict[str, Any]]:
    """Compute weighted inclusive/leaf samples with repository-relative filenames."""
    frames = raw["shared"]["frames"]

    def key(index: int) -> tuple[str, str]:
        frame = frames[index]
        path = frame.get("file", "").replace("\\", "/")
        path = next(
            (
                path[path.index(marker) :]
                for marker in ("operational/", "src/e_jepa_ttc/", "site-packages/")
                if marker in path
            ),
            path.rsplit("/", 1)[-1],
        )
        return frame["name"], path

    result = []
    watched = ("_producer_loss", "forward", "transport_physical_features", "_summarise_one")
    for profile in raw["profiles"]:
        if profile["type"] != "sampled":
            continue
        samples = profile["samples"]
        weights = profile.get("weights", [1.0] * len(samples))
        total = sum(weights)
        inclusive: Counter = Counter()
        leaf: Counter = Counter()
        selected: Counter = Counter()
        for stack, weight in zip(samples, weights, strict=True):
            keys = {key(index) for index in stack}
            names = {item[0] for item in keys}
            for item in keys:
                inclusive[item] += weight
            if stack:
                leaf[key(stack[-1])] += weight
            for name in watched:
                if name in names:
                    selected[name] += weight

        def top(counter: Counter, count: int, denominator: float) -> list[dict[str, Any]]:
            return [
                {
                    "name": name,
                    "file": path,
                    "weight": weight,
                    "percent": 100 * weight / denominator if denominator else 0,
                }
                for (name, path), weight in counter.most_common(count)
            ]

        name_parts = profile.get("name", "").split('"')
        thread = name_parts[1] if len(name_parts) > 1 and name_parts[1] else "unnamed"
        result.append(
            {
                "thread": thread,
                "samples": len(samples),
                "sample_weight_sum": total,
                "unit": profile.get("unit"),
                "top_inclusive": top(inclusive, 12, total),
                "top_leaf": top(leaf, 5, total),
                "selected_inclusive_percent": {
                    name: 100 * selected[name] / total if total else 0 for name in watched
                },
            }
        )
    return result


def export(evidence: Path, run: Path) -> dict[str, Any]:
    baseline, baseline_sha = read(evidence / "BASELINE_TELEMETRY.json")
    live, live_sha = read(evidence / "LIVE_TELEMETRY.json")
    pointer, pointer_sha = read(evidence / "ORIGIN_CHECKPOINT_POINTER.json")
    fit = run / "fits/R_A5"
    origin, origin_sha = read(fit / "rgb_overlap_checkpoints/origin_002124.json")
    runtime, runtime_sha = read(fit / "RGB_OVERLAP_RUNTIME.json")
    stacks, stacks_sha = read(evidence / "LIVE_STACKS.json")
    queue, queue_sha = read(evidence / "QUEUE_VERIFICATION.json")
    baseline = public_rows([row for row in baseline if row["updates"] < ORIGIN_UPDATE])
    live = public_rows([row for row in live if row["updates"] > ORIGIN_UPDATE])
    expected = pointer["checkpoint_sha256"]
    if (
        pointer["completed_updates"] != ORIGIN_UPDATE
        or origin["origin"]["completed_updates"] != ORIGIN_UPDATE
        or origin["checkpoint_sha256"] != expected
        or origin["origin"]["checkpoint_sha256"] != expected
        or runtime["origin"]["checkpoint_sha256"] != expected
        or runtime["completed_updates"] <= ORIGIN_UPDATE
        or runtime["freeze_sha256"] != origin["freeze_sha256"]
    ):
        raise ValueError("Restore origin or subsequent durable runtime differs")
    for item in (origin, runtime):
        if (
            item["decoder"]["workers"] != 4
            or item["prefetch"]["prefetch_depth"] != 2
            or item["geometry"]["enabled"] is not False
            or item["objective_changed"] is not False
        ):
            raise ValueError("Observed runtime does not match the admitted 4/depth2 recipe")
    window(baseline)
    window(live)
    return {
        "schema": "rgb_overlap_live_observation_v1",
        "implementation_commit": "2b38c31",
        "origin_update": ORIGIN_UPDATE,
        "target_updates": TARGET_UPDATES,
        "baseline": baseline,
        "optimized": live,
        "origin_receipt": receipt(origin),
        "last_runtime_receipt": receipt(runtime),
        "profile_summary": summarize_stacks(stacks),
        "queue_verification": {
            "time": queue["time"],
            "pause": queue["pause"],
            "stop_request": queue["stop_request"],
            "journal": {
                name: queue["journal"][name]
                for name in (
                    "completed_updates",
                    "durable_updates",
                    "pending_update_upper",
                    "recovery_upper",
                )
            },
            "module_process_counts": dict(
                Counter(
                    process["module"]
                    for process in queue["processes"]
                    if process["module"].startswith("operational.rgb_port_rgb_overlap.")
                )
            ),
        },
        "source_sha256": {
            "BASELINE_TELEMETRY.json": baseline_sha,
            "LIVE_TELEMETRY.json": live_sha,
            "ORIGIN_CHECKPOINT_POINTER.json": pointer_sha,
            "origin_002124.json": origin_sha,
            "RGB_OVERLAP_RUNTIME.json": runtime_sha,
            "LIVE_STACKS.json": stacks_sha,
            "QUEUE_VERIFICATION.json": queue_sha,
        },
        "filtering": {
            "baseline": "updates < 2124; omit deliberate final pause plateau",
            "optimized": "updates > 2124; omit initial restore/startup samples",
        },
        "limits": "Sequential observations, different batches and host load; not controlled ABBA. "
        "ETA applies only to R_A5. Resource samples do not measure exact process peaks.",
    }


def render(result: dict[str, Any]) -> str:
    baseline, live = result["baseline"], result["optimized"]
    old, current = window(baseline), window(live)
    last = live[-1]
    runtime = result["last_runtime_receipt"]
    elapsed = float(runtime["elapsed_s_since_restore"])
    wait = float(runtime["prefetch"]["source_wait_s"])
    processes = [
        max(row["processes"], key=lambda item: item.get("private") or 0)
        for row in live
        if row["processes"]
    ]
    lines = [
        "# R_A5 reanudado: observación del entrenamiento completo",
        "",
        "Regenerado desde `LIVE_RESULT.json` por `publish_live.py`, sin argumentos. "
        "Esta es una ventana registrada, no un monitor actualizado automáticamente.",
        "",
        f"Implementación: `{result['implementation_commit']}`. Restauración desde "
        f"**{ORIGIN_UPDATE} updates durables**, con el hash exacto del checkpoint conservado.",
        "",
        "| Ventana | Inicio | Fin | Duración (s) | Avance | Updates/min |",
        "|---|---|---|---:|---:|---:|",
    ]
    for name, value in (("Anterior", old), ("Overlap", current)):
        lines.append(
            f"| {name} | {value['start']} | {value['end']} | {value['seconds']:.2f} | "
            f"{value['updates']} | {value['updates_per_minute']:.2f} |"
        )
    eta = current["eta_R_A5_hours"]
    lines.extend(
        [
            "",
            f"Última muestra: **{last['updates']}/{TARGET_UPDATES}**; "
            f"durables **{last['durable']}**, recuperación acumulada **{last['recovery']}**.",
            "",
            "Extrapolación para terminar sólo R_A5 al ritmo de toda la ventana: "
            + (f"**{eta:.2f} horas**." if eta is not None else "no disponible, sin avance."),
            "",
            "La referencia excluye el plateau de la pausa voluntaria en 2.124; "
            "la ventana nueva comienza en la primera muestra con avance tras restaurar. "
            "Las ventanas tienen distintos batches, duraciones y carga del host. "
            "No constituyen una comparación controlada ni incluyen las fases posteriores de V13.",
            "",
            "## Ventanas recientes",
            "",
            "| Solicitada | Duración realmente disponible (s) | Updates/min | Cobertura completa |",
            "|---|---:|---:|---|",
        ]
    )
    for minutes in (5, 10, 15):
        cutoff = stamp(last) - minutes * 60
        before = [i for i, row in enumerate(live) if stamp(row) <= cutoff]
        subset = live[before[-1] :] if before else live
        measured = window(subset)
        lines.append(
            f"| {minutes} min | {measured['seconds']:.2f} | "
            f"{measured['updates_per_minute']:.2f} | {'Sí' if before else 'No'} |"
        )
    lines.extend(
        [
            "",
            "Si falta historial, la tabla muestra la ventana disponible y lo indica. "
            "No se fabrican muestras ni ritmos de 10/15 minutos.",
            "",
            "## Checkpoint y preparación",
            "",
            f"Recibo durable observado: **{runtime['completed_updates']} updates**; "
            f"**{elapsed:.2f} s** desde la restauración hasta el recibo.",
            "",
            f"Espera acumulada del consumidor por entradas: **{wait:.3f} s** "
            f"(**{100 * wait / elapsed:.2f}%** de ese intervalo). "
            "Esto mide espera en los futuros, no toda la preparación ni una descomposición "
            "del cómputo GPU. Preparación especulativa y trabajo solapado siguen incluidos "
            "en sus contadores separados.",
            "",
            "El recibo confirma **cuatro decodificadores, dos batches anticipados y "
            "optimización geométrica desactivada**. BF16, pérdida, sampler y estados "
            "del entrenamiento se mantienen.",
        ]
    )
    verification = result["queue_verification"]
    lines.extend(
        [
            "",
            f"Verificación independiente de la cola: {verification['time']}; "
            f"**{verification['journal']['completed_updates']} updates**, "
            f"pausa **{verification['pause']}**, STOP_REQUEST **{verification['stop_request']}**. "
            "Los procesos observados incluyen launcher y ejecutable Python para cada módulo; "
            "no representan entrenamientos duplicados.",
            "",
            "## Foco restante según pilas",
            "",
        ]
    )
    main_thread = next(
        (item for item in result["profile_summary"] if item["thread"] == "MainThread"), None
    )
    if main_thread:
        lines.extend(
            [
                f"Hilo principal: **{main_thread['samples']} muestras**. "
                "Porcentajes inclusivos dentro de sus muestras registradas:",
                "",
                "| Función presente en la pila | Muestras ponderadas (%) |",
                "|---|---:|",
            ]
        )
        for name in ("_producer_loss", "forward", "transport_physical_features", "_summarise_one"):
            percent = main_thread["selected_inclusive_percent"][name]
            lines.append(f"| `{name}` | {percent:.1f} |")
    lines.extend(
        [
            "",
            "Los porcentajes inclusivos se solapan: **no deben sumarse ni interpretarse "
            "como tiempo CUDA o tiempo de pared**. Los lectores siguen descomprimiendo datos; "
            "esa actividad se solapa con el entrenamiento y la espera medida del consumidor "
            "es pequeña en este recibo. El foco siguiente es el trabajo del modelo, sus "
            "operaciones geométricas y el lanzamiento de operaciones, que requieren perfiles "
            "adicionales antes de atribuir tiempos GPU concretos.",
            "",
            "## Memoria observada",
            "",
        ]
    )
    if processes:
        rss = [item["rss"] / 1024**3 for item in processes]
        private = [item["private"] / 1024**3 for item in processes if item["private"] is not None]
        lines.extend(
            [
                "Se selecciona por muestra el proceso de entrenamiento con más memoria privada; "
                "se conservan también las mediciones del launcher en el JSON.",
                "",
                "| Medida | Mínimo (GiB) | Mediana (GiB) | Máximo muestreado (GiB) |",
                "|---|---:|---:|---:|",
            ]
        )
        available = [row["available_ram"] / 1024**3 for row in live]
        for name, values in (("RSS", rss), ("Privada", private), ("RAM disponible", available)):
            if values:
                lines.append(
                    f"| {name} | {min(values):.3f} | {statistics.median(values):.3f} | "
                    f"{max(values):.3f} |"
                )
    lines.extend(
        [
            "",
            "Son muestras periódicas, no picos exactos ni una garantía de estabilidad "
            "prolongada. La memoria privada no equivale a RSS ni al margen global de "
            "memoria comprometida de Windows.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-root", type=Path)
    parser.add_argument("--run", type=Path)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if bool(args.evidence_root) != bool(args.run):
        parser.error("--evidence-root and --run must be supplied together")
    if args.check and args.evidence_root:
        parser.error("--check uses the existing public JSON only")
    target = HERE / "LIVE_RESULT.json"
    if args.evidence_root:
        result = export(args.evidence_root, args.run)
        target.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    else:
        result, _ = read(target)
    text = render(result)
    markdown = HERE / "LIVE.md"
    if args.check:
        if markdown.read_text(encoding="utf-8") != text:
            raise ValueError("LIVE.md differs from public evidence")
        print("Public live observation verified")
    else:
        markdown.write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

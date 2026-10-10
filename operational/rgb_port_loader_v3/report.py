"""Regenerate the paired CPU trial table and exact-parity admission evidence."""

from __future__ import annotations

import argparse
import json
import statistics
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any


def build(root: Path, output: Path) -> dict[str, Any]:
    """Fail on parity or QA errors; report both matched repeats without selection."""
    suites = ET.parse(root / "QA_JUNIT.xml").getroot()
    counts = {
        name: sum(int(suite.attrib.get(name, 0)) for suite in suites.iter("testsuite"))
        for name in ("tests", "failures", "errors", "skipped")
    }
    if counts["failures"] or counts["errors"] or counts["tests"] < 6:
        raise ValueError("Required loader regression QA failed or absent")
    trials, comparisons = {}, []
    for fit in ("A5", "C2F"):
        baseline_times, candidate_times = [], []
        for repeat in (1, 2):
            baseline, candidate = [
                json.loads((root / f"matched_{mode}_{repeat}" / f"{fit}.json").read_text())
                for mode in ("baseline", "prefetch")
            ]
            for key in ("samples", "order_sha256", "batch_sha256", "shard_reads", "torch_threads"):
                if baseline[key] != candidate[key]:
                    raise ValueError(f"Parity mismatch {fit}/{repeat}/{key}")
            for mode, value in (("baseline", baseline), ("prefetch", candidate)):
                trials[f"{fit}/{mode}/{repeat}"] = value
            baseline_times.append(baseline["wall_s_including_hash"])
            candidate_times.append(candidate["wall_s_including_hash"])
        before, after = statistics.mean(baseline_times), statistics.mean(candidate_times)
        comparisons.append(
            {
                "fit": fit,
                "samples_per_repeat": baseline["samples"],
                "baseline_mean_s": before,
                "prefetch_mean_s": after,
                "wall_reduction_percent": 100 * (1 - after / before),
                "baseline_s": baseline_times,
                "prefetch_s": candidate_times,
            }
        )
    result = {
        "status": "PASSED",
        "exact_batch_parity": True,
        "qa": counts,
        "method": "ABBA sequence, two independent CPU consumers; four decoder threads total; "
        "two queued batches per fit; distinct fixed orders; two torch threads per fit.",
        "comparisons": comparisons,
        "trials": trials,
        "limitations": [
            "No optimizer updates or GPU work in these trials.",
            "Wall time includes SHA256 work overlapping prefetch.",
            "OS file cache was not flushed; these are warm input-cache trials.",
            "Only two logical groups per fit, not the entire epoch.",
            "The live baseline trial was excluded because C2F paused during it.",
        ],
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "CPU_COMPARISON.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    lines = [
        "# V13: cuatro decodificadores y dos batches anticipados",
        "",
        "Tabla regenerada por `operational/rgb_port_loader_v3/report.py`.",
        "",
        "| Fit | Muestras/repetición | Base (s) | Paralelo + prefetch (s) | Reducción |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in comparisons:
        lines.append(
            f"| {row['fit']} | {row['samples_per_repeat']} | "
            f"{row['baseline_mean_s']:.3f} | {row['prefetch_mean_s']:.3f} | "
            f"{row['wall_reduction_percent']:.2f}% |"
        )
    lines += [
        "",
        f"{counts['tests']} pruebas aprobadas; igualdad exacta de todos los campos "
        "de cada batch (bytes, dtype, shape y metadatos), orden y lecturas.",
        "",
        "## Alcance",
        "",
        result["method"],
        "",
        *["- " + item for item in result["limitations"]],
        "",
        "## Implementación",
        "",
        "`operational/rgb_port_loader_v3/loader.py` conserva el lector NPZ original, "
        "las comprobaciones de teacher/tokens y el collate completo. Dos hilos por fit "
        "descomprimen shards en paralelo; un coordinador por fit anticipa dos batches. "
        "Los hilos comparten arrays, sin serialización entre procesos. No se omiten "
        "validaciones ni se cambia la precisión, pérdida, sampler u optimizador.",
        "",
        "`producer.py` espera a que termine la restauración y el prewarm CUDA antes de "
        "activar prefetch. `contracts.py` vincula código, QA y freeze histórico; "
        "`queue.py` conserva propiedad de procesos, límites y reintentos anteriores.",
        "",
        "El ensayo CPU se realizó con una reserva de commit de Windows de 3 GiB. "
        "La revisión posterior solicitada por el usuario usa 1 GiB; véase MEMORY_RESERVE.md. "
        "El ensayo inicial con "
        "entrenamientos vivos hizo que C2F se pausara de forma segura en 19.081 updates "
        "al detectar 2,606 GiB de margen. A5 se pausó después de forma solicitada en "
        "27.633 para aislar las mediciones. RAM libre y margen de commit son distintos.",
        "",
        "La evidencia CPU no demuestra todavía una mejora del entrenamiento GPU. "
        "Los recibos `fits/<fit>/loader_v3_checkpoints/` registran esa continuación "
        "por separado cuando se ejecuta.",
        "",
    ]
    (output / "README.md").write_text("\n".join(lines), encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    build(args.root, args.output)

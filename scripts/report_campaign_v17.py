"""Generate campaign state, cost scope and accounting without inventing a candidate."""

from __future__ import annotations

from collections import defaultdict
from datetime import UTC, datetime

import pandas as pd

from operational.range_transfer.common import read, sha, write
from operational.representation_audit.run import ROOT, RUN, accounting


def main() -> None:
    counters = accounting()
    specs = {
        "A": "NEXT_DECISION.json",
        "B": "PRECISION_NEXT_DECISION.json",
        "D": "D_NEXT_DECISION.json",
        "E": "GEOMETRY_SCREEN_DECISION.json",
        "F": "TEMP_GEOMETRY_SCREEN_DECISION.json",
        "G": "SPATIAL_SCREEN_DECISION.json",
        "H": "NOCLIP_SCREEN_DECISION.json",
        "I": "NONLINEAR_SCREEN_DECISION.json",
    }
    decisions = {
        key: read(RUN / path) if (RUN / path).exists() else {"status": "PENDING"}
        for key, path in specs.items()
    }
    for key in ("E", "F", "G", "H", "I"):
        replica_path = RUN / specs[key].replace("SCREEN", "REPLICAS")
        if replica_path.exists():
            screen_status = decisions[key]["status"]
            decisions[key] = {**read(replica_path), "screen_status": screen_status}
    selected_path = RUN / "SELECTED_REPLICATION_FAMILY.json"
    other_family = selected_path.exists() and read(selected_path)["family"] != "NONLINEAR"
    if decisions["I"]["status"] == "PENDING" and (
        other_family
        or any(decisions[key]["status"] not in ("PENDING", "NO_CANDIDATE") for key in ("G", "H"))
    ):
        decisions["I"] = {"status": "NOT_TRIGGERED", "reason": "G/H admitted a family"}
    admitted = [
        key
        for key, value in decisions.items()
        if value["status"] not in ("PENDING", "NO_CANDIDATE", "NOT_TRIGGERED")
    ]
    state = (
        "INTEGRATION_OR_REPLICATION_REQUIRED"
        if admitted
        else (
            "RUNNING"
            if any(d["status"] == "PENDING" for d in decisions.values())
            else "NO_CANDIDATE"
        )
    )
    book = read(RUN / "GPU_BUDGET.json")
    settled = defaultdict(float)
    reserved = 0.0
    for row in book["entries"]:
        scope = row["scope"]
        category = (
            "feature_extraction"
            if scope.startswith("spatial_extract")
            else "evaluation"
            if "evaluat" in scope
            else "technical_or_parity"
            if any(x in scope.lower() for x in ("continuous", "split", "qa"))
            else "scientific_training"
        )
        if row["status"] == "SETTLED":
            settled[category] += row["charged_s"]
        else:
            reserved += row["charged_s"]
    cost = {
        "scope": (
            "wall seconds inside reserved GPU scopes, including enclosed I/O; "
            "not CUDA kernel time or E2E latency"
        ),
        "settled_seconds_by_category": dict(settled),
        "settled_total_s": sum(settled.values()),
        "reserved_upper_s": reserved,
        "charged_upper_s": sum(settled.values()) + reserved,
        "cap_s": 64800,
    }
    write(RUN / "COST_SUMMARY.json", cost)
    fits = []
    for path in sorted((RUN / "fits").glob("*/CHECKPOINT_RECEIPT.json")):
        receipt = read(path)
        fits.append(
            {
                "fit": path.parent.name,
                "status": receipt["status"],
                "updates": receipt["completed_updates"],
                "checkpoint_sha256": receipt["checkpoint_sha256"],
            }
        )
    closed = 24 if (RUN / "group_probe/RESULTS.json").exists() else 0
    calibrations = [
        RUN / "geometry/P_HEIGHT_CALIBRATION.json",
        RUN / "temporal_geometry/P_HEIGHT_CALIBRATION.json",
        RUN / "spatial_geometry/SP0_CALIBRATION.json",
        RUN / "spatial_geometry/SP1_CALIBRATION.json",
    ]
    closed += sum(path.exists() for path in calibrations)
    status = {
        "status": state,
        "SOTA": False,
        "utc": datetime.now(UTC).isoformat(),
        "queues": decisions,
        "accounting": counters,
        "closed_form_fits": closed,
        "cost": cost,
        "next_action": (
            "Complete pending registered queues; integrate only admitted families. "
            "Preserve all negatives."
        )
        if state != "NO_CANDIDATE"
        else (
            "No promotion, threshold rescue or external selection. "
            "Keep the complete negative evidence."
        ),
    }
    write(RUN / "CAMPAIGN_NEXT_DECISION.json", status)
    pd.DataFrame(fits).to_csv(RUN / "FIT_LEDGER.csv", index=False)
    lines = [
        "# V17: estado y contabilidad",
        "",
        f"Estado: **{state}**. No hay afirmación SOTA.",
        "",
        f"Generado: {status['utc']}. Los estados pendientes no son resultados negativos "
        "ni tareas completadas.",
        "",
        "| Cola | Estado |",
        "|---|---|",
    ]
    lines += [f"| {key} | {value['status']} |" for key, value in decisions.items()]
    lines += [
        "",
        f"Updates científicos: **{counters['scientific']}**; "
        f"técnicos: **{counters['technical']}**; "
        f"recuperación superior: **{counters['recovery_upper']}**; "
        f"pendiente superior: **{counters['pending_upper']}**. "
        f"Total físico superior: **{counters['physical_upper']} / 1.000.000**.",
        "",
        f"Ajustes cerrados adicionales: **{closed}**. Se contabilizan como entrenamiento "
        "aunque no usan optimizer.step. Las réplicas comparten encoders congelados; "
        "no son representaciones entrenadas independientemente.",
        "",
        "| Scope GPU | Segundos liquidados |",
        "|---|---:|",
    ]
    lines += [f"| {key} | {value:.3f} |" for key, value in sorted(settled.items())]
    lines += [
        "",
        f"Total liquidado: **{cost['settled_total_s']:.3f} s**. "
        f"Reserva viva/conservadora: **{reserved:.3f} s**. "
        f"Cargo superior: **{cost['charged_upper_s']:.3f} / 64.800 s**.",
        "",
        "Este coste incluye carga y guardado dentro de los scopes GPU. "
        "No mide energía, tiempo puro de kernels ni latencia completa desde eventos. "
        "El equipo es compartido y se preservan los propietarios ajenos.",
        "",
        "## Entrenamientos",
        "",
        "| Fit | Estado | Updates |",
        "|---|---|---:|",
    ]
    lines += [f"| {r['fit']} | {r['status']} | {r['updates']} |" for r in fits]
    lines += [
        "",
        "Los controles U0 de B se reutilizan en H; Q1 seed7 de E se reutiliza en F. "
        "No se cuentan como nuevos entrenamientos ni como evidencia independiente "
        "al aparecer en otra tabla.",
        "",
        "## Evidencia y límites",
        "",
        "- `RESULTS.md`: comparación A y reproducción de métricas.",
        "- `QUEUES_RESULTS.md`: todas las colas adicionales, incluyendo NA por cobertura inválida.",
        "- `MECHANISMS.md`: hipótesis, alcance de los diagnósticos y resultados negativos.",
        "- `PUBLIC_COMPARABILITY.md`: por qué estas cifras no acreditan SOTA oficial.",
        "- `REPRODUCTION.md`: reanudación real, pesos y dependencias.",
        "",
        "V es desarrollo interno expuesto. FCWD/Dev32 no han entrenado ni seleccionado estos fits. "
        "No se realizan submissions ni publicaciones automáticas.",
    ]
    (ROOT / "docs/representation_audit_20261012/CAMPAIGN_STATUS.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    write(
        RUN / "REPORT_GENERATION.json",
        {
            "source_sha256": sha(__import__("pathlib").Path(__file__)),
            "utc": status["utc"],
            "status": state,
        },
    )


if __name__ == "__main__":
    main()

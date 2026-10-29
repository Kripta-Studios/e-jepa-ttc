"""Record a specific observed resource dependency without classifying pending science."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    run = root / "artifacts/representation_audit_20261012"
    evidence = sorted((run / "resource_observation").glob("*.json"))[-1]
    observation = json.loads(evidence.read_text(encoding="utf-8"))
    denied = [r for r in observation["first_and_denied_snapshots"] if not r["allowed"]]
    receipt = json.loads((run / "fits/SP1-S7/CHECKPOINT_RECEIPT.json").read_text())
    if receipt["status"] == "COMPLETE":
        if denied:
            raise ValueError("The latest observed resume did not pass every resource check")
        result = {
            "status": "RESOLVED",
            "utc": datetime.now(UTC).isoformat(),
            "fit": "SP1-S7",
            "completed_updates": receipt["completed_updates"],
            "checkpoint_sha256": receipt["checkpoint_sha256"],
            "successful_observation": str(evidence.relative_to(run)),
            "guard_calls": observation["guard_calls"],
            "denied_checks": 0,
            "historical_dependency_preserved": "RESOURCE_DEPENDENCY.json",
            "resolution": "shared memory became available; unchanged resource guard and recipe",
        }
        (run / "RESOURCE_RESOLVED.json").write_text(
            json.dumps(result, indent=2) + "\n", encoding="utf-8"
        )
        report = root / "docs/representation_audit_20261012/RESOURCE_STATE.md"
        historical = report.read_text(encoding="utf-8").split("\n## Resolución verificada\n")[0]
        report.write_text(
            historical + "\n## Resolución verificada\n\n"
            "La pausa anterior es histórica. SP1 terminó sus6000updates tras liberarse "
            "memoria compartida, con la receta y el guard sin cambios. "
            f"La reanudación observada pasó sus{observation['guard_calls']} comprobaciones "
            "de recursos, sin rechazos. G y H completaron después sus evaluaciones. "
            "`RESOURCE_RESOLVED.json` conserva el hash final y la evidencia; "
            "`RESOURCE_DEPENDENCY.json` conserva la instantánea original. "
            "El estado actual de toda la campaña se regenera en CAMPAIGN_STATUS.md.\n",
            encoding="utf-8",
        )
        return
    if not denied:
        raise ValueError("No observed resource refusal")
    if receipt["status"] == "COMPLETE" or (run / "PROCESS.lock").exists():
        raise ValueError("Do not label a complete or live fit as blocked")
    guard = denied[-1]
    command = f"pwsh -NoProfile -File '{root / 'scripts/Resume-ResearchV17.ps1'}'"
    result = {
        "status": "RESOURCE_DEPENDENCY",
        "utc": datetime.now(UTC).isoformat(),
        "fit": "SP1-S7",
        "completed_updates": receipt["completed_updates"],
        "checkpoint_sha256": receipt["checkpoint_sha256"],
        "expected_updates": 6000,
        "observed_commit_headroom_bytes": guard["commit_headroom_bytes"],
        "minimum_in_fit_commit_headroom_bytes": 2 * 1024**3,
        "observed_available_ram_bytes": guard["available_bytes"],
        "observed_own_rss_bytes": guard["v14_tree_rss_bytes"],
        "evidence": str(evidence.relative_to(run)),
        "resume_command": command,
        "pending": [
            "finish SP1",
            "freeze/evaluate G",
            "evaluate completed H",
            "I only if G and H are rejected; replicate/integrate only admitted family",
        ],
        "scientific_negative_claimed_for_pending_work": False,
        "dependency": "sufficient shared Windows commit headroom under unchanged resource guard",
        "next_restart": "wait for released commit; compute budget does not reserve system memory",
    }
    (run / "RESOURCE_DEPENDENCY.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "# V17: dependencia de recursos observada",
        "",
        f"Instantánea: {result['utc']}. Consultar CAMPAIGN_STATUS para el estado actual.",
        "",
        f"SP1 conserva **{receipt['completed_updates']}/6000 updates** en un checkpoint durable. "
        "La evaluación del contraste G está cerrada hasta completar ambos endpoints. "
        "H tiene sus6000updates completos, pero su evaluación espera el orden registrado. "
        "I tiene implementación y pruebas CPU; sólo se activa si G y H son rechazadas.",
        "",
        "| Medida en el rechazo observado | Bytes |",
        "|---|---:|",
        f"| Margen de commit Windows | {guard['commit_headroom_bytes']} |",
        f"| Mínimo del guard durante fit | {2 * 1024**3} |",
        f"| RAM física disponible | {guard['available_bytes']} |",
        f"| RSS del proceso propio | {guard['v14_tree_rss_bytes']} |",
        "",
        "El bloqueo observado fue el margen de memoria comprometida, aunque había RAM física "
        "disponible. La consulta al driver devolvió carga diferida de módulos, modo2. "
        "[NVIDIA documenta LAZY como predeterminado en estas versiones]"
        "(https://docs.nvidia.com/cuda/cuda-programming-guide/"
        "04-special-topics/lazy-loading.html). "
        "No se cambió la configuración global ni la de otros propietarios.",
        "",
        "Las pausas preservan pesos, optimizador, scheduler, RNG y diario. "
        "El observador sólo registra el valor devuelto por el guard; no cambia su decisión. "
        "Su coste entra en los scopes normales de tiempo GPU. La auditoría de muestras "
        "reproduce la secuencia completa desde cada semilla y comprueba los prefijos retenidos.",
        "",
        "## Reanudación real",
        "",
        "```powershell",
        command,
        "```",
        "",
        "El comando reutiliza fits completos y shards verificados, sigue la cola registrada y "
        "vuelve a detenerse si los recursos son insuficientes. Una reserva de presupuesto "
        "de GPU no reserva memoria del sistema. No se reducen cohortes, batch, precisión ni "
        "gates para sortear este bloqueo.",
        "",
        "Las ramas pendientes no son resultados negativos ni trabajo terminado. "
        "No se afirma SOTA. Los resultados positivos parciales y los rechazos anteriores "
        "permanecen en los informes de A y de las demás colas.",
    ]
    (root / "docs/representation_audit_20261012/RESOURCE_STATE.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()

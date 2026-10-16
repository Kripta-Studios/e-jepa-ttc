"""Build an evidence-bound Spanish report for the RGB-PORT campaign."""

# ruff: noqa: E501

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .accounting import atomic_write_json, read_json_shared, sha256_file

FIT_IDS = (
    "E_A5_MATCHED",
    "E_C2F_MATCHED",
    "R_A5",
    "R_C2F",
    "PAIR_E_MATCHED",
    "PAIR_R",
    "E_H1_MATCHED",
    "E_CTX_MATCHED",
    "R_H1",
    "R_CTX",
    "F_TRUE",
    "F_ZERO",
)
V_SCHEMA = "rgb_port_v_campaign_evaluation_v1"
EXPECTED_CONTRASTS = (
    "E_CTX_MATCHED=E_H1_MATCHED",
    "R_CTX=R_H1",
    "F_TRUE=F_ZERO",
    "F_TRUE=E_CTX_MATCHED",
)


def _json_if(path: Path, consumed: dict[str, str]) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    value = read_json_shared(path)
    consumed[str(path.resolve())] = sha256_file(path)
    return value


def _write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(f".{path.name}.{os.getpid()}.pending")
    with pending.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(value.rstrip() + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(pending, path)


def _fmt(value: object) -> str:
    if value is None:
        return "no disponible"
    if isinstance(value, bool):
        return "sí" if value else "no"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _relative(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _endpoint_state(run: Path, fit_id: str, consumed: dict[str, str]) -> dict[str, Any]:
    fit = run / "fits" / fit_id
    receipt_path = fit / "CHECKPOINT_RECEIPT.json"
    journal_path = fit / "UPDATE_JOURNAL.json"
    provenance_path = fit / "RUN_PROVENANCE.json"
    freeze_path = fit / "ENDPOINT_FREEZE.json"
    receipt = _json_if(receipt_path, consumed)
    journal = _json_if(journal_path, consumed)
    provenance = _json_if(provenance_path, consumed)
    freeze = _json_if(freeze_path, consumed)
    freeze_valid = bool(
        freeze
        and freeze.get("schema") == "rgb_port_fit_endpoint_freeze_v1"
        and all(
            (fit / name).is_file() and sha256_file(fit / name) == digest
            for name, digest in freeze.get("files", {}).items()
        )
        and set(freeze.get("files", {}))
        == {
            "checkpoint_last.pt",
            "CHECKPOINT_RECEIPT.json",
            "UPDATE_JOURNAL.json",
            "RUN_PROVENANCE.json",
        }
    )
    checkpoint_path: Path | None = None
    checkpoint_sha: str | None = None
    checkpoint_valid = False
    if receipt and isinstance(receipt.get("checkpoint_path"), str):
        checkpoint_path = Path(receipt["checkpoint_path"])
        if checkpoint_path.is_file():
            checkpoint_sha = sha256_file(checkpoint_path)
            consumed[str(checkpoint_path.resolve())] = checkpoint_sha
            checkpoint_valid = checkpoint_sha == receipt.get("checkpoint_sha256")
    complete = bool(
        receipt
        and receipt.get("status") == "COMPLETE"
        and receipt.get("scientific_endpoint") is True
        and checkpoint_valid
        and freeze_valid
    )
    return {
        "fit_id": fit_id,
        "status": "COMPLETE" if complete else str((receipt or {}).get("status", "MISSING")),
        "completed_updates": int(
            (journal or receipt or {}).get("completed_updates", 0)
        ),
        "checkpoint_durable_updates": int(
            (journal or {}).get("durable_updates", (receipt or {}).get("completed_updates", 0))
        ),
        "journal_completed_updates": int((journal or {}).get("completed_updates", 0)),
        "recovery_durable_upper": int((journal or {}).get("recovery_upper", 0)),
        "pending_update_upper": int((journal or {}).get("pending_update_upper", 0)),
        "checkpoint_path": str(checkpoint_path.resolve()) if checkpoint_path else None,
        "checkpoint_sha256": checkpoint_sha,
        "checkpoint_receipt_match": checkpoint_valid,
        "identity_sha256": (receipt or {}).get("identity_sha256"),
        "endpoint_freeze_sha256": sha256_file(freeze_path) if freeze is not None else None,
        "endpoint_freeze_valid": freeze_valid,
        "run_provenance_sha256": sha256_file(provenance_path) if provenance is not None else None,
        "ancestry": (freeze or {}).get("parents", (freeze or {}).get("parent_bindings")),
    }


def _curve_payload(run: Path, fit_id: str, state: Mapping[str, Any], consumed: dict[str, str]) -> dict[str, Any]:
    fit = run / "fits" / fit_id
    sources = sorted({*fit.glob("curve*.json"), *fit.glob("epoch*.json")})
    records: list[dict[str, Any]] = []
    for source in sources:
        value = _json_if(source, consumed)
        if value is not None:
            records.append(
                {
                    "path": str(source.resolve()),
                    "sha256": sha256_file(source),
                    "payload": value,
                }
            )
    return {
        "schema": "rgb_port_consolidated_curve_v1",
        "fit_id": fit_id,
        "status": "COMPLETE" if state["status"] == "COMPLETE" else "INCOMPLETE",
        "endpoint": dict(state),
        "source_count": len(records),
        "sources": records,
        "invented_points": 0,
    }


def _accounting(run: Path, states: Mapping[str, Mapping[str, Any]], consumed: dict[str, str]) -> dict[str, Any]:
    technical_sources: list[dict[str, Any]] = []
    for path in (run / "TECHNICAL_JOURNAL.json", run / "qa" / "GLOBAL_TECHNICAL_JOURNAL.json"):
        journal = _json_if(path, consumed)
        if journal is not None:
            technical_sources.append(
                {"path": str(path.resolve()), "sha256": sha256_file(path), "completed": int(journal.get("completed", 0))}
            )
    technical = max((item["completed"] for item in technical_sources), default=0)
    scientific = sum(int(state["completed_updates"]) for state in states.values())
    recovery = sum(int(state["recovery_durable_upper"]) for state in states.values())
    ledger_path = run / "ACCOUNTING.json"
    ledger = _json_if(ledger_path, consumed)
    ledger_totals = {"scientific": 0, "technical": 0, "recovery": 0, "physical": 0}
    if ledger:
        for event in ledger.get("events", {}).values():
            category = str(event.get("category"))
            charged = int(event.get("charged_updates", 0))
            if category in ledger_totals:
                ledger_totals[category] += charged
            ledger_totals["physical"] += charged
    return {
        "source_of_truth": "durable per-fit update journals plus technical journals",
        "scientific_durable": scientific,
        "technical_durable": technical,
        "recovery_durable_upper": recovery,
        "physical_durable_upper": scientific + technical + recovery,
        "technical_sources": technical_sources,
        "ledger_snapshot": ledger_totals if ledger is not None else None,
        "ledger_matches_journals": bool(
            ledger is not None
            and ledger_totals["scientific"] == scientific
            and ledger_totals["technical"] == technical
            and ledger_totals["recovery"] == recovery
        ),
    }


def _metric_line(name: str, scope: str, metric: Mapping[str, Any]) -> str:
    diag = metric.get("diagnostics", {})
    return (
        f"| {name} | {scope} | {_fmt(metric.get('group_macro_bucket_MiD'))} | "
        f"{_fmt(diag.get('rte_pct'))} | {_fmt(diag.get('crucial_mae_s'))} | "
        f"{_fmt(diag.get('sign_error_rate'))} | {_fmt(diag.get('p90_ae_s'))} | "
        f"{_fmt(diag.get('p95_ae_s'))} | {_fmt(metric.get('coverage'))} |"
    )


def _evaluation_section(value: Mapping[str, Any] | None) -> list[str]:
    if not value or value.get("schema") != V_SCHEMA or value.get("status") != "COMPLETE":
        return ["## Evaluación V", "", "INCOMPLETA: no existe un análisis V científico completo y congelado."]
    lines = [
        "## Evaluación V",
        "",
        "El target es el proxy TTC signed de escala aparente original de eAP/Garl; no representa contacto físico calibrado.",
        "",
        "| endpoint | alcance | MiD macro por grupo/bucket | RTE % | MAE crucial s | error signo | p90 AE s | p95 AE s | cobertura |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, scopes in sorted(value.get("scores", {}).items()):
        for key, label in (("native", "nativo"), ("common_cap60", "común ±60 s")):
            metric = scopes.get(key)
            if isinstance(metric, Mapping):
                lines.append(_metric_line(name, label, metric))
    lines.extend(["", "### Cuatro contrastes preregistrados", ""])
    gates = value.get("screening_gates", {})
    for contrast in EXPECTED_CONTRASTS:
        gate = gates.get(contrast)
        if isinstance(gate, Mapping):
            checks = gate.get("checks", {})
            rendered = ", ".join(f"{key}={_fmt(item)}" for key, item in sorted(checks.items()))
            lines.append(f"- `{contrast}`: passed={_fmt(gate.get('passed'))}; {rendered}; bootstrap={_fmt((gate.get('bootstrap') or {}).get('status'))}.")
        else:
            lines.append(f"- `{contrast}`: ausente; no se interpreta como resultado negativo.")
    return lines


def _powershell_quote(value: str | Path) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def _resume_commands(
    run: Path, repository: Path, consumed: dict[str, str]
) -> tuple[str, list[tuple[str, str]]]:
    python = Path(sys.executable).resolve()
    pythonpath = os.pathsep.join((str(repository / "src"), str(repository)))
    primary = (
        f"Set-Location -LiteralPath {_powershell_quote(repository)}; "
        f"$env:PYTHONPATH={_powershell_quote(pythonpath)}; "
        f"& {_powershell_quote(python)} -m operational.rgb_port.run resume "
        f"--run {_powershell_quote(run)}"
    )
    state = _json_if(run / "RGB_PORT_STATE.json", consumed)
    direct: list[tuple[str, str]] = []
    if not state:
        return primary, direct
    for fit_id in FIT_IDS:
        record = state.get("tasks", {}).get(fit_id, {})
        command = record.get("resolved_command")
        if not isinstance(command, list) or not command or "--freeze" not in command:
            continue
        freeze_index = command.index("--freeze") + 1
        if freeze_index >= len(command) or not Path(str(command[freeze_index])).is_file():
            continue
        direct.append((fit_id, subprocess.list2cmdline([str(item) for item in command])))
    return primary, direct


def _model_card(branch: str, states: Mapping[str, Mapping[str, Any]], source_sha: str | None) -> str:
    event = branch == "E"
    fits = (
        ("E_A5_MATCHED", "E_C2F_MATCHED", "PAIR_E_MATCHED", "E_H1_MATCHED", "E_CTX_MATCHED")
        if event
        else ("R_A5", "R_C2F", "PAIR_R", "R_H1", "R_CTX", "F_TRUE", "F_ZERO")
    )
    title = "E" if event else "RGB+E"
    lines = [
        f"# Model card {title}",
        "",
        "## Alcance",
        "",
        "Predice el proxy TTC signed de escala aparente original de eAP/Garl. No es una estimación de contacto físico calibrada.",
        "",
        "La partición P/H/V usa sequence_proxy provisional y queda expuesta; no sostiene una afirmación SOTA oficial.",
        "",
        "## Temporalidad y entrada",
        "",
    ]
    if event:
        lines.append("E usa como máximo 4 consultas nativas dentro de 650 ms, con un span máximo observado de 300 ms. La capacidad 8 no equivale al H8 histórico.")
    else:
        lines.append("RGB usa como máximo 5 tripletas distintas reales a 10 Hz dentro de 650 ms. Los deltas internos proceden del reloj real del sensor RGB.")
        lines.append("")
        lines.append("En ausencia de RGB, las rutas de fusión congeladas usan exactamente la predicción E_CTX; la rama R aislada queda no disponible.")
    lines.extend(["", "## Endpoints y ascendencia", "", "| endpoint | estado | checkpoint SHA-256 | freeze SHA-256 |", "|---|---|---|---|"])
    for fit_id in fits:
        state = states[fit_id]
        lines.append(f"| {fit_id} | {state['status']} | {_fmt(state['checkpoint_sha256'])} | {_fmt(state['endpoint_freeze_sha256'])} |")
    lines.extend(
        [
            "",
            f"SOURCE_FREEZE SHA-256: `{source_sha or 'ausente'}`.",
            "",
            "## Límites",
            "",
            "Una rama incompleta o bloqueada significa que falta evidencia ejecutada; no es un fracaso de rendimiento.",
            "Dev32 es diagnóstico expuesto. Sus crops RGB propios coinciden con la receta train, pero no recalculan representaciones Garl ni convierten comparadores legacy al protocolo nativo.",
            "La transferencia FCWD RGB requiere calibración RGB→evento; si falta, se conserva como BLOCKED_EXTERNAL sin cancelar las demás ramas.",
        ]
    )
    return "\n".join(lines)


def build_report(*, run: Path, config_path: Path, output: Path) -> dict[str, Any]:
    """Generate report artifacts only from already materialized campaign evidence."""
    run = run.resolve()
    config_path = config_path.resolve(strict=True)
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    consumed: dict[str, str] = {}
    config = _json_if(config_path, consumed)
    if config is None or config.get("schema") != "rgb_port_execution_v1":
        raise ValueError("Unknown RGB-PORT execution config")
    _json_if(run / "SOURCE_FREEZE.json", consumed)
    migration = _json_if(run / "AUDIT_CODE_MIGRATION.json", consumed)
    split = _json_if(run / "SPLIT_MANIFEST.json", consumed)
    evaluation = _json_if(run / "evaluation" / "V_RESULTS.json", consumed)
    costs = _json_if(run / "profiles" / "ROUTE_COSTS.json", consumed)
    dev32 = _json_if(run / "transfer" / "dev32_score" / "RESULT.json", consumed)
    fcwd = _json_if(run / "transfer" / "fcwd" / "FCWD_RGB_STATUS.json", consumed)
    next_decision = _json_if(run / "NEXT_DECISION.json", consumed)
    states = {fit_id: _endpoint_state(run, fit_id, consumed) for fit_id in FIT_IDS}

    curve_dir = output / "curves"
    curve_dir.mkdir(parents=True, exist_ok=True)
    for fit_id in FIT_IDS:
        atomic_write_json(curve_dir / f"{fit_id}.json", _curve_payload(run, fit_id, states[fit_id], consumed))
    accounting = _accounting(run, states, consumed)
    scientific_v = bool(
        evaluation
        and evaluation.get("schema") == V_SCHEMA
        and evaluation.get("status") == "COMPLETE"
        and set(evaluation.get("scores", {})) == set(FIT_IDS[-6:])
        and set(evaluation.get("screening_gates", {})) == set(EXPECTED_CONTRASTS)
    )
    grouping = (split or {}).get("grouping_level", (split or {}).get("grouping", "ausente"))
    lines = [
        "# Informe RGB-PORT",
        "",
        f"Estado del informe: **{'COMPLETE' if scientific_v else 'INCOMPLETE'}**.",
        "",
        f"Partición: `{grouping}`; la independencia por adquisición no demostrada queda como limitación. Una rama incompleta no se interpreta como resultado negativo.",
        "",
        *_evaluation_section(evaluation),
        "",
        "## Costes de rutas",
        "",
    ]
    if costs and costs.get("status") == "COMPLETE":
        for scope, profile in sorted(costs.get("profiles", {}).items()):
            lines.append(f"- `{scope}`: alcance `{profile.get('scope', scope)}`, mediana={_fmt(profile.get('warm_ms', {}).get('median', profile.get('median_ms', profile.get('latency_median_ms'))))} ms, p95={_fmt(profile.get('warm_ms', {}).get('p95', profile.get('p95_ms', profile.get('latency_p95_ms'))))} ms; batch semántico={_fmt(profile.get('batch_semantics'))}.")
        lines.append("Los costes excluidos se mantienen separados; no se suman offline cache/weights/teacher a la latencia de ruta.")
    else:
        lines.append("INCOMPLETO: PROFILE_ROUTES no aportó un recibo COMPLETE; no se estiman costes.")
    if migration:
        lines.extend([
            "", "## Correcciones de código y precisión", "",
            "Código corregido directamente con migración verificable de hashes. Los estados event conservan su objetivo, pesos, optimizador y cursor; RGB incorpora el router de luminancia/gradiente y la reducción por batch efectivo.",
            "La geometría de entrenamiento sigue en BF16 por decisión explícita. Los campos TTC q10/q90 heredados son alias de los límites del intervalo de fase transformado, no cuantiles marginales TTC garantizados. Los cruces por fase cero se exportan como no disponibles y se conserva el intervalo de fase.",
            "Dev32 usa la cadencia event de 100 ms e historias RGB de frames reales. La ROI retrospectiva está disponible en el query; esta adaptación sigue siendo transferencia exploratoria, no equivalencia con el protocolo nativo ni holdout.",
        ])
    lines.extend(["", "## Transferencia", ""])
    lines.append(
        f"- Dev32 expuesto: {_fmt((dev32 or {}).get('status', 'MISSING'))}. Es exploratorio, no holdout ni selección oficial."
    )
    lines.append(
        f"- FCWD RGB+fusión: {_fmt((fcwd or {}).get('status', 'MISSING'))}; motivo={_fmt((fcwd or {}).get('reason'))}. Este bloqueo externo no invalida V ni las otras ramas."
    )
    lines.extend(
        [
            "",
            "## Contabilidad durable",
            "",
            f"- actualizaciones científicas: {accounting['scientific_durable']}",
            f"- actualizaciones técnicas: {accounting['technical_durable']}",
            f"- cota durable de recuperación: {accounting['recovery_durable_upper']}",
            f"- cota física total: {accounting['physical_durable_upper']}",
            f"- ledger coincide con journals: {_fmt(accounting['ledger_matches_journals'])}",
            "",
            "## Estado de los 12 endpoints",
            "",
            "| endpoint | estado | updates durables | checkpoint SHA-256 |",
            "|---|---|---:|---|",
        ]
    )
    for fit_id in FIT_IDS:
        state = states[fit_id]
        lines.append(f"| {fit_id} | {state['status']} | {state['completed_updates']} | {_fmt(state['checkpoint_sha256'])} |")
    lines.extend(["", "## Reanudación", ""])
    repository = config_path.parent.parent.parent
    primary_resume, direct_resumes = _resume_commands(run, repository, consumed)
    lines.extend(
        [
            "Comando principal ejecutable en PowerShell, con cwd y PYTHONPATH explícitos:",
            "",
            "```powershell",
            primary_resume,
            "```",
        ]
    )
    if direct_resumes:
        lines.extend(["", "Comandos directos ya resueltos y con admisión materializada:", ""])
        lines.extend(f"- `{fit_id}`: `{command}`" for fit_id, command in direct_resumes)
    else:
        lines.extend(["", "No hay comandos directos resueltos con admisión materializada."])
    lines.extend(["", "## Decisión de queue", ""])
    lines.append(
        "NEXT_DECISION permanece bajo propiedad de queue. "
        + (f"Se leyó el snapshot SHA-256 `{consumed[str((run / 'NEXT_DECISION.json').resolve())]}`." if next_decision else "No había snapshot; este generador no lo crea ni lo modifica.")
    )

    report_path = output / "REPORT.md"
    source_sha = consumed.get(str((run / "SOURCE_FREEZE.json").resolve()))
    e_card = output / "model_cards" / "E_MODEL_CARD.md"
    rgb_card = output / "model_cards" / "RGB_E_MODEL_CARD.md"
    _write_text(report_path, "\n".join(lines))
    _write_text(e_card, _model_card("E", states, source_sha))
    _write_text(rgb_card, _model_card("RGB_E", states, source_sha))

    output_files = [report_path, e_card, rgb_card, *(curve_dir / f"{fit_id}.json" for fit_id in FIT_IDS)]
    manifest = {
        "schema": "rgb_port_report_hash_manifest_v1",
        "outputs": [
            {"path": _relative(path, output), "bytes": path.stat().st_size, "sha256": sha256_file(path)}
            for path in output_files
        ],
        "inputs": [{"path": path, "sha256": digest} for path, digest in sorted(consumed.items())],
    }
    manifest_path = output / "HASH_MANIFEST.json"
    atomic_write_json(manifest_path, manifest)
    receipt = {
        "schema": "rgb_port_report_receipt_v1",
        "status": "COMPLETE" if scientific_v else "INCOMPLETE",
        "scientific_v_analysis_available": scientific_v,
        "optimizer_updates": 0,
        "report_path": str(report_path),
        "model_cards": {"E": str(e_card), "RGB_E": str(rgb_card)},
        "curve_artifacts": {fit_id: str(curve_dir / f"{fit_id}.json") for fit_id in FIT_IDS},
        "endpoint_states": states,
        "accounting": accounting,
        "optional_branches": {
            "PROFILE_ROUTES": (costs or {}).get("status", "MISSING"),
            "DEV32_SCORE": (dev32 or {}).get("status", "MISSING"),
            "FCWD_RGB_STATUS": (fcwd or {}).get("status", "MISSING"),
        },
        "fcwd_block_is_nonfatal": True,
        "next_decision_preserved": next_decision is not None,
        "hash_manifest_path": str(manifest_path),
        "hash_manifest_sha256": sha256_file(manifest_path),
    }
    atomic_write_json(output / "REPORT_RECEIPT.json", receipt)
    return receipt


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    receipt = build_report(run=args.run, config_path=args.config, output=args.output)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["FIT_IDS", "build_report", "main"]

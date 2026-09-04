"""Analyze frozen results and emit the required report and next decision."""

# ruff: noqa: E501

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

from e_jepa_ttc.artifacts.hashing import sign_artifact


def _load(path: Path) -> dict[str, Any] | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def _mid(gate: dict[str, Any] | None, arm: str) -> str:
    if gate is None:
        return "no ejecutado"
    return f"{gate['summaries'][arm]['sequence_macro_paper_MiD_overall']:.6f}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    args = parser.parse_args()
    stage61 = _load(args.campaign_root / "stage61/aggregate_seed7/STAGE61_GATE.json")
    stage62 = _load(args.campaign_root / "stage62/aggregate_seed7/STAGE62_GATE.json")
    x3 = _load(args.campaign_root / "X3_FEASIBILITY.json")
    if stage61 is None or x3 is None:
        raise ValueError("analysis requires Stage 61 and X3 terminal artifacts")
    if stage61["gate_passed"]:
        decision = "STAGE61_SEED7_SUPPORTED_REPLICATION_PENDING_OR_BLOCKED"
    elif stage62 is not None and stage62["gates"]["locality"] and stage62["gates"]["association"]:
        decision = (
            "STAGE61_NOT_SUPPORTED_X2_LOCAL_FIELD_SUPPORTED"
            if stage62["gate_passed"]
            else "STAGE61_NOT_SUPPORTED_X2_MECHANISM_ONLY"
        )
    elif x3["decision"] == "X3_DATA_READY":
        decision = "STAGE61_AND_X2_NEGATIVE_X3_DATA_READY"
    else:
        decision = "STAGE61_AND_X2_NEGATIVE_X3_BLOCKED"
    analysis_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=args.repo, text=True
    ).strip()
    training_commit = str(stage61["code_commit"])
    next_decision = sign_artifact(
        {
            "artifact_type": "scientific_recovery_v9_stage61_stage62_next_decision_v1",
            "decision": decision,
            "training_commit": training_commit,
            "analysis_commit": analysis_commit,
            "stages": {
                "stage61": "completed",
                "stage62": "completed" if stage62 is not None else "skipped_stage61_gate_passed",
                "x3_feasibility": "completed_read_only",
            },
            "seeds": {"stage61": [7], "stage62": [7] if stage62 is not None else []},
            "replication": {
                "seed13": "blocked_missing_matched_A5_C2F_producers",
                "seed23": "blocked_missing_24_of_24_prerequisites",
            },
            "sealed_status": {
                "public_validation_opened": False,
                "private_test_opened": False,
                "evttc_test_opened": False,
                "codabench_opened": False,
            },
            "stage61_gate_artifact_sha256": stage61["artifact_sha256"],
            "stage62_gate_artifact_sha256": stage62["artifact_sha256"] if stage62 else None,
            "x3_artifact_sha256": x3["artifact_sha256"],
        }
    )
    (args.repo / "NEXT_DECISION.json").write_text(
        json.dumps(next_decision, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    report = f"""# E-JEPA-TTC — informe final Stage 61 → Stage 62

## Resultado ejecutivo

Decisión: `{decision}`.

Se ejecutó Stage 61 seed 7 con nested A5/C2F/PAIR estricto sobre 8192 filas y nueve secuencias. Stage 62 fue {"ejecutado" if stage62 else "omitido porque Stage 61 pasó su gate"}. La auditoría X3 fue siempre ejecutada en modo read-only y decidió `{x3["decision"]}`.

## Evidencia preregistrada

| Brazo | MiD sequence-macro |
|---|---:|
| Router R | {_mid(stage61, "RouterR")} |
| Stage61 R1 BASE8 | {_mid(stage61, "R1")} |
| Stage61 R2 PHASE17 | {_mid(stage61, "R2")} |
| Stage61 R2 SHUFFLE | {_mid(stage61, "R2_SHUFFLE")} |
| X2 A5 replay | {_mid(stage62, "X2_A5_REPLAY")} |
| X2 GLOBALPOOL | {_mid(stage62, "X2_GLOBALPOOL")} |
| X2 LOCALFIELD | {_mid(stage62, "X2_LOCALFIELD")} |
| X2 SHUFFLEFIELD | {_mid(stage62, "X2_SHUFFLEFIELD")} |
| X2 TIMESWAP | {_mid(stage62, "X2_TIMESWAP")} |

Stage 61 gate: `{stage61["status"]}`. Stage 62 gate: `{stage62["status"] if stage62 else "not_executed"}`. Los intervalos y probabilidades completos están en los artifacts bootstrap firmados del bundle.

## Gates preregistrados

- Stage 61: gate global `false`; mejoró R2 frente a R1 en media bootstrap, pero su IC95 cruzó cero. El falsificador R2 frente a R2-SHUFFLE sí pasó; R2 frente a Router R no superó simultáneamente magnitud, IC95 y probabilidad.
- Stage 62: `locality=false`, `association=false`, `time_order=false`, `utility=true`, `system=false`. LOCALFIELD sólo superó el control A5 replay; no demostró ventaja frente a GLOBALPOOL, SHUFFLEFIELD, TIMESWAP ni Router R.
- Por lo anterior no se autorizó replicación de Stage 61 y se ejecutó Stage 62 completo.

## Integridad y acceso sellado

- Commit de entrenamiento Stage 61/62: `{training_commit}`.
- Commit del analizador: `{analysis_commit}`. La diferencia corresponde al lector read-only X3 y no altera modelos ni predicciones.
- PAIR se entrenó sólo desde caches 133-D producidas por el A5 exacto de cada inner/final fold.
- Cada router se ajustó sólo con inner-OOF; outer-dev se abrió una vez tras congelar los heads y routers.
- GLOBAL/LOCAL/SHUFFLE compartieron inicialización, batches, optimizador, presupuesto y productor.
- No se ejecutaron X1 antiguo, DYN-W, Track M, recurrent depth ni entrenamiento X3.
- No se abrieron public validation, private test, EvTTC test ni CodaBench; no hubo push.

## Replicación

Seeds 13 y 23 no se fabricaron: no existe el universo físico matched de productores A5/C2F nested requerido. La ausencia queda declarada como bloqueo de prerequisitos, no como resultado multiseed.

## X3 feasibility

La cache train8192 preserva tensores voxelizados de tres endpoints, pero la auditoría exige eventos raw con timestamps y polaridad ligados por token. Resultado: `{x3["decision"]}`. No se entrenó ningún modelo X3.

## Intentos abortados y QA

Dos intentos preliminares se conservaron como telemetría no seleccionable. El primero abortó antes de outer-dev porque un derangement aleatorio no garantizaba solución; se sustituyó por una construcción determinista. El segundo abortó tras la primera evaluación outer porque un campo TTC auxiliar del productor V8 era NaN; se corrigió ligando la predicción puntual exacta, su SHA, token y target. Ningún resultado de esos intentos entra en los gates.

Los tests específicos de Stage 61/62 pasan 10/10; Ruff, `git diff --check`, los tres esquemas JSON y sus firmas internas pasan. La suite histórica completa conserva 39 fallos ajenos a esta continuación, documentados en `qa/QA_FULL_PYTEST.log`: artefactos V7/V8 ausentes en este checkout, expectativas legacy de mutación de esquemas y rutas subprocess afectadas por la codificación del perfil Windows.

## Claims permitidos y prohibidos

Los resultados describen únicamente el protocolo local Garl-TTC firmado. Cruzar 144.353, si ocurriera, sería sólo una barra descriptiva pendiente de replicación; no implica SOTA. No se permite extrapolar a public/private test ni atribuir causalidad fuera de los controles preregistrados.
"""
    (args.repo / "CODEX_STAGE61_STAGE62_FINAL_REPORT.md").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()

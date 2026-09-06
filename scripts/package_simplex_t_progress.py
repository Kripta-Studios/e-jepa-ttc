"""Package an honest, resumable first-fold checkpoint, never scientific completion."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import zipfile
from datetime import UTC, datetime
from pathlib import Path

import psutil

from e_jepa_ttc.artifacts.simplex_t_preflight import sha256, write_new_json
from e_jepa_ttc.simplex_t.coordination import shared_write_admission
from e_jepa_ttc.simplex_t.lifecycle import admitted


def main() -> None:
    """Collect immutable available evidence with explicit missing scientific outputs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--other-reserved-bytes", required=True, type=int)
    args = parser.parse_args()
    root = Path.cwd()
    campaign = root / "artifacts/simplex_t"
    if args.output.exists():
        raise FileExistsError("preserve earlier deliveries")
    if (campaign / "T1/CURRENT_REPLAY.lock").exists():
        raise ValueError("snapshot only at a closed replay boundary")
    if any((campaign / stage).exists() for stage in ("T2", "T3", "T4", "T5")):
        raise ValueError("first-fold packager must not summarize scientific runs")
    required = [
        "T0/EXPANSION_INPUT_BINDING_AUDIT_V2.json",
        "T0/EXPANSION_EXPOSURE_TIMING.json",
        "T0/real_context_cpu_resume/RESUME_QA.json",
        "T0/real_context_cpu_resume/CONTRACT.json",
        "T0/real_context_cpu_resume/TECHNICAL_JOURNAL.json",
        "T0/qa_expansion_metadata.xml",
        "T0/qa_source_analysis_integration_021d449.xml",
        "T1/CONTEXT_CACHE_AUDIT_8192.json",
        "T1/RESOURCE_PAUSE_8165.json",
        "T1/context_features_fp32/IDENTITY.json",
        "T1/compiled_context/outer0/COMPILED.json",
        "TECHNICAL_BUDGET.json",
    ]
    files = {campaign / name for name in required}
    resume = json.loads((campaign / required[2]).read_text())
    cache = json.loads((campaign / "T1/CONTEXT_CACHE_AUDIT_8192.json").read_text())
    budget = json.loads((campaign / "TECHNICAL_BUDGET.json").read_text())
    if not resume["exact_complete_state_match"] or resume["executed_technical_updates"] != 20:
        raise ValueError("real complete-state resume proof missing")
    if cache["completed_query_blocks"] != 8192 or cache["observations"] != 130915:
        raise ValueError("unexpected cache snapshot; use a new delivery analysis")
    if sum(budget["reservations"].values()) != 645:
        raise ValueError("technical accounting changed; reconcile before packaging")
    for pattern in (
        "src/e_jepa_ttc/simplex_t/*.py",
        "scripts/*simplex_t*.py",
        "scripts/*simplex_t*.ps1",
        "tests/unit/test_simplex_t*.py",
        "docs/SIMPLEX_T*",
        "configs/experiment/simplex_t*",
        "artifacts/simplex_t/T1/query_context_index/*",
        "artifacts/simplex_t/T1/query_context_dedup/*",
        "artifacts/simplex_t/T1/compiled_context/outer0/*",
        "artifacts/simplex_t/T0/real_context_cpu_resume/*/checkpoint_last.pt",
        "artifacts/simplex_t/T1/*PARITY*.json",
        "artifacts/simplex_t/T1/*FAILURE*.json",
        "artifacts/simplex_t/T1/*RESOLUTION*.json",
        "artifacts/simplex_t/T0/current_expert_replay_64/*QA*.json",
    ):
        files.update(path for path in root.glob(pattern) if path.is_file())
    reservation = 2 * sum(path.stat().st_size for path in files) + 64 * 1024**2
    snapshot = admitted([root])
    if args.other_reserved_bytes < 0 or not snapshot["has_headroom"] or not shared_write_admission(
        snapshot["written_volume_free_bytes"][0], args.other_reserved_bytes + reservation
    ):
        raise RuntimeError("RESOURCE_PAUSE before progress package")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    inventory = {
        str(path.relative_to(root)).replace("\\", "/"): sha256(path) for path in sorted(files)
    }
    active = []
    for process in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            command = " ".join(process.info["cmdline"] or [])
            if "python" in (process.info["name"] or "").lower() and (
                "run_scientific_recovery_v10_stage70_76.py" in command
            ):
                active.append({"pid": process.pid, "command": command})
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    decision = {
        "status": "RESUMABLE_RESOURCE_COORDINATION_PAUSE" if active else "INCOMPLETE_FIRST_FOLD",
        "scientific_complete": False,
        "scientific_freeze": False,
        "code_commit": commit,
        "observed_at_utc": datetime.now(UTC).isoformat(),
        "scientific_fits": 0,
        "scientific_optimizer_updates": 0,
        "executed_technical_updates": 645,
        "reserved_technical_updates": 645,
        "completed_query_blocks": 8192,
        "required_D0_query_blocks": 24576,
        "cached_observations": 130915,
        "compiled_outer_folds": [0],
        "real_train_resume": resume,
        "D1_input_queries_verified": 27307,
        "D1_sequences_verified": 22,
        "D1_expert_cache_generated": False,
        "resource_snapshot": admitted([root]),
        "observed_stage70_processes": active,
        "stage70_process_activity_is_not_score_or_verified_GPU_occupancy": True,
        "missing": [
            "D0 remaining two folds of expert replay",
            "D1 owner temporal acknowledgment, transitive producers, pool/loader/cache integration",
            "matched-density availability resolution before scores",
            "full scientific CLI integration, failure-ID QA and pre-fit freeze",
            "all scientific fits, predictions, controls, factor inference and T6 conclusions",
        ],
        "next_gpu_action": "reconfirm exclusive boundary with active Stage70 executor",
        "owner_request": "docs/SIMPLEX_T_STAGE70_EXPANSION_ACK_REQUEST.md",
        "background_completion_claimed": False,
    }
    analysis = hashlib.sha256(
        json.dumps({"decision": decision, "inventory": inventory}, sort_keys=True).encode()
    ).hexdigest()[:12]
    args.output.mkdir(parents=True)
    write_new_json(args.output / "NEXT_DECISION_SIMPLEX_T.json", decision)
    write_new_json(args.output / "CONTENT_MANIFEST.json", inventory)
    report = f"""# SIMPLEX-T: estado reanudable, no resultado científico

Commit: {commit}. Análisis: {analysis}.

D0 conserva sus 8192 consultas OLD y las exclusiones originales. El primer
fold está compilado: 8192 bloques, 130915 observaciones. Faltan dos folds.
El cargador real verificó 5461 consultas TRAIN; 5452 tienen H8 completo.
La prueba CPU 10 frente a 5+5 pasó con igualdad exacta del estado completo.
Total: 645 actualizaciones técnicas; 0 fits y 0 actualizaciones científicas.

D1: verificadas 27307 consultas, 54614 ventanas y 22 secuencias de expansión.
Los cinco hashes de entradas coinciden. Las identidades de selección prospectiva
y filas utilizables coinciden. El validador upstream consulta TTC; no eliminó
ninguna de estas consultas. Esto no convierte los pares en historia primaria.
Las exposiciones RGB añaden 1003–19992 us; no se supone edad cero ni latencia
online conocida. El contexto autorizado usa ROI actual retrospectiva, no tracking.

Se conserva la pausa real de RAM a los 8165 bloques: 8534745088 bytes disponibles,
por debajo de 8 GiB. Se reanudó sin cambios numéricos y se completó el fold.
Ahora se observan {len(active)} procesos del ejecutor Stage70. Su presencia no
demuestra por sí sola ocupación GPU; requiere coordinar el siguiente turno
exclusivo. No se detuvo ni modificó Stage70 y no se consultaron sus scores.

QA nueva: cinco tests de metadatos pasados, Ruff y tipos de los scripts pasados.
El paquete incluye el recibo previo de 50 tests de integración/análisis. Esto
no sustituye la comparación final de failure IDs con el baseline del repositorio.
Se preservan el fallo de capacidad de lectura y la extensión explícita de PAIR
infinito mediante documentación e inventario; no son resultados científicos.

Pendiente: reconocimiento temporal D1, exclusiones transitivas completas,
cargador/caché adicional, controles de cantidad/diversidad, integración científica
end-to-end, QA final, freeze y T2–T6. No se abrieron confirmación ni holdout.
No hay pesos científicos, predicciones de candidatos, efectos factoriales ni
intervalos científicos que reportar. Los checkpoints incluidos son SOLO técnicos.

La petición concreta al propietario está en
docs/SIMPLEX_T_STAGE70_EXPANSION_ACK_REQUEST.md. Los paths, hashes, estado y
dependencias exactas están en NEXT_DECISION_SIMPLEX_T.json y CONTENT_MANIFEST.json.
Esta entrega no declara el goal completado ni promete ejecución en background.
"""
    report_path = args.output / "CODEX_SIMPLEX_T_FINAL_REPORT.md"
    with report_path.open("x", encoding="utf-8") as stream:
        stream.write(report)
    archive = args.output / f"E_JEPA_TTC_SIMPLEX_T_ESSENTIAL_RESULTS_{analysis}.zip"
    with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=1) as bundle:
        for relative, digest in inventory.items():
            if sha256(root / relative) != digest:
                raise ValueError("snapshot source changed during packaging")
            bundle.write(root / relative, relative)
        for name in (
            "CODEX_SIMPLEX_T_FINAL_REPORT.md",
            "NEXT_DECISION_SIMPLEX_T.json",
            "CONTENT_MANIFEST.json",
        ):
            bundle.write(args.output / name, name)
    with zipfile.ZipFile(archive) as bundle:
        if bundle.testzip() is not None:
            raise ValueError("archive CRC verification failed")
    digest = sha256(archive)
    with archive.with_suffix(".zip.sha256").open("x", encoding="utf-8") as stream:
        stream.write(f"{digest}  {archive.name}\n")
    print(json.dumps({"zip": str(archive), "sha256": digest, "scientific_complete": False}))


if __name__ == "__main__":
    main()

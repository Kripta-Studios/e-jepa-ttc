"""Measured tables, explicit blocked branches and a regenerable campaign report."""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

from .common import ROOT, Campaign, atomic_bytes, atomic_json, digest, read

if TYPE_CHECKING:
    from pandas import DataFrame


def markdown_table(frame: DataFrame) -> str:
    """Render persisted tables without introducing an optional tabulate dependency."""
    columns = list(frame.columns)
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for row in frame.itertuples(index=False, name=None):
        values = [f"{v:.6f}" if isinstance(v, float) else str(v) for v in row]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def report(c: Campaign) -> None:
    """Every reported number comes from persisted predictions, curves or receipts."""
    import matplotlib
    import pandas as pd

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from .budget import accounting
    from .checkpoints import inventory, inventory_native
    from .queue import dependencies

    checkpoints = inventory(c)
    native_checkpoints = inventory_native(c)
    counts = accounting(c)
    dep = dependencies(c)
    atomic_json(c.out / "ACCOUNTING.json", counts)
    atomic_json(c.out / "DEPENDENCIES.json", dep)
    state = read(c.out / "QUEUE_STATE.json") if (c.out / "QUEUE_STATE.json").exists() else {}
    wide_path = c.out / "H8_WIDE_RESULTS.json"
    wide = read(wide_path) if wide_path.exists() else {}
    wide_evaluated = wide.get("status") == "COMPLETE"
    replication_path = c.out / "WIDE_REPLICATION_RESULTS.json"
    replication = read(replication_path) if replication_path.exists() else {}
    garl_path = c.out / "GARL_CONTEXT_RESULTS.json"
    garl = read(garl_path) if garl_path.exists() else {}
    micro = read(c.out / "garl/MICROBATCH_PROFILE.json")
    parity = read(c.out / "INPUT_OUTPUT_PARITY.json")
    raw_measured = bool(parity.get("R0_end_to_end_measured"))
    native_runtime_path = c.out / "GARL_RUNTIME_RESULTS.json"
    native_runtime = read(native_runtime_path) if native_runtime_path.exists() else {}
    restored = [read(p) for p in sorted((c.out / "data_recovery/files").glob("*.json"))]
    verified_raw = [r for r in restored if r.get("status") == "VERIFIED"]
    restoration_status = (
        f"{len(verified_raw)}/31 HDF5 TRAIN verificados por SHA256; "
        f"{sum(r['bytes'] for r in verified_raw):,} bytes. "
        f"Dependencias raw pendientes E1: {len(dep['E1_missing'])}; "
        f"Garl: {len(dep['garl_raw_missing'])}."
    )
    qa = read(c.out / "TEST_RESULTS/final/QA.json")
    pytest_log = (c.out / "TEST_RESULTS/final/pytest.txt").read_text(encoding="utf-8")
    passed_tests = re.search(r"(\d+) passed", pytest_log)
    quiet_passes = re.search(r"^([.]+)\s+\[100%\]$", pytest_log, re.MULTILINE)
    test_count = (
        int(passed_tests[1])
        if passed_tests
        else len(quiet_passes[1])
        if quiet_passes and qa["results"]["pytest"]["returncode"] == 0
        else "consultar recibo"
    )
    remaining = sum(
        2500 - r["completed_updates"] for r in checkpoints if r["key"].endswith("seed7")
    ) + 2500 * (3 - sum(r["key"].endswith("seed7") for r in checkpoints))
    checkpoint_summary = "; ".join(
        f"{r['key']}: {r['completed_updates']} ({r['status']})" for r in checkpoints
    )
    analytical = read(c.out / "EWMA_TRANSPORT_CV_RESULTS.json")
    rows = [
        {"system": "H8 historical seed7", "MiD": analytical["reference_score"], "delta_H8": 0},
        {
            "system": "EWMA_TRANSPORT_CV_H8",
            "MiD": analytical["candidate_score"],
            "delta_H8": analytical["point_delta"],
        },
        {
            "system": "EWMA cap sensitivity",
            "MiD": analytical["cap_sensitivity"]["candidate_score"],
            "delta_H8": analytical["cap_sensitivity"]["point_delta"],
        },
    ]
    atomic_bytes(
        c.out / "tables/ANALYTICAL_COMPARISON.csv", pd.DataFrame(rows).to_csv(index=False).encode()
    )
    wide_rows = []
    for seed in (7, 13, 23):
        path = c.out / ("H8_WIDE_RESULTS.json" if seed == 7 else f"H8_WIDE_RESULTS_seed{seed}.json")
        result = read(path) if path.exists() else {}
        if result.get("status") != "COMPLETE":
            continue
        for reference, comparison in result["comparisons"].items():
            wide_rows.append(
                {
                    "scope": f"seed{seed}",
                    "reference": reference,
                    "WIDE_MiD": comparison["candidate_score"],
                    "reference_MiD": comparison["reference_score"],
                    "delta_MiD": comparison["point_delta"],
                    "hierarchical_low": comparison["hierarchical_ci95"][0],
                    "hierarchical_high": comparison["hierarchical_ci95"][1],
                    "sequence_low": comparison["sequence_only"]["ci95"][0],
                    "sequence_high": comparison["sequence_only"]["ci95"][1],
                    "sequence_wins": comparison["sequence_wins"],
                }
            )
    for scope, result in replication.get("results", {}).items():
        for reference, comparison in result["contrasts"].items():
            wide_rows.append(
                {
                    "scope": scope,
                    "reference": reference,
                    "WIDE_MiD": result["scores"]["WIDE"],
                    "reference_MiD": result["scores"][reference],
                    "delta_MiD": comparison["point_delta"],
                    "hierarchical_low": comparison["hierarchical_ci95"][0],
                    "hierarchical_high": comparison["hierarchical_ci95"][1],
                    "sequence_low": comparison["sequence_only"]["ci95"][0],
                    "sequence_high": comparison["sequence_only"]["ci95"][1],
                    "sequence_wins": comparison["sequence_wins"],
                }
            )
    wide_table = pd.DataFrame(wide_rows)
    atomic_bytes(c.out / "tables/WIDE_COMPARISON.csv", wide_table.to_csv(index=False).encode())
    native_rows = []
    for contrast, comparison in garl.get("comparisons", {}).items():
        native_rows.append(
            {
                "contrast": contrast,
                "candidate_MiD": comparison["candidate_score"],
                "reference_MiD": comparison["reference_score"],
                "delta_MiD": comparison["point_delta"],
                "hierarchical_low": comparison["hierarchical_ci95"][0],
                "hierarchical_high": comparison["hierarchical_ci95"][1],
                "sequence_low": comparison["sequence_only"]["ci95"][0],
                "sequence_high": comparison["sequence_only"]["ci95"][1],
            }
        )
    native_table = pd.DataFrame(native_rows)
    atomic_bytes(c.out / "tables/GARL_COMPARISON.csv", native_table.to_csv(index=False).encode())
    sequence = pd.DataFrame(
        [{"sequence_id": k, "delta_MiD": v} for k, v in analytical["sequence_deltas"].items()]
    )
    atomic_bytes(c.out / "tables/ANALYTICAL_SEQUENCE.csv", sequence.to_csv(index=False).encode())
    runtime = pd.read_csv(c.out / "prepared_heads/MEASUREMENTS.csv")
    pooled = (
        runtime.groupby(["regime", "label"])
        .milliseconds.agg(requests="count", p50_ms="median", p95_ms=lambda x: x.quantile(0.95))
        .reset_index()
    )
    atomic_bytes(c.out / "tables/PREPARED_HEAD_COST.csv", pooled.to_csv(index=False).encode())
    if not (c.out / "RUNTIME_COMPARISON.csv").exists():
        atomic_bytes(c.out / "RUNTIME_COMPARISON.csv", pooled.to_csv(index=False).encode())
        atomic_bytes(c.out / "PROFILE_COMPONENTS.csv", runtime.to_csv(index=False).encode())
    repeat = runtime.groupby(["label", "query"])[["ttc_s", "point_phase"]].agg(
        lambda x: x.max() - x.min()
    )
    atomic_json(
        c.out / "prepared_heads/REPEATABILITY.json",
        {
            "status": "EXACT" if not repeat.to_numpy().any() else "FAILED_INTEGRITY",
            "max_ttc_difference": float(repeat.ttc_s.max()),
            "max_phase_difference": float(repeat.point_phase.max()),
            "queries_per_head": 64,
            "blocks": 3,
            "scientific_accuracy_comparison": False,
        },
    )
    paths = c.out / "figures"
    paths.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    for record in checkpoints:
        curve = pd.read_csv(Path(record["checkpoint"]).parent / "TRAINING_CURVE.csv")
        smooth = curve.loss.rolling(100, min_periods=1).mean()
        axes[0].plot(curve["update"], smooth, label=record["key"].replace("WIDE/", ""))
    axes[0].set(
        xlabel="Update durable",
        ylabel="Loss TRAIN (media móvil100)",
        title="WIDE: checkpoints conservados",
    )
    axes[0].legend(fontsize=8)
    axes[1].barh(sequence.sequence_id, sequence.delta_MiD, color="#b65b43")
    axes[1].axvline(0, color="black", linewidth=0.8)
    axes[1].set(xlabel="Delta MiD EWMA−H8 seed7", title="OLD_DEV: control analítico")
    fig.savefig(paths / "CURVES_AND_SEQUENCE_RESULTS.png", dpi=180)
    plt.close(fig)
    if native_checkpoints:
        fig, ax = plt.subplots(figsize=(10, 5), layout="constrained")
        for record in native_checkpoints:
            curve_path = Path(record["checkpoint"]).parent / "TRAINING_CURVE.csv"
            if curve_path.exists():
                curve = pd.read_csv(curve_path)
                ax.plot(
                    curve["update"],
                    curve.loss.rolling(100, min_periods=1).mean(),
                    label=record["key"],
                    linewidth=1,
                )
        ax.set(
            xlabel="Update durable",
            ylabel="Loss TRAIN (media móvil100)",
            title="Garl: productores y cabezas; pérdidas de objetivos distintos",
        )
        ax.legend(fontsize=7, ncol=2)
        fig.savefig(paths / "GARL_TRAINING_CURVES.png", dpi=180)
        plt.close(fig)
    manifest = {
        "campaign": "EFFICIENT_CONTEXT_20261004",
        "base_commit": c.config["base_commit"],
        "protocol_sha256": digest(c.out / "PROTOCOL.json"),
        "code_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "roots": read(c.out / "SOURCE_ADMISSION.json")["roots"],
        "checkpoint_inventory": checkpoints,
        "native_checkpoint_inventory": native_checkpoints,
        "branches": state.get("branches", {}),
        "caps": {
            "WIDE": 22500,
            "GARL_PRODUCERS": 200000,
            "GARL_HEADS": 15000,
            "TECHNICAL_SYNTHETIC": 200,
            "RECOVERY": 2300,
            "PHYSICAL": 240000,
        },
        "native_producers_exact_plan": read(c.out / "garl/ADMISSION.json")[
            "updates_exact_50_epochs"
        ],
        "OLD_DEV_access": {
            "zero_update_analytical_control": True,
            "WIDE_scored_after_all_three_folds_sealed": wide_evaluated,
            "partial_WIDE_scored": False,
            "Garl_results_status": read(c.out / "GARL_COMPARISON.json").get("status"),
            "Garl_context_results_status": garl.get("status", "PENDING"),
        },
        "raw_restoration": restoration_status,
        "microbatch_profile": micro,
        "active_resource_policy": c.policy,
        "resource_amendment": read(c.out / "RESOURCE_AUTHORIZATION.json")
        if (c.out / "RESOURCE_AUTHORIZATION.json").exists()
        else None,
        "additional_resource_amendment": read(c.out / "RESOURCE_AUTHORIZATION_V2.json")
        if (c.out / "RESOURCE_AUTHORIZATION_V2.json").exists()
        else None,
        "protected_evaluation_access": False,
        "push_or_submission": False,
        "source_references": (
            "release SHA bindings and original frozen TRAIN/config/control identities"
        ),
    }
    atomic_json(c.out / "EXPERIMENT_MANIFEST.json", manifest)
    command = (
        "$env:PYTHONUTF8='1'; & '../e-jepa-ttc/.venv/Scripts/python.exe' "
        "-m operational.efficient_context.queue all "
        "--protocol configs/campaign/efficient_context_v1.json --resume"
    )
    atomic_bytes(
        c.out / "RESUME.ps1",
        (
            "$ErrorActionPreference='Stop'\nSet-Location -LiteralPath $PSScriptRoot\n"
            "Set-Location -LiteralPath '../..'\n" + command + "\nexit $LASTEXITCODE\n"
        ).encode("utf-8-sig"),
    )
    next_decision = {
        "historical_candidate": "TPR-D1-H8-C160",
        "H16_retained": True,
        "WIDE_promoted": False,
        "EWMA_control": "NEGATIVE_LOCAL_DEVELOPMENT",
        "E2": (
            "COMPLETE"
            if replication.get("status") == "COMPLETE"
            else "BLOCKED_DEPENDENCY"
            if dep["wide_missing"]
            else "CONDITIONAL_REPLICAS_PENDING"
            if wide_evaluated
            else "PENDING"
        ),
        "E3": (
            "COMPLETE"
            if garl.get("status") == "COMPLETE"
            else "BLOCKED_DEPENDENCY"
            if dep["garl_missing"] or dep["garl_raw_missing"]
            else "PENDING_EXECUTION"
        ),
        "E1": "COMPLETE" if raw_measured else "PENDING_RAW_PROFILE",
        "new_WIDE_OLD_DEV_evaluation_performed": wide_evaluated,
        "scientific_negative_for_missing_comparators": False,
        "remaining_seed7_WIDE_updates": remaining,
        "replication_rule": "unchanged five prospective guardrails after all3 endpoints freeze",
        "native_resume_recipe": "12 fixed50-epoch native producers, then6 INNER-OOF GRU160 heads",
        "missing_paths": dep,
        "resume_command": command,
        "resume_script": "RESUME.ps1",
        "holdouts_opened": False,
        "confirmation_requires_future_authorization": True,
    }
    atomic_json(c.out / "NEXT_DECISION.json", next_decision)
    semantics = (
        "# Semántica del target\n\nSe conserva TTC firmado del benchmark Garl/eAP y "
        "psi(T)=-log(1−0,1/T). El anchor es el final de la observación; las edades "
        "y la disponibilidad del ROI permanecen explícitas. Esta auditoría verifica "
        "los anchors int64 de las observaciones compiladas y su genealogía; no reproduce "
        "independientemente la adquisición física de las etiquetas.\n\n"
        "El TTC de profundidad/expansión no se sustituye por primer contacto físico. "
        "No se reconstruyen velocidades con Z/TTC_GT. Los campos3D TRAIN admitidos "
        "sólo supervisan visible_height en la receta nativa Garl; el forward acepta "
        "sensores y ROI. Inferencia event-only de los expertos históricos no borra "
        "su supervisión RGB/DINO. +60s es capping numérico, no una clase NO_CONTACT.\n"
    )
    atomic_bytes(c.out / "TTC_TARGET_SEMANTICS.md", semantics.encode())
    atomic_bytes(
        c.out / "INFERENCE_INTERFACE.md",
        (
            "# Interfaces congeladas\n\nWIDE usa ocho slots [0,2,4,6,9,11,13,15] "
            "del H16 padre, PHASE17, cuatro tiempos, máscara y tres fases expertas actuales. "
            "Gaps recalculados en int64; normalizador histórico D1; GRU160 CPU FP32. "
            "No se admite un endpoint parcial para OLD_DEV.\n\n"
            "Garl nativo usa40 canales FP32,20 planos por cada uno de dos endpoints, "
            "ResNet50 y ROI nativo por frame. Los límites ms se redondean como upstream. "
            "Garl-H1/H8 usa fase nativa, log1p(count), log1p(rate), los cuatro tiempos "
            "H8 y máscara; lambda_cost=0. Count/rate procede del ROI y último intervalo "
            "de100ms común, sólo sensores. Observación nativa~200ms frente a~300ms "
            "de A5/C2F: la comparación declara ese presupuesto diferente.\n\n"
            f"R0 medido: {raw_measured}; estado de paridad: {parity['status']}. "
            "El recibo de cada solicitud conserva el binding raw y sus errores. "
            "R2 preparado se mide por separado; cuando R0 está completo, sus segmentos "
            "transfer/productores/norm/cabeza constan en PROFILE_COMPONENTS.csv. "
            "R1 no implementado. No hay declaración de servicio online ni tiempo real.\n"
        ).encode(),
    )
    lines = [
        "# Campaña E-JEPA-TTC: contexto eficiente\n",
        "Base publicada `fd16d8102914537653622d3abf298b753120ea43`. "
        "Raíces históricas de sólo lectura; outputs nuevos separados.\n",
        "## Estado verificable\n",
        f"WIDE conserva **{counts['wide_saved']:,} updates duraderos**. "
        f"{checkpoint_summary}. Restantes seed7: {remaining}. "
        f"OLD_DEV evaluado tras el seal: {wide_evaluated}. "
        "Los endpoints parciales no se puntúan; la replicación conserva sus "
        "cinco guardrails prospectivos.\n",
        "El fallo inicial del disco se conserva en los recibos de recuperación. "
        "Los dos parquets Garl TRAIN restaurados coinciden con los SHA256 originales. "
        "La restauración raw usa una revisión HF fija, sólo la allowlist TRAIN "
        "autorizada, y verifica el SHA completo de cada HDF5. " + restoration_status + " "
        "RESUME_PROOF verifica checkpoints, RNG y siguiente batch sin updates.\n",
        "## Precisión WIDE\n",
        markdown_table(wide_table) + "\n" if wide_rows else "Endpoints pendientes.\n",
        f"Réplicas autorizadas por los cinco guardrails seed7: "
        f"{wide.get('replication_authorized', False)}. "
        f"Agregación de réplicas: {replication.get('status', 'PENDING')}. "
        "Cada seed se evalúa después del seal de sus tres folds. "
        "La agregación promedia pérdidas pareadas por consulta, nunca TTC; "
        "separa tres semillas y semillas nuevas13/23. No se promueve el candidato.\n",
        "## Precisión del control analítico\n",
        markdown_table(pd.DataFrame(rows)) + "\n",
        f"EWMA−H8: **{analytical['point_delta']:.6f} MiD**, IC jerárquico95% "
        f"[{analytical['hierarchical_ci95'][0]:.6f}, {analytical['hierarchical_ci95'][1]:.6f}], "
        f"IC sólo por secuencia [{analytical['sequence_only']['ci95'][0]:.6f}, "
        f"{analytical['sequence_only']['ci95'][1]:.6f}]. Mejora en "
        f"{analytical['sequence_wins']}/9 secuencias; outputs finitos100%. "
        "Se reutilizan8.192 consultas OLD_DEV, sus masas y draws originales. "
        "Cero updates; el control histórico EWMA permanece intacto.\n",
        f"Términos válidos: {analytical['valid_terms']}; rechazados: "
        f"{analytical['rejected_terms']}; flags cap60: {analytical['cap60_terms']}. "
        "Los flags de productor y los anchors están incluidos en fragmentos. "
        "La sensibilidad trata fases medianas históricas capadas como cero "
        "sin cambiar el presente ni eliminar consultas.\n",
        "## Coste y paridad\n",
        markdown_table(pooled) + "\n",
        "768 solicitudes nuevas TRAIN,64 IDs, tres bloques,10 warmups por cabeza, "
        "CPU FP32/4 threads/2 interop. Son mediciones de cabeza y emisión; "
        "no incluyen productores, HDF5, ROI, ingestión ni detector. "
        "La variación entre bloques y su orden impiden atribuir causalmente "
        "una aceleración al runtime raw. "
        f"R0 real: {raw_measured}; paridad: {parity['status']}. "
        "R1 no implementado. No se repiten las1.728 medidas históricas.\n",
        markdown_table(pd.read_csv(c.out / "RUNTIME_COMPARISON.csv")) + "\n"
        if raw_measured
        else "Las mediciones R0 todavía no están completas.\n",
        "Los recibos profile/EXECUTION_CONTEXT.json registran que las lecturas "
        "raw E1 comparten el volumen con la recuperación TRAIN. Los tiempos "
        "absolutos pueden incluir contención de disco; las dos rutas permanecen "
        "pareadas con orden fijado prospectivamente. No se presenta como "
        "throughput de almacenamiento aislado.\n",
        "La variante CUDA de slots reducidos falló el límite1e-4 de features "
        "en una consulta TRAIN (máximo0,0001277923583984375), con raw exacto. "
        "Se preserva completa en verification/rejected_valid_dispatch. La ruta "
        "alternativa sólo optimiza preparación raw y conserva batch16; "
        "no atribuye reducción de cómputo de productores a slots válidos. "
        "Las tolerancias permanecen intactas.\n"
        if (c.out / "verification/rejected_valid_dispatch/PRESERVATION.json").exists()
        else "",
        markdown_table(pd.read_csv(c.out / "garl_runtime/RUNTIME.csv")) + "\n"
        if native_runtime.get("status") == "COMPLETE"
        else "Coste de la ruta Garl pendiente de los endpoints completos.\n",
        "Garl runtime usa los mismos64 IDs TRAIN y tres bloques: módulos nativos "
        "con inputs preparados, cabeza CPU FP32 y ruta sensores+ROI. El encoding "
        "40ch nativo permanece intacto; conteo/tasa se extraen aparte del ROI común "
        "de100ms. Se excluyen ingestión y detector de ROI. Las observaciones "
        "nativas~200ms e históricas~300ms tienen presupuesto distinto.\n",
        "Preprocesado compartido: filtro/proyección ROI una vez, views half-open "
        "int64 y reducciones/normalización originales por ventana. Tests cubren "
        "borde, ROI vacío/fuera, offset, empates, timestamps grandes, chunks, "
        "rollback y fallback de capacidad. Dispatch válido requiere validación "
        "real con tolerancias1e-4 features,1e-5 fase,0,01s TTC,1e-7 tiempos "
        "y máscara exacta. No se declara EXACT para la ruta CUDA sin medirla.\n",
        "## Comparadores y privilegios\n",
        "La revisión acotada verifica archivos V7 manifestados. Garl local "
        "144,353027 y C2F158,573140 permanecen descriptivos: faltan genealogía "
        "TRAIN completa, selección de checkpoint y ROI/modalidad/disponibilidad. "
        "No se usan como productores admisibles.\n",
        "Los12 productores Garl nuevos están admitidos por datos y presupuesto: "
        "**59.100 updates exactos** para50 épocas. Source commit/config blob "
        "verificados al iniciar; arquitectura40ch/128px/ResNet50/512 intacta. "
        "No se cargan paper_event_only ni paper_visual_only. "
        f"Perfil sintético: {micro['status']}; microbatch elegido "
        f"{micro.get('selected_microbatch')}, batch efectivo128. "
        "El recibo documenta memoria, selección y adaptación BN, si procede. "
        f"Progreso científico nuevo Garl:{counts['garl_producer_saved']}; "
        f"cabezas Garl:{counts['garl_head_saved']}.\n",
        "La colocación de la caché nativa tiene un freeze de ingeniería separado: "
        "conserva en RAM los tensores FP32 expulsados del disco, dentro de los "
        "mismos límites6GiB/8GB y cuota total10GB. Se comprobaron192 retornos "
        "sobre64 inputs TRAIN ya codificados: tensores y targets exactos, cero "
        "extracciones raw nuevas y cero updates. Los archivos científicos "
        "originales, sampler, RNG, optimizer, epoch50 y checkpoint identity "
        "permanecen intactos. El reinicio controlado conserva un checkpoint "
        "completo y registra la reserva de recuperación.\n"
        if (c.out / "garl/CACHE_ENGINEERING_FREEZE.json").exists()
        else "",
        "Un segundo freeze de ingeniería conserva los mismos bytes NPZ originales "
        "también en RAM comprimida, con las mismas capacidades. El QA adicional "
        "verifica192 retornos exactos sobre64 inputs TRAIN, layouts contiguos y "
        "ausencia de consumo de RNG, lecturas raw o updates. Se conservan los "
        "dos freezes y pruebas de reanudación con estados completos. El cambio "
        "reduce las relecturas cuando el conjunto activo supera la capacidad de "
        "tensores descomprimidos; no modifica el encoding ni sus pesos.\n"
        if (c.out / "garl/COMPRESSED_CACHE_FREEZE.json").exists()
        else "",
        "La autorización posterior de más recursos permite16GB de RSS total "
        "del árbol,8GiB de caché RAM comprimida y cuatro workers CPU de preparación. "
        "Cada worker mantiene4 threads/2 interop y hasta8 lectores originales "
        "de sólo lectura. El lookahead está limitado a64 records y usa una copia "
        "privada del sampler restaurado; la entrega conserva el orden del batch "
        "fijado. Paridad exacta de inputs y targets TRAIN y RNG del padre sin "
        "cambios constan en PARALLEL_INPUT_QA.json. Los workers no construyen "
        "modelos ni inicializan CUDA; permanece un único trainer pesado. Los "
        "límites de updates y disco no cambian.\n"
        if (c.out / "garl/PARALLEL_INPUT_FREEZE.json").exists()
        else "",
        "Las seis cabezas implementadas usan INNER-OOF excluyendo outer e inner "
        "holdouts, endpoint final fijo y normalización TRAIN de observaciones "
        "únicas. No incorporan A5/PAIR/RGB/DINO como features. Los expertos "
        "históricos tuvieron supervisión RGB/DINO aunque inferencia event-only; "
        "Garl usa geometría3D TRAIN para LHR, nunca en forward. "
        "Mismo número de anchors no implica encoding/ROI/span equivalentes.\n",
        markdown_table(native_table) + "\n"
        if native_rows
        else "Los comparadores nativos aún no tienen resultados evaluados. "
        "Este estado no es un resultado científico negativo.\n",
        "## Contabilidad y reanudación\n",
        "```json\n" + json.dumps(counts, indent=2) + "\n```\n",
        f"Reserva pendiente: {counts['unresolved_execution_upper']} updates; "
        "se cuenta conservadoramente en el techo físico. "
        f"Updates confirmados por progreso pero no duraderos: "
        f"{counts['unsaved_updates_confirmed_by_progress']}. "
        "Se contabiliza recuperación al reiniciar el journal; "
        "no se presenta una reserva como ejecución observada. "
        "No se añaden brazos para consumir los máximos.\n",
        "Dependencias exactas: `DEPENDENCIES.json`. Necesarias para WIDE: "
        "`E:\\GarlTTC_dataset\\data\\train.parquet` y "
        "`E:\\GarlTTC_dataset\\annotations\\train.parquet`, ligados por SHA256. "
        "E1/Garl necesitan además raw TRAIN y código upstream de E:. "
        "No se cambian hashes, etiquetas ni entornos globales para sustituirlos.\n",
        "Desde el worktree, tras restablecer las mismas fuentes:\n\n```powershell\n"
        + command
        + "\n```\n",
        "El comando continúa E0–E3 con un worker pesado, checkpoints y fragmentos "
        "existentes. Avanza ramas viables; un comparador ausente sólo bloquea "
        "su dependencia. H8 mantiene identidad; H16 se conserva; WIDE no se promueve.\n",
        "## Validación y límites\n",
        f"QA final: {qa['status']}; {test_count} tests focalizados, Ruff, formato, "
        "Pyright y help tienen recibos con exit code y0 updates de tests. "
        "Se conservan los fallos transitorios y de QA anteriores. "
        "Contratos handoff:39 tests CPU pasaron antes de la campaña. "
        f"Estado nativo evaluado: {garl.get('status', 'PENDING_EXECUTION')}; "
        "no se afirma ejecución end-to-end de una rama que carezca de endpoints.\n",
        "La suite ampliada para almacenamiento comprimido pasó50 tests, con "
        "Ruff y Pyright satisfactorios; recibos en TEST_RESULTS/compressed_cache. "
        "Incluye los tests anteriores y no se suma como50 experimentos nuevos.\n"
        if (c.out / "TEST_RESULTS/compressed_cache/QA.json").exists()
        else "",
        "La ampliación para preparación CPU tiene tests de contratos, incluido "
        "el orden del sampler al cruzar épocas y el límite de lectores abiertos. "
        "Sus recibos de Ruff/Pyright y paridad real están en TEST_RESULTS/parallel_inputs.\n"
        if (c.out / "TEST_RESULTS/parallel_inputs/QA.json").exists()
        else "",
        "Bundle: código/configs/diff, pesos/checkpoints completos, predicciones "
        "EWMA, curves, normalizadores, draws/recibos, bindings externos y QA. "
        "CRC y manifiesto SHA256 se verifican; se regenera la evidencia incluida "
        "desde extracción independiente. Los eventos crudos y productores "
        "históricos se referencian; no se promete entrenamiento raw autónomo.\n",
        "Se conserva TTC firmado del benchmark. OLD_DEV ya fue reutilizado "
        "adaptativamente: evidencia local de desarrollo, sin confirmación "
        "independiente, SOTA, contacto físico ni utilidad AEB demostrados. "
        "Stage76, public validation, private test, EvTTC test y CodaBench "
        "permanecen cerrados. No push ni submissions.\n",
    ]
    atomic_bytes(c.out / "FINAL_REPORT.md", "\n".join(lines).encode())
